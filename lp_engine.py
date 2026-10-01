"""
lp_engine.py
============
Core engine for the Linear Programming Solver toolkit.

Responsibilities:
  * Parse friendly constraint strings such as "2x1 + 3x2 <= 100"
    (the unicode forms "≤", "≥" and "−" are accepted too).
  * Build an LPModel: 1-8 decision variables, 1-12 constraints,
    non-negativity assumed automatically.
  * Solve it with SciPy's HiGHS solver (scipy.optimize.linprog).
  * Reproduce the "by hand" corner-point method for ANY number of
    variables: every combination of n boundaries (constraint lines /
    planes plus the axes x_j = 0) is intersected, the feasible
    intersection points are kept, and Z is evaluated at each corner.
    With 2 variables this is exactly the graphical method from the
    teaching slides; with 3-8 variables it is the same idea in
    higher dimensions.
  * Detect infeasible / unbounded / multiple-optimal / degenerate cases.
  * Report sensitivity information from HiGHS: the shadow price of every
    constraint and the reduced cost of every variable left at zero.
  * Produce a human-readable, step-by-step explanation of the solution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations, islice
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

MAX_VARIABLES = 8
MAX_CONSTRAINTS = 12
TOL = 1e-7
CORNER_ROWS_SHOWN = 10          # corner points listed in the report for 3+ variables


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Constraint:
    coeffs: List[float]
    sense: str          # '<=', '>=' or '='
    rhs: float
    label: str = ""

    def render(self, var_names: List[str], include_label: bool = True) -> str:
        lhs = _join_terms(self.coeffs, var_names) or "0"
        tag = f"   ({self.label})" if (include_label and self.label) else ""
        return f"{lhs} {self.sense} {_num(self.rhs)}{tag}"

    def name(self, var_names: List[str]) -> str:
        """Short display name: the label if given, else the equation."""
        if self.label:
            return self.label
        return self.render(var_names, include_label=False)


@dataclass
class LPModel:
    var_names: List[str]
    objective: List[float]
    goal: str                       # 'max' or 'min'
    constraints: List[Constraint]

    def __post_init__(self):
        self.goal = self.goal.strip().lower()
        if self.goal not in ("max", "min"):
            raise ValueError("goal must be 'max' or 'min'.")
        n = len(self.var_names)
        if not (1 <= n <= MAX_VARIABLES):
            raise ValueError(
                f"This tool supports 1 to {MAX_VARIABLES} decision variables (received {n})."
            )
        if len(set(self.var_names)) != n:
            raise ValueError("Variable names must be unique.")
        if len(self.constraints) == 0:
            raise ValueError("At least one constraint is required.")
        if len(self.constraints) > MAX_CONSTRAINTS:
            raise ValueError(
                f"This tool supports up to {MAX_CONSTRAINTS} constraints "
                f"(received {len(self.constraints)})."
            )
        if len(self.objective) != n:
            raise ValueError(
                f"Objective has {len(self.objective)} coefficients but there are {n} variables."
            )
        if not all(math.isfinite(v) for v in self.objective):
            raise ValueError("Objective coefficients must be ordinary finite numbers.")
        for i, c in enumerate(self.constraints, start=1):
            if len(c.coeffs) != n:
                raise ValueError(f"Constraint {i} has a different number of coefficients than variables.")
            if c.sense not in ("<=", ">=", "="):
                raise ValueError(f"Constraint {i} has an unknown sign '{c.sense}'.")
            if not all(math.isfinite(v) for v in c.coeffs) or not math.isfinite(c.rhs):
                raise ValueError(f"Constraint {i} contains a value that is not a finite number.")

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    def objective_expr(self) -> str:
        return _join_terms(self.objective, self.var_names) or "0"

    def objective_str(self) -> str:
        return f"{self.goal.capitalize()}imize  Z = {self.objective_expr()}"


@dataclass
class LPSolution:
    status: str                                  # optimal | multiple_optimal | infeasible | unbounded | error
    x: Optional[List[float]]
    objective_value: Optional[float]
    slack: Optional[List[float]]
    binding: Optional[List[bool]]
    message: str
    corner_points: Optional[List[Tuple[float, ...]]] = None   # every vertex of the feasible region
    corner_values: Optional[List[float]] = None               # Z at each vertex
    candidates_checked: int = 0                                # boundary combinations tried
    degenerate: bool = False                                   # more boundaries than needed meet at the optimum
    shadow_prices: Optional[List[float]] = None                # dZ per +1 unit of each constraint's RHS
    reduced_costs: Optional[List[float]] = None                # dZ per unit a zero variable is forced in

    @property
    def is_solved(self) -> bool:
        return self.status in ("optimal", "multiple_optimal")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

_TERM_RE = re.compile(r"^([+-]?\d*\.?\d*)\*?([a-zA-Z_]\w*)$")

_UNICODE_FIXES = {
    "\u2212": "-",      # − minus sign
    "\u2013": "-",      # – en dash
    "\u2264": "<=",     # ≤
    "\u2265": ">=",     # ≥
    "\u00d7": "*",      # ×
    "\u00b7": "*",      # ·
}


def _normalise(text: str) -> str:
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    # accept the reversed spellings people often type
    text = text.replace("=<", "<=").replace("=>", ">=").replace("==", "=")
    return text


def parse_linear_expression(expr: str, var_names: List[str]) -> List[float]:
    """Parse '2x1 + 3.5x2 - x3' into a coefficient list ordered like var_names."""
    expr = _normalise(expr).replace(" ", "")
    if not expr:
        raise ValueError("Empty expression.")
    coeffs = {v: 0.0 for v in var_names}
    tokens = re.findall(r"[+-]?[^+-]+", expr)
    for tok in tokens:
        m = _TERM_RE.match(tok)
        if not m:
            if re.fullmatch(r"[+-]?\d*\.?\d+", tok):
                raise ValueError(
                    f"'{tok}' is a plain number on the variable side. Move constants to the "
                    "right-hand side, e.g.  2x + y <= 100."
                )
            raise ValueError(f"Could not parse term '{tok}'.")
        coef_str, var = m.group(1), m.group(2)
        if var not in coeffs:
            raise ValueError(f"Unknown variable '{var}'. Expected one of {var_names}.")
        if coef_str in ("", "+"):
            coef = 1.0
        elif coef_str == "-":
            coef = -1.0
        else:
            try:
                coef = float(coef_str)
            except ValueError:
                raise ValueError(f"Could not read the number in term '{tok}'.") from None
        coeffs[var] += coef
    return [coeffs[v] for v in var_names]


def parse_constraint(line: str, var_names: List[str]) -> Constraint:
    """Parse 'x1 + 2x2 <= 100 # label' into a Constraint."""
    label = ""
    if "#" in line:
        line, label = line.split("#", 1)
        label = label.strip()
    line = _normalise(line)
    # A strict '<' or '>' can't be modelled exactly in LP; treat it as '<=' / '>='.
    for sense, stored in (("<=", "<="), (">=", ">="), ("=", "="), ("<", "<="), (">", ">=")):
        if sense in line:
            lhs, rhs = line.split(sense, 1)
            if not lhs.strip():
                raise ValueError("The left-hand side is empty -- put the variables before the sign.")
            coeffs = parse_linear_expression(lhs, var_names)
            try:
                rhs_val = float(rhs.strip())
            except ValueError:
                raise ValueError(
                    f"Right-hand side '{rhs.strip()}' must be a single number. "
                    "Keep all variables on the left, e.g.  x - y >= 0."
                ) from None
            return Constraint(coeffs=coeffs, sense=stored, rhs=rhs_val, label=label)
    raise ValueError(f"Constraint '{line.strip()}' must contain <=, >= or =.")


def parse_objective(text: str, var_names: List[str]) -> List[float]:
    """Accept either '40x + 30y' or '40, 30' for the objective coefficients."""
    text = _normalise(text).strip()
    if "=" in text:                       # tolerate "Z = 40x + 30y" / "max Z = ..."
        text = text.split("=", 1)[1]
    else:                                 # tolerate a leading "max" / "min" word
        text = re.sub(r"^\s*(max|min)\w*\s+", "", text, flags=re.IGNORECASE)
    if not text.strip():
        raise ValueError("The objective is empty.")
    if re.search(r"[a-zA-Z_]", text):
        return parse_linear_expression(text, var_names)
    try:
        values = [float(p) for p in re.split(r"[,;\s]+", text.strip()) if p]
    except ValueError:
        raise ValueError("Enter the objective as  40x + 30y  or as numbers like  40, 30.") from None
    if len(values) != len(var_names):
        raise ValueError(f"Expected {len(var_names)} objective coefficients, got {len(values)}.")
    return values


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _num(x: float) -> str:
    r = round(x)
    if abs(x - r) < 1e-9:
        return str(int(r))
    return f"{x:.4g}"


def _term(a: float, v: str) -> str:
    if abs(a - 1) < 1e-9:
        return f"+ {v}"
    if abs(a + 1) < 1e-9:
        return f"- {v}"
    sign = "+" if a >= 0 else "-"
    return f"{sign} {_num(abs(a))}{v}"


def _join_terms(coeffs: Sequence[float], var_names: Sequence[str]) -> str:
    """Join signed terms into 'a - b + c' form, no double signs, no leading '+'."""
    parts = [_term(a, v) for a, v in zip(coeffs, var_names) if abs(a) > TOL]
    text = " ".join(parts)
    if text.startswith("+ "):
        text = text[2:]
    elif text.startswith("- "):
        text = "-" + text[2:]
    return text


def format_point(point: Sequence[float], var_names: Sequence[str] | None = None) -> str:
    """(40, 20) for 2 variables; (t = 20, c = 0, d = 10) when names are given."""
    if var_names is None:
        return "(" + ", ".join(_num(v) for v in point) + ")"
    return "(" + ", ".join(f"{n} = {_num(v)}" for n, v in zip(var_names, point)) + ")"


# --------------------------------------------------------------------------- #
# Corner-point (vertex) enumeration -- works for any number of variables
# --------------------------------------------------------------------------- #

def _arrays(model: LPModel):
    A = np.array([c.coeffs for c in model.constraints], dtype=float)
    b = np.array([c.rhs for c in model.constraints], dtype=float)
    senses = [c.sense for c in model.constraints]
    return A, b, senses


def _feasible_mask(P: np.ndarray, A: np.ndarray, b: np.ndarray, senses: List[str],
                   upper: Optional[np.ndarray] = None, tol: float = 1e-6) -> np.ndarray:
    """Vectorised check: which rows of P satisfy every constraint (and x >= 0)?"""
    mask = (P >= -tol).all(axis=1)
    if upper is not None:
        mask &= (P <= upper + tol * np.maximum(1.0, np.abs(upper))).all(axis=1)
    lhs = P @ A.T
    scale = tol * np.maximum(1.0, np.abs(b))
    for i, s in enumerate(senses):
        if s == "<=":
            mask &= lhs[:, i] <= b[i] + scale[i]
        elif s == ">=":
            mask &= lhs[:, i] >= b[i] - scale[i]
        else:
            mask &= np.abs(lhs[:, i] - b[i]) <= scale[i]
    return mask


def feasible_vertices(model: LPModel, upper: Optional[Sequence[float]] = None,
                      chunk: int = 20000) -> Tuple[np.ndarray, int]:
    """Return (vertices, number_of_combinations_tried).

    Each vertex of an n-variable feasible region is a point where n boundary
    hyperplanes meet. We try every combination of n boundaries drawn from
    the constraints and the axes x_j = 0 (plus the box x_j = upper_j when
    `upper` is given -- used by the plots to clip unbounded regions), solve
    the n x n system, and keep the solutions that satisfy every constraint.
    Worst case (8 variables, 12 constraints) is C(20, 8) = 125,970 small
    systems, solved in vectorised batches.
    """
    n = model.n_vars
    A, b, senses = _arrays(model)

    H_parts, h_parts = [A, np.eye(n)], [b, np.zeros(n)]
    up = None
    if upper is not None:
        up = np.asarray(upper, dtype=float)
        H_parts.append(np.eye(n))
        h_parts.append(up)
    H = np.vstack(H_parts)
    h = np.concatenate(h_parts)

    total = math.comb(len(H), n)
    row_norms = np.linalg.norm(H, axis=1)
    combos = combinations(range(len(H)), n)
    found = []
    while True:
        batch = list(islice(combos, chunk))
        if not batch:
            break
        idx = np.array(batch)
        M = H[idx]                                  # (k, n, n)
        r = h[idx]                                  # (k, n)
        norms = row_norms[idx].prod(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(np.linalg.det(M)) / np.where(norms > 0, norms, np.inf)
        ok = ratio > 1e-9                           # boundaries genuinely cross at one point
        if not ok.any():
            continue
        P = np.linalg.solve(M[ok], r[ok][..., None])[..., 0]
        found.append(P[_feasible_mask(P, A, b, senses, up)])

    if not found or sum(len(f) for f in found) == 0:
        return np.empty((0, n)), total

    pts = np.round(np.vstack(found), 6) + 0.0       # "+ 0.0" turns -0.0 into 0.0
    pts = np.unique(pts, axis=0)
    return pts, total


def order_polygon(points: np.ndarray) -> np.ndarray:
    """Order 2-D points counter-clockwise around their centroid (for drawing)."""
    if len(points) < 3:
        return points
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    return points[np.argsort(ang)]


def enumerate_corner_points(model: LPModel):
    """The corner-point method for any number of variables.

    Returns (corner_points, corner_values, combinations_tried).
    2 variables  -> points ordered around the region (ready to draw a polygon).
    3+ variables -> points ordered best Z first (ready to list in a report).
    """
    pts, total = feasible_vertices(model)
    if len(pts) == 0:
        return [], [], total

    obj = np.array(model.objective, dtype=float)
    if model.n_vars == 2:
        pts = order_polygon(pts)
    else:
        vals = pts @ obj
        order = np.argsort(-vals if model.goal == "max" else vals, kind="stable")
        pts = pts[order]
    values = (pts @ obj).tolist()
    return [tuple(float(v) for v in p) for p in pts], values, total


def _check_alternate_optima(model: LPModel, obj_val: float, corner_values) -> bool:
    if not corner_values:
        return False
    best = max(corner_values) if model.goal == "max" else min(corner_values)
    tol = 1e-6 * max(1.0, abs(best))
    count = sum(1 for v in corner_values if abs(v - best) < tol and abs(v - obj_val) < tol)
    return count > 1


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #

def _build_matrices(model: LPModel):
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for c in model.constraints:
        if c.sense == "<=":
            A_ub.append(c.coeffs)
            b_ub.append(c.rhs)
        elif c.sense == ">=":
            A_ub.append([-a for a in c.coeffs])
            b_ub.append(-c.rhs)
        else:
            A_eq.append(c.coeffs)
            b_eq.append(c.rhs)
    return (A_ub or None, b_ub or None, A_eq or None, b_eq or None)


def solve(model: LPModel) -> LPSolution:
    n = model.n_vars
    c = np.array(model.objective, dtype=float)
    c_signed = -c if model.goal == "max" else c   # linprog always minimizes
    A_ub, b_ub, A_eq, b_eq = _build_matrices(model)
    bounds = [(0, None)] * n

    # The by-hand corner-point method runs for every model, so the report can
    # show the corners even when the problem turns out infeasible or unbounded.
    corners, corner_vals, tried = enumerate_corner_points(model)
    common = dict(corner_points=corners, corner_values=corner_vals, candidates_checked=tried)

    res = linprog(c_signed, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if res.status == 2:
        return LPSolution(
            status="infeasible", x=None, objective_value=None, slack=None, binding=None,
            message="No point satisfies every constraint at the same time: the problem is infeasible.",
            **common,
        )
    if res.status == 3:
        return LPSolution(
            status="unbounded", x=None, objective_value=None, slack=None, binding=None,
            message="The objective can be improved without limit inside the feasible region: "
                    "the problem is unbounded.",
            **common,
        )
    if res.status != 0:
        return LPSolution(
            status="error", x=None, objective_value=None, slack=None, binding=None,
            message=f"Solver could not find a solution ({res.message}).",
            **common,
        )

    x = [max(0.0, v) for v in res.x.tolist()]   # clip tiny negative numerical noise
    x = [0.0 if abs(v) < 1e-9 else v for v in x]
    obj_val = float(np.dot(c, x))

    slacks, binding = [], []
    for cons in model.constraints:
        lhs = sum(a * xi for a, xi in zip(cons.coeffs, x))
        if cons.sense == "<=":
            s = cons.rhs - lhs
        elif cons.sense == ">=":
            s = lhs - cons.rhs
        else:
            s = 0.0
        tol = 1e-6 * max(1.0, abs(cons.rhs))
        slacks.append(0.0 if abs(s) < tol else s)
        binding.append(abs(s) < tol)

    shadow, reduced = _sensitivity(model, res)

    # Degenerate: more boundaries pass through the optimum than are needed to pin it down.
    active = sum(binding) + sum(1 for v in x if abs(v) < 1e-9)
    degenerate = active > n

    status = "multiple_optimal" if _check_alternate_optima(model, obj_val, corner_vals) else "optimal"
    message = (
        "Optimal solution found."
        if status == "optimal"
        else "Multiple optimal solutions exist: more than one corner point (and every point "
             "between them) achieves the same best value of Z."
    )

    return LPSolution(
        status=status, x=x, objective_value=obj_val, slack=slacks, binding=binding,
        message=message, degenerate=degenerate, shadow_prices=shadow, reduced_costs=reduced,
        **common,
    )


def _clean(v: float) -> float:
    return 0.0 if abs(v) < 1e-9 else float(v)


def _sensitivity(model: LPModel, res) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Shadow prices and reduced costs in the user's own terms (for Z, max or min).

    HiGHS reports marginals for the problem it actually solved: always a
    minimisation, with every '>=' row negated into a '<=' row. Undoing those
    two sign flips gives dZ / d(rhs) for each constraint as the user wrote it,
    and dZ per unit of a zero variable forced into the plan.
    """
    try:
        ineq = list(res.ineqlin.marginals) if getattr(res, "ineqlin", None) is not None else []
        eq = list(res.eqlin.marginals) if getattr(res, "eqlin", None) is not None else []
        lower = list(res.lower.marginals)
    except AttributeError:
        return None, None                          # older SciPy without marginals
    goal_sign = -1.0 if model.goal == "max" else 1.0
    shadow, i_ub, i_eq = [], 0, 0
    for c in model.constraints:
        if c.sense == "=":
            m, i_eq = eq[i_eq], i_eq + 1
        else:
            m, i_ub = ineq[i_ub], i_ub + 1
        row_sign = -1.0 if c.sense == ">=" else 1.0
        shadow.append(_clean(goal_sign * row_sign * m))
    reduced = [_clean(goal_sign * m) for m in lower]
    return shadow, reduced


# --------------------------------------------------------------------------- #
# Step-by-step explanation
# --------------------------------------------------------------------------- #

def _corner_step(model: LPModel, solution: LPSolution) -> str:
    n, m = model.n_vars, len(model.constraints)
    corners = solution.corner_points or []

    if n == 2:
        header = "Step 2 -- Corner points of the feasible region"
        if not corners:
            return (header + "\nNo pair of boundary lines produces a point that satisfies every "
                    "constraint: the feasible region is empty.")
        rows = [f"   {format_point(p)}  ->  Z = {_num(v)}"
                for p, v in zip(corners, solution.corner_values)]
        return (header + "\nEvery boundary line is intersected with every other one; points that "
                "satisfy all constraints are kept, and Z is evaluated at each:\n" + "\n".join(rows))

    header = "Step 2 -- Corner points (vertices) of the feasible region"
    if n == 1:
        intro = (
            f"With one decision variable the feasible region is an interval on the "
            f"{model.var_names[0]} number line, and its 'corners' are the interval's end points. "
            f"Each of the {m + 1} boundaries (the constraints and {model.var_names[0]} = 0) was "
            "solved on its own"
        )
    else:
        what = "3-D space" if n == 3 else f"{n}-dimensional space"
        drawable = ("can be drawn as a solid" if n == 3 else "can't be drawn on paper")
        intro = (
            f"With {n} decision variables the region lives in {what}, so it {drawable} "
            f"-- and the corner-point idea works exactly as in 2-D. Each corner is a point where "
            f"{n} boundaries meet (constraints or axes {model.var_names[0]} = 0, ...). All "
            f"C({m}+{n}, {n}) = {solution.candidates_checked:,} combinations of {n} boundaries "
            "were intersected"
        )
    if not corners:
        return (header + "\n" + intro + ", and none of the intersection points satisfies every "
                "constraint: the feasible region is empty.")

    shown = min(CORNER_ROWS_SHOWN, len(corners))
    rows = [f"   {i + 1:>2}. {format_point(p, model.var_names)}  ->  Z = {_num(v)}"
            for i, (p, v) in enumerate(zip(corners[:shown], solution.corner_values[:shown]))]
    more = (f"\n   ... and {len(corners) - shown} more corner(s) with a "
            f"{'lower' if model.goal == 'max' else 'higher'} Z." if len(corners) > shown else "")
    return (header + "\n" + intro + f"; {len(corners)} of them satisfy every constraint. "
            f"Z at the best {shown} corner(s):\n" + "\n".join(rows) + more
            + "\nHiGHS (SciPy's simplex / interior-point solver) is used to confirm the optimum.")


def step_by_step(model: LPModel, solution: LPSolution) -> List[str]:
    steps = []

    model_lines = [model.objective_str(), "subject to:"]
    model_lines += ["   " + c.render(model.var_names) for c in model.constraints]
    model_lines.append("   " + ", ".join(model.var_names) + " >= 0")
    steps.append("Step 1 -- The model\n" + "\n".join(model_lines))

    steps.append(_corner_step(model, solution))

    if solution.is_solved:
        assignment = ", ".join(f"{v} = {_num(x)}" for v, x in zip(model.var_names, solution.x))
        extra = ""
        if solution.status == "multiple_optimal":
            extra += "\n   (Multiple optimal solutions exist -- any point between the tying corners is equally optimal.)"
        if solution.degenerate:
            extra += "\n   (Degenerate corner -- more boundaries pass through this point than are needed to fix it.)"
        steps.append(
            "Step 3 -- Optimal solution\n"
            f"   {assignment}\n"
            f"   Z = {_num(solution.objective_value)}" + extra
        )
        binding_names = [c.name(model.var_names)
                         for c, b in zip(model.constraints, solution.binding) if b]
        slack_rows = [
            f"   {c.label or f'Constraint {i+1}'}: "
            + ("surplus" if c.sense == ">=" else "slack") + f" = {_num(s)}"
            for i, (c, s) in enumerate(zip(model.constraints, solution.slack))
        ]
        steps.append(
            "Step 4 -- Binding constraints and slack\n"
            + (f"Binding (exactly at the limit): {', '.join(binding_names)}\n" if binding_names
               else "No constraint is exactly binding at this optimum.\n")
            + "\n".join(slack_rows)
        )
        sens = _sensitivity_step(model, solution)
        if sens:
            steps.append(sens)
    else:
        steps.append(f"Step 3 -- Result\n   {solution.message}")

    return steps


def _sensitivity_step(model: LPModel, solution: LPSolution) -> Optional[str]:
    if solution.shadow_prices is None:
        return None
    names = [c.label or f"Constraint {i + 1}" for i, c in enumerate(model.constraints)]
    width = min(max(len(n) for n in names), 32)
    rows = []
    for name, sp in zip(names, solution.shadow_prices):
        if sp == 0:
            effect = "not limiting -- changing it slightly leaves Z unchanged"
        else:
            effect = f"+1 unit on the right-hand side {'raises' if sp > 0 else 'lowers'} Z by {_num(abs(sp))}"
        rows.append(f"   {name[:width]:<{width}}  {_num(sp):>9}   {effect}")
    lines = ["Step 5 -- Sensitivity (shadow prices)",
             "How much Z changes when one constraint's right-hand side grows by 1 unit, holding "
             "the others fixed (valid for small changes):"] + rows
    rc_rows = [f"   {v}: forcing 1 unit into the plan changes Z by {_num(rc)}"
               for v, x, rc in zip(model.var_names, solution.x, solution.reduced_costs or [])
               if abs(x) < 1e-9 and rc != 0]
    if rc_rows:
        worse = "lower" if model.goal == "max" else "higher"
        lines.append(f"Reduced costs -- variables left at 0 (using them would make Z {worse}):")
        lines += rc_rows
    if solution.degenerate:
        lines.append("   Note: the optimum is degenerate, so some shadow prices hold in one direction only.")
    return "\n".join(lines)
