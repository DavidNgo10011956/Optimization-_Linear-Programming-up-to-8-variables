"""Input guide tables, example problems and terminal banner for the LP solver.

Laid out the same way as integral_input_guide.py so every program in the
toolkit shows its guide in the same format. The layout helpers (WIDTH, hr,
format_table) live in lp_utils.py and are shared with the integral guide:
    hr() · banner(show_special) · input / symbol / example tables ·
    a special-cases guide · a detector that returns a plain-English note.
"""

from __future__ import annotations

from lp_engine import MAX_CONSTRAINTS, MAX_VARIABLES   # single source of truth
from lp_utils import WIDTH, hr, format_table


# ---------------------------------------------------------------------------
# Special-case catalogue  (shown when the user wants the advanced guide)
# ---------------------------------------------------------------------------

SPECIAL_CASES: list[dict[str, str]] = [
    {
        "name":    "Infeasible",
        "symbol":  "∅ region",
        "meaning": "The constraints contradict each other — no plan satisfies all of them.",
        "spot":    "No green region is drawn and no optimal solution is reported.",
        "example": "x + y <= 10  together with  x + y >= 20",
    },
    {
        "name":    "Unbounded",
        "symbol":  "Z → ∞",
        "meaning": "The region is open in a direction that keeps improving Z forever.",
        "spot":    "Usually a missing capacity limit (max) or minimum requirement (min).",
        "example": "max Z = x + y  with only  x - y <= 10",
    },
    {
        "name":    "Multiple optima",
        "symbol":  "Z tie",
        "meaning": "Two or more corners give the same best Z — so does every point between them.",
        "spot":    "The objective line is parallel to a binding constraint.",
        "example": "max Z = 2x + 4y  with  x + 2y <= 10,  x <= 6",
    },
    {
        "name":    "Degenerate",
        "symbol":  "extra line",
        "meaning": "More boundaries pass through the optimal corner than are needed to fix it.",
        "spot":    "A binding constraint also has a variable equal to 0 at the optimum.",
        "example": "max Z = 3x + 2y  with  x + y <= 4,  x <= 4",
    },
    {
        "name":    "Redundant",
        "symbol":  "slack > 0",
        "meaning": "A constraint that never touches the feasible region — removing it changes nothing.",
        "spot":    "Its line lies completely outside the green region; its slack is always positive.",
        "example": "x + y <= 8  makes  2x + 2y <= 20  redundant",
    },
]


def special_case_guide() -> str:
    """Return a formatted guide for the special / advanced LP outcomes."""
    lines: list[str] = [
        "",
        "SPECIAL CASES  (advanced — most textbook problems have one clear optimum)",
        "-" * 72,
        "These outcomes are detected automatically. The program still reports every",
        "corner point it found and explains the result in plain English.",
        "",
    ]
    for case in SPECIAL_CASES:
        lines.append(f"  {case['name']:<18}  {case['symbol']}")
        lines.append(f"  {'':18}  Meaning     : {case['meaning']}")
        lines.append(f"  {'':18}  How to spot : {case['spot']}")
        lines.append(f"  {'':18}  Example     : {case['example']}")
        lines.append("")
    return "\n".join(lines)


def is_special_case(status: str, degenerate: bool = False) -> str | None:
    """Return a plain-English note if the solver status is a special case."""
    checks = {
        "infeasible":       "Infeasible — the constraints contradict each other, so no plan satisfies "
                            "them all. Check the direction of each <= / >= sign.",
        "unbounded":        "Unbounded — Z can be improved forever. A real problem almost always has a "
                            "limit you haven't entered yet (capacity, budget, demand, ...).",
        "multiple_optimal": "Multiple optima — the objective is parallel to a binding constraint, so "
                            "several plans tie for the best Z. Any of them is a correct answer.",
    }
    notes = [checks[status]] if status in checks else []
    if degenerate:
        notes.append("Degenerate — more boundaries meet at the optimal corner than are needed. "
                     "The answer is still correct; the simplex method may just pause there.")
    return "\n".join(notes) if notes else None


# ---------------------------------------------------------------------------
# Example problems  (Basic = teaching-slide examples, larger ones use 3-8 variables)
# ---------------------------------------------------------------------------

def example_problems() -> list[dict]:
    return [
        # Basic -- from the "Linear Programming -- Teaching Notes" slide deck
        {
            "level": "Basic",
            "name": "Worked example — The Factory Problem",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [40, 30],
            "constraints": [
                "2x + y <= 100 # Machine 1 capacity",
                "x + 2y <= 80 # Machine 2 capacity",
            ],
        },
        {
            "level": "Basic",
            "name": "Exercise 1 — Warm-Up",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [5, 4],
            "constraints": [
                "x + y <= 8",
                "2x + y <= 10",
            ],
        },
        {
            "level": "Basic",
            "name": "Exercise 2 — Production Planning",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [60, 45],
            "constraints": [
                "3x + 2y <= 120 # labour",
                "2x + 4y <= 160 # material",
            ],
        },
        {
            "level": "Basic",
            "name": "Exercise 3 — Farmer's Land Allocation",
            "var_names": ["w", "c"],
            "goal": "max",
            "objective": [500, 400],
            "constraints": [
                "w + c <= 100 # hectares available",
                "3w + 2c <= 240 # labour units",
            ],
        },
        # Intermediate -- 3 and 4 variables
        {
            "level": "Intermediate",
            "name": "Furniture Workshop (tables, chairs, desks)",
            "var_names": ["t", "c", "d"],
            "goal": "max",
            "objective": [70, 30, 90],
            "constraints": [
                "4t + 2c + 6d <= 240 # carpentry hours",
                "2t + c + 2d <= 100 # finishing hours",
                "t + c + d <= 45 # storage space",
            ],
        },
        {
            "level": "Intermediate",
            "name": "Diet Problem — cheapest healthy mix",
            "var_names": ["oats", "milk", "eggs", "beans"],
            "goal": "min",
            "objective": [0.3, 0.9, 0.5, 0.6],
            "constraints": [
                "10oats + 8milk + 13eggs + 15beans >= 60 # protein (g)",
                "380oats + 150milk + 155eggs + 340beans >= 2000 # energy (kcal)",
                "10oats + 0milk + 0eggs + 16beans >= 25 # fibre (g)",
                "oats <= 4 # max servings of oats",
            ],
        },
        # Advanced -- 5 to 8 variables
        {
            "level": "Advanced",
            "name": "Investment Portfolio (5 funds, $100k)",
            "var_names": ["bonds", "blue", "growth", "intl", "cash"],
            "goal": "max",
            "objective": [0.04, 0.07, 0.11, 0.095, 0.02],
            "constraints": [
                "bonds + blue + growth + intl + cash = 100 # total invested ($k)",
                "growth + intl <= 40 # high-risk cap",
                "bonds + cash >= 25 # safe-asset floor",
                "cash >= 5 # emergency cash",
                "growth - blue <= 0 # growth no more than blue-chip",
            ],
        },
        {
            "level": "Advanced",
            "name": "Transportation — 2 factories to 3 stores",
            "var_names": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "goal": "min",
            "objective": [4, 6, 9, 5, 3, 7],
            "constraints": [
                "a1 + a2 + a3 <= 250 # factory A supply",
                "b1 + b2 + b3 <= 300 # factory B supply",
                "a1 + b1 >= 150 # store 1 demand",
                "a2 + b2 >= 180 # store 2 demand",
                "a3 + b3 >= 170 # store 3 demand",
            ],
        },
        {
            "level": "Advanced",
            "name": "Bakery Product Mix (8 products)",
            "var_names": ["bread", "rolls", "bagel", "croiss", "muffin", "cookie", "cake", "pie"],
            "goal": "max",
            "objective": [2.0, 1.5, 1.8, 2.6, 2.2, 1.2, 9.0, 7.5],
            "constraints": [
                "0.5bread + 0.3rolls + 0.4bagel + 0.35croiss + 0.25muffin + 0.1cookie + 1.2cake + 0.9pie <= 180 # flour (kg)",
                "0.05croiss + 0.04muffin + 0.03cookie + 0.3cake + 0.25pie + 0.01bread <= 30 # butter (kg)",
                "0.1bread + 0.08rolls + 0.12bagel + 0.2croiss + 0.15muffin + 0.05cookie + 0.6cake + 0.5pie <= 90 # oven hours",
                "0.05bread + 0.05rolls + 0.1bagel + 0.25croiss + 0.1muffin + 0.04cookie + 0.8cake + 0.6pie <= 80 # labour hours",
                "bread >= 60 # standing bread orders",
                "cake <= 20 # cake demand",
                "pie <= 25 # pie demand",
                "cookie >= 100 # cookie contract",
            ],
        },
        # Special -- the unusual outcomes described in the special-cases guide
        {
            "level": "Special",
            "name": "Infeasible example",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [1, 1],
            "constraints": [
                "x + y <= 10",
                "x + y >= 20",
            ],
        },
        {
            "level": "Special",
            "name": "Unbounded example",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [1, 1],
            "constraints": [
                "x - y <= 10",
            ],
        },
        {
            "level": "Special",
            "name": "Degenerate optimum",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [3, 2],
            "constraints": [
                "x + y <= 4",
                "x <= 4",
            ],
        },
        {
            "level": "Special",
            "name": "Multiple optimal solutions",
            "var_names": ["x", "y"],
            "goal": "max",
            "objective": [2, 4],
            "constraints": [
                "x + 2y <= 10",
                "x <= 6",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Standard input / symbol guides
# ---------------------------------------------------------------------------

def input_guide_rows() -> list[dict[str, str]]:
    """Professional guide table to reduce input errors."""
    return [
        {"Mathematical expression": "Max Z = 40x + 30y",   "Meaning": "objective to maximise",            "Enter as": "max, then 40x + 30y"},
        {"Mathematical expression": "Min C = 3a + 5b",     "Meaning": "objective to minimise",            "Enter as": "min, then 3a + 5b"},
        {"Mathematical expression": "Z = 40x + 30y",       "Meaning": "objective as a number list",       "Enter as": "40, 30"},
        {"Mathematical expression": "2x + y ≤ 100",        "Meaning": "at most (capacity / limit)",       "Enter as": "2x + y <= 100"},
        {"Mathematical expression": "x + 2y ≥ 40",         "Meaning": "at least (requirement)",           "Enter as": "x + 2y >= 40"},
        {"Mathematical expression": "x + y = 50",          "Meaning": "exactly (fixed total)",            "Enter as": "x + y = 50"},
        {"Mathematical expression": "3x₁ + 2x₂ − x₃ ≤ 60", "Meaning": "three variables, minus sign",      "Enter as": "3x1 + 2x2 - x3 <= 60"},
        {"Mathematical expression": "0.5x + 1.25y ≤ 20",   "Meaning": "decimal coefficients",             "Enter as": "0.5x + 1.25y <= 20"},
        {"Mathematical expression": "x ≤ 30",              "Meaning": "limit on a single variable",       "Enter as": "x <= 30"},
        {"Mathematical expression": "x ≥ y",               "Meaning": "compare two variables",            "Enter as": "x - y >= 0"},
        {"Mathematical expression": "2x + 3y ≤ 100 Labour","Meaning": "name the resource for the report", "Enter as": "2x + 3y <= 100 # Labour"},
        {"Mathematical expression": "x₁, x₂, …, x₈",       "Meaning": f"1 to {MAX_VARIABLES} variable names",           "Enter as": "x1, x2, x3"},
        {"Mathematical expression": "x, y ≥ 0",            "Meaning": "non-negativity",                   "Enter as": "automatic — skip it"},
    ]


def symbol_guide_rows() -> list[dict[str, str]]:
    """Guide table explaining the symbols the constraint parser understands."""
    return [
        {"Symbol": "<=",    "Meaning": "less than or equal (≤)",      "Example": "2x + y <= 100"},
        {"Symbol": ">=",    "Meaning": "greater than or equal (≥)",   "Example": "x + y >= 20"},
        {"Symbol": "=",     "Meaning": "exactly equal",               "Example": "x + y = 50"},
        {"Symbol": "+  -",  "Meaning": "add / subtract terms",        "Example": "3x1 - x2 + 2x3"},
        {"Symbol": "*",     "Meaning": "multiplication (optional)",   "Example": "3*x and 3x both work"},
        {"Symbol": ".",     "Meaning": "decimal point",               "Example": "0.75x + 1.5y <= 12"},
        {"Symbol": ",",     "Meaning": "separates names / numbers",   "Example": "x, y, z   or   40, 30, 25"},
        {"Symbol": "#",     "Meaning": "starts an optional label",    "Example": "x + y <= 8 # hours"},
        {"Symbol": "<  >",  "Meaning": "read as <= / >= (LP can't do strict)", "Example": "x < 30  is solved as  x <= 30"},
    ]


def example_rows() -> list[dict[str, str]]:
    """The example problems as table rows, grouped by difficulty."""
    rows = []
    for i, ex in enumerate(example_problems(), start=1):
        rows.append({
            "Level": ex["level"],
            "Example": f"{i:>2}. {ex['name']}",
            "Size": f"{len(ex['var_names'])} × {len(ex['constraints'])}",
        })
    return rows



def banner(show_special: bool = False) -> str:
    """Return the professional startup/help banner.

    Parameters
    ----------
    show_special:
        When True, append the special-cases reference section.
    """
    lines: list[str] = [
        hr(),
        f"{'PROFESSIONAL LINEAR PROGRAMMING SOLVER':^{WIDTH}}",
        f"{'Interactive Terminal Edition':^{WIDTH}}",
        f"{f'Up to {MAX_VARIABLES} decision variables · up to {MAX_CONSTRAINTS} constraints':^{WIDTH}}",
        hr(),
        "",
        "INPUT GUIDE",
        "",
    ]
    lines.extend(
        format_table(
            ["Mathematical expression", "Meaning", "Enter as"],
            input_guide_rows(),
            ["Mathematical expression", "Meaning", "Enter as"],
            [23, 33, 26],
        )
    )
    lines.extend(["", "SYMBOL GUIDE", ""])
    lines.extend(
        format_table(
            ["Symbol", "Meaning", "Example"],
            symbol_guide_rows(),
            ["Symbol", "Meaning", "Example"],
            [10, 28, 36],
        )
    )
    lines.extend(["", "EXAMPLE PROBLEMS TO TRY", ""])
    lines.extend(
        format_table(
            ["Level", "Example", "Vars × Cons"],
            example_rows(),
            ["Level", "Example", "Size"],
            [14, 50, 16],
        )
    )
    lines.extend(
        [
            "",
            "Commands:",
            "  example        - load one of the example problems above",
            "  own            - build your own model step by step",
            "  help           - show this guide again",
            "  special        - show the special-cases guide (infeasible, unbounded, ...)",
            "  quit  / exit   - close the program",
            "",
            "Tip: Keep variables on the left and a single number on the right of each constraint.",
            "Tip: The app accepts many natural forms — try  3x + 2y ≤ 60  or  3*x+2*y=<60.",
            "Tip: 2 variables draw the feasible region; 3 draw it in 3-D; 4-8 draw a solution chart.",
            "Tip: Step 5 of every report lists shadow prices -- what one more unit of each resource is worth.",
            "Tip: Type  quit  at any prompt while building a model to go back to this menu.",
        ]
    )
    if show_special:
        lines.append(special_case_guide())
    lines.append(hr())
    return "\n".join(lines)
