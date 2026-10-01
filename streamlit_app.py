"""
streamlit_app.py
================
Web front end for the Linear Programming Solver toolkit.

It reuses the same modules as the terminal program (lp_app.py):
    lp_engine.py       -- parsing, solving, corner points, shadow prices
    lp_input_guide.py  -- example problems, input guides, special cases
    lp_plotting.py     -- the pictures (feasible region, 3-D region, plan)
    lp_utils.py        -- shared helpers

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy: push this folder to GitHub and point Streamlit Community Cloud at
streamlit_app.py (see README.md).
"""

from __future__ import annotations

import io
import os
import sys

# Streamlit Cloud has no display: pick the non-GUI backend BEFORE pyplot is imported anywhere.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Make sibling modules importable however the app is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lp_engine import (LPModel, MAX_CONSTRAINTS, MAX_VARIABLES, _num, format_point,  # noqa: E402
                       parse_constraint, parse_objective, solve, step_by_step)
from lp_input_guide import (SPECIAL_CASES, example_problems, input_guide_rows,  # noqa: E402
                            is_special_case, symbol_guide_rows)
from lp_plotting import plot_explanation, plot_model  # noqa: E402
from lp_utils import parse_name_list  # noqa: E402


st.set_page_config(page_title="Linear Programming Solver", page_icon="📐", layout="wide")


# --------------------------------------------------------------------------- #
# Model building and solving
# --------------------------------------------------------------------------- #

def build_model(var_names, goal, objective_text, constraint_lines) -> LPModel:
    """Turn the text the user typed into an LPModel (raises ValueError on bad input)."""
    objective = parse_objective(objective_text, var_names)
    lines = [ln for ln in constraint_lines if ln.strip()]
    if not lines:
        raise ValueError("Enter at least one constraint.")
    if len(lines) > MAX_CONSTRAINTS:
        raise ValueError(f"At most {MAX_CONSTRAINTS} constraints are supported (you entered {len(lines)}).")
    constraints = []
    for i, line in enumerate(lines, start=1):
        try:
            cons = parse_constraint(line, var_names)
        except ValueError as exc:
            raise ValueError(f"Constraint {i} ({line.strip()}): {exc}") from None
        if not any(cons.coeffs):
            raise ValueError(f"Constraint {i} ({line.strip()}) has no variables in it.")
        constraints.append(cons)
    return LPModel(var_names=var_names, objective=objective, goal=goal, constraints=constraints)


@st.cache_data(show_spinner=False)
def solve_cached(var_names: tuple, goal: str, objective_text: str, constraint_lines: tuple):
    """Solve once per distinct model; Streamlit reruns the script on every click."""
    model = build_model(list(var_names), goal, objective_text, list(constraint_lines))
    return model, solve(model)


def figure_png(model, solution):
    """Draw the matching picture and return (png_bytes, figure) or (None, None)."""
    fig = plot_model(model, solution)          # no save_path, no show -> just the figure
    if fig is None:
        return None, None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue(), fig


# --------------------------------------------------------------------------- #
# Sidebar: choose an example or build your own
# --------------------------------------------------------------------------- #

EXAMPLES = example_problems()


def example_inputs():
    labels = [f"{ex['level']} · {ex['name']}" for ex in EXAMPLES]
    choice = st.sidebar.selectbox("Example problem", range(len(EXAMPLES)), format_func=lambda i: labels[i])
    ex = EXAMPLES[choice]
    objective_text = ", ".join(_num(c) for c in ex["objective"])
    st.sidebar.caption(f"{len(ex['var_names'])} variables × {len(ex['constraints'])} constraints")
    return ex["var_names"], ex["goal"], objective_text, ex["constraints"]


def own_inputs():
    n = st.sidebar.number_input("Number of decision variables", 1, MAX_VARIABLES, 2, step=1)
    default_names = "x, y" if n == 2 else ("x" if n == 1 else ", ".join(f"x{i}" for i in range(1, n + 1)))
    names_text = st.sidebar.text_input("Variable names (comma-separated)", default_names, key=f"names_{n}")
    goal = st.sidebar.radio("Goal", ["max", "min"], horizontal=True,
                            format_func=lambda g: "Maximize" if g == "max" else "Minimize")
    objective_text = st.sidebar.text_input("Objective Z =", "40x + 30y" if n == 2 else "",
                                           placeholder="e.g. 40x + 30y   or   40, 30")
    constraints_text = st.sidebar.text_area(
        f"Constraints (one per line, up to {MAX_CONSTRAINTS})",
        "2x + y <= 100 # Machine 1\nx + 2y <= 80 # Machine 2" if n == 2 else "",
        height=180,
        help="Use <=, >= or =. Add a label after #. Non-negativity is automatic.",
    )
    var_names = parse_name_list(names_text)          # may raise ValueError -> shown in main area
    if len(var_names) != n:
        raise ValueError(f"Expected {n} variable name{'s' if n > 1 else ''}, got {len(var_names)}.")
    return var_names, goal, objective_text, constraints_text.splitlines()


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #

STATUS_TEXT = {
    "optimal": ("Optimal solution found", "success"),
    "multiple_optimal": ("Multiple optimal solutions", "info"),
    "infeasible": ("Infeasible — no plan satisfies every constraint", "error"),
    "unbounded": ("Unbounded — Z can improve without limit", "warning"),
    "error": ("The solver could not finish", "error"),
}


def show_status(solution):
    text, kind = STATUS_TEXT.get(solution.status, (solution.message, "info"))
    getattr(st, kind)(text)
    note = is_special_case(solution.status, solution.degenerate)
    if note:
        with st.expander("Special case explained", expanded=True):
            for line in note.splitlines():
                st.markdown(f"- {line}")


def show_headline(model, solution):
    if not solution.is_solved:
        return
    cols = st.columns(min(model.n_vars, 4) + 1)
    cols[0].metric(f"Optimal Z ({model.goal})", _num(solution.objective_value))
    for i, (name, value) in enumerate(zip(model.var_names, solution.x)):
        cols[(i % min(model.n_vars, 4)) + 1].metric(name, _num(value))


def steps_tab(model, solution):
    for step in step_by_step(model, solution):
        title, _, body = step.partition("\n")
        st.markdown(f"**{title.replace(' -- ', ' — ')}**")
        st.code(body, language=None)


def corners_tab(model, solution):
    if not solution.corner_points:
        st.write("The feasible region has no corner points (it is empty).")
        return
    rows = [dict(zip(model.var_names, p)) | {"Z": z}
            for p, z in zip(solution.corner_points, solution.corner_values)]
    df = pd.DataFrame(rows)
    if solution.is_solved:
        best = solution.objective_value
        df["Optimal"] = [("★" if abs(z - best) <= 1e-6 * max(1, abs(best)) else "") for z in df["Z"]]
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"{len(df)} corner point(s) found from {solution.candidates_checked:,} boundary combinations.")


def sensitivity_tab(model, solution):
    if not solution.is_solved:
        st.write("Sensitivity information is only available when an optimum exists.")
        return
    rows = []
    for i, cons in enumerate(model.constraints):
        rows.append({
            "Constraint": cons.label or f"Constraint {i + 1}",
            "Equation": cons.render(model.var_names, include_label=False),
            "Slack / surplus": solution.slack[i],
            "Binding": "yes" if solution.binding[i] else "",
            "Shadow price": solution.shadow_prices[i] if solution.shadow_prices else None,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Shadow price = change in Z when that constraint's right-hand side grows by 1 unit, "
               "holding the others fixed (valid for small changes).")
    if solution.degenerate:
        st.warning("The optimum is degenerate, so these shadow prices may hold in one direction only: "
                   "raising and lowering a right-hand side can change Z by different amounts. "
                   "Re-solve with the changed value before relying on a shadow price.")


def plot_tab(model, solution):
    if model.n_vars < 2:
        st.write("With one variable there is no region to draw — see the step-by-step report.")
        return
    png, fig = figure_png(model, solution)
    if png is None:
        st.write("There is nothing meaningful to draw for this result.")
        return
    st.pyplot(fig, width="stretch")
    plt.close(fig)                                  # free memory between reruns
    st.download_button("Download plot (PNG)", png, file_name="lp_plot.png", mime="image/png")
    st.markdown("**What the picture shows**")
    st.text(plot_explanation(model, solution))


def guides_page():
    st.header("Input guide")
    st.dataframe(pd.DataFrame(input_guide_rows()), width="stretch", hide_index=True)
    st.header("Symbol guide")
    st.dataframe(pd.DataFrame(symbol_guide_rows()), width="stretch", hide_index=True)
    st.header("Special cases")
    df = pd.DataFrame(SPECIAL_CASES).rename(columns={
        "name": "Case", "symbol": "Symbol", "meaning": "Meaning", "spot": "How to spot", "example": "Example"})
    st.dataframe(df, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def main():
    st.title("Linear Programming Solver")
    st.write(f"Solve models with 1–{MAX_VARIABLES} decision variables and up to {MAX_CONSTRAINTS} "
             "constraints, with corner points, shadow prices and a picture of the result.")

    st.sidebar.header("Model")
    mode = st.sidebar.radio("Start from", ["Example problem", "Build your own", "Guides"])

    if mode == "Guides":
        guides_page()
        return

    try:
        var_names, goal, objective_text, constraint_lines = (
            example_inputs() if mode == "Example problem" else own_inputs())
    except ValueError as exc:
        st.error(f"Input error: {exc}")
        return

    # After the first Solve, results follow the sidebar inputs live (solving is cached).
    if st.sidebar.button("Solve", type="primary", width="stretch"):
        st.session_state["solved_once"] = True
    if not st.session_state.get("solved_once"):
        st.info("Choose or enter a model in the sidebar, then press **Solve**.")
        return

    key = (tuple(var_names), goal, objective_text, tuple(constraint_lines))

    try:
        with st.spinner("Solving…"):
            model, solution = solve_cached(*key)
    except ValueError as exc:
        st.error(f"Input error: {exc}")
        return
    except Exception as exc:                        # never crash the page
        st.error(f"Solver error: {exc}")
        return

    show_status(solution)
    show_headline(model, solution)

    tabs = st.tabs(["Picture", "Step by step", "Corner points", "Sensitivity"])
    with tabs[0]:
        try:
            plot_tab(model, solution)
        except Exception as exc:                    # a plotting problem shouldn't hide the answer
            st.warning(f"Couldn't draw the plot: {exc}")
    with tabs[1]:
        steps_tab(model, solution)
    with tabs[2]:
        corners_tab(model, solution)
    with tabs[3]:
        sensitivity_tab(model, solution)


main()
