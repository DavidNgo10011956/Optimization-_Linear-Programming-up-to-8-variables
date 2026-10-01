"""
lp_app.py
=========
Terminal application for the Linear Programming Solver toolkit.
Solves models with 1-8 decision variables and up to 12 constraints,
prints a step-by-step explanation (including shadow prices), and saves a
picture of the result:
  * 2 variables   -> the feasible region (graphical method)
  * 3 variables   -> the feasible region in 3-D + resource usage
  * 4-8 variables -> the optimal plan + resource usage

Run with:
    python lp_app.py                 # plots are saved and opened
    python lp_app.py --no-window     # plots are saved only
    python lp_app.py --no-plot       # text report only
    python lp_app.py --outdir plots  # save plots into ./plots

Requires: numpy, scipy (and matplotlib for the plots).
All modules (lp_engine.py, lp_input_guide.py, lp_plotting.py, lp_utils.py)
must be in the same folder as this file.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make sibling modules importable even when started from another folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lp_engine import (LPModel, parse_constraint, parse_objective, solve, step_by_step,  # noqa: E402
                       MAX_CONSTRAINTS, MAX_VARIABLES)
from lp_input_guide import banner, example_problems, is_special_case, special_case_guide  # noqa: E402
from lp_utils import hr, parse_name_list  # noqa: E402

COMMANDS = {
    "example": "example", "e": "example",
    "own": "own", "o": "own", "": "own",
    "help": "help", "h": "help", "?": "help",
    "special": "special", "s": "special",
    "quit": "exit", "exit": "exit", "q": "exit", "x": "exit",
}

GOALS = {"max": "max", "maximize": "max", "maximise": "max",
         "min": "min", "minimize": "min", "minimise": "min"}

SAMPLE_OBJ = (40, 30, 25, 20, 15, 12, 10, 8)
SAMPLE_CON = (2, 3, 5, 4, 6, 2, 3, 7)


class BackToMenu(Exception):
    """Raised when the user types 'quit' (or Ctrl-C) part-way through building a model."""


def ask(prompt: str) -> str:
    """input() that lets the user abandon the current model with 'quit'/'exit' or Ctrl-C."""
    try:
        text = input(prompt).strip()
    except KeyboardInterrupt:
        print()
        raise BackToMenu from None
    if text.lower() in ("quit", "exit", "q"):
        raise BackToMenu
    return text


def ask_int(prompt: str, low: int, high: int) -> int:
    while True:
        text = ask(prompt)
        try:
            value = int(text)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"  Please enter a whole number from {low} to {high}.")


def choose_mode() -> str:
    """Ask what to do next. Returns 'example', 'own', 'help', 'special', or 'exit'."""
    while True:
        choice = input("Command (example / own / help / special / quit) [own]: ").strip().lower()
        if choice in COMMANDS:
            return COMMANDS[choice]
        print("  Unknown command -- type  help  to see the guide.")


def prompt_example() -> LPModel:
    examples = example_problems()
    print()
    level = None
    for i, ex in enumerate(examples, start=1):
        if ex["level"] != level:
            level = ex["level"]
            print(f"  {level}")
        size = f"{len(ex['var_names'])} vars × {len(ex['constraints'])} cons"
        print(f"    {i:>2}. {ex['name']:<48} {size}")
    ex = examples[ask_int("Choose a number: ", 1, len(examples)) - 1]
    var_names = ex["var_names"]
    constraints = [parse_constraint(line, var_names) for line in ex["constraints"]]
    return LPModel(var_names=var_names, objective=list(ex["objective"]), goal=ex["goal"],
                   constraints=constraints)


def prompt_own() -> LPModel:
    print("\nBuild your own model  (type  quit  at any prompt to go back to the menu)\n")
    n = ask_int(f"Number of decision variables (1-{MAX_VARIABLES}): ", 1, MAX_VARIABLES)

    default_names = [f"x{i}" for i in range(1, n + 1)] if n > 1 else ["x"]
    while True:
        try:
            text = ask(f"Variable names, comma-separated ({n} name{'s' if n > 1 else ''}) "
                       f"[{', '.join(default_names)}]: ")
            var_names = parse_name_list(text) if text else default_names
            if len(var_names) != n:
                raise ValueError(f"Expected {n} name{'s' if n > 1 else ''}, got {len(var_names)}.")
            break
        except ValueError as exc:
            print(f"  {exc}")

    while True:
        goal = GOALS.get(ask("Maximize or minimize Z? (max/min): ").lower())
        if goal:
            break
        print("  Please type  max  or  min.")

    example_obj = " + ".join(f"{c}{v}" for c, v in zip(SAMPLE_OBJ, var_names))
    example_list = ", ".join(str(c) for c in SAMPLE_OBJ[:n])
    while True:
        try:
            objective = parse_objective(
                ask(f"Objective  Z =  (e.g. {example_obj}  or  {example_list}): "), var_names)
            if not any(objective):
                print("  Note: every objective coefficient is 0, so every feasible plan is optimal.")
            break
        except ValueError as exc:
            print(f"  {exc}")

    m = ask_int(f"Number of constraints (1-{MAX_CONSTRAINTS}): ", 1, MAX_CONSTRAINTS)
    hint = " + ".join(f"{c}{v}" for c, v in zip(SAMPLE_CON, var_names))
    print(f"\nEnter each constraint, e.g.  {hint} <= 100   or   {var_names[0]} >= 5 # my label")
    print("(Non-negativity is automatic -- don't enter it.)")
    constraints = []
    for i in range(m):
        while True:
            try:
                raw = ask(f"  Constraint {i + 1}: ")
                cons = parse_constraint(raw, var_names)
                if not any(cons.coeffs):
                    print("    That constraint has no variables in it. Please re-enter.")
                    continue
                if ("<" in raw or ">" in raw) and "=" not in raw.replace("≤", "=").replace("≥", "="):
                    print(f"    Note: read as  {cons.sense}  (linear programs can't use strict inequalities).")
                constraints.append(cons)
                break
            except ValueError as exc:
                print(f"    {exc}  Please re-enter.")

    return LPModel(var_names=var_names, objective=objective, goal=goal, constraints=constraints)


def _plot(model, solution, run_number: int, args) -> None:
    try:
        import matplotlib
        if args.no_window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lp_plotting import plot_model, plot_explanation, show_or_open
    except ImportError:
        print("\n(matplotlib is not installed, so no plot was drawn -- install it with  "
              "pip install matplotlib  to get pictures.)")
        return

    prefix = "lp_feasible_region" if model.n_vars == 2 else "lp_solution"
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{prefix}_{run_number}.png")
    try:
        fig = plot_model(model, solution, save_path=path)
    except Exception as exc:                      # a plotting problem shouldn't hide the answer
        print(f"\n(Couldn't draw the plot: {exc})")
        return

    if fig is not None:
        abs_path = os.path.abspath(path)
        if args.no_window:
            print(f"\nPlot saved to {abs_path}\n")
        else:
            result = show_or_open(fig, path)
            if result == "window":
                print(f"\nOpened the plot in a window (close it to continue). "
                      f"It was also saved to {abs_path}.\n")
            elif result == "opened":
                print(f"\nOpened {abs_path} in your default image viewer.\n")
            else:
                print(f"\nCouldn't open a viewer automatically -- open this file to see the plot:\n"
                      f"  {abs_path}\n")
        plt.close(fig)

    print(plot_explanation(model, solution))


def solve_and_report(model: LPModel, run_number: int, args) -> None:
    solution = solve(model)

    print("\n" + hr("─"))
    print("\n\n".join(step_by_step(model, solution)))

    note = is_special_case(solution.status, solution.degenerate)
    if note:
        print("\nSpecial case (type  special  for the full guide):")
        for line in note.splitlines():
            print(f"  * {line}")

    if model.n_vars >= 2 and not args.no_plot:
        _plot(model, solution, run_number, args)
    print(hr("─"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Interactive linear programming solver (1-8 variables).")
    p.add_argument("--no-plot", action="store_true", help="print the text report only")
    p.add_argument("--no-window", action="store_true", help="save plots without opening them")
    p.add_argument("--outdir", default=".", help="folder for the saved plots (default: current folder)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Never crash on a console that can't print ≤, ≥, × ... (e.g. some Windows code pages).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    print(banner())

    run_number = 0
    try:
        while True:
            mode = choose_mode()
            if mode == "exit":
                break
            if mode == "help":
                print(banner())
                continue
            if mode == "special":
                print(special_case_guide())
                continue

            try:
                model = prompt_example() if mode == "example" else prompt_own()
            except BackToMenu:
                print("  Back to the menu.\n")
                continue
            except ValueError as exc:
                print(f"\nInput error: {exc}\n")
                continue

            run_number += 1
            try:
                solve_and_report(model, run_number, args)
            except Exception as exc:
                print(f"\nSolver error: {exc}")

            print()  # spacing before the next menu
    except (EOFError, KeyboardInterrupt):
        print()

    print("Goodbye!")


if __name__ == "__main__":
    main()
