"""
lp_plotting.py
==============
Pictures for the LP solver, chosen by the number of decision variables:

  * 2 variables  -> the graphical method from the teaching slides: boundary
                    lines, shaded feasible region, corner points, optimum.
  * 3 variables  -> the feasible region as a 3-D solid, next to a
                    resource-usage chart.
  * 4-8 variables -> a solution chart: the optimal value of every variable
                    (and its share of Z) next to the resource-usage chart.

Unbounded regions are clipped to the plotting window and flagged, so an
open region is never drawn as if it were closed.
"""

from __future__ import annotations

import textwrap

import numpy as np
import matplotlib.pyplot as plt

from lp_engine import LPModel, LPSolution, feasible_vertices, order_polygon, _num


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _axis_limits(model: LPModel, solution: LPSolution) -> np.ndarray:
    """Plot window per variable: 35 % beyond the furthest corner (and intercepts)."""
    n = model.n_vars
    pts = np.array(solution.corner_points) if solution.corner_points else np.zeros((1, n))
    top = pts.max(axis=0)
    if not solution.corner_points:
        # empty region: size the window from the axis intercepts so the lines are visible
        for c in model.constraints:
            for j, a in enumerate(c.coeffs):
                if abs(a) > 1e-9 and c.rhs / a > 0:
                    top[j] = max(top[j], c.rhs / a)
    limits = np.maximum(top, 1.0) * 1.35 + 1
    return np.maximum(limits, 0.6 * limits.max())   # keep every axis readable


def _region_is_clipped(clipped: np.ndarray, limits: np.ndarray) -> bool:
    """True if the clipped region touches the plot window -> the real region is open."""
    return bool(len(clipped)) and bool(np.any(np.isclose(clipped, limits, rtol=1e-6, atol=1e-6)))


def _wrapped_title(model: LPModel, prefix: str) -> str:
    return "\n".join(textwrap.wrap(f"{prefix} -- {model.objective_str()}", 90))


def _resource_panel(ax, model: LPModel, solution: LPSolution) -> None:
    """Horizontal bars: amount used (LHS) vs. the limit / requirement (RHS)."""
    names, lhs, rhs, colors = [], [], [], []
    for i, (c, b) in enumerate(zip(model.constraints, solution.binding)):
        used = sum(a * x for a, x in zip(c.coeffs, solution.x))
        names.append(c.label or f"C{i + 1}: {c.render(model.var_names, include_label=False)}")
        lhs.append(used)
        rhs.append(c.rhs)
        colors.append("tab:red" if b else "tab:blue")

    y = np.arange(len(names))[::-1]
    ax.barh(y, rhs, height=0.6, color="lightgray", edgecolor="gray", label="Limit / requirement (RHS)")
    ax.barh(y, lhs, height=0.34, color=colors, label="Used at the optimum (LHS)")
    span = max([abs(v) for v in lhs + rhs] + [1.0])
    for yi, u, r, c in zip(y, lhs, rhs, model.constraints):
        ax.text(max(u, r) + span * 0.02, yi, f"{_num(u)} {c.sense} {_num(r)}",
                va="center", fontsize=8, color="dimgray")
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(n, 28)) for n in names], fontsize=8)
    ax.set_xlim(min(0, min(lhs + rhs) * 1.1), span * 1.35)
    ax.set_title("Resources: used vs. available at the optimum", fontsize=10)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="lightgray", edgecolor="gray", label="Limit / requirement (RHS)"),
               Patch(facecolor="tab:red", label="Used -- binding (slack = 0)"),
               Patch(facecolor="tab:blue", label="Used -- has slack")]
    ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              frameon=False)
    ax.grid(axis="x", alpha=0.3)


# --------------------------------------------------------------------------- #
# 2 variables -- graphical method
# --------------------------------------------------------------------------- #

def plot_feasible_region(model: LPModel, solution: LPSolution, save_path: str = None, show: bool = False):
    if model.n_vars != 2:
        raise ValueError("plot_feasible_region needs exactly two variables -- use plot_model().")

    corner_points = solution.corner_points or []
    x_max, y_max = _axis_limits(model, solution)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    xg = np.linspace(0, x_max, 400)

    colors = plt.cm.tab10.colors
    for i, c in enumerate(model.constraints):
        a, b, r = c.coeffs[0], c.coeffs[1], c.rhs
        label = c.label or c.render(model.var_names)
        color = colors[i % len(colors)]
        if abs(b) > 1e-9:
            ax.plot(xg, (r - a * xg) / b, color=color, linewidth=1.6, label=label)
        elif abs(a) > 1e-9:
            ax.axvline(r / a, color=color, linewidth=1.6, label=label)

    # Shade the region clipped to the window, so open regions stay open.
    clipped, _ = feasible_vertices(model, upper=[x_max, y_max])
    if len(clipped) >= 3:
        poly = order_polygon(clipped)
        open_region = _region_is_clipped(clipped, np.array([x_max, y_max]))
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.22, color="tab:green", zorder=0,
                label="Feasible region" + (" (continues beyond plot)" if open_region else ""))
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            on_window = (np.isclose(p[0], x_max) and np.isclose(q[0], x_max)) or \
                        (np.isclose(p[1], y_max) and np.isclose(q[1], y_max))
            ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:green", linewidth=1.2, zorder=1,
                    linestyle=":" if on_window else "-")
    elif len(clipped):
        ax.plot(clipped[:, 0], clipped[:, 1], color="tab:green", linewidth=3, label="Feasible region")

    for p in corner_points:
        ax.annotate(f"({_num(p[0])}, {_num(p[1])})", p, textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="dimgray")

    if solution.is_solved and solution.x is not None:
        if solution.status == "multiple_optimal":
            tol = 1e-6 * max(1.0, abs(solution.objective_value))
            ties = np.array([p for p, v in zip(corner_points, solution.corner_values)
                             if abs(v - solution.objective_value) < tol])
            if len(ties) >= 2:
                ax.plot(ties[:, 0], ties[:, 1], color="red", linewidth=4, alpha=0.6, zorder=4,
                        label="Every point on this edge is optimal")
        ax.scatter([solution.x[0]], [solution.x[1]], color="red", s=90, zorder=5,
                   label=f"Optimum Z={solution.objective_value:.4g}")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(model.var_names[0])
    ax.set_ylabel(model.var_names[1])
    ax.set_title(_wrapped_title(model, "Feasible Region"))
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# 3 variables -- 3-D feasible region
# --------------------------------------------------------------------------- #

def _region_3d_panel(ax, model: LPModel, solution: LPSolution) -> bool:
    """Draw the 3-D region; return True if it had to be clipped (open region)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull

    limits = _axis_limits(model, solution)
    clipped, _ = feasible_vertices(model, upper=limits)
    open_region = _region_is_clipped(clipped, limits)

    if len(clipped) >= 4:
        try:
            hull = ConvexHull(clipped, qhull_options="QJ")   # QJ copes with flat (= constraint) regions
            faces = [clipped[s] for s in hull.simplices]
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.18, facecolor="tab:green",
                                                 edgecolor="none"))
        except Exception:
            pass
    if len(clipped):
        ax.scatter(*clipped.T, color="tab:green", s=8, alpha=0.5)

    corners = np.array(solution.corner_points) if solution.corner_points else np.empty((0, 3))
    if len(corners):
        ax.scatter(*corners.T, color="black", s=18, depthshade=False, label="Corner points")
        for p in corners[:8]:
            ax.text(*p, f" ({_num(p[0])}, {_num(p[1])}, {_num(p[2])})", fontsize=6, color="dimgray")

    if solution.is_solved:
        ax.scatter(*solution.x, color="red", s=90, depthshade=False,
                   label=f"Optimum Z={solution.objective_value:.4g}")

    ax.set_xlim(0, limits[0]); ax.set_ylim(0, limits[1]); ax.set_zlim(0, limits[2])
    ax.set_xlabel(model.var_names[0]); ax.set_ylabel(model.var_names[1]); ax.set_zlabel(model.var_names[2])
    ax.set_title("Feasible region in 3-D" + ("\n(open region, clipped to the view)" if open_region else ""),
                 fontsize=10)
    ax.view_init(elev=22, azim=35)
    try:
        ax.set_box_aspect(None, zoom=0.85)      # leave room for the vertical-axis label
    except TypeError:
        pass                                    # matplotlib < 3.6 has no zoom argument
    ax.legend(fontsize=7, loc="upper left")
    return open_region


# --------------------------------------------------------------------------- #
# 4-8 variables -- solution chart
# --------------------------------------------------------------------------- #

def _variables_panel(ax, model: LPModel, solution: LPSolution) -> None:
    x = np.array(solution.x)
    contrib = np.array(model.objective) * x
    bars = ax.bar(model.var_names, x, color=["tab:green" if v > 1e-9 else "lightgray" for v in x])
    top = max(x.max(), 1.0)
    for bar, v, cz in zip(bars, x, contrib):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02,
                f"{_num(v)}\nZ{'+' if cz >= 0 else '-'}{_num(abs(cz))}" if v > 1e-9 else "0",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, top * 1.25)
    ax.set_title(f"Optimal plan (Z = {solution.objective_value:.4g})\n"
                 "bar = value, label = contribution to Z", fontsize=10)
    ax.tick_params(axis="x", labelrotation=30 if model.n_vars > 5 else 0, labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_solution_summary(model: LPModel, solution: LPSolution, save_path: str = None, show: bool = False):
    """3 variables: 3-D region + resources.  4-8 variables: plan + resources."""
    if model.n_vars < 3:
        raise ValueError("plot_solution_summary is for models with 3 or more variables.")
    if not solution.is_solved:
        raise ValueError("A solution chart needs an optimal solution.")

    fig = plt.figure(figsize=(13, 6))
    if model.n_vars == 3:
        left = fig.add_subplot(1, 2, 1, projection="3d")
        _region_3d_panel(left, model, solution)
    else:
        left = fig.add_subplot(1, 2, 1)
        _variables_panel(left, model, solution)
    _resource_panel(fig.add_subplot(1, 2, 2), model, solution)

    fig.suptitle(_wrapped_title(model, "LP Solution"), fontsize=11)
    fig.tight_layout()
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# Dispatcher / output
# --------------------------------------------------------------------------- #

def plot_model(model: LPModel, solution: LPSolution, save_path: str = None, show: bool = False):
    """Pick the right picture for the model. Returns the figure, or None if
    there's nothing meaningful to draw (1 variable, or 3+ variables without
    an optimal solution)."""
    if model.n_vars == 2:
        return plot_feasible_region(model, solution, save_path, show)
    if model.n_vars >= 3 and solution.is_solved:
        return plot_solution_summary(model, solution, save_path, show)
    if model.n_vars == 3 and solution.corner_points:
        # e.g. unbounded: still worth seeing the (clipped) open region
        fig = plt.figure(figsize=(7.5, 6.5))
        _region_3d_panel(fig.add_subplot(1, 1, 1, projection="3d"), model, solution)
        fig.suptitle(_wrapped_title(model, "Feasible Region"), fontsize=11)
        fig.tight_layout()
        return _finish(fig, save_path, show)
    return None


def _finish(fig, save_path, show):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return fig


def show_or_open(fig, save_path: str) -> str:
    """Try to pop up an interactive window for the figure. If no GUI backend
    is available (headless terminal, no Tk/Qt installed, etc.), fall back to
    opening the saved PNG with the operating system's default image viewer.
    Returns 'window', 'opened', or 'saved' depending on what actually happened.
    """
    import sys
    import os
    import subprocess
    import matplotlib

    backend = matplotlib.get_backend().lower()
    non_interactive_prefixes = ("agg", "pdf", "svg", "ps", "cairo", "template")
    looks_interactive = not any(backend.startswith(p) for p in non_interactive_prefixes)

    if looks_interactive:
        try:
            plt.show(block=True)
            return "window"
        except Exception:
            pass

    if save_path and os.path.exists(save_path):
        try:
            if sys.platform.startswith("darwin"):
                proc = subprocess.run(["open", save_path], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(save_path)  # type: ignore[attr-defined]
                return "opened"
            else:
                proc = subprocess.run(["xdg-open", save_path], check=False,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if proc.returncode == 0:
                return "opened"
        except Exception:
            pass

    return "saved"


# --------------------------------------------------------------------------- #
# Plain-language explanation of the picture
# --------------------------------------------------------------------------- #

def plot_explanation(model: LPModel, solution: LPSolution) -> str:
    """A plain-language walkthrough of what the plot shows, written so a
    student can connect the picture back to the numbers."""
    if model.n_vars == 2:
        return _explain_2d(model, solution)
    return _explain_nd(model, solution)


def _binding_line(model: LPModel, solution: LPSolution, what: str) -> list[str]:
    binding = [c.name(model.var_names) for c, b in zip(model.constraints, solution.binding) if b]
    if not binding:
        return []
    return [
        f"  * The optimum sits exactly on the {what} for " + ", ".join(binding) + " -- those "
        "constraints are exactly at their limit (a resource fully used, or a requirement met "
        "with nothing to spare), which is why the optimum lands on a corner rather than in the "
        "open middle of the region."
    ]


def _explain_2d(model: LPModel, solution: LPSolution) -> str:
    x_name, y_name = model.var_names
    lines = [
        f"How to read this plot\n"
        f"  * The horizontal axis is {x_name}, the vertical axis is {y_name} -- "
        f"every point ({x_name}, {y_name}) on the page is one possible production/decision plan."
    ]

    boundary_bits = [f"'{c.name(model.var_names)}' is the line "
                     f"{c.render(model.var_names, include_label=False).replace('<=', '=').replace('>=', '=')}"
                     for c in model.constraints]
    lines.append(
        "  * Each colored line is one constraint's boundary -- the plan that uses that "
        "resource up completely (or exactly meets that requirement):\n      "
        + "\n      ".join(boundary_bits)
    )

    if solution.corner_points:
        lines.append(
            "  * The shaded green area is the feasible region: every plan inside or on its "
            f"edge satisfies ALL constraints (and {x_name}, {y_name} >= 0) at the same time. "
            "Its corners are the only points that can possibly be optimal."
        )
        limits = _axis_limits(model, solution)
        clipped, _ = feasible_vertices(model, upper=limits)
        if _region_is_clipped(clipped, limits):
            lines.append(
                "  * The region is OPEN: it carries on past the edge of the picture (dotted "
                "border). That's normal for 'min' problems with >= constraints -- but for a "
                "'max' problem it usually means Z can grow forever."
            )
    else:
        lines.append(
            "  * No green region is shown because no plan satisfies every constraint at once -- "
            "the feasible region is empty."
        )

    if solution.is_solved and solution.x is not None:
        x_val, y_val = solution.x
        lines.append(
            f"  * The red dot at ({_num(x_val)}, {_num(y_val)}) is the optimal plan: among every "
            f"point in the feasible region, this corner gives the {model.goal}imum possible value, "
            f"Z = {solution.objective_value:.4g}."
        )
        lines += _binding_line(model, solution, "line(s)")
        lines.append(
            f"  * Why a corner, not the middle? Z = {model.objective_expr()} is a flat (linear) "
            "objective. Sliding its line across the feasible region, the last point it touches "
            "before leaving the region is always a corner -- never an interior point."
        )
    else:
        lines.append(f"  * {solution.message}")

    return "\n".join(lines)


def _explain_nd(model: LPModel, solution: LPSolution) -> str:
    n = model.n_vars
    if not solution.is_solved and n == 3 and solution.corner_points:
        return ("How to read this plot\n"
                "  * The green solid is the feasible region, clipped to the picture, and the black "
                "dots are its corner points.\n"
                f"  * {solution.message}\n"
                "  * The region is open, so no corner is the best one: moving further out keeps "
                "improving Z. Look for a missing capacity, budget or demand limit.")
    if not solution.is_solved:
        return (f"No chart was drawn: {solution.message}\n"
                f"  * With {n} variables the chart shows the optimal plan, so there is nothing to "
                "plot until the model has an optimal solution.")

    lines = ["How to read this chart"]
    if n == 3:
        a, b, c = model.var_names
        lines += [
            f"  * Left: the three axes are {a}, {b} and {c}. Each constraint is now a flat PLANE "
            "instead of a line, and the green solid is the feasible region -- every plan inside "
            "it satisfies all constraints at once.",
            f"  * The black dots are the {len(solution.corner_points)} corner points (where three "
            "planes meet). Exactly as in 2-D, the optimum is always at one of them.",
            f"  * The red dot at ({', '.join(_num(v) for v in solution.x)}) is the optimal plan, "
            f"Z = {solution.objective_value:.4g}. Rotate the window with the mouse to see it from "
            "other angles.",
        ]
    else:
        zero = [v for v, x in zip(model.var_names, solution.x) if x <= 1e-9]
        lines += [
            f"  * With {n} variables the feasible region has {n} dimensions and can't be drawn, so "
            "the chart shows the answer instead.",
            f"  * Left: the height of each bar is that variable's value in the optimal plan; the "
            f"label above it shows how much of Z = {solution.objective_value:.4g} it earns.",
        ]
        if zero:
            lines.append(
                "  * Grey bars (" + ", ".join(zero) + ") are 0 -- at the optimum it doesn't pay to "
                "use them: " + ("the resources they need earn more elsewhere."
                                if model.goal == "max" else
                                "the other variables meet the requirements more cheaply.")
            )
    lines += [
        "  * Right: for each constraint the grey bar is the limit or requirement (right-hand "
        "side) and the thin bar is how much the optimal plan actually uses (left-hand side).",
    ]
    lines += _binding_line(model, solution, "boundaries")
    lines.append(
        f"  * The corner-point method checked {solution.candidates_checked:,} combinations of "
        f"{n} boundaries and found {len(solution.corner_points)} corners. The optimal corner is where the "
        "binding (red) constraints meet the axes of the variables that are 0."
    )
    return "\n".join(lines)
