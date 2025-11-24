from __future__ import annotations

from typing import Sequence, Tuple

import flax.nnx as nnx
import jax
import numpy as np


def print_parameter_counts(module: nnx.Module, module_name: str = "Model", max_depth: int = 3):
    """
    Print parameter counts for each layer in an nnx module with hierarchical formatting.
    """

    def format_number(n: int) -> str:
        return f"{n:,}"

    def format_percentage(part: int, total: int) -> str:
        if total == 0:
            return "0.0%"
        return f"{100.0 * part / total:.1f}%"

    def get_param_count(state_dict) -> int:
        count = 0
        for value in jax.tree_util.tree_leaves(state_dict):
            if hasattr(value, "size"):
                count += int(value.size)
        return count

    def traverse_module(mod, name: str, depth: int, prefix: str = ""):
        try:
            state = nnx.state(mod, nnx.Param)
        except Exception:
            return []

        param_count = get_param_count(state)
        results = []
        if param_count > 0 or depth == 0:
            results.append({"name": name, "depth": depth, "prefix": prefix, "params": param_count})

        if depth >= max_depth:
            return results

        module_attrs = []
        try:
            obj_dict = vars(mod)
        except TypeError:
            obj_dict = {}

        for attr_name, attr_value in obj_dict.items():
            if not isinstance(attr_name, str) or attr_name.startswith("_"):
                continue
            try:
                if isinstance(attr_value, nnx.Module):
                    module_attrs.append((attr_name, attr_value))
                elif isinstance(attr_value, (list, tuple, nnx.List)):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            module_attrs.append((f"{attr_name}[{idx}]", item))
            except Exception:
                continue

        module_attrs.sort(key=lambda x: x[0])

        for i, (attr_name, submod) in enumerate(module_attrs):
            is_last = i == len(module_attrs) - 1
            branch = "└── " if is_last else "├── "
            sub_results = traverse_module(submod, attr_name, depth + 1, prefix + ("    " if is_last else "│   "))
            for result in sub_results:
                if result["depth"] == depth + 1:
                    result["prefix"] = prefix + branch
                results.append(result)

        return results

    total_params = get_param_count(nnx.state(module, nnx.Param))
    results = traverse_module(module, module_name, 0)

    print("=" * 80)
    print(f"Parameter Count Analysis: {module_name}")
    print("=" * 80)
    print()

    if not results:
        print("No parameters found.")
        return

    max_name_len = max(len(r["prefix"] + r["name"]) for r in results)

    for result in results:
        full_name = result["prefix"] + result["name"]
        params = result["params"]
        pct = format_percentage(params, total_params)
        name_str = full_name.ljust(max_name_len + 2)
        param_str = format_number(params).rjust(12)
        pct_str = pct.rjust(8)
        bar_width = 30
        filled = int(bar_width * params / total_params) if total_params > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"{name_str} {param_str} params  {pct_str}  [{bar}]")

    print()
    print("─" * 80)
    print(f"{'TOTAL'.ljust(max_name_len + 2)} {format_number(total_params).rjust(12)} params  100.0%")
    print("=" * 80)
    print()

    if len(results) > 1:
        param_counts = [r["params"] for r in results if r["depth"] == 1]
        if param_counts:
            print("Summary Statistics (depth=1 modules):")
            print(f"  - Number of modules: {len(param_counts)}")
            print(f"  - Largest module: {format_number(max(param_counts))} params")
            print(f"  - Smallest module: {format_number(min(param_counts))} params")
            print(f"  - Average module: {format_number(int(np.mean(param_counts)))} params")
            print()


class LiveLossPlot:
    """
    Lightweight live plotting helper for notebooks.

    Call ``update(time_slice, step, loss, t, delta_t, trajectories)`` during training.
    Plots loss plus per-stage 2D trajectory snapshots (first two dims) of the current chains.
    """

    def __init__(
        self,
        title: str = "Training",
        per_slice: bool = True,
        max_points: int = 200,
        traj_bounds: Tuple[float, float] | None = (-5.0, 5.0),
    ):
        self.title = title
        self.per_slice = per_slice
        self.max_points = max(1, max_points)
        if traj_bounds is not None and (
            len(traj_bounds) != 2 or traj_bounds[0] >= traj_bounds[1]
        ):
            raise ValueError("traj_bounds must be (min, max) with min < max or None.")
        self.traj_bounds = traj_bounds
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from IPython import display  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LiveLossPlot requires matplotlib and IPython in notebooks") from exc
        self.plt = plt
        self.display = display
        # time_slice -> dict with steps, losses, fig, axes, handle
        self.series = {}
        if not per_slice:
            self.series["shared"] = {
                "steps": [],
                "losses": [],
                "fig": None,
                "ax_loss": None,
                "traj_axes": [],
                "traj_count": 0,
                "handle": None,
            }

    def _get_series(self, time_slice: int):
        if self.per_slice:
            if time_slice not in self.series:
                self.series[time_slice] = {
                    "steps": [],
                    "losses": [],
                    "fig": None,
                    "ax_loss": None,
                    "traj_axes": [],
                    "traj_count": 0,
                    "handle": None,
                }
            return self.series[time_slice]
        return self.series["shared"]

    def _build_fig(self, traj_count: int):
        traj_cols = min(3, max(1, traj_count)) if traj_count > 0 else 1
        traj_rows = 0 if traj_count == 0 else (traj_count + traj_cols - 1) // traj_cols
        rows = 1 + traj_rows
        fig = self.plt.figure(figsize=(4 * traj_cols, 4 * rows))
        gs = fig.add_gridspec(rows, traj_cols)
        ax_loss = fig.add_subplot(gs[0, :])
        traj_axes = []
        for idx in range(traj_count):
            r = 1 + idx // traj_cols
            c = idx % traj_cols
            traj_axes.append(fig.add_subplot(gs[r, c]))
        return fig, ax_loss, traj_axes

    def _ensure_axes(self, series, traj_count: int):
        same_axes = series.get("traj_count") == traj_count and series.get("fig") is not None
        if same_axes:
            return
        # rebuild figure/axes
        old_fig = series.get("fig")
        if old_fig is not None:
            self.plt.close(old_fig)
        fig, ax_loss, traj_axes = self._build_fig(traj_count)
        series.update(
            {
                "fig": fig,
                "ax_loss": ax_loss,
                "traj_axes": traj_axes,
                "traj_count": traj_count,
            }
        )
        if series.get("handle") is None:
            series["handle"] = self.display.display(fig, display_id=True)
        else:
            series["handle"].update(fig)

    def _plot_trajectories(
        self, traj_axes: Sequence, trajectories: Sequence[Tuple[str, np.ndarray]] | None
    ):
        if trajectories is None or not trajectories:
            for ax in traj_axes:
                ax.clear()
                ax.set_axis_on()
                if self.traj_bounds is not None:
                    ax.set_xlim(*self.traj_bounds)
                    ax.set_ylim(*self.traj_bounds)
                ax.set_title("trajectories (first 2 dims)")
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal", adjustable="box")
            return

        for idx, (ax, (name, pts)) in enumerate(zip(traj_axes, trajectories)):
            ax.clear()
            ax.set_axis_on()
            pts = np.asarray(pts)
            if pts.ndim != 2 or pts.shape[1] < 2:
                ax.set_axis_off()
                ax.set_title(name)
                continue
            pts = pts[: self.max_points]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                alpha=0.65,
                label=name,
                color=self.plt.get_cmap("tab10")(idx % 10),
                edgecolors="none",
            )
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_title(name)
            if self.traj_bounds is not None:
                ax.set_xlim(*self.traj_bounds)
                ax.set_ylim(*self.traj_bounds)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

    def update(
        self,
        time_slice: int,
        step: int,
        loss: float,
        t: float,
        delta_t: float,
        trajectories: Sequence[Tuple[str, np.ndarray]] | None = None,
    ):
        series = self._get_series(time_slice)
        traj_count = len(trajectories) if trajectories is not None else series.get("traj_count", 0)
        self._ensure_axes(series, traj_count)

        steps, losses, fig, ax_loss, traj_axes, handle = (
            series["steps"],
            series["losses"],
            series["fig"],
            series["ax_loss"],
            series["traj_axes"],
            series["handle"],
        )
        steps.append(step)
        losses.append(loss)

        ax_loss.clear()
        ax_loss.plot(steps, losses, label=f"slice {time_slice}")
        ax_loss.set_xlabel("updates (within slice)" if self.per_slice else "updates")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title(self.title if not self.per_slice else f"{self.title} (slice {time_slice})")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_loss.text(
            0.01,
            0.98,
            f"t={t:.3f}, dt={delta_t:.3f}",
            transform=ax_loss.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        self._plot_trajectories(traj_axes, trajectories)

        if handle is None:
            series["handle"] = self.display.display(fig, display_id=True)
        else:
            handle.update(fig)
