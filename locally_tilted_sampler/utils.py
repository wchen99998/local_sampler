from __future__ import annotations

import numpy as np

import flax.nnx as nnx
import jax
from typing import Callable, Optional


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
    Lightweight live loss plotting helper for notebooks.

    Call ``update(time_slice, step, loss, t, delta_t)`` during training.
    By default plots a separate figure per time slice to avoid mis-connected lines.
    """

    def __init__(self, title: str = "Training loss", per_slice: bool = True):
        self.title = title
        self.per_slice = per_slice
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from IPython import display  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LiveLossPlot requires matplotlib and IPython in notebooks") from exc
        self.plt = plt
        self.display = display
        # time_slice -> dict with steps, losses, fig, ax, handle
        self.series = {}
        if not per_slice:
            fig, ax = plt.subplots(figsize=(6, 4))
            handle = display.display(fig, display_id=True)
            self.series["shared"] = {
                "steps": [],
                "losses": [],
                "fig": fig,
                "ax": ax,
                "handle": handle,
            }

    def _get_series(self, time_slice: int):
        if self.per_slice:
            if time_slice not in self.series:
                fig, ax = self.plt.subplots(figsize=(6, 4))
                handle = self.display.display(fig, display_id=True)
                self.series[time_slice] = {
                    "steps": [],
                    "losses": [],
                    "fig": fig,
                    "ax": ax,
                    "handle": handle,
                }
            return self.series[time_slice]
        return self.series["shared"]

    def update(self, time_slice: int, step: int, loss: float, t: float, delta_t: float):
        series = self._get_series(time_slice)
        steps, losses, fig, ax, handle = (
            series["steps"],
            series["losses"],
            series["fig"],
            series["ax"],
            series["handle"],
        )
        steps.append(step)
        losses.append(loss)
        ax.clear()
        ax.plot(steps, losses, label=f"slice {time_slice}")
        ax.set_xlabel("updates (within slice)" if self.per_slice else "updates")
        ax.set_ylabel("loss")
        ax.set_title(self.title if not self.per_slice else f"{self.title} (slice {time_slice})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if handle is None:
            series["handle"] = self.display.display(fig, display_id=True)
        else:
            handle.update(fig)
