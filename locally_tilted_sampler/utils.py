from __future__ import annotations

import numpy as np

import flax.nnx as nnx
import jax


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
