"""Grid-sampled composition space generation for high-entropy alloy (HEA) systems.

Generates a Cartesian-grid simplex sample for all subsystems up to a chosen
order, then deduplicates and reorders by active-element group to minimize
repeated thermodynamic initializations in downstream workflows.
"""

from __future__ import annotations

from itertools import combinations, compress

import numpy as np
import pandas as pd


def generate_compositions(
    elements: list[str],
    n_comps: int = 20,
    max_order: int | None = None,
    only_subsystems: list[list[str]] | None = None,
) -> pd.DataFrame:
    """Generate a uniformly sampled HEA composition space.

    Enumerates the integer-grid simplex points at resolution ``1/n_comps``
    for every subsystem of ``elements`` up to ``max_order`` components.
    Rows are grouped by their set of active (non-zero) elements to minimize
    repeated Thermo-Calc or CALPHAD initializations.

    Args:
        elements (list[str]): Pool of element symbols (will be sorted).
        n_comps (int): Grid resolution; each fraction is a multiple of
            ``1/n_comps``. Defaults to 20 (5 at.% spacing).
        max_order (int | None): Largest subsystem size. Defaults to
            ``len(elements)``.
        only_subsystems (list[list[str]] | None): If provided, only include
            subsystems whose element sets intersect this list. Defaults to
            ``None`` (all subsystems).

    Returns:
        DataFrame with one column per element. Rows sum to 1.0; absent
        elements are 0.0. Sorted by active-element group.

    Raises:
        ValueError: If ``max_order`` is less than 2 or exceeds element count.
    """
    elements = sorted(elements)
    if max_order is None:
        max_order = len(elements)
    if max_order < 2:
        raise ValueError(f"max_order must be >= 2, got {max_order}")
    if max_order > len(elements):
        raise ValueError(
            f"max_order ({max_order}) exceeds number of elements ({len(elements)})"
        )

    subsystems = [list(s) for s in combinations(elements, max_order)]
    if only_subsystems is not None:
        filter_sets = [set(s) for s in only_subsystems]
        subsystems = [s for s in subsystems if set(s) & set.union(*filter_sets)]

    grid = _simplex_grid(max_order, n_comps)

    frames: list[pd.DataFrame] = []
    for subset in subsystems:
        df = pd.DataFrame(grid, columns=subset)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=elements)

    result = pd.concat(frames, ignore_index=True).fillna(0.0)
    result = result.reindex(columns=elements, fill_value=0.0)
    result = result.drop_duplicates().reset_index(drop=True)
    result = _sort_by_active_group(result, elements)
    return result


def _simplex_grid(n_components: int, n_comps: int) -> np.ndarray:
    """Enumerate integer-grid simplex points for an n-component system.

    Args:
        n_components (int): Number of components.
        n_comps (int): Grid resolution.

    Returns:
        Array of shape ``(N, n_components)`` with rows summing to 1.0.
    """
    points: list[list[int]] = []
    for index in np.ndindex(*[n_comps + 1 for _ in range(n_components)]):
        if sum(index) == n_comps:
            points.append(list(index))
    return np.array(points, dtype=float) / n_comps


def _sort_by_active_group(df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """Sort rows by their active-element group for efficient downstream use.

    Groups compositions sharing the same set of non-zero elements together.
    Within each group, original order is preserved.

    Args:
        df (pd.DataFrame): Composition DataFrame with element columns.
        elements (list[str]): Ordered element list.

    Returns:
        Reordered DataFrame with the same rows, reset index.
    """
    seen: list[list[str]] = []
    group_order: dict[int, list[int]] = {}

    for row_idx in range(len(df)):
        comp = df.iloc[row_idx][elements]
        active = list(compress(elements, (comp > 0).tolist()))
        try:
            g = seen.index(active)
        except ValueError:
            g = len(seen)
            seen.append(active)
            group_order[g] = []
        group_order[g].append(row_idx)

    ordered_indices = [i for g in group_order.values() for i in g]
    return df.iloc[ordered_indices].reset_index(drop=True)
