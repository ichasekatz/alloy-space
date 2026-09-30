"""HEA composition space generation with alloy_space.

Generates a uniformly sampled grid over a refractory multi-component alloy
space, grouping rows by active-element set for efficient thermodynamic evaluation.
"""

from __future__ import annotations

from pathlib import Path

from alloy_space import generate_compositions

elements = ["Hf", "Mo", "Nb", "Ta", "Ti", "Zr"]

if __name__ == "__main__":
    # All quinary subsystems at 5 at.% resolution
    df = generate_compositions(elements=elements, n_comps=20, max_order=5)
    print(f"Generated {len(df):,} compositions over {len(elements)}-element space")
    print(df.head())

    out = Path("hea_compositions.csv")
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
