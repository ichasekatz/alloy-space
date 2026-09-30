<div align="center">

# alloy-space

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**HEA composition space sampling and BCC solid-solution strength prediction.**

</div>

## Overview

`alloy-space` provides two tools for high-entropy alloy (HEA) design:

- **`generate_compositions`** — enumerate a uniform simplex grid over N-component alloy systems, grouped by active-element set for efficient downstream thermodynamic evaluation (Thermo-Calc, CALPHAD, etc.)
- **`model_control` / `model` / `temp_model`** — Maresca–Curtin BCC solid-solution strengthening model: predict the athermal yield stress and temperature-dependent flow stress from composition-weighted elastic constants and atomic volumes

Consolidates the `Alloy-Space` and `Graph-Generator` repositories.

## Installation

```bash
git clone https://github.com/ichasekatz/alloy-space.git
cd alloy-space
uv sync
```

## Quick Start

```python
from alloy_space import generate_compositions, model_control

elements = ["Hf", "Mo", "Nb", "Ta", "W"]

# Composition space
df = generate_compositions(elements=elements, n_comps=20, max_order=5)
print(f"{len(df):,} compositions")

# Strength prediction for equimolar
tau_y0, delta_Eb, tau_1573 = model_control(
    elements=elements,
    fractions=[0.2, 0.2, 0.2, 0.2, 0.2],
    temperature=1573.0,
)
print(f"tau_y0 = {tau_y0:.1f} MPa, tau(1573 K) = {tau_1573:.1f} MPa")
```

## API

### Composition Generation

```python
generate_compositions(
    elements: list[str],
    n_comps: int = 20,        # 1/n_comps at.% spacing
    max_order: int | None = None,
    only_subsystems: list[list[str]] | None = None,
) -> pd.DataFrame
```

### Strength Model

```python
model(composition: dict[str, float], alpha: float = 1/12) -> dict
temp_model(results: dict, eps_dot: float, temperature: float | np.ndarray) -> float | np.ndarray
model_control(elements, fractions, temperature=1573.0, eps_dot=0.001) -> tuple[float, float, float]
```

Returns `(tau_y_0 [MPa], delta_Eb [eV], tau_T [MPa])`.

**Supported elements:** W, Mo, Ta, Nb, V, Al, Ti, Zr, Hf, Cr, Fe, Ni, Co, Re, Ru, Cu, Mn, Au, Ag, Pt, B

## Examples

| Script | Description |
|--------|-------------|
| `examples/01_generate_compositions.py` | Generate refractory HEA composition space |
| `examples/02_strength_model.py` | Predict yield stress vs. temperature |

```bash
uv run python examples/02_strength_model.py
```

## Running Tests

```bash
uv run pytest -v
```

## References

Maresca, F.; Curtin, W.A. *Mechanistic origin of high retained strength in refractory BCC high entropy alloys up to 1900K.* Acta Mater. **2020**, 182, 235–249. https://doi.org/10.1016/j.actamat.2019.10.015

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{katz_alloy_space_2026,
  author = {Katz, Chase},
  title  = {alloy-space: HEA composition sampling and BCC strength modeling},
  year   = {2026},
  url    = {https://github.com/ichasekatz/alloy-space},
}
```
