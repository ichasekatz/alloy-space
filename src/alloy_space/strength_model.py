"""BCC solid-solution strengthening model (Fleischer/Labusch approach).

Predicts the athermal yield stress and thermal activation barrier for BCC
refractory high-entropy alloys from composition-weighted elastic constants
and atomic volumes.

References:
    Maresca, F.; Curtin, W.A. Mechanistic origin of high retained strength
    in refractory BCC high entropy alloys up to 1900K. Acta Mater. 2020,
    182, 235-249. https://doi.org/10.1016/j.actamat.2019.10.015
"""

from __future__ import annotations

import numpy as np

# Elastic constants [C11, C12, C44] in GPa for common BCC and FCC metals.
# Values from DFT calculations compiled in Maresca & Curtin (2020).
_elastic_constants: dict[str, list[float]] = {
    "W": [517.8, 201.7, 139.4],
    "Mo": [466.0, 165.2, 99.5],
    "Ta": [260.9, 165.2, 70.4],
    "Nb": [247.2, 140.0, 14.2],
    "V": [272.0, 144.8, 17.6],
    "Al": [38.7, 79.2, 33.0],
    "Ti": [95.9, 115.9, 40.3],
    "Zr": [81.8, 94.3, 30.2],
    "Hf": [73.7, 117.0, 51.7],
    "Cr": [247.6, 73.4, 48.3],
    "Fe": [279.2, 148.8, 93.0],
    "Ni": [214.3, 148.6, 75.0],
    "Co": [129.3, 140.9, 93.5],
    "Re": [325.0, 380.2, 158.6],
    "Ru": [46.6, 401.1, 173.4],
    "Cu": [168.0, 121.0, 75.0],
    "Mn": [256.9, 272.2, 105.4],
    "Au": [192.9, 163.0, 42.0],
    "Ag": [124.0, 93.4, 46.1],
    "Pt": [304.0, 255.0, 54.0],
    "B": [448.0, 152.0, 237.0],
}

# Atomic volumes in Å³ from DFT relaxed BCC unit cells.
_atomic_volumes: dict[str, float] = {
    "W": 16.229,
    "Mo": 15.956,
    "Ta": 18.313,
    "Nb": 18.342,
    "V": 13.453,
    "Al": 17.08,
    "Ti": 17.123,
    "Zr": 22.885,
    "Hf": 22.128,
    "Cr": 11.575,
    "Fe": 11.358,
    "Ni": 11.012,
    "Co": 11.07,
    "Re": 15.135,
    "Cu": 12.077,
    "Mn": 10.985,
    "Ru": 14.348,
    "Au": 18.42,
    "Ag": 10.29,
    "Pt": 9.095,
    "B": 7.3,
}

# Conversion factor: GPa·Å³ → eV
_gpa_angstrom3_to_ev = 1.0 / 160.2176621


def model(
    composition: dict[str, float],
    alpha: float = 1 / 12,
    elastic_constants: dict[str, list[float]] | None = None,
    atomic_volumes: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compute BCC solid-solution strengthening parameters for an alloy composition.

    Uses the Maresca–Curtin model to predict the athermal yield stress
    ``tau_y_0`` and the thermal activation barrier ``delta_Eb`` from
    composition-weighted elastic constants and misfit volumes.

    Args:
        composition (dict[str, float]): Mapping of element symbol to atomic
            fraction (values must sum to 1.0).
        alpha (float): Dislocation interaction parameter. Defaults to 1/12.
        elastic_constants (dict[str, list[float]] | None): Override for
            ``[C11, C12, C44]`` in GPa per element. Defaults to built-in table.
        atomic_volumes (dict[str, float] | None): Override for atomic volumes
            in Å³ per element. Defaults to built-in table.

    Returns:
        Dictionary with keys:
            - ``tau_y_0`` (float): Athermal yield stress in MPa.
            - ``delta_Eb`` (float): Thermal activation barrier in eV.
            - ``mu_bar`` (float): Effective shear modulus in GPa.
            - ``nu_bar`` (float): Effective Poisson ratio.
            - ``bar_V`` (float): Average atomic volume in Å³.
            - ``misfit`` (dict[str, float]): Volume misfit per element in Å³.

    Raises:
        KeyError: If an element is not found in the elastic constants or
            atomic volume tables.
    """
    c_table = elastic_constants if elastic_constants is not None else _elastic_constants
    v_table = atomic_volumes if atomic_volumes is not None else _atomic_volumes

    bar_c11 = sum(c_table[el][0] * fr for el, fr in composition.items())
    bar_c12 = sum(c_table[el][1] * fr for el, fr in composition.items())
    bar_c44 = sum(c_table[el][2] * fr for el, fr in composition.items())
    bar_v = sum(v_table[el] * fr for el, fr in composition.items())

    misfit = {el: v_table[el] - bar_v for el in composition}
    misfit_vol_factor = sum(fr * misfit[el] ** 2 for el, fr in composition.items())

    mu_bar = np.sqrt(0.5 * bar_c44 * (bar_c11 - bar_c12))
    # lattice parameter a (Å), then <111>/2 Burgers vector
    b_bar_actual = (2 * bar_v) ** (1 / 3) * np.sqrt(3) / 2
    nu_bar = (3 * (bar_c11 + 2 * bar_c12) / 3 - 2 * mu_bar) / (2 * ((bar_c11 + 2 * bar_c12) / 3 + mu_bar))

    tau_y_0 = (
        0.040
        * alpha ** (-1 / 3)
        * mu_bar
        * ((1 + nu_bar) / (1 - nu_bar)) ** (4 / 3)
        * (misfit_vol_factor / b_bar_actual**6) ** (2 / 3)
    )

    delta_eb_gpa_a3 = (
        2.00
        * alpha ** (1 / 3)
        * mu_bar
        * b_bar_actual**3
        * ((1 + nu_bar) / (1 - nu_bar)) ** (2 / 3)
        * (misfit_vol_factor / b_bar_actual**6) ** (1 / 3)
    )
    delta_eb_ev = delta_eb_gpa_a3 * _gpa_angstrom3_to_ev

    return {
        "tau_y_0": tau_y_0,
        "delta_Eb": delta_eb_ev,
        "mu_bar": mu_bar,
        "nu_bar": nu_bar,
        "bar_V": bar_v,
        "misfit": misfit,
    }


def temp_model(
    results: dict[str, object],
    eps_dot: float,
    temperature: float | np.ndarray,
    approx: bool = True,
) -> float | np.ndarray:
    """Compute temperature-dependent yield stress from athermal model results.

    Applies the thermally-activated flow model of Maresca & Curtin to
    compute the resolved shear stress at a given strain rate and temperature.

    Args:
        results (dict[str, object]): Output of :func:`model`.
        eps_dot (float): Applied strain rate in s⁻¹.
        temperature (float | np.ndarray): Temperature(s) in K.
        approx (bool): Use the exponential approximation (``True``) valid at
            high temperature, or the piecewise low/high-T model (``False``).
            Defaults to ``True``.

    Returns:
        Resolved shear stress ``tau`` in MPa (same shape as ``temperature``).
    """
    eps_dot_0 = 1e4  # reference strain rate, s⁻¹
    boltzmann = 8.617e-5  # eV/K

    tau_y_0: float = results["tau_y_0"]  # type: ignore[assignment]
    delta_eb: float = results["delta_Eb"]  # type: ignore[assignment]

    kt_ratio = boltzmann * temperature / delta_eb
    log_ratio = np.log(eps_dot_0 / eps_dot)

    if approx:
        return tau_y_0 * np.exp(-(1 / 0.55) * kt_ratio * log_ratio)

    tau_low = tau_y_0 * (1 - (kt_ratio * log_ratio) ** (2 / 3))
    tau_high = tau_y_0 * np.exp(-(1 / 0.55) * kt_ratio * log_ratio)

    temperature = np.asarray(temperature)
    tau = np.where(tau_low / tau_y_0 > 0.5, tau_low, tau_high)
    return float(tau) if tau.ndim == 0 else tau


def model_control(
    elements: list[str],
    fractions: list[float],
    temperature: float = 1573.0,
    eps_dot: float = 0.001,
    elastic_constants: dict[str, list[float]] | None = None,
    atomic_volumes: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """High-level wrapper: compute yield stress for an alloy at one temperature.

    Args:
        elements (list[str]): Element symbols.
        fractions (list[float]): Corresponding atomic fractions (must sum ~1).
        temperature (float): Temperature in K. Defaults to 1573 K.
        eps_dot (float): Strain rate in s⁻¹. Defaults to 0.001.
        elastic_constants (dict[str, list[float]] | None): Override table.
        atomic_volumes (dict[str, float] | None): Override table.

    Returns:
        Tuple of ``(tau_y_0, delta_Eb, tau_T)`` where ``tau_y_0`` is the
        athermal yield stress in MPa, ``delta_Eb`` is the activation barrier
        in eV, and ``tau_T`` is the temperature-dependent yield stress in MPa.
    """
    composition = dict(zip(elements, fractions, strict=True))
    result = model(composition, elastic_constants=elastic_constants, atomic_volumes=atomic_volumes)
    tau_t = temp_model(result, eps_dot=eps_dot, temperature=temperature, approx=True)
    return result["tau_y_0"], result["delta_Eb"], tau_t
