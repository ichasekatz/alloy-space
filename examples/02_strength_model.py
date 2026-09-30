"""BCC solid-solution strength prediction with alloy_space.

Demonstrates the Maresca-Curtin strength model for a CrMoNbVW refractory HEA.
Computes the athermal yield stress and temperature-dependent flow stress.
"""

from __future__ import annotations

import numpy as np

from alloy_space import model_control
from alloy_space.strength_model import model, temp_model

# Cr5Mo45Nb30V10Ti10 (at.%)
elements = ["Cr", "Mo", "Nb", "V", "Ti"]
fractions = [0.05, 0.45, 0.30, 0.10, 0.10]

if __name__ == "__main__":
    tau_y0, delta_eb, tau_1573 = model_control(
        elements=elements,
        fractions=fractions,
        temperature=1573.0,
        eps_dot=0.001,
    )
    print(f"Athermal yield stress:       tau_y0    = {tau_y0:.1f} MPa")
    print(f"Activation barrier:          delta_Eb  = {delta_eb:.4f} eV")
    print(f"Yield stress at 1573 K:      tau(T)    = {tau_1573:.1f} MPa")

    # Temperature sweep 300–1800 K
    composition = dict(zip(elements, fractions, strict=True))
    result = model(composition)
    temps = np.linspace(300, 1800, 100)
    tau_T = temp_model(result, eps_dot=0.001, temperature=temps)
    print(f"\nTemp sweep: tau_y at 300K = {tau_T[0]:.1f} MPa, at 1800K = {tau_T[-1]:.1f} MPa")
