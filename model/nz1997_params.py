"""Noureddini H, Zhu D (1997) Kinetics of transesterification of soybean oil.
J Am Oil Chem Soc 74(11):1457-1463. doi:10.1007/s11746-997-0254-2

Verified 16 Nov 2026 against a clean scan of the original (Tables 1 and 2, Eq. 2-4, Fig. 1).
Conditions: NaOH 0.20 wt% of oil, MeOH:oil 6:1, 50 C, N_Re = 6200 (300 rpm), 1500 mL reactor.
Concentrations in mol/L (Fig. 1 axis), time in min -> second-order k in L mol^-1 min^-1.
Table 2 prints no units; the units follow from Eq. 3 and Fig. 1.
"""
import numpy as np

R_CAL = 1.98720  # cal mol^-1 K^-1
T_REF = 323.15   # 50 C, the temperature of Table 2

# Table 2, "Without shunt reaction" (the authors conclude the shunt is not needed)
K_REF = dict(k1=0.050, k2=0.110, k3=0.215, k4=1.228, k5=0.242, k6=0.007)
# Table 2, "With shunt reaction" (for reference only)
K_REF_SHUNT = dict(k1=0.049, k2=0.102, k3=0.218, k4=1.280, k5=0.239, k6=0.007,
                   k7=7.84e-5, k8=1.58e-5)

# Table 1, Arrhenius k = A exp(-E/RT), N_Re = 6200, cal/mol
EA_CAL = dict(k1=13145, k2=9932, k3=19860, k4=14639, k5=6421, k6=9588)
# Table 1, other columns (for sensitivity checks)
EA_CAL_MODARR_6200 = dict(k1=11707, k2=8482, k3=18439, k4=13433, k5=7937, k6=10992)
EA_CAL_ARR_12400 = dict(k1=13600, k2=9783, k3=18769, k4=11177, k5=5182, k6=9873)

EA_J = {k: v * 4.184 for k, v in EA_CAL.items()}


def k_at(T_K, kref=K_REF, ea=EA_CAL):
    """Rate constants at T_K, anchored on the 50 C values of Table 2."""
    return {k: kref[k] * np.exp(-ea[k] / R_CAL * (1.0 / T_K - 1.0 / T_REF)) for k in kref}
