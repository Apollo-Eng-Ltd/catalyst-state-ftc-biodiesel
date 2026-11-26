"""Batch transesterification plant model.

Mole-based states in a single reacting (ester/oil/methanol) phase, with glycerol
partitioning into a separate phase once it exceeds a solubility limit (option A,
chosen 16 Nov 2026). Units: mol, L, min, K, J.

Reactions (rates in mol L^-1 min^-1, on the reacting-phase concentrations):
  r1  TG + A <-> DG + E          k1, k2   (Noureddini and Zhu 1997)
  r2  DG + A <-> MG + E          k3, k4
  r3  MG + A <-> GL + E          k5, k6   (reverse uses dissolved glycerol only)
  rN  FFA + Cat -> Soap + A      k_N      neutralisation of free fatty acid by methoxide
  rW  W + Cat -> OH + A          k_S      feed water hydrolyses methoxide to hydroxide
  rH  Cat -> OH                  k_H      methoxide -> hydroxide by water released from KHCO3
                                          (K2CO3 + MeOH <-> MeOK + KHCO3); first order in total base
  rSE OH + E -> Soap + A         k_SE     saponification by hydroxide only (methoxide cannot saponify)
Only methoxide (Cat) catalyses transesterification; hydroxide is counted as spent catalyst.

Rate constants: k_i = a * eta * g(c_cat,act)/C_REF * k_i,NZ(T), g(c) = w softplus((c - c_th)/w), and
C_REF = 0.0365 mol/L (0.20 wt% NaOH in the 1997 runs).

Glycerol phase (algebraic):
  c_GL,s = GL_sat * (1 - exp(-c_GL,tot / GL_sat))     dissolved, enters r3 reverse
  n_GL,g = n_GL,tot - c_GL,s V                          separate glycerol phase
  c_cat,act = c_cat / (1 + K_cg * n_GL,g / V)            optional catalyst loss to glycerol phase

Conserved (checked by `invariants`):
  glyceride backbone  TG + DG + MG + GL
  fatty-acid chains   3TG + 2DG + MG + E + FFA + Soap
  potassium (base)    Cat + OH + Soap - (catalyst dosed)
"""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.integrate import solve_ivp
from nz1997_params import K_REF, EA_CAL, R_CAL, T_REF

SP = ["TG", "DG", "MG", "E", "A", "GL", "Cat", "FFA", "W", "Soap", "OH"]
IX = {s: i for i, s in enumerate(SP)}
I_V, I_T, I_TJ, I_CATIN = len(SP), len(SP) + 1, len(SP) + 2, len(SP) + 3
NX = len(SP) + 4

MW = dict(K2CO3=138.205, MeOH=32.042, H2O=18.015, KOH=56.106, GL=92.094)
RHO = dict(oil=0.914, MeOH=0.791)  # kg/L
C_REF = 0.0365                     # mol/L, base in Noureddini-Zhu runs


@dataclass
class Params:
    # fitted to the 2013 yields (fit_kinetics.py)
    a: float = 1.0          # catalytic activity factor (methoxide fraction, K vs Na)
    GL_sat: float = 1e6     # mol/L dissolved glycerol limit; 1e6 = single phase (NZ limit)
    k_N: float = 1.0e3      # L mol^-1 min^-1, fast acid-base
    k_S: float = 1.0        # L mol^-1 min^-1 (literature value to be set; not identifiable from 2013 data)
    k_SE: float = 0.0       # L mol^-1 min^-1, hydroxide saponification of ester
    k_H: float = 0.0        # min^-1, methoxide -> hydroxide
    Ea_SE: float = 0.0      # J/mol
    K_cg: float = 0.0       # L/mol, catalyst uptake by glycerol phase
    c_th: float = 0.0       # mol/L, base held inactive (KHCO3 / undissolved K2CO3); 0 = linear
    w_th: float = 2e-3      # mol/L, smoothing width of the threshold
    # fault parameters (nominal values)
    eta: float = 1.0        # agitation multiplier on k_i
    phi: float = 1.0        # heater effectiveness
    b_T: float = 0.0        # temperature sensor bias, K (measurement model only)
    # thermal (lab validation runs use isothermal=True)
    isothermal: bool = True
    rho_cp: float = 1.9e3   # J L^-1 K^-1, oil/methanol mixture (PLACEHOLDER, to be sourced)
    UA: float = 600.0       # J min^-1 K^-1 (PLACEHOLDER, set for the pilot reactor)
    V_J: float = 0.5        # L
    rho_cp_J: float = 4.18e3  # J L^-1 K^-1, water
    dH: dict = field(default_factory=lambda: dict(r1=0.0, r2=0.0, r3=0.0, rN=-5.0e4, rW=0.0, rH=0.0, rSE=0.0))  # J/mol (PLACEHOLDER)


def rate_constants(T, p):
    k = {kk: K_REF[kk] * np.exp(-EA_CAL[kk] / R_CAL * (1.0 / T - 1.0 / T_REF)) for kk in K_REF}
    k_SE = p.k_SE * np.exp(-p.Ea_SE / 8.314462 * (1.0 / T - 1.0 / 333.15))
    return k, k_SE


def algebraic(x, p):
    V = x[I_V]
    c = x[:len(SP)] / V
    gl_s = p.GL_sat * (-np.expm1(-c[IX["GL"]] / p.GL_sat))
    n_gl_g = x[IX["GL"]] - gl_s * V
    cat_act = c[IX["Cat"]] / (1.0 + p.K_cg * n_gl_g / V)
    return c, gl_s, n_gl_g, cat_act


def rates(x, p):
    c, gl_s, _, cat = algebraic(x, p)
    c = np.maximum(c, 0.0)          # no negative concentrations inside rate laws
    cat = max(cat, 0.0)
    k, k_SE = rate_constants(x[I_T], p)
    z = (cat - p.c_th) / p.w_th
    g = p.w_th * (np.logaddexp(0.0, z) - (np.logaddexp(0.0, -p.c_th / p.w_th) if p.c_th == 0 else 0.0))
    f = p.a * p.eta * max(g, 0.0) / C_REF
    TG, DG, MG, E, A = (c[IX[s]] for s in ["TG", "DG", "MG", "E", "A"])
    r = dict(
        r1=f * (k["k1"] * TG * A - k["k2"] * DG * E),
        r2=f * (k["k3"] * DG * A - k["k4"] * MG * E),
        r3=f * (k["k5"] * MG * A - k["k6"] * gl_s * E),
        rN=p.k_N * c[IX["FFA"]] * cat,
        rW=p.k_S * c[IX["W"]] * cat,
        rH=p.k_H * c[IX["Cat"]],
        rSE=k_SE * E * c[IX["OH"]],
    )
    return r


# stoichiometry: species x reaction
NU = {
    "r1": dict(TG=-1, A=-1, DG=+1, E=+1),
    "r2": dict(DG=-1, A=-1, MG=+1, E=+1),
    "r3": dict(MG=-1, A=-1, GL=+1, E=+1),
    "rN": dict(FFA=-1, Cat=-1, Soap=+1, A=+1),
    "rW": dict(W=-1, Cat=-1, OH=+1, A=+1),
    "rH": dict(Cat=-1, OH=+1),
    "rSE": dict(E=-1, OH=-1, Soap=+1, A=+1),
}


def rhs(t, x, p, u):
    """u(t) -> dict(u_cat mol/min base, F_A mol/min MeOH, F_W mol/min water, T_Jin K, F_J L/min, Q_h J/min)."""
    ui = u(t)
    V = x[I_V]
    r = rates(x, p)
    dx = np.zeros(NX)
    for rx, nu in NU.items():
        for s, v in nu.items():
            dx[IX[s]] += v * r[rx] * V
    # inputs (methanol dosing at efficiency lambda is applied by the caller through F_A)
    dx[IX["Cat"]] += ui.get("u_cat", 0.0)
    dx[I_CATIN] = ui.get("u_cat", 0.0)
    F_A = ui.get("F_A", 0.0)          # methanol in (+) or lost (-), mol/min
    F_W = ui.get("F_W", 0.0)          # water ingress, mol/min
    dx[IX["A"]] += F_A
    dx[IX["W"]] += F_W
    dx[I_V] = F_A * MW["MeOH"] / 1000.0 / RHO["MeOH"] + F_W * MW["H2O"] / 1000.0
    if p.isothermal:
        dx[I_T] = 0.0
        dx[I_TJ] = 0.0
    else:
        T, TJ = x[I_T], x[I_TJ]
        q_rx = sum(-p.dH[rx] * r[rx] * V for rx in r)
        dx[I_T] = (q_rx + p.UA * (TJ - T)) / (p.rho_cp * V)
        dx[I_TJ] = (ui.get("F_J", 0.0) * p.rho_cp_J * (ui.get("T_Jin", TJ) - TJ)
                    - p.UA * (TJ - T) + p.phi * ui.get("Q_h", 0.0)) / (p.rho_cp_J * p.V_J)
    return dx


def initial_state(m_oil_g=100.0, MW_oil=870.0, ratio=6.0, cat_g=4.0, AV=0.44,
                  water_wt=0.0002, T_C=60.0, extra_MeOH_L=0.0):
    """Lab batch of the 2013 study. AV in mg KOH/g; FFA moles = AV m / 56.106 mg/mmol."""
    n_TG = m_oil_g / MW_oil
    n_A = ratio * n_TG + extra_MeOH_L * RHO["MeOH"] * 1000.0 / MW["MeOH"]
    V = m_oil_g / 1000.0 / RHO["oil"] + n_A * MW["MeOH"] / 1000.0 / RHO["MeOH"]
    x = np.zeros(NX)
    x[IX["TG"]] = n_TG
    x[IX["A"]] = n_A
    x[IX["Cat"]] = cat_g / MW["K2CO3"]            # base equivalents (K2CO3 -> CH3OK + KHCO3)
    x[IX["FFA"]] = AV * m_oil_g / 1000.0 / MW["KOH"]
    x[IX["W"]] = water_wt * m_oil_g / MW["H2O"]
    x[I_V], x[I_T], x[I_TJ] = V, T_C + 273.15, T_C + 273.15
    return x


def preneutralise(x0):
    """Apply rN instantaneously (k_N -> infinity): base consumed 1:1 by FFA at t = 0."""
    x = x0.copy()
    n = min(x[IX["FFA"]], x[IX["Cat"]])
    x[IX["FFA"]] -= n; x[IX["Cat"]] -= n; x[IX["Soap"]] += n; x[IX["A"]] += n
    return x


def invariants(x, x0):
    n = {s: x[IX[s]] for s in SP}
    n0 = {s: x0[IX[s]] for s in SP}
    gly = lambda d: d["TG"] + d["DG"] + d["MG"] + d["GL"]
    chain = lambda d: 3 * d["TG"] + 2 * d["DG"] + d["MG"] + d["E"] + d["FFA"] + d["Soap"]
    pot = lambda d: d["Cat"] + d["OH"] + d["Soap"]
    return dict(glyceride=gly(n) - gly(n0), chain=chain(n) - chain(n0),
                potassium=pot(n) - pot(n0) - (x[I_CATIN] - x0[I_CATIN]))


def X_E(x, x0):
    """Ester conversion on the initial fatty-chain basis (3 TG0)."""
    return x[IX["E"]] / (3.0 * x0[IX["TG"]])


def simulate(x0, p, t_end, u=lambda t: {}, t_eval=None, method="Radau", rtol=1e-8, atol=1e-12):
    return solve_ivp(rhs, (0.0, t_end), x0, args=(p, u), method=method, t_eval=t_eval,
                     rtol=rtol, atol=atol, dense_output=True)
