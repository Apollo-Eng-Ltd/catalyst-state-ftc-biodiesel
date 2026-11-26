"""PyTorch version of the plant rate laws and balances (for the physics loss).

Mirrors model/plant.py with the adopted parameters (vegetable-oil group, hydroxide pathway,
k_SE from Eze et al. 2015) and the pilot settings (model/pilot.py). Checked against plant.rhs in
test_torch_plant.py. States are scaled moles z_i = n_i / n_TG0 (species) and z_V = V / V0.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from nz1997_params import K_REF, EA_CAL, R_CAL, T_REF   # noqa: E402
from plant import C_REF                                   # noqa: E402

SP = ["TG", "DG", "MG", "E", "A", "GL", "Cat", "FFA", "W", "Soap", "OH"]
IX = {s: i for i, s in enumerate(SP)}
_th = json.load(open(os.path.join(HERE, "..", "model", "fit_kinetics_lit_veg.run.json")))["theta"]
PAR = dict(a=_th["a"], c_th=_th["c_th"], k_H=_th["k_H"], k_SE=_th["k_SE"], Ea_SE=_th["Ea_SE"],
           GL_sat=1e-3, w_th=2e-3, k_N=50.0, k_S=50.0, dH=-9.8e3)
KN = ["k1", "k2", "k3", "k4", "k5", "k6"]
K50 = torch.tensor([K_REF[k] for k in KN], dtype=torch.float64)
EAC = torch.tensor([EA_CAL[k] for k in KN], dtype=torch.float64)
# stoichiometry rows: r1 r2 r3 rN rW rH rSE ; columns SP
NU = torch.zeros(7, len(SP), dtype=torch.float64)
for j, d in enumerate([dict(TG=-1, A=-1, DG=1, E=1), dict(DG=-1, A=-1, MG=1, E=1), dict(MG=-1, A=-1, GL=1, E=1),
                       dict(FFA=-1, Cat=-1, Soap=1, A=1), dict(W=-1, Cat=-1, OH=1, A=1), dict(Cat=-1, OH=1),
                       dict(E=-1, OH=-1, Soap=1, A=1)]):
    for s, v in d.items():
        NU[j, IX[s]] = v


def softplus_thr(c, c_th, w):
    g = w * torch.nn.functional.softplus((c - c_th) / w, beta=1.0)
    if c_th == 0:
        g = g - w * np.log(2.0)
    return torch.clamp(g, min=0.0)


def rates(c, T, a_rel, eta, kSE60=None, kHm=None, cthm=None):
    """c: (..., 11) concentrations mol/L; T: (...) K; a_rel, eta: (...) multipliers. Optional per-sample
    uncertain parameters: kSE60 (k_SE at 60 C, L/mol/min), kHm, cthm (multipliers on k_H, c_th).
    Returns (..., 7)."""
    c = torch.clamp(c, min=0.0)
    k = K50 * torch.exp(-EAC / R_CAL * (1.0 / T.unsqueeze(-1) - 1.0 / T_REF))
    kSE = (PAR["k_SE"] if kSE60 is None else kSE60) * torch.exp(-PAR["Ea_SE"] / 8.314462 * (1.0 / T - 1.0 / 333.15))
    gl_s = PAR["GL_sat"] * (-torch.expm1(-c[..., IX["GL"]] / PAR["GL_sat"]))
    cat = c[..., IX["Cat"]]
    if cthm is None:
        g = softplus_thr(cat, PAR["c_th"], PAR["w_th"])
    else:
        g = torch.clamp(PAR["w_th"] * torch.nn.functional.softplus((cat - PAR["c_th"] * cthm) / PAR["w_th"]), min=0.0)
    f = PAR["a"] * a_rel * eta * g / C_REF
    TG, DG, MG, E, A = (c[..., IX[s]] for s in ["TG", "DG", "MG", "E", "A"])
    r = torch.stack([
        f * (k[..., 0] * TG * A - k[..., 1] * DG * E),
        f * (k[..., 2] * DG * A - k[..., 3] * MG * E),
        f * (k[..., 4] * MG * A - k[..., 5] * gl_s * E),
        PAR["k_N"] * c[..., IX["FFA"]] * cat,
        PAR["k_S"] * c[..., IX["W"]] * cat,
        PAR["k_H"] * (1.0 if kHm is None else kHm) * cat,
        kSE * E * c[..., IX["OH"]],
    ], dim=-1)
    return r


def dz_dt(z, zV, T, a_rel, eta, F_W, k_loss, nTG0, V0, kSE60=None, kHm=None, cthm=None):
    """Scaled species balance. z (...,11), zV (...), T K, F_W mol/min, k_loss min^-1 (methanol loss)."""
    nTG0 = torch.as_tensor(nTG0, dtype=z.dtype); V0 = torch.as_tensor(V0, dtype=z.dtype)
    V = zV * V0                                       # same shape as zV
    n3 = nTG0.unsqueeze(-1) if nTG0.dim() > 0 else nTG0
    c = z * n3 / V.unsqueeze(-1)
    r = rates(c, T, a_rel, eta, kSE60, kHm, cthm)     # mol L^-1 min^-1
    dn = (r @ NU) * V.unsqueeze(-1)                   # mol/min
    dn = dn.clone()
    dn[..., IX["W"]] = dn[..., IX["W"]] + F_W
    lossA = k_loss * z[..., IX["A"]] * nTG0
    dn[..., IX["A"]] = dn[..., IX["A"]] - lossA
    dV = (-lossA * 32.042 / 1000.0 / 0.791 + F_W * 18.015 / 1000.0)
    return dn / n3, dV / V0, r


def dT_dt(T, TJ, r, zV, V0, rho_cp, UA):
    """Reactor energy balance, K/min. r (...,7) mol L^-1 min^-1."""
    V = zV * torch.as_tensor(V0, dtype=zV.dtype)
    q = -PAR["dH"] * (r[..., 0] + r[..., 1] + r[..., 2]) * V   # J/min
    return (q + UA * (TJ - T)) / (rho_cp * V)
