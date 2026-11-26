"""CasADi version of the pilot plant. Mirrors model/plant.py + model/pilot.py.

State x (14): moles of TG, DG, MG, E, A, GL, Cat, FFA, W, Soap, OH; V (L); T, T_J (K).
Control u (3): T_cmd (K, temperature-unit setpoint), u_cat (mol base/min dosed), F_A (mol MeOH/min dosed).
Parameter q (5): a_rel, eta, F_W (mol/min water ingress), k_loss (1/min methanol loss), phi (TCU).
Constants per batch (dict c): rho_cp, UA, V_J, rho_cp_J, F_J, and the nominal kinetic parameters.
Checked against plant.rhs in test_casadi_plant.py.
"""
import os, sys, json
import numpy as np
import casadi as ca
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from nz1997_params import K_REF, EA_CAL, R_CAL, T_REF   # noqa: E402
from plant import C_REF, MW, RHO                         # noqa: E402

SP = ["TG", "DG", "MG", "E", "A", "GL", "Cat", "FFA", "W", "Soap", "OH"]
IX = {s: i for i, s in enumerate(SP)}
iV, iT, iTJ = 11, 12, 13
NX, NU, NQ = 14, 3, 5
KN = ["k1", "k2", "k3", "k4", "k5", "k6"]
_th = json.load(open(os.path.join(HERE, "..", "model", "fit_kinetics_lit_veg.run.json")))["theta"]
KIN = dict(a=_th["a"], c_th=_th["c_th"], k_H=_th["k_H"], k_SE=_th["k_SE"], Ea_SE=_th["Ea_SE"],
           GL_sat=1e-3, w_th=2e-3, k_N=50.0, k_S=50.0, dH=-9.8e3)
NU_ = {"r1": dict(TG=-1, A=-1, DG=1, E=1), "r2": dict(DG=-1, A=-1, MG=1, E=1), "r3": dict(MG=-1, A=-1, GL=1, E=1),
       "rN": dict(FFA=-1, Cat=-1, Soap=1, A=1), "rW": dict(W=-1, Cat=-1, OH=1, A=1), "rH": dict(Cat=-1, OH=1),
       "rSE": dict(E=-1, OH=-1, Soap=1, A=1)}


def softplus(z):
    return ca.if_else(z > 30, z, ca.log1p(ca.exp(ca.fmin(z, 30))))


def f_expr(x, u, q, c, kin=KIN):
    V = x[iV]; T = x[iT]; TJ = x[iTJ]
    n = [ca.fmax(x[i], 0) for i in range(11)]
    cc = [ni / V for ni in n]
    k = [K_REF[kk] * ca.exp(-EA_CAL[kk] / R_CAL * (1 / T - 1 / T_REF)) for kk in KN]
    kSE = kin["k_SE"] * ca.exp(-kin["Ea_SE"] / 8.314462 * (1 / T - 1 / 333.15))
    gl_s = kin["GL_sat"] * (1 - ca.exp(-cc[IX["GL"]] / kin["GL_sat"]))
    cat = cc[IX["Cat"]]
    g = ca.fmax(kin["w_th"] * softplus((cat - kin["c_th"]) / kin["w_th"]), 0)
    fa = kin["a"] * q[0] * q[1] * g / C_REF
    TG, DG, MG, E, A = (cc[IX[s]] for s in ["TG", "DG", "MG", "E", "A"])
    r = dict(r1=fa * (k[0] * TG * A - k[1] * DG * E), r2=fa * (k[2] * DG * A - k[3] * MG * E),
             r3=fa * (k[4] * MG * A - k[5] * gl_s * E), rN=kin["k_N"] * cc[IX["FFA"]] * cat,
             rW=kin["k_S"] * cc[IX["W"]] * cat, rH=kin["k_H"] * cat, rSE=kSE * E * cc[IX["OH"]])
    dn = [0] * 11
    for rx, nu in NU_.items():
        for s, v in nu.items():
            dn[IX[s]] = dn[IX[s]] + v * r[rx] * V
    F_W, kl = q[2], q[3]
    lossA = kl * n[IX["A"]]
    dn[IX["W"]] = dn[IX["W"]] + F_W
    dn[IX["A"]] = dn[IX["A"]] - lossA + u[2]
    dn[IX["Cat"]] = dn[IX["Cat"]] + u[1]
    dV = (u[2] - lossA) * MW["MeOH"] / 1000 / RHO["MeOH"] + F_W * MW["H2O"] / 1000
    Tin = TJ + q[4] * (u[0] - TJ)
    q_rx = -kin["dH"] * (r["r1"] + r["r2"] + r["r3"]) * V
    dT = (q_rx + c["UA"] * (TJ - T)) / (c["rho_cp"] * V)
    dTJ = (c["F_J"] * c["rho_cp_J"] * (Tin - TJ) - c["UA"] * (TJ - T)) / (c["rho_cp_J"] * c["V_J"])
    return ca.vertcat(*dn, dV, dT, dTJ)


def make_f(c, kin=KIN):
    x = ca.MX.sym("x", NX); u = ca.MX.sym("u", NU); q = ca.MX.sym("q", NQ)
    return ca.Function("f", [x, u, q], [f_expr(x, u, q, c, kin)])


def make_integrator(c, dt, kin=KIN, method="cvodes", n_fe=10):
    x = ca.MX.sym("x", NX); u = ca.MX.sym("u", NU); q = ca.MX.sym("q", NQ)
    dae = {"x": x, "p": ca.vertcat(u, q), "ode": f_expr(x, u, q, c, kin)}
    if method == "cvodes":
        return ca.integrator("F", "cvodes", dae, 0.0, dt, {"abstol": 1e-8, "reltol": 1e-6})
    if method == "collocation":
        return ca.integrator("F", "collocation", dae, 0.0, dt, {"number_of_finite_elements": n_fe,
                                                               "interpolation_order": 3, "collocation_scheme": "radau"})
    # explicit RK4 with n_fe substeps (expanded to SX for speed)
    xs = ca.SX.sym("x", NX); us = ca.SX.sym("u", NU); qs = ca.SX.sym("q", NQ)
    fs = ca.Function("fs", [xs, us, qs], [f_expr(xs, us, qs, c, kin)])
    h = dt / n_fe; xk = xs
    for _ in range(n_fe):
        k1 = fs(xk, us, qs); k2 = fs(xk + h / 2 * k1, us, qs); k3 = fs(xk + h / 2 * k2, us, qs); k4 = fs(xk + h * k3, us, qs)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Fr = ca.Function("Fr", [xs, ca.vertcat(us, qs)], [xk], ["x0", "p"], ["xf"])
    return Fr
