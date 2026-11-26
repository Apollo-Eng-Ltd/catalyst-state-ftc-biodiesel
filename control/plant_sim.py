"""Stepwise closed-loop plant simulator.

Same plant, faults, hidden mismatch and measurement model as faults/faults.py, but advanced one
control interval at a time so that a controller can act on what it has measured. Inputs per step:
  mode "pi":   T_sp (K) for the inner PI loop on the TCU setpoint (Kc = 6, tau_I = 8 min, back-calc. AW)
  mode "direct": T_cmd (K) given directly (NMPC)
  plus u_cat (mol base/min, catalyst top-up) and F_A (mol MeOH/min, make-up), held over the step.
Measurements every 0.25 min: T (+bias), T_J, T_Jin, T_cmd, Q_J, P_st, pH, m_ld, and the known inputs
u_cat, F_Ain. Truth (states, fault parameters) is kept for evaluation only.
Check: test_plant_sim.py reproduces faults.simulate_batch under PI to solver tolerance.
"""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from plant import rhs, X_E, IX, I_T, I_TJ, I_V, algebraic       # noqa: E402
from pilot import pilot_case, PILOT, AV_2013                      # noqa: E402
from faults import (magnitude, ramp, P0_W, KC, TAUI, DT, PH_REF, C_B_REF, MW_FAME, RHO_FAME)  # noqa: E402

MEAS = ["T", "T_J", "T_Jin", "T_cmd", "Q_J", "P_st", "pH", "m_ld", "u_cat", "F_Ain"]


class Batch:
    def __init__(self, case, seed=0):
        self.case = case
        cls = case.get("cls", "N"); self.cls = cls
        m = magnitude(cls, case.get("s", 0.0), case.get("sign", 1.0)); self.m = m
        AV0 = case.get("AV", AV_2013[case["oil"]]) + (m if cls == "F1" else 0.0)
        x0, p = pilot_case(case["oil"], ratio=case.get("ratio", 6.0), cat_wt=case.get("cat_wt", 4.0), AV=AV0,
                           water_wt=case.get("water_wt", 0.0002), T0_C=case.get("T0_C", 55.0),
                           group=case.get("group", None))
        mm = case.get("mismatch", {})
        p.a *= mm.get("a", 1.0); p.k_H *= mm.get("k_H", 1.0); p.c_th *= mm.get("c_th", 1.0); p.UA *= mm.get("UA", 1.0)
        if "k_SE" in mm:
            p.k_SE = mm["k_SE"]
        if cls == "F3":
            p.a *= m
        self.p, self.x0 = p, x0
        self.z = np.append(x0, 0.0)                    # plant state + PI integral
        self.t = 0.0
        self.t_f, self.shape, self.tau_r = case.get("t_f", 0.0), case.get("shape", "abrupt"), case.get("tau_r", 20.0)
        self.rng = np.random.default_rng(seed)
        self.m0 = PILOT["m_oil_kg"] + x0[IX["A"]] * 32.042 / 1000.0 + x0[IX["Cat"]] * 138.205 / 1000.0
        self.mass_in = 0.0                              # cumulative dosed and lost mass, kg
        self.log = {k: [] for k in MEAS + ["t", "X_E", "cat_act", "T_true", "state", "fpar", "fault_active"]}
        self.cat_dosed = 0.0; self.meoh_dosed = 0.0
        self._record(self.z, 0.0, dict(mode="pi", T_sp=self.z[I_T], u_cat=0.0, F_A=0.0))

    def faults_at(self, t, x):
        cls, m = self.cls, self.m
        r = ramp(t, self.t_f, self.shape, self.tau_r) if cls in ("F2", "E1", "E2", "E3", "E4") else 0.0
        phi = 1.0 - r * (1.0 - m) if cls == "E1" else 1.0
        nu = 1.0 - r * (1.0 - m) if cls == "E2" else 1.0
        b = r * m if cls == "E4" else 0.0
        F_W = r * (m / 100.0 * PILOT["m_oil_kg"] * 1000.0 / 18.015) / 10.0 if cls == "F2" else 0.0
        F_loss = r * m * x[IX["A"]] if cls == "E3" else 0.0
        return phi, nu, b, F_W, F_loss

    def _tcmd(self, z, b, act):
        if act["mode"] == "direct":
            return min(max(act["T_cmd"], PILOT["T_Jin_min_K"]), PILOT["T_Jin_max_K"]), act["T_cmd"]
        raw = act["T_sp"] + KC * ((act["T_sp"] - (z[I_T] + b)) + z[-1] / TAUI)
        return min(max(raw, PILOT["T_Jin_min_K"]), PILOT["T_Jin_max_K"]), raw

    def _f(self, t, z, act):
        x = z[:-1]
        phi, nu, b, F_W, F_loss = self.faults_at(t, x)
        self.p.eta = nu
        Tc, raw = self._tcmd(z, b, act)
        Tin = x[I_TJ] + phi * (Tc - x[I_TJ])
        dx = rhs(t, x, self.p, lambda tt: dict(T_Jin=Tin, F_J=PILOT["F_J_Lmin"], F_W=F_W,
                                               F_A=act["F_A"] - F_loss, u_cat=act["u_cat"]))
        dI = ((act["T_sp"] - (x[I_T] + b)) + (Tc - raw) / KC) if act["mode"] == "pi" else 0.0
        return np.append(dx, dI)

    def _record(self, z, t, act):
        x = z[:-1]
        phi, nu, b, F_W, F_loss = self.faults_at(t, x)
        Tc, _ = self._tcmd(z, b, act)
        Tin = x[I_TJ] + phi * (Tc - x[I_TJ])
        _, _, _, cat = algebraic(x, self.p)
        V_bp = (x[IX["A"]] * 32.042 / 0.791 + x[IX["E"]] * MW_FAME / RHO_FAME) / 1000.0
        pH = PH_REF + np.log10(max(x[IX["Cat"]] + x[IX["OH"]], 1e-12) / V_bp / C_B_REF)
        L = self.log
        n = lambda sd: self.rng.normal(0.0, sd)
        Tm, TJm, Tinm = x[I_T] + b + n(0.05), x[I_TJ] + n(0.05), Tin + n(0.05)
        L["t"].append(t); L["T"].append(Tm - 273.15); L["T_J"].append(TJm - 273.15); L["T_Jin"].append(Tinm - 273.15)
        L["T_cmd"].append(Tc - 273.15)
        L["Q_J"].append(PILOT["F_J_Lmin"] * self.p.rho_cp_J * (Tinm - TJm) / 60.0 / 1000.0)
        L["P_st"].append(P0_W * nu ** 3 + n(0.02 * P0_W)); L["pH"].append(pH + n(0.02))
        L["m_ld"].append(self.m0 + self.mass_in + n(0.2)); L["u_cat"].append(act["u_cat"]); L["F_Ain"].append(act["F_A"])
        L["X_E"].append(X_E(x, self.x0)); L["cat_act"].append(cat); L["T_true"].append(x[I_T] - 273.15)
        L["state"].append(x.copy())
        L["fpar"].append([nu, phi, b, F_W, -F_loss, self.m if self.cls == "F3" else 1.0, self.m if self.cls == "F1" else 0.0])
        L["fault_active"].append(float(self.cls != "N" and t >= self.t_f))

    def step(self, act, dt=5.0):
        """act: dict(mode="pi", T_sp=K) or dict(mode="direct", T_cmd=K); plus u_cat, F_A (default 0)."""
        act = dict(dict(u_cat=0.0, F_A=0.0), **act)
        if act["mode"] == "direct":
            act["T_sp"] = 0.0
        t0 = self.t; t_ev = t0 + DT * np.arange(1, int(round(dt / DT)) + 1)
        s = solve_ivp(self._f, (t0, t0 + dt), self.z, args=(act,), method="LSODA", t_eval=t_ev,
                      rtol=1e-6, atol=1e-9, max_step=1.0)
        if not s.success:
            raise RuntimeError(s.message)
        for k, tk in enumerate(s.t):
            zk = s.y[:, k]
            # mass bookkeeping for the load cell (trapezoid on the 0.25-min grid)
            xprev = self.z[:-1] if k == 0 else s.y[:-1, k - 1]
            fl = lambda x_, tt: (act["F_A"] - self.faults_at(tt, x_)[4]) * 32.042 / 1000.0 \
                + self.faults_at(tt, x_)[3] * 18.015 / 1000.0 + act["u_cat"] * 138.205 / 1000.0
            self.mass_in += 0.5 * (fl(xprev, tk - DT) + fl(zk[:-1], tk)) * DT
            self._record(zk, tk, act)
        self.z = s.y[:, -1]; self.t = t0 + dt
        self.cat_dosed += act["u_cat"] * dt; self.meoh_dosed += act["F_A"] * dt
        self.p.eta = 1.0
        return self

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.log.items()}
