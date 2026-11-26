"""Receding-horizon economic NMPC for the batch.

Multiple shooting, CVODES on each 5-min interval, IPOPT. Horizon H intervals (default 6 = 30 min),
truncated at the latest batch end (t_max). Decision per interval: T_cmd (TCU setpoint), u_cat (catalyst
top-up, mol base/min), F_A (methanol make-up, mol/min); all bounded, with rate limits on T_cmd.
Objective (minimise):
   -w_X X_E(end of horizon)                         maximise ester conversion
   + w_cat * sum(u_cat dt) / n_base0                catalyst used, as a fraction of the charge
   + w_A   * sum(F_A dt)   / n_A0                   methanol used, fraction of the charge
   + w_du  * sum((dT_cmd / 10 K)^2)                 move suppression
   + w_s   * sum(slack^2)                           soft upper temperature limit T <= T_hi
Parameters q = [a_rel, eta, F_W, k_loss, phi] come from the estimator (FT-NMPC) or are nominal
(nominal NMPC). The state is supplied by the caller.
"""
import numpy as np, casadi as ca
import casadi_plant as cp

DT_C = 5.0


class NMPC:
    def __init__(self, c, H=6, T_hi=273.15 + 63.5, ucat_max=1.0, FA_max=20.0, dTmax=15.0,
                 w=dict(X=100.0, cat=5.0, A=1.0, du=0.05, s=1e3), method="cvodes", n_fe=10):
        self.H, self.T_hi, self.w = H, T_hi, w
        F = cp.make_integrator(c, DT_C, method=method, n_fe=n_fe)
        nx, nu, nq = cp.NX, cp.NU, cp.NQ
        X0 = ca.MX.sym("X0", nx); Q = ca.MX.sym("Q", nq); P = ca.MX.sym("P", 4)  # [xscale TG0, n_base0, n_A0, Hact]
        xs = ca.MX.sym("xs", nx)                                                  # state scaling
        U_prev = ca.MX.sym("Up", nu)
        w_, lbw, ubw, g, lbg, ubg = [], [], [], [], [], []
        J = 0
        Xk = X0
        Up = U_prev
        mask = ca.MX.sym("mask", H)                                               # 1 = interval active
        self.ix = []
        for k in range(H):
            Uk = ca.MX.sym(f"U{k}", nu); w_ += [Uk]
            lbw += [298.15, 0.0, 0.0]; ubw += [353.15, ucat_max, FA_max]
            Xn = F(x0=Xk, p=ca.vertcat(Uk, Q))["xf"]
            Xn = mask[k] * Xn + (1 - mask[k]) * Xk                                # inactive intervals hold
            Sk = ca.MX.sym(f"S{k}", nx); w_ += [Sk]; lbw += [-np.inf] * nx; ubw += [np.inf] * nx
            g += [(Sk - Xn) / xs]; lbg += [0.0] * nx; ubg += [0.0] * nx           # shooting continuity
            sl = ca.MX.sym(f"sl{k}"); w_ += [sl]; lbw += [0.0]; ubw += [np.inf]
            g += [Sk[cp.iT] - sl]; lbg += [-np.inf]; ubg += [T_hi]
            g += [Uk[0] - Up[0]]; lbg += [-dTmax]; ubg += [dTmax]
            J += mask[k] * (w["cat"] * Uk[1] * DT_C / P[1] + w["A"] * Uk[2] * DT_C / P[2]
                            + w["du"] * ((Uk[0] - Up[0]) / 10.0) ** 2 + w["s"] * sl ** 2)
            Xk, Up = Sk, Uk
        J += -w["X"] * Xk[cp.IX["E"]] / (3 * P[0])
        nlp = {"x": ca.vertcat(*w_), "f": J, "g": ca.vertcat(*g), "p": ca.vertcat(X0, Q, P, xs, U_prev, mask)}
        self.solver = ca.nlpsol("nmpc", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0,
                                                        "ipopt.max_iter": 150, "ipopt.tol": 1e-6})
        self.lbw, self.ubw, self.lbg, self.ubg = lbw, ubw, lbg, ubg
        self.nw = len(lbw); self.w0 = None

    def solve(self, x, q, TG0, n_base0, n_A0, u_prev, n_active):
        H = self.H
        mask = np.array([1.0 if k < n_active else 0.0 for k in range(H)])
        xs = np.r_[np.full(11, TG0), x[cp.iV], 300.0, 300.0]
        if self.w0 is None or len(self.w0) != self.nw:
            w0 = []
            for k in range(H):
                w0 += list(u_prev) + list(x) + [0.0]
            self.w0 = np.array(w0)
        par = np.r_[x, q, [TG0, n_base0, n_A0, n_active], xs, u_prev, mask]
        import time as _t; _t0 = _t.perf_counter()
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=par)
        wopt = np.array(sol["x"]).ravel()
        stats = self.solver.stats()
        # warm start: shift one interval
        blk = 3 + cp.NX + 1
        self.w0 = np.r_[wopt[blk:], wopt[-blk:]]
        u0 = wopt[:3]
        xH = wopt[(n_active - 1) * blk + 3:(n_active - 1) * blk + 3 + cp.NX] if n_active > 0 else x
        return u0, dict(success=stats["success"], status=stats["return_status"], t_wall=_t.perf_counter() - _t0, iters=stats.get("iter_count"),
                        XE_pred_end=float(xH[cp.IX["E"]] / (3 * TG0)))
