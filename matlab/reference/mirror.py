"""NumPy mirror of the MATLAB supervisor (ftc_rhs.m, RK4, single-shooting NMPC, GRU estimator).
Used to check the MATLAB algorithm against the CasADi/IPOPT NMPC before porting it line by line."""
import os, sys, time
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["model", "control", "estimator", "faults"]:
    sys.path.insert(0, os.path.join(HERE, "..", d))
import casadi_plant as cp

K50 = np.array([0.05, 0.11, 0.215, 1.228, 0.242, 0.007]); EA = np.array([13145, 9932, 19860, 14639, 6421, 9588.0])
NU = np.array([[-1, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0], [0, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0], [0, 0, -1, 1, -1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, -1, -1, 0, 1, 0], [0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1], [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1],
               [0, 0, 0, -1, 1, 0, 0, 0, 0, 1, -1]], float)


def params(c):
    k = cp.KIN
    return dict(a=k["a"], c_th=k["c_th"], k_H=k["k_H"], k_SE=k["k_SE"], Ea_SE=k["Ea_SE"], GL_sat=1e-3, w_th=2e-3,
                k_N=50.0, k_S=50.0, dH=-9.8e3, C_REF=0.0365, **c)


def rhs(x, u, q, P):
    n = np.maximum(x[:11], 0); V, T, TJ = x[11], x[12], x[13]
    c = n / V
    k = K50 * np.exp(-EA / 1.9872 * (1 / T - 1 / 323.15))
    kSE = P["k_SE"] * np.exp(-P["Ea_SE"] / 8.314462 * (1 / T - 1 / 333.15))
    gl = P["GL_sat"] * (1 - np.exp(-c[5] / P["GL_sat"]))
    z = (c[6] - P["c_th"]) / P["w_th"]
    sp = z if z > 30 else np.log1p(np.exp(z))
    g = max(P["w_th"] * sp, 0.0)
    fa = P["a"] * q[0] * q[1] * g / P["C_REF"]
    r = np.array([fa * (k[0] * c[0] * c[4] - k[1] * c[1] * c[3]), fa * (k[2] * c[1] * c[4] - k[3] * c[2] * c[3]),
                  fa * (k[4] * c[2] * c[4] - k[5] * gl * c[3]), P["k_N"] * c[7] * c[6], P["k_S"] * c[8] * c[6],
                  P["k_H"] * c[6], kSE * c[3] * c[10]])
    dn = NU.T @ r * V
    lossA = q[3] * n[4]
    dn[8] += q[2]; dn[4] += -lossA + u[2]; dn[6] += u[1]
    dV = (u[2] - lossA) * 32.042 / 1000 / 0.791 + q[2] * 18.015 / 1000
    Tin = TJ + q[4] * (u[0] - TJ)
    dT = (-P["dH"] * (r[0] + r[1] + r[2]) * V + P["UA"] * (TJ - T)) / (P["rho_cp"] * V)
    dTJ = (P["F_J"] * P["rho_cp_J"] * (Tin - TJ) - P["UA"] * (TJ - T)) / (P["rho_cp_J"] * P["V_J"])
    return np.r_[dn, dV, dT, dTJ]


def rk4(x, u, q, P, dt=5.0, n=40):
    h = dt / n
    for _ in range(n):
        k1 = rhs(x, u, q, P); k2 = rhs(x + h / 2 * k1, u, q, P); k3 = rhs(x + h / 2 * k2, u, q, P); k4 = rhs(x + h * k3, u, q, P)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class SSNMPC:
    """Single shooting: decisions U (H x 3); soft T limit as a quadratic penalty (= optimal slack)."""
    def __init__(self, P, H=6, T_hi=273.15 + 63.5, ucat_max=1.0, FA_max=20.0, dTmax=15.0, n_rk=40,
                 w=dict(X=100.0, cat=5.0, A=1.0, du=0.05, s=1e3)):
        self.P, self.H, self.T_hi, self.w, self.n_rk = P, H, T_hi, w, n_rk
        self.lb = np.tile([298.15, 0.0, 0.0], H); self.ub = np.tile([353.15, ucat_max, FA_max], H)
        self.dTmax = dTmax; self.U0 = None
        self.sc = np.tile([10.0, ucat_max, FA_max], H)        # decision scaling

    def _sim(self, U, x, q, nact):
        X = [x];
        for k in range(nact):
            x = rk4(x, U[k], q, self.P, 5.0, self.n_rk); X.append(x)
        return X

    def solve(self, x, q, TG0, nb0, nA0, u_prev, nact):
        H, w = self.H, self.w
        if self.U0 is None:
            self.U0 = np.tile(u_prev, H)
        def J(v):
            U = (v * self.sc).reshape(H, 3); X = self._sim(U, x, q, nact)
            j, up = 0.0, u_prev
            for k in range(nact):
                sl = max(X[k + 1][12] - self.T_hi, 0.0)
                j += w["cat"] * U[k, 1] * 5 / nb0 + w["A"] * U[k, 2] * 5 / nA0 + w["du"] * ((U[k, 0] - up[0]) / 10) ** 2 + w["s"] * sl ** 2
                up = U[k]
            return j - w["X"] * X[-1][3] / (3 * TG0)
        A = np.zeros((2 * H, 3 * H)); b = np.zeros(2 * H)
        cons = []
        for k in range(nact):
            def g(v, k=k):
                U = (v * self.sc).reshape(H, 3); prev = u_prev[0] if k == 0 else U[k - 1, 0]
                return np.array([self.dTmax - (U[k, 0] - prev), self.dTmax + (U[k, 0] - prev)])
            cons.append(dict(type="ineq", fun=g))
        t0 = time.perf_counter()
        r = minimize(J, self.U0 / self.sc, method="SLSQP", bounds=list(zip(self.lb / self.sc, self.ub / self.sc)),
                     constraints=cons, options=dict(maxiter=100, ftol=1e-7))
        U = (r.x * self.sc).reshape(H, 3)
        self.U0 = np.r_[U[1:].ravel(), U[-1]]
        X = self._sim(U, x, q, nact)
        return U[0], dict(t_wall=time.perf_counter() - t0, XE_pred_end=float(X[-1][3] / (3 * TG0)), success=r.success, nit=r.nit)
