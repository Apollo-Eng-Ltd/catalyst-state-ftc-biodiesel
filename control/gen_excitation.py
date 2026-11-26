"""Closed-loop excitation dataset for the estimator.

Same fault design, severity bands, onset windows and hidden mismatch as faults/generate_dataset.py,
but every batch also carries random manipulated-input excitation, as the NMPC will apply:
  setpoint: new value U(55, 61) C every 10 min (inner PI on the TCU); U(55, 63) put 10 % of batches
            above 64.7 C, outside the model's validity
  catalyst top-up: with p = 0.5, one 5-min dose of 0.25-1.5 wt% K2CO3 (of oil) at U(5, 60) min
  methanol make-up: with p = 0.5, U(0, 20) mol/min for U(10, 30) min starting at U(5, 60) min
Output data/cl_<split>_X.npz, _S.npz, _meta.csv in the format of the open-loop dataset, with two extra
measured channels (u_cat, F_Ain). Usage: python3 gen_excitation.py [n_train] [n_test]
"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from generate_dataset import make_cases, OUT, STATE_NAMES, FPAR   # noqa: E402
from plant_sim import Batch, MEAS                                  # noqa: E402
from faults import DT                                              # noqa: E402

TRUTH = ["X_E", "cat_act", "fault_active"]


def run(c):
    rng = np.random.default_rng(c["seed"] + 777)
    sched_T = 273.15 + rng.uniform(55, 61, 12)
    dose = (rng.uniform(5, 60), rng.uniform(0.25, 1.5)) if rng.random() < 0.5 else None
    meoh = (rng.uniform(5, 60), rng.uniform(10, 30), rng.uniform(0, 20)) if rng.random() < 0.5 else None
    try:
        b = Batch(c, seed=c["seed"])
        for k in range(24):
            t = 5.0 * k
            u_cat = (dose[1] / 100 * 600e3 / 138.205 / 5.0) if dose and dose[0] <= t < dose[0] + 5 else 0.0
            F_A = meoh[2] if meoh and meoh[0] <= t < meoh[0] + meoh[1] else 0.0
            b.step(dict(mode="pi", T_sp=sched_T[k // 2], u_cat=u_cat, F_A=F_A))
        A = b.arrays()
    except Exception as e:
        return c, None, str(e)
    X = np.stack([A[k] for k in MEAS] + [A["X_E"], A["cat_act"], A["fault_active"]], 1).astype(np.float32)
    st = np.asarray(A["state"])
    S = np.concatenate([st[:, :15] if st.shape[1] >= 15 else np.c_[st, np.zeros((len(st), 15 - st.shape[1]))],
                        np.asarray(A["fpar"])], 1).astype(np.float32)
    info = dict(dose_t=dose[0] if dose else np.nan, dose_wt=dose[1] if dose else 0.0,
                meoh_t=meoh[0] if meoh else np.nan, meoh_rate=meoh[2] if meoh else 0.0,
                Tmax=float(np.max(A["T_true"])))
    return dict(c, **info), (X, S), None


def main(n_train, n_test):
    t0 = time.time(); log = {}
    for split, n, seed in [("train", n_train, 11), ("test_interp", n_test, 12), ("test_extrap", n_test, 13)]:
        cases = make_cases(split, n, seed)
        with Pool(os.cpu_count()) as pool:
            res = pool.map(run, cases, chunksize=4)
        ok = [(c, a) for c, a, e in res if e is None]; fails = [e for c, a, e in res if e is not None]
        X = np.stack([a[0] for _, a in ok]); S = np.stack([a[1] for _, a in ok])
        rows = []
        for c, _ in ok:
            r = {k: v for k, v in c.items() if not k.startswith("_") and k != "mismatch"}
            r.update({f"mm_{k}": v for k, v in c["mismatch"].items()}); rows.append(r)
        meta = pd.DataFrame(rows); meta.insert(0, "batch_id", [f"cl_{split}_{i:05d}" for i in range(len(meta))])
        meta.to_csv(f"{OUT}/cl_{split}_meta.csv", index=False)
        np.savez_compressed(f"{OUT}/cl_{split}_X.npz", X=X, channels=np.array(MEAS + TRUTH), dt=DT)
        np.savez_compressed(f"{OUT}/cl_{split}_S.npz", S=S, channels=np.array(STATE_NAMES + FPAR), dt=DT)
        log[split] = dict(n=len(meta), failed=len(fails), fail_examples=fails[:3], T_over_64p7=int((meta.Tmax > 64.7).sum()))
        print(split, log[split], round(time.time() - t0), "s", flush=True)
    json.dump(dict(run_utc=datetime.datetime.now(datetime.UTC).isoformat(), splits=log), open(f"{OUT}/cl_dataset.run.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250, int(sys.argv[2]) if len(sys.argv) > 2 else 80)
