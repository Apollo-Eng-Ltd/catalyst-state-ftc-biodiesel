"""Pilot-scale batch reactor for the fault and control studies.

Sources (checked 16 Nov 2026):
  [U]   Overall coefficient, jacketed carbon/stainless vessel: 400 W m-2 K-1 heating, 350 cooling.
        Thermopedia, "Agitated vessel heat transfer", doi 10.1615/AtoZ.a.agitated_vessel_heat_transfer, Table 2.
  [CPO] Soybean oil cp = 2.038 kJ kg-1 K-1 at 333.15 K; rho = 1039.225 - 0.397 T kg m-3.
        Garcia Rojas EE, Coimbra JSR, Telis-Romero J (2013) Int J Food Prop 16(7):1620-1629,
        doi 10.1080/10942912.2011.604889, Table 3 and correlations.
  [CPM] Methanol liquid Cp = 86.2 J mol-1 K-1 at 323 K (valid 323-353 K), Hough et al. (1950),
        via NIST Chemistry WebBook, SRD 69 -> 2.69 kJ kg-1 K-1.
  [DH]  Heat of transesterification -9.8 +/- 0.9 kJ per mol ester (rapeseed oil + MeOH, 40 C, ITC).
        Fjerbaek Sotoft L, Westh P, Christensen KV, Norddahl B (2010) Thermochimica Acta, pp. 84-90.
        Enzymatic route; same overall reaction, used as the per-step enthalpy.
  Neutralisation, methoxide hydrolysis and saponification heats are set to 0 in the base case.
  OPEN ITEM: not yet sourced. Methoxide -> hydroxide is close to thermoneutral; FFA neutralisation
  and ester saponification are exothermic. A crude -57 kJ/mol bound on every side step gives up to
  ~35 % of the main exotherm in nominal batches, so a sensitivity case is required (test_pilot.py).

Design choices (stated, not sourced):
  Vessel 1.2 m3 nominal, D = 1.0 m; charge 600 kg oil + methanol at the chosen ratio (~0.84 m3).
  Wetted jacket area = pi D H + 1.08 pi D^2/4 (dished bottom).
  Jacket: water, 50 mm annulus -> V_J = A x 0.05 m, fed at F_J = 150 L/min by a temperature
  control unit (TCU) that delivers T_J,in in [25, 80] C. T_J,in is the manipulated variable.
  The exotherm (-9.8 kJ/mol ester) gives an adiabatic rise of ~13 K, released mostly in the first
  10 min, so the jacket must cool as well as heat; a heater-only jacket overshoots past 64.7 C.
  Kinetics assume the lab's kinetically controlled regime (eta = 1) is reached at pilot scale.
"""
import json, os
import numpy as np
from plant import Params, initial_state, MW, RHO

HERE = os.path.dirname(os.path.abspath(__file__))

PILOT = dict(
    D_m=1.0,
    m_oil_kg=600.0,
    U_heat_W_m2K=400.0,         # [U]
    U_cool_W_m2K=350.0,         # [U]
    cp_oil_J_kgK=2038.0,        # [CPO]
    cp_MeOH_J_kgK=86.2 / 32.042 * 1000.0,  # [CPM]
    rho_cp_water_J_LK=4.18e3 * 0.983,      # water at 60 C
    jacket_gap_m=0.05,
    F_J_Lmin=150.0,             # recirculation, 9 m3/h
    Q_h_max_W=40e3,             # used only for the heat-up check
    T_Jin_min_K=298.15,
    T_Jin_max_K=353.15,
    dH_trans_J_mol=-9.8e3,      # [DH]
    T_max_K=273.15 + 64.7,      # methanol normal boiling point
)

FEEDS = dict(  # adopted kinetic fits (hydroxide pathway, k_SE from Eze et al. 2015)
    veg=os.path.join(HERE, "fit_kinetics_lit_veg.run.json"),
    jco=os.path.join(HERE, "fit_kinetics_lit_jco_kH.run.json"),
)
AV_2013 = dict(CSFO=0.452, SBFO=0.396, BFVO=0.441, CWVO=0.920, SWVO=1.088, BWVO=1.017, JCO=2.606)
MW_OIL = dict(CSFO=848.3, CWVO=848.3, SBFO=874.0, SWVO=874.0, BFVO=861.2, BWVO=861.2, JCO=870.0)


def geometry(V_L, D=PILOT["D_m"]):
    H = (V_L / 1000.0) / (np.pi * D ** 2 / 4)
    A = np.pi * D * H + 1.08 * np.pi * D ** 2 / 4
    return H, A


def feed_theta(group):
    return json.load(open(FEEDS[group]))["theta"]


def pilot_case(oil="SWVO", ratio=6.0, cat_wt=4.0, AV=None, water_wt=0.0002, T0_C=60.0, extra=None, group=None):
    """Return (x0, Params) for a pilot batch. AV defaults to the 2013 value for the oil.
    Nominal water 0.02 wt% (pre-dried feed; the 2013 oils were dried to < 0.05 wt%)."""
    AV = AV_2013[oil] if AV is None else AV
    m_g = PILOT["m_oil_kg"] * 1000.0
    x0 = initial_state(m_oil_g=m_g, MW_oil=MW_OIL[oil], ratio=ratio, cat_g=cat_wt / 100 * m_g,
                       AV=AV, water_wt=water_wt, T_C=T0_C)
    V = x0[-4]  # I_V
    m_MeOH = ratio * m_g / MW_OIL[oil] * MW["MeOH"] / 1000.0
    rho_cp = (PILOT["m_oil_kg"] * PILOT["cp_oil_J_kgK"] + m_MeOH * PILOT["cp_MeOH_J_kgK"]) / V  # J L-1 K-1
    _, A = geometry(V)
    th = feed_theta(group or ("jco" if oil == "JCO" else "veg"))   # group="veg": operator's nominal model
    p = Params(a=th["a"], GL_sat=1e-3, c_th=th["c_th"], k_SE=th["k_SE"], Ea_SE=th["Ea_SE"], k_H=th["k_H"],
               k_N=50.0, k_S=50.0,  # fast: tau < 0.1 min at 0.2 mol/L base; larger values only add stiffness
               isothermal=False, rho_cp=rho_cp,
               UA=PILOT["U_heat_W_m2K"] * A * 60.0,                  # J min-1 K-1
               V_J=A * PILOT["jacket_gap_m"] * 1000.0,               # L
               rho_cp_J=PILOT["rho_cp_water_J_LK"],
               dH=dict(r1=PILOT["dH_trans_J_mol"], r2=PILOT["dH_trans_J_mol"], r3=PILOT["dH_trans_J_mol"],
                       rN=0.0, rW=0.0, rH=0.0, rSE=0.0))
    for k, v in (extra or {}).items():
        setattr(p, k, v)
    return x0, p
