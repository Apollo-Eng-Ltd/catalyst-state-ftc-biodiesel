# Catalyst-state estimation and fault-tolerant NMPC for batch biodiesel transesterification

Code for the simulation study of a K2CO3-catalysed batch transesterification reactor with feedstock and equipment faults. It covers the plant model and its fit to laboratory yields, the fault dataset, fault detection and classification, a recurrent catalyst-state estimator (with an EKF for comparison), an economic NMPC with a calibrated stop rule, the Monte Carlo comparison, and a Simulink version of the closed loop.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Extraction of the 2013 yield data |
| `data/` | Yield points and replicates, metadata of the generated batch sets |
| `model/` | Plant model (Noureddini and Zhu kinetics with neutralisation, hydroxide formation and saponification), kinetic fit, pilot reactor |
| `faults/` | Fault library, measurement model, dataset generation |
| `classifier/` | Residuals, detection statistic, classifiers and confusion matrices |
| `estimator/` | GRU estimator, physics loss, unlabelled adaptation, EKF, trained weights |
| `control/` | CasADi plant, NMPC, stop rules, Monte Carlo runs, statistics, economics, sensitivity runs |
| `figures/` | Tables and figures built from the logged results |
| `matlab/` | Script-built Simulink model `ftc_biodiesel.slx`, NMPC (fmincon), GRU port, headline runs |
| `matlab/reference/` | Python references and the NumPy mirror used to check the MATLAB port |

## Setup

Python 3.11 or later.

```
pip install -r requirements.txt
```

The MATLAB part was run in MATLAB Online with Simulink and the Optimization Toolbox.

## Running

Scripts are run from their own folder. In the order used for the paper:

```
cd model      && python3 test_plant.py && python3 fit_kinetics.py lit_veg && python3 fit_kinetics.py lit_jco_kH && python3 test_pilot.py
cd faults     && python3 generate_dataset.py && python3 gen_pool.py
cd classifier && python3 residuals.py && python3 train_classifier.py
cd control    && python3 gen_excitation.py
cd estimator  && python3 train_estimator.py data 0 60 && python3 train_estimator.py data_cl 0 60 --cl && python3 evaluate.py
cd control    && python3 calibrate_stoprule.py && python3 calibrate_stoprule_ekf.py
cd control    && python3 monte_carlo2.py 30 && python3 mc2_stats.py && python3 economics.py
cd control    && python3 weight_sensitivity.py && python3 mc2_sideheat.py
cd control    && python3 batch_calibration.py && python3 revision_runs.py && python3 revision_stats.py
cd classifier && python3 loadcell_noise.py
cd figures    && python3 build_tables.py && python3 make_paper_figures.py
```

The other estimator variants (label fraction, physics weight, adaptation to the Jatropha pool) are listed in the usage line of `train_estimator.py`. The generated datasets are not in the repository; `generate_dataset.py` and `gen_excitation.py` rebuild them from fixed seeds. The Monte Carlo results (`control/*.jsonl`) and trained weights (`*.pt`) are included, so the tables and figures can be rebuilt without rerunning the simulations.

In MATLAB:

```
cd matlab
build_ftc_sim        % builds ftc_biodiesel.slx
ftc_run_headline     % four headline batches under four controllers, compared with matlab/reference/refs_headline.json
```

`ftc_sup_data.json` holds the estimator weights, scaling and noise sequence exported by `matlab/reference/make_refs.py`.

## Data source

The laboratory yields come from P. Boakye, MSc dissertation, KNUST (2013). The values in `data/` were transcribed from its appendix tables; see `scripts/extract_boakye2013.py`.
