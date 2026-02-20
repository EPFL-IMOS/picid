# Datasources

This repository provides unified access to a wide collection of **datasources** for prognostics, health management (PHM), forecasting, and anomaly detection. Each datasource is implemented as a dedicated loader (under `picid.data.datasources`) that standardizes raw data into the framework’s container format, enabling reproducible experiments across domains.

In this framework, a **datasource** is *data-centric* (raw files, I/O, split layout). The data it produces is held in [data objects](dataobject.md) (e.g. `SplitDatasetContainer`). Task-centric access—e.g. sliding windows or fit-predict tables—is provided by [datasets](datasets.md), which consume that container output.

---

## Summary table

| Datasource | Module | Domain / type | Tasks supported | Key features |
|------------|--------|----------------|-----------------|--------------|
| **agtf30k** | `agtf30k` | Industrial time-series | Forecasting, anomaly detection | Large-scale equipment sensor data; multiple channels and operating conditions |
| **battery** | `battery` | Electrochemical / energy storage | SOH estimation, RUL prediction, degradation modeling | Multiple chemistries and cycling protocols; voltage, current, temperature |
| **concepts_n_cmapss** | `concepts_n_cmapss` | Turbofan engine prognostics | Degradation prediction, feature engineering, predictive maintenance | N-CMAPSS with domain concepts and extended feature representations |
| **ETTh** | `ETTh` | Energy / transformer temperature | Time-series forecasting, trend and seasonal analysis | Hourly (ETTh1, ETTh2) and 15-minute (ETTm1, ETTm2) variants; ETT benchmark |
| **MZVAV** | `MZVAV` | Building / HVAC | Forecasting, fault detection | Multi-zone VAV system sensor and control data |
| **nb14** | `nb14` | Battery / NASA | RUL prediction, degradation | NASA battery randomized cycling; ragged cycle-length structure |
| **phmd_battery** | `phmd_battery` | Battery PHM | Failure prediction, lifecycle modeling | PHM community benchmarks for battery degradation and RUL |
| **phmd_cbmv3** | `phmd_cbmv3` | Condition-based maintenance | Diagnostics, prognostics | CBM benchmark (v3) multi-source loader |
| **phmd_hsf15** | `phmd_hsf15` | PHM challenge | Diagnostics, prognostics | HSF15 benchmark; multi-source loader |
| **phmd_n_cmapss** | `phmd_n_cmapss` | Turbofan engine | Fault detection, diagnostics, prognostics | Extended CMAPSS with additional sensors and failure modes |
| **phmd_phme20** | `phmd_phme20` | PHM challenge | Diagnostics, prognostics | PHM 2020 benchmark; multi-source loader |
| **phmd_pronostia** | `phmd_pronostia` | Bearings / rotating machinery | Fault diagnosis, RUL estimation, early fault detection | Pronostia test rig; vibration under varied loads and speeds |
| **phmd_pubd16** | `phmd_pubd16` | Battery PHM | Degradation and failure analysis, SOH and RUL | Battery experiments; end-of-life and degradation patterns |
| **phmd_xjtu_sy** | `phmd_xjtu_sy` | Bearings / rotating machinery | Fault detection, prognostics | XJTU-SY test rig; vibration and rotational speed |
| **picasso** | `picasso` | Time-series / contextual | Forecasting, contextual modeling | Picasso loader; past/future context and target features |
| **railway** | `railway` | Railway equipment and infrastructure | Predictive maintenance, anomaly detection, reliability | Multisensor monitoring across railway subsystems |
| **unibo** | `unibo` | Industrial / power tools | RUL, degradation | UNIBO powertools data; ragged cycle-length structure |
| **kubric_movie** | `kubric_movie` | Video / synthetic | Research, multi-modal | Kubric MOVi dataset loader; video/simulation data |

---

## Datasource descriptions

### agtf30k

Large-scale industrial time-series dataset for forecasting and anomaly detection. Captures diverse equipment and sensor types, enabling benchmarking of predictive models.

### battery

Collection of battery datasets covering multiple chemistries and experimental protocols. Includes voltage, current, temperature, and internal resistance. Supports research on State-of-Health (SOH), Remaining Useful Life (RUL), and degradation modeling.

### concepts_n_cmapss

Augmented version of the **N-CMAPSS** dataset for turbofan engine prognostics. Enriches the original with domain concepts and extended feature representations for concept-based modeling and predictive maintenance.

### ETTh

**Electricity Transformer Temperature (ETT)** dataset: oil temperature and load data over two years. Hourly (ETTh1, ETTh2) and 15-minute (ETTm1, ETTm2) variants. Widely used for long-horizon time-series forecasting benchmarks.

### MZVAV

Multi-zone variable air volume (VAV) building/HVAC dataset. Provides sensor and control time-series for forecasting and fault detection in building systems.

### nb14 (NASA Battery)

NASA randomized battery cycling data. Each unit has **features**, **targets**, and **unit_id** as ragged arrays:

- **Features:** `(n_cycles, cycle_length, n_features)` — `cycle_length` is ragged per cycle.
- **Targets:** `(n_cycles, 1)` — one target value per cycle.
- **Unit ID:** `(1,)`.

### phmd_battery

Battery datasets from the PHM community for degradation modeling, lifecycle analysis, and early failure detection.

### phmd_cbmv3

Condition-based maintenance (CBM) benchmark v3. Multi-source PHM loader for diagnostics and prognostics experiments.

### phmd_hsf15

HSF15 PHM challenge dataset. Multi-source loader for diagnostics and prognostics.

### phmd_n_cmapss

Extended **CMAPSS** with additional sensors and fault modes for advanced turbofan engine diagnostics and prognostics.

### phmd_phme20

PHM 2020 challenge dataset. Multi-source loader for diagnostics and prognostics benchmarks.

### phmd_pronostia

Bearing dataset from the Pronostia test rig. Vibration data under different loads and speeds; supports RUL estimation and fault classification.

### phmd_pubd16

Battery dataset for degradation and failure analysis. Cycling data under varied conditions for SOH and RUL prediction.

### phmd_xjtu_sy

Bearing dataset from the **XJTU-SY** test rig. Vibration and rotational speed signals for fault detection and prognostics.

### picasso

Time-series datasource with past/future context and target features. Used for contextual forecasting and related tasks.

### railway

Railway equipment and infrastructure data. Multisensor monitoring across subsystems for predictive maintenance and anomaly detection.

### unibo (UNIBO Powertools)

UNIBO powertools dataset. Each unit has **features**, **targets**, and **unit_id** as ragged arrays:

- **Features:** `(n_cycles, cycle_length, n_features)` — ragged `cycle_length`.
- **Targets:** `(n_cycles, 1)` — one target per cycle.
- **Unit ID:** `(1,)`.

### kubric_movie

Loader for the Kubric MOVi dataset. Provides video/synthetic data for research and multi-modal experiments.

---

## References

- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. [ETT dataset repository](https://github.com/zhouhaoyi/ETDataset)
- IEEE PHM Society Data Challenge archives (battery, turbofan, bearing datasets)
- Nectoux et al., *PRONOSTIA: An experimental platform for bearings accelerated degradation tests*, PHM 2012
- XJTU-SY Bearing dataset: Xi’an Jiaotong University test rig benchmark for rotating machinery
