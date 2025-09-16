# Datasets

This repository provides unified access to a wide collection of datasets for prognostics, health management, forecasting, and anomaly detection.  
Each dataset is accessible through dedicated loaders that standardize preprocessing and enable reproducible experiments.

---

## Summary Table

| Dataset            | Domain / Type                         | Tasks Supported                                                        | Key Features                                                                 |
|--------------------|----------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **agtf30k**        | Industrial time-series                 | Forecasting, anomaly detection                                          | Large-scale equipment sensor data; multiple channels and operating conditions |
| **battery**        | Electrochemical / Energy storage       | SOH estimation, RUL prediction, degradation modeling                    | Diverse battery chemistries, cycling protocols, current/voltage/temperature   |
| **concepts_n_cmapss**| Turbofan engine prognostics           | Degradation prediction, feature engineering, predictive maintenance     | N-CMAPSS with domain concepts and extended feature representations            |
| **ETTh**           | Energy / Transformer temperature       | Time-series forecasting, trend and seasonal analysis                    | Hourly and minute-level variants; widely used in long-horizon forecasting     |
| **phmd_battery**   | Battery PHM                            | Failure prediction, lifecycle modeling                                  | Benchmark datasets from PHM community for battery degradation and RUL studies |
| **phmd_n_cmapss**  | Turbofan engine diagnostics            | Fault detection, diagnostics, prognostics                               | Extended CMAPSS with additional sensors and failure modes                     |
| **phmd_pronostia** | Bearings / Rotating machinery          | Fault diagnosis, RUL estimation, early fault detection                  | Rolling bearing experiments with varied loads, speeds, and degradation paths  |
| **phmd_pubd16**    | Battery PHM                            | Degradation and failure analysis, SOH and RUL prediction                | Battery experiments focused on end-of-life and degradation patterns           |
| **phmd_xjtu_sy**   | Bearings / Rotating machinery          | Fault detection, prognostics                                            | Bearing vibration and speed signals from XJTU-SY test rig                     |
| **railway**        | Railway equipment and infrastructure   | Predictive maintenance, anomaly detection, reliability analysis         | Multisensor monitoring across railway subsystems and components               |

---

## Datasources

### agtf30k
Large-scale industrial time-series dataset for forecasting and anomaly detection.  
Captures diverse equipment and sensor types, enabling benchmarking of predictive models.

### battery
Collection of battery datasets covering multiple chemistries and experimental protocols.  
Includes measurements such as voltage, current, temperature, and internal resistance.  
Supports research on State-of-Health (SOH), Remaining Useful Life (RUL), and degradation modeling.

### concepts_n_cmapss
An augmented version of the **N-CMAPSS** dataset for turbofan engine prognostics.  
Enriches the original dataset with domain concepts, improving feature engineering  
and enabling concept-based modeling for predictive maintenance.

### ETTh
**Electricity Transformer Temperature (ETT)** dataset containing oil temperature and load data  
collected over two years. Provides hourly (ETTh1, ETTh2) and 15-minute (ETTm1, ETTm2) variants.  
Widely used as a benchmark for long-horizon time-series forecasting.

### phmd_battery
Battery datasets from the Prognostics and Health Management (PHM) community.  
Designed for experiments on degradation modeling, lifecycle analysis, and early failure detection.

### phmd_n_cmapss
Extended version of the **CMAPSS** dataset with additional sensors and fault modes.  
Supports advanced diagnostics and prognostics tasks in turbofan engine health management.

### phmd_pronostia
Bearing dataset collected on the Pronostia test rig.  
Includes vibration data under different loads and speeds, capturing fault progression  
and supporting remaining useful life estimation and fault classification.

### phmd_pubd16
Battery dataset focused on degradation and failure analysis.  
Provides cycling data under varied operating conditions, enabling battery health assessment  
and predictive modeling of SOH and RUL.

### phmd_xjtu_sy
Bearing dataset from the **XJTU-SY** test rig.  
Contains vibration signals and rotational speed measurements for rotating machinery.  
Supports fault detection, early fault prediction, and prognostics.

### railway
Railway equipment dataset for predictive maintenance and anomaly detection.  
Covers multiple subsystems and sensor modalities, supporting comprehensive reliability analysis.

---

## References

- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. [ETT dataset repository](https://github.com/zhouhaoyi/ETDataset)  
- IEEE PHM Society Data Challenge archives (battery, turbofan, and bearing datasets)  
- Nectoux et al., *PRONOSTIA: An experimental platform for bearings accelerated degradation tests*, PHM 2012  
- XJTU-SY Bearing dataset: Xi’an Jiaotong University test rig benchmark for rotating machinery  
