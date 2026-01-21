#!/bin/bash
bash ./container/rt_run_task.sh paper-diagnostics-baselines-s1  ./scripts/paper_diagnostics/baselines_s1.sh
bash ./container/rt_run_task.sh paper-diagnostics-baselines-s2  ./scripts/paper_diagnostics/baselines_s2.sh
bash ./container/rt_run_task.sh paper-diagnostics-fit-predict-s1  ./scripts/paper_diagnostics/fit_predict_s1.sh
bash ./container/rt_run_task.sh paper-diagnostics-fit-predict-s2  ./scripts/paper_diagnostics/fit_predict_s2.sh


bash ./container/rt_run_task.sh paper-prognostics-baselines-s1  ./scripts/paper_prognostics/baselines_s1.sh
bash ./container/rt_run_task.sh paper-prognostics-baselines-s2  ./scripts/paper_prognostics/baselines_s2.sh
bash ./container/rt_run_task.sh paper-prognostics-fit-predict-s1  ./scripts/paper_prognostics/fit_predict_s1.sh
bash ./container/rt_run_task.sh paper-prognostics-fit-predict-s2  ./scripts/paper_prognostics/fit_predict_s2.sh