#!/bin/bash

uv run python picid_report/scaling_laws_plots.py --all --no-title --all  --errorbar pi,95
uv run python picid_report/seq_len_plots.py --all --no-title
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 72
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 88
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 666
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 101
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 226688
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 0
uv run python picid_report/class_balance_plot.py --data-dir ~/datasets --subset-blocks none --subset-seed 72 --no-title

uv run python picid_report/quantile_distribution_plots.py