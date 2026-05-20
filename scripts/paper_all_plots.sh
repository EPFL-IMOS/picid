#!/bin/bash

python picid_report/scaling_laws_plots.py --all --no-title --all  --errorbar pi,95
python picid_report/seq_len_plots.py --all --no-title
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 72
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 88
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 666
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 101
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 226688
python picid_report/class_balance_plot.py --data-dir ~/datasets --no-title --subset-seed 0
python picid_report/class_balance_plot.py --data-dir ~/datasets --subset-blocks none --subset-seed 72 --no-title