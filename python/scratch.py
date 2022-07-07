import importlib
import time
import multiprocessing
import os
from subprocess import call
import process_data as p_d
import make_figures as mf
import run_statistics as rs

# t0 = time.time()
# for k in range(1,13):
# 	in_file = '../data/human/1.0/processed/trials.csv'
# 	p_d.append_kmeans(in_file, k, label_cols=True)
# 	p_d.print_special(f'finished k={k} ({time.time()-t0:.1f} sec)')

# t0 = time.time()
# for k in range(1,13):
# 	in_file = '../data/model/exp1/processed/trials.csv'
# 	p_d.append_kmeans(in_file, k, sim_trials_per_human_trial=10, label_cols=True)
# 	p_d.print_special(f'finished k={k} ({time.time()-t0:.1f} sec)')


