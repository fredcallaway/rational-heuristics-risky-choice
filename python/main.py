import os
import importlib
import cfg
import process_data as p_d
import make_figures as mf
import run_statistics as rs

def data_processing():

    p_d.run_process_data(which_experiment='both')
    p_d.print_special(f'finished processing data ({cfg.timer()})', header=True)

def figures(save=True, show=False):

    # Experiment 1
    cfg.exp1.figs.save = save
    cfg.exp1.figs.show = show
    mf.exp1_centroids(cfg.exp1)
    mf.exp1_strategies(cfg.exp1)
    mf.exp1_heatmaps(cfg.exp1)
    mf.exp1_condition_lines(cfg.exp1)
    mf.strategyVsKmeans_confusion_matrix(cfg.exp1)
    mf.under_performance_pie(cfg.exp1)
    mf.under_performance_byStrat(cfg.exp1)
    mf.centroids_1_k(cfg.exp1)
    mf.lda(cfg.exp1)

    # Experiment 2
    cfg.exp2.figs.save = save
    cfg.exp2.figs.show = show
    mf.exp2_centroids(cfg.exp2)
    mf.exp2_strategies(cfg.exp2)
    mf.exp2_condition_bars(cfg.exp2)
    mf.under_performance_pie(cfg.exp2)
    mf.under_performance_byStrat(cfg.exp2)
    mf.under_performance_pie(cfg.exp2, exclude=2)
    mf.under_performance_byStrat(cfg.exp2, exclude=2)
    mf.exp2_clicks_dispersion_cost(cfg.exp2)
    mf.exp2_clicks_dispersion_cost_3d(cfg.exp1)

    p_d.print_special(f'finished making figures ({cfg.timer()})', header=True)

def statistics(print_summary=True):

    # # Experiment 1
    cfg.exp1.stats.print_summary = print_summary
    rs.exp1_strategy_logistic_regression(cfg.exp1)
    rs.exp1_strategy_table(cfg.exp1)
    rs.exp1_behavioral_features(cfg.exp1)
    rs.under_performance(cfg.exp1.human)
    rs.under_performance(cfg.exp1.human, exclude=True)

    # Experiment 2
    cfg.exp2.stats.print_summary = print_summary
    rs.exp2_strategies(cfg.exp2)
    rs.exp2_behavioral_features(cfg.exp2)
    rs.under_performance(cfg.exp2.human_exp)
    rs.under_performance(cfg.exp2.human_con)
    rs.under_performance(cfg.exp2.human_exp, exclude=True)
    rs.under_performance(cfg.exp2.human_con, exclude=True)

    rs.participant_demographics(cfg.exp1)
    rs.participant_demographics(cfg.exp2)

    p_d.print_special(f'finished running statistics ({cfg.timer()})', header=True)


model_runs = ['A/','B/','C/']
TASK_ID = os.getenv('SLURM_ARRAY_TASK_ID')
if TASK_ID != None:
    '''need to resolve human files being written over in parallel;
    could make duplicate human copies and use basedir.human = '../data/human/'
    but better to eliminate human redundancies'''
    cfg.basedir(model_runs[int(TASK_ID)])
    importlib.reload(cfg)
    data_processing()
    figures()
    statistics()
else:
    for m in ['A/','B/','C/']:#model_runs:
        cfg.basedir(m)
        importlib.reload(cfg)
        data_processing()
        figures()
        statistics()

