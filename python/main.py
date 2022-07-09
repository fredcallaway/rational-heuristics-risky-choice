# import importlib
# import time
# import multiprocessing
# import os
# from subprocess import call


def run_all():
    data_processing()
    figures()
    statistics()


def data_processing():
    import process_data as p_d
    p_d.run_process_data(which_experiment='both')


def figures(save=True, close=True):
    import make_figures as mf
    import matplotlib.pyplot as plt

    # Experiment 1
    model_file = '../data/model/exp1/processed/trials.csv'
    human_file = '../data/human/1.0/processed/trials.csv'
    fig_dir = '../figs/exp1/'

    mf.exp1_centroids(save=save)
    mf.exp1_strategies(save=save)
    mf.exp1_heatmaps(save=save)
    if close: plt.close('all')
    mf.exp1_condition_lines(save=save)
    if close: plt.close('all')
    mf.strategyVsKmeans_confusion_matrix(model_file, human_file, fig_dir, save=save)
    mf.under_performance_pie(human_file, 'exclude', fig_dir, save=save)
    mf.under_performance_byStrat(human_file, 'exclude', fig_dir, save=save)
    mf.centroids_1_k(model_file, fig_dir, save=save)
    mf.centroids_1_k(human_file, fig_dir, save=save)
    mf.lda(model_file, human_file, fig_dir, save=save)
    if close: plt.close('all')

    # Experiment 2
    model_file = '../data/model/exp2/processed/trials.csv'
    human_file1 = '../data/human/2.3/processed/trials_exp.csv'
    human_file2 = '../data/human/2.3/processed/trials_con.csv'
    fig_dir = '../figs/exp2/'

    mf.exp2_centroids(save=save)
    mf.exp2_strategies(save=save)
    mf.exp2_condition_bars(save=save)
    mf.under_performance_pie(human_file1, human_file2, fig_dir, exclude=True, save=save)
    mf.under_performance_byStrat(human_file1, human_file2, fig_dir, exclude=True, save=save)

    mf.exp2_clicks_dispersion_cost(save=save)
    mf.exp2_clicks_dispersion_cost_3d(save=save)
    if close: plt.close('all')



def statistics(print_summary=True):
    import run_statistics as rs

    human_file = '../data/human/1.0/processed/trials.csv'
    stats_dir = '../stats/exp1/'
    rs.exp1_strategy_logistic_regression(print_summary=print_summary)
    rs.exp1_strategy_table(print_summary=print_summary)
    rs.exp1_behavioral_features(print_summary=print_summary)
    rs.under_performance(human_file, stats_dir)
    rs.under_performance(human_file, stats_dir, exclude_participants=True)


    human_file1 = '../data/human/2.3/processed/trials_exp.csv'
    human_file2 = '../data/human/2.3/processed/trials_con.csv'
    stats_dir = '../stats/exp2/'
    rs.exp2_strategies(print_summary=print_summary)
    rs.exp2_behavioral_features(print_summary=print_summary)
    rs.under_performance(human_file1, stats_dir, exclude_participants=2)
    rs.under_performance(human_file2, stats_dir, exclude_participants=2)


    rs.participant_demographics('../data/human/1.0/participants.csv')
    rs.participant_demographics('../data/human/2.3/participants.csv')






