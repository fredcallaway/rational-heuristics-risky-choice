# import importlib
# import time
# import multiprocessing
# import os
# from subprocess import call
import process_data as p_d
import make_figures as mf
import run_statistics as rs


def run_all():
    data_processing()
    figures()
    statistics()


def data_processing():
    p_d.run_process_data(which_experiment='both')


def figures(save=True):

    # Experiment 1
    model_file = '../data/model/exp1/processed/trials.csv'
    human_file = '../data/human/1.0/processed/trials.csv'
    fig_dir = '../figs/exp1/'

    mf.centroids(model_file, human_file, fig_dir, save)
    mf.strategies(model_file, human_file, fig_dir, save)
    mf.heatmaps(model_file, human_file, fig_dir, save)
    mf.condition_lines(model_file, human_file, fig_dir, save)
    mf.strategyVsKmeans_confusion_matrix(model_file, human_file, fig_dir, save)
    mf.under_performance_pie(human_file, 'exclude', fig_dir, save)
    mf.under_performance_byStrat(human_file, 'exclude', fig_dir, save)

    mf.centroids_1_k(model_file, fig_dir, save)
    mf.centroids_1_k(human_file, fig_dir, save)

    # Experiment 2
    model_file = '../data/model/exp2/processed/trials.csv'
    human_file1 = '../data/human/2.3/processed/trials_exp.csv'
    human_file2 = '../data/human/2.3/processed/trials_con.csv'
    fig_dir = '../figs/exp2/'

    mf.centroids_exp2(model_file, human_file1, human_file2, fig_dir, save)
    mf.strategies_exp2(model_file, human_file1, human_file2, fig_dir, save)
    mf.condition_bars_exp2(model_file, human_file1, human_file2, fig_dir, save)
    mf.under_performance_pie(human_file1, human_file2, fig_dir, save, exclude=True)
    mf.under_performance_byStrat(human_file1, human_file2, fig_dir, save, exclude=True)

    mf.clicks_dispersion_cost_exp2(fig_dir, save)
    mf.clicks_dispersion_cost_3d_exp2(fig_dir, save)


    # LDA to illustrate clusters
    # mf.lda(model_file, human_file, fig_dir, save)
    # mf.lda(isHuman=True, pca=False, save=False, df=df_trials)


def statistics():

    human_file = '../data/human/1.0/processed/trials.csv'
    stats_dir = '../stats/exp1/'
    rs.strategy_postHoc_cohenD()
    rs.behavioral_features()
    rs.under_performance(human_file, stats_dir)
    rs.under_performance(human_file, stats_dir, exclude_participants=True)


    human_file1 = '../data/human/2.3/processed/trials_exp.csv'
    human_file2 = '../data/human/2.3/processed/trials_con.csv'
    stats_dir = '../stats/exp2/'
    rs.exp2_strategy()
    rs.exp2_behavior()
    rs.under_performance(human_file1, stats_dir, exclude_participants=2)
    rs.under_performance(human_file2, stats_dir, exclude_participants=2)


    human_file = '../data/human/1.0/processed/trials.csv'
    rs.participant_demographics(human_file)
    human_file = '../data/human/2.3/processed/trials.csv'
    rs.participant_demographics(human_file)






