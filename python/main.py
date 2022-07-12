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
    mf.exp1_centroids()
    mf.exp1_strategies()
    mf.exp1_heatmaps()
    mf.exp1_condition_lines()
    mf.strategyVsKmeans_confusion_matrix()
    mf.under_performance_pie()
    mf.under_performance_byStrat()
    mf.centroids_1_k()
    mf.lda()

    # Experiment 2
    cfg.exp2.figs.save = save
    cfg.exp2.figs.show = show
    mf.exp2_centroids()
    mf.exp2_strategies()
    mf.exp2_condition_bars()
    mf.under_performance_pie(cfg.exp2)
    mf.under_performance_byStrat(cfg.exp2)
    mf.under_performance_pie(cfg.exp2, exclude=2)
    mf.under_performance_byStrat(cfg.exp2, exclude=2)
    mf.exp2_clicks_dispersion_cost()
    mf.exp2_clicks_dispersion_cost_3d()

    p_d.print_special(f'finished making figures ({cfg.timer()})', header=True)

def statistics(print_summary=True):

    # Experiment 1
    cfg.exp1.stats.print_summary = print_summary
    rs.exp1_strategy_logistic_regression()
    rs.exp1_strategy_table()
    rs.exp1_behavioral_features()
    rs.under_performance()
    rs.under_performance(exclude=True)

    # Experiment 2
    cfg.exp2.stats.print_summary = print_summary
    rs.exp2_strategies()
    rs.exp2_behavioral_features()
    rs.under_performance(cfg.exp2.human_exp)
    rs.under_performance(cfg.exp2.human_con)
    rs.under_performance(cfg.exp2.human_exp, exclude=True)
    rs.under_performance(cfg.exp2.human_con, exclude=True)

    rs.participant_demographics(cfg.exp1)
    rs.participant_demographics(cfg.exp2)

    p_d.print_special(f'finished running statistics ({cfg.timer()})', header=True)


data_processing()
figures()
statistics()