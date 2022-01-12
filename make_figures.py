import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sn;
import pdb
import pickle
import process_data as p_d
import os



def strategy_bars(save=False, in_dir='data/model/no_implicit_cost/', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix=''):

    df_m = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    # df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl')
    # trials_h = pd.read_pickle(in_dir_human+'trials.pkl')
    #########
    df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl') #########
    #########
    df = pd.read_pickle(in_dir_human+'mean_by_condition_EVminTime.pkl')

    fig = plt.figure(figsize=(16, 12))

    ind = 0
    width = 0.2
    offset = [-0.22,0,0.22]
    labels = ['Model','Exp.','Control']
    for i in df.index.levels[0]:
        for j in df.index.levels[1]:
            for k in df.index.levels[2]:
                ind += 1
                for x, d in enumerate([df_m, df, df_h]):
                    if x == 1:
                        trials = pd.read_pickle(in_dir_human+'trials.pkl')
                    elif x == 2:
                        trials = pd.read_pickle(in_dir_human+'trials_EVminTime.pkl')
                    bottom = 0
                    pp = []
                    plt.gca().set_prop_cycle(None)
                    for s in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']:
                        if not(np.any(d.columns==s)):
                            d[s] = 0
                        dat = d.iloc[(d.T.columns.get_level_values('sigma')==i)&(d.T.columns.get_level_values('alpha')<(j+.01))&(d.T.columns.get_level_values('alpha')>(j-.01))&(d.T.columns.get_level_values('cost')==k)][s]
                        if x == 0:
                            sem = 0
                        else:
                            sem = trials.groupby(['pid']).mean()[s].sem()
                        p = plt.bar(ind+offset[x], dat, width, bottom=bottom, yerr=sem)
                        pp.append(p)
                        bottom += dat

    plt.ylabel('Strategy Frequency', fontsize=32)
    plt.yticks(fontsize=20)
    plt.xticks([i+j for i in [1,2,3,4] for j in [-.22,0,.22]],np.tile(labels,4),rotation=90,fontsize=24)
    plt.figtext(.15, 0, 'Alpha=$10^{-0.5}$\nCost=1',verticalalignment='top',horizontalalignment='left',fontsize=28)
    plt.figtext(.35, 0, 'Alpha=$10^{-0.5}$\nCost=4',verticalalignment='top',horizontalalignment='left',fontsize=28)
    plt.figtext(.55, 0, 'Alpha=$10^{0.5}$\nCost=1',verticalalignment='top',horizontalalignment='left',fontsize=28)
    plt.figtext(.75, 0, 'Alpha=$10^{0.5}$\nCost=4',verticalalignment='top',horizontalalignment='left',fontsize=28)

    plt.legend((pp), ('SAT-TTB+','SAT-TTB','TTB','WADD','Rand','Other'), fontsize=18, loc=8)

    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_dir+'strategyBars'+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)

    plt.show()

def behav_bars(save=False, in_dir='data/model/no_implicit_cost/', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix=''):

    df_m = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    # df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl')
    #########
    df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl') #########
    #########
    df = pd.read_pickle(in_dir_human+'mean_by_condition_EVminTime.pkl')

    behav = ['nr_clicks', 'net_payoff', 'click_var_gamble', 'click_var_outcome']
    behav_labels = ['Information Gathered', 'Reward', 'Alternative Variance', 'Attribute Variance']

    ind = 0
    width = 0.2
    offset = [-0.22,0,0.22]
    labels = ['Model','Exp.','Control']
    colors = ['mediumturquoise','mediumaquamarine','teal']
    for s_, s in enumerate(behav):

        if not(np.remainder(s_,2)):
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(2, 1)
        plt.sca(plt.subplot(gs[np.remainder(s_,2)]))
        for i in df.index.levels[0]:
            for j in df.index.levels[1]:
                for k in df.index.levels[2]:
                    ind += 1
                    pp = []
                    for x, d in enumerate([df_m, df, df_h]):
                        if x == 0:
                            sem = 0
                        elif x == 1:
                            trials = pd.read_pickle(in_dir_human+'trials.pkl')
                            sem = trials.groupby(['pid']).mean()[s].sem()
                        elif x == 2:
                            trials = pd.read_pickle(in_dir_human+'trials_EVminTime.pkl')
                            sem = trials.groupby(['pid']).mean()[s].sem()
                        dat = d.iloc[(d.T.columns.get_level_values('sigma')==i)&(d.T.columns.get_level_values('alpha')<(j+.01))&(d.T.columns.get_level_values('alpha')>(j-.01))&(d.T.columns.get_level_values('cost')==k)][s]
                        p = plt.bar(ind+offset[x], dat, width, color=colors[x], yerr=sem)
                        pp.append(p)

        plt.ylabel(behav_labels[s_], fontsize=32)
        plt.yticks(fontsize=20)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
        if np.remainder(s_,2):
            plt.figtext(.15, 0, 'Alpha=$10^{-0.5}$\nCost=1',verticalalignment='bottom',horizontalalignment='left',fontsize=28)
            plt.figtext(.345, 0, 'Alpha=$10^{-0.5}$\nCost=4',verticalalignment='bottom',horizontalalignment='left',fontsize=28)
            plt.figtext(.54, 0, 'Alpha=$10^{0.5}$\nCost=1',verticalalignment='bottom',horizontalalignment='left',fontsize=28)
            plt.figtext(.725, 0, 'Alpha=$10^{0.5}$\nCost=4',verticalalignment='bottom',horizontalalignment='left',fontsize=28)
        else:
            plt.legend((pp), labels, fontsize=18)

        if np.remainder(s_,2):
            if save:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                ttl = out_dir+behav_labels[s_-1]+'+'+behav_labels[s_]+'_bars'+out_suffix+'.png'
                plt.savefig(ttl.replace(' ','_'),bbox_inches='tight',pad_inches=0.05)
            plt.show()

def centroids_exp2(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human1='data/human/', in_suffix1='', centroid_order1='', in_dir_human2='data/human/', in_suffix2='', centroid_order2='', centroid_order_mod=''):

    kmeans = pickle.load(open(in_dir_human1+'kmeans'+in_suffix1+'.pkl', 'rb'))
    km1 = kmeans['click_embedding_prob']['kmeans'][0]
    kmeans = pickle.load(open(in_dir_human2+'kmeans'+in_suffix2+'.pkl', 'rb'))
    km2 = kmeans['click_embedding_prob']['kmeans'][0]
    kmeans = pickle.load(open(in_dir+'kmeans_model.pkl', 'rb'))
    km_mod = kmeans['click_embedding_prob']['kmeans'][0]
    print('using ',km1.n_clusters,' participant1 clusters, ',km2.n_clusters,' participant2 clusters, ',km_mod.n_clusters,' model clusters')

    fig = plt.figure(figsize=(16, 7.5))
    plt.figtext(0.5, 0.94, 'Model', ha='center', va='center', fontsize=30, fontweight='bold')
    plt.figtext(0.5, 0.63, 'Experimental Participants', ha='center', va='center', fontsize=30, fontweight='bold')
    plt.figtext(0.5, 0.362, 'Control Participants', ha='center', va='center', fontsize=30, fontweight='bold')
    gs = gridspec.GridSpec(3, 5)

    if centroid_order_mod=='':
        sp = np.arange(km_mod.n_clusters)
    else:
        sp = centroid_order_mod

    for i in range(km_mod.n_clusters):
        print('centroid ',i,'n=',sum(km_mod.labels_==sp[i]),'(',np.round(100*sum(km_mod.labels_==sp[i])/len(km_mod.labels_),2),'%)')
        plt.sca(plt.subplot(gs[i]))
        plt.title('Centroid '+str(i+1),fontsize=26)
        plt.imshow(np.reshape(km_mod.cluster_centers_[sp[i]],(4,6)), vmin=0, vmax=1)
        if (i==0):
            plt.ylabel('Prob. of\noutcome',fontsize=17)
            plt.yticks([0,3], ['higher','lower'], fontsize=16, rotation='vertical', va='center')
            plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
        else:
            plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
    plt.sca(plt.subplot(gs[4]))
    plt.axis('off')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="20%", pad=0)
    plt.colorbar(cax=cax)
    plt.figtext(0.86, 0.77, 'Centroid values\n(Prob. of click)', ha='center', va='center', fontsize=17, rotation='vertical')

    if centroid_order1=='':
        sp = np.arange(km1.n_clusters)
    else:
        sp = centroid_order1
    
    for i in range(km1.n_clusters):
        print('centroid ',i,'n=',sum(km1.labels_==sp[i]),'(',np.round(100*sum(km1.labels_==sp[i])/len(km1.labels_),2),'%)')
        plt.sca(plt.subplot(gs[i+5]))
        plt.imshow(np.reshape(km1.cluster_centers_[sp[i]],(4,6)), vmin=0, vmax=1)
        if (i==0):
            plt.ylabel('Prob. of\noutcome',fontsize=17)
            plt.yticks([0,3], ['higher','lower'], fontsize=16, rotation='vertical', va='center')
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        if (i==4):
            plt.title('Centroid 5',fontsize=25)

    if centroid_order2=='':
        sp = np.arange(km2.n_clusters)
    else:
        sp = centroid_order2
    
    for i in range(km2.n_clusters):
        print('centroid ',i,'n=',sum(km2.labels_==sp[i]),'(',np.round(100*sum(km2.labels_==sp[i])/len(km2.labels_),2),'%)')
        plt.sca(plt.subplot(gs[i+10]))
        plt.imshow(np.reshape(km2.cluster_centers_[sp[i]],(4,6)), vmin=0, vmax=1)
        if (i==0):
            plt.ylabel('Prob. of\noutcome',fontsize=17)
            plt.yticks([0,3], ['higher','lower'], fontsize=16, rotation='vertical', va='center')
        else:
            plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.xlabel('Total prob. of\nobserved outcomes',fontsize=17)
        plt.xticks([0,5], ['higher','lower'], fontsize=16)
        if (i==4):
            plt.title('Centroid 5',fontsize=25)

    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_dir+'centroids'+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)
    
    plt.show()

def centroids(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix='', centroid_order='', centroid_order_mod=''):

    kmeans = pickle.load(open(in_dir_human+'kmeans'+in_suffix+'.pkl', 'rb'))
    km = kmeans['click_embedding_prob']['kmeans'][0]
    kmeans = pickle.load(open(in_dir+'kmeans_model.pkl', 'rb'))
    km_mod = kmeans['click_embedding_prob']['kmeans'][0]#[1]
    print('using ',km.n_clusters,' participant clusters, ',km_mod.n_clusters,' model clusters')

    fontsize_ticks = 32 #20
    fontsize_labels = 42 #32
    fontsize_legend = 36 #18

    fig = plt.figure(figsize=(32, 10))
    plt.figtext(0.5, 0.96, 'Model', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
    plt.figtext(0.5, 0.49, 'Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
    gs = gridspec.GridSpec(2, 5)

    # make subplot for common labels
    # ax0 = fig.add_subplot(111)
    # ax0.spines['top'].set_color('none')
    # ax0.spines['bottom'].set_color('none')
    # ax0.spines['left'].set_color('none')
    # ax0.spines['right'].set_color('none')
    # ax0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # # put common labels
    # plt.sca(plt.subplot(ax0))
    # ax0.set_xlabel('Total prob. of observed outcomes',fontsize=fontsize_labels)
    # ax0.set_ylabel('Prob. of outcome',fontsize=fontsize_labels)

    if centroid_order_mod=='':
        sp = np.arange(km_mod.n_clusters)
    else:
        sp = centroid_order_mod #[0,1,4,3,2]

    for i in range(km_mod.n_clusters):
        print('centroid ',i,'n=',sum(km_mod.labels_==sp[i]),'(',np.round(100*sum(km_mod.labels_==sp[i])/len(km_mod.labels_),2),'%)')
        plt.sca(plt.subplot(gs[i]))
        plt.title('Centroid '+str(i+1),fontsize=fontsize_labels)
        plt.imshow(np.reshape(km_mod.cluster_centers_[sp[i]],(4,6)), vmin=0, vmax=1)
        if (i==0):
            # plt.ylabel('Prob. of\noutcome',fontsize=fontsize_labels)
            plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
            plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
        else:
            plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
    plt.sca(plt.subplot(gs[4]))
    plt.axis('off')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="20%", pad=0)
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize=20)
    plt.figtext(0.86, 0.71, 'Centroid values\n(Prob. of click)', ha='center', va='center', fontsize=fontsize_legend, rotation='vertical')

    if centroid_order=='':
        sp = np.arange(km.n_clusters)
    else:
        sp = centroid_order #[0,1,4,3,2]
    
    for i in range(km.n_clusters):
        print('centroid ',i,'n=',sum(km.labels_==sp[i]),'(',np.round(100*sum(km.labels_==sp[i])/len(km.labels_),2),'%)')
        plt.sca(plt.subplot(gs[i+5]))
        plt.imshow(np.reshape(km.cluster_centers_[sp[i]],(4,6)), vmin=0, vmax=1)
        if (i==0):
            # plt.ylabel('Prob. of\noutcome',fontsize=fontsize_labels)
            plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
        else:
            plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        # plt.xlabel('Total prob. of\nobserved outcomes',fontsize=fontsize_labels)
        if i==0:
            plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
        else:
            plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
        if (i==4):
            plt.title('Centroid 5',fontsize=fontsize_labels)

    fig.text(0.5, 0.04, 'Total prob. of observed outcomes', fontsize=fontsize_labels, ha='center', va='center')
    fig.text(0.087, 0.5, 'Prob. of outcome', fontsize=fontsize_labels, ha='center', va='center', rotation='vertical')

    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_dir+'centroids'+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)
    
    plt.show()

    return plt.gca



def strategies(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix=''):

    df = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    y_out = []

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]
    strategies = ['SAT-TTB+','SAT-TTB','TTB','WADD','Rand','Other']

    if plot_kmeans:
        param = [i+'_strategy_click_embedding_prob_k4' for i in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']]
    else:
        param = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']

    fig = plt.figure(figsize=(16, 12))
    plt.figtext(0.5, 0.9, 'Model', ha='center', va='center', fontsize=32, fontweight='bold')
    plt.figtext(0.5, 0.49, 'Participants', ha='center', va='center', fontsize=32, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 5, 5]) 

    plt.sca(plt.subplot(gs[0]))
    y = p_d.get_means_by_cond(df, cond='sigma', param=param)
    plt.plot(y, lw=4)
    plt.ylabel('Strategy Frequency', fontsize=32)
    plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
    plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
    plt.xlim((-.2,1.2))
    plt.ylim((-.02,.72))
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.sca(plt.subplot(gs[1]))
    y = p_d.get_means_by_cond(df, cond='alpha', param=param)
    plt.plot(y, lw=4)
    plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=20)
    plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
    plt.ylim((-.02,.72))
    plt.grid(True)

    plt.sca(plt.subplot(gs[2]))
    y = p_d.get_means_by_cond(df, cond='cost', param=param)
    plt.plot(y, lw=4)
    plt.xticks(np.arange(len(costs)), costs, fontsize=20)
    plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
    plt.legend(['SAT-TTB+','SAT-TTB','TTB','WADD','random','other'], fontsize=18, loc='upper right')
    plt.ylim((-.02,.72))
    plt.grid(True)

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')

    if plot_kmeans:
        param = [i+'_strategy_click_embedding_prob_k5' for i in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']]
    else:
        param = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']

    plt.sca(plt.subplot(gs[3]))
    y = p_d.get_means_by_cond(df, cond='sigma', param=param)
    y_out.append(y)
    plt.plot(y, lw=4)
    if not(plot_kmeans):
        plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=param).T[2], 'g:', lw=4)
    plt.xlabel('Stakes [$\sigma$]', fontsize=32)
    plt.ylabel('Strategy Frequency', fontsize=32)
    plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
    plt.xlim((-.2,1.2))
    plt.ylim((-.02,.72))
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.sca(plt.subplot(gs[4]))
    y = p_d.get_means_by_cond(df, cond='alpha', param=param)
    y_out.append(y)
    plt.plot(y, lw=4)
    plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=32)
    plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=20)
    plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
    plt.ylim((-.02,.72))
    plt.grid(True)

    plt.sca(plt.subplot(gs[5]))
    y = p_d.get_means_by_cond(df, cond='cost', param=param)
    y_out.append(y)
    plt.plot(y, lw=4)
    plt.xlabel('Cost [$\lambda$]', fontsize=32)
    plt.xticks(np.arange(len(costs)), costs, fontsize=20)
    plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
    plt.ylim((-.02,.72))
    plt.grid(True)
    
    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_dir+'strategies'+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)

    plt.show()

    return y_out

def condition_plots(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix=''):

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    df_mod_trials = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    paramT = ['nr_clicks','click_var_gamble']
    paramB = ['net_payoff','click_var_outcome']
    labelT = ['Information Gathered','Alternative Variance']
    labelB = ['Reward','Attribute Variance']
    limT = [(2,20), (0,.06)]
    limB = [(20,110), (0,.2)]

    for p in range(len(paramT)):
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 5, 5])

        plt.sca(plt.subplot(gs[0]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.ylabel(labelT[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.xlim((-.2,1.2))
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[3]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Stakes [$\sigma$]', fontsize=32)
        plt.ylabel(labelB[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.xlim((-.2,1.2))
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[1]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(dispersion)), dispersion, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[4]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=32)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[2]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limT[p])
        plt.legend(['Model','Participants'], fontsize=18, loc='upper right')
        plt.grid(True)

        plt.sca(plt.subplot(gs[5]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Cost [$\lambda$]', fontsize=32)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ttl = out_dir+labelT[p]+'+'+labelB[p]+out_suffix+'.png'
            plt.savefig(ttl.replace(' ','_'),bbox_inches='tight',pad_inches=0.05)

        plt.show()

def relative_performance(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix=''):

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    df_mod_trials = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    paramT = ['nr_clicks']
    paramB = ['payoff_gross']
    labelT = ['Information Gathered']
    labelB = ['Relative Performance']
    limT = [(2,20)]
    limB = [(0,1.03)]

    for p in range(len(paramT)):
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 5, 5])

        plt.sca(plt.subplot(gs[0]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.ylabel(labelT[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.xlim((-.2,1.2))
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[3]))
        perfect = p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_gross']) / perfect, color='#17becf', lw=8)#/ perfect, 
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB[p])#, perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]) #/ perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        plt.xlabel('Stakes [$\sigma$]', fontsize=32)
        plt.ylabel(labelB[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.xlim((-.2,1.2))
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[1]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(dispersion)), dispersion, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[4]))
        perfect = p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_gross']) / perfect, color='#17becf', lw=8)#/ perfect, 
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB[p])#, perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]) #/ perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=32)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[2]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limT[p])
        plt.legend(['Model','Participants'], fontsize=18, loc='upper right')
        plt.grid(True)

        plt.sca(plt.subplot(gs[5]))
        perfect = p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_gross']) / perfect, color='#17becf', lw=8, label='_nolegend_')#/ perfect, 
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB[p])#, perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]) #/ perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        # plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        # plt.legend(['Perfect Gamble','Random Gamble'], fontsize=18, loc='upper right')
        plt.xlabel('Cost [$\lambda$]', fontsize=32)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ttl = out_dir+labelT[p]+'+'+labelB[p]+out_suffix+'.png'
            plt.savefig(out_dir+'Information_Gathered+Reward_relative_sem.png',bbox_inches='tight',pad_inches=0.05)

        plt.show()

def relative_performance_old(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix=''):

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    df_mod_trials = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    paramT = ['nr_clicks']
    paramB = ['payoff_gross']
    labelT = ['Information Gathered']
    labelB = ['Reward']
    limT = [(2,20)]
    limB = [(-2,140)]

    for p in range(len(paramT)):
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 5, 5])

        plt.sca(plt.subplot(gs[0]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.ylabel(labelT[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.xlim((-.2,1.2))
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[3]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        plt.xlabel('Stakes [$\sigma$]', fontsize=32)
        plt.ylabel(labelB[p], fontsize=32)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=20)
        plt.xlim((-.2,1.2))
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[1]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(dispersion)), dispersion, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.ylim(limT[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[4]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=32)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        plt.sca(plt.subplot(gs[2]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limT[p])
        plt.legend(['Model','Participants'], fontsize=18, loc='upper right')
        plt.grid(True)

        plt.sca(plt.subplot(gs[5]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramB[p]]), color='#17becf', lw=8, label='_nolegend_')
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_perfect']), color='#696969', linestyle='dashed', lw=4)
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_rand']), color='#696969', linestyle='dotted', lw=4)
        plt.legend(['Perfect Gamble','Random Gamble'], fontsize=18, loc='upper right')
        plt.xlabel('Cost [$\lambda$]', fontsize=32)
        plt.xticks(np.arange(len(costs)), costs, fontsize=20)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=20)
        plt.grid(True)

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ttl = out_dir+labelT[p]+'+'+labelB[p]+out_suffix+'.png'
            plt.savefig(out_dir+'Information_Gathered+Reward_relative_sem.png',bbox_inches='tight',pad_inches=0.05)

        plt.show()

def heatmaps(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix=''):

    from scipy.stats import pearsonr
    from sklearn.utils import shuffle
    from sklearn.metrics.pairwise import cosine_similarity

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_csv(in_dir+'mean_by_condition_model.csv')
    # df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    if plot_kmeans:
        for s in ['TTB','TTB_SAT','SAT_TTB','WADD']:
            df[[s+'_k'][0]] = df[[s+'_strategy_click_embedding_prob_k5'][0]]
            df_mod[[s+'_k'][0]] = df_mod[[s+'_strategy_click_embedding_prob_k4'][0]]
        # df['strategy_k'] = df['strategy_click_embedding_prob_k5']
        # df_mod['strategy_k'] = df_mod['strategy_click_embedding_prob_k4']
        paramL = ['nr_clicks','TTB_k','TTB_SAT_k','click_var_outcome']
        paramR = ['net_payoff','SAT_TTB_k','WADD_k','click_var_gamble']
    else:
        paramL = ['nr_clicks','TTB','SAT_TTB','click_var_outcome','payoff_relative']
        paramR = ['processing_pattern','WADD','TTB_SAT','click_var_gamble','payoff_relative'] #payoff_gross
    titleL = ['Information Gathered','TTB Frequency','SAT-TTB Frequency','Attribute Variance','Relative Reward']
    titleR = ['Alternative vs. Attribute','WADD Frequency','SAT-TTB+ Frequency','Alternative Variance','w/ implicit cost'] #Relative Reward

    for p in range(len(paramL)):

        if paramR[p]=='payoff_relative':
            df['payoff_relative'] = df['payoff_gross'] / df['payoff_perfect']
            df_mod['payoff_relative'] = df_mod['payoff_gross'] / df_mod['payoff_perfect']
            # df_trials['payoff_relative'] = df_trials['payoff_gross'] / df_trials['payoff_perfect']
            df_mod_2 = pd.read_pickle('data/model/human_trials_fitcost/mean_by_condition_model.pkl')
            df_mod_2['payoff_relative'] = df_mod_2['payoff_gross'] / df_mod_2['payoff_perfect']
        
        array1_shuffled, array2_shuffled = shuffle(df[paramL[p]], df_mod[paramL[p]])
        print(paramL[p],' shuffled correlation: ',pearsonr(array1_shuffled, array2_shuffled))
        v1 = np.array(df[paramL[p]]).reshape(1, -1)
        v2 = np.array(df_mod[paramL[p]]).reshape(1, -1)
        css = cosine_similarity(v1,v2)
        
        try:
            print(paramL[p],' cosine sim ',css)
            print(paramL[p],' correlation: ',pearsonr(df[paramL[p]], df_mod[paramL[p]]))
            print(paramR[p],' correlation: ',pearsonr(df[paramR[p]], df_mod[paramR[p]]))
        #     p_d.get_3d_correlations(df, df_mod, paramL[p])
        #     p_d.get_3d_correlations(df, df_mod, paramR[p])
        except Exception:
            pass
        
        fig = plt.figure(figsize=(16,10))
        gs = gridspec.GridSpec(4,4, width_ratios=[10,11.5,10,11.5], height_ratios=[0,100,100,100])
        ax = [None]*12

        plt.sca(plt.subplot(gs[4]))
        try:
            tmp = p_d.get_means_by_cond(df_mod, cond=[], param=[paramL[p]])
        except Exception:
            tmp = np.reshape(np.zeros(50),(2,5,5))
        if paramL[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        cb_range = [min(np.ndarray.flatten(tmp)), max(np.abs(np.ndarray.flatten(tmp)))]
        ax[0] = sn.heatmap(np.flip(tmp[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
        ax[0].invert_yaxis()
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.sca(plt.subplot(gs[5]))
        ax[1] = sn.heatmap(np.flip(tmp[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
        ax[1].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[6]))
        try:
            tmp = p_d.get_means_by_cond(df_mod, cond=[], param=[paramR[p]])
        except Exception:
            tmp = np.reshape(np.zeros(50),(2,5,5))
        if paramR[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        if paramR[p]=='payoff_relative':
            tmp = p_d.get_means_by_cond(df_mod_2, cond=[], param=[paramR[p]])
        cb_range = [min(np.ndarray.flatten(tmp)), max(np.abs(np.ndarray.flatten(tmp)))]
        ax[2] = sn.heatmap(np.flip(tmp[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
        ax[2].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.sca(plt.subplot(gs[7]))
        ax[3] = sn.heatmap(np.flip(tmp[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
        ax[3].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[8]))
        tmp = p_d.get_means_by_cond(df, cond=[], param=[paramL[p]])
        if paramL[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        cb_range = [min(np.ndarray.flatten(tmp)), max(np.abs(np.ndarray.flatten(tmp)))]
        ax[4] = sn.heatmap(np.flip(tmp[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
        ax[4].invert_yaxis()
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        plt.sca(plt.subplot(gs[9]))
        ax[5] = sn.heatmap(np.flip(tmp[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
        ax[5].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[10]))
        tmp = p_d.get_means_by_cond(df, cond=[], param=[paramR[p]])
        if paramR[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        # if paramR[p]=='payoff_relative':
        #     tmp = p_d.get_means_by_cond(df_mod_2, cond=[], param=[paramR[p]])
        cb_range = [min(np.ndarray.flatten(tmp)), max(np.abs(np.ndarray.flatten(tmp)))]
        ax[6] = sn.heatmap(np.flip(tmp[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
        ax[6].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.sca(plt.subplot(gs[11]))
        ax[7] = sn.heatmap(np.flip(tmp[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
        ax[7].invert_yaxis()
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[12]))
        try:
            tmp = p_d.get_means_by_cond(df, cond=[], param=[paramL[p]]) - p_d.get_means_by_cond(df_mod, cond=[], param=[paramL[p]])
        except Exception:
            tmp = np.reshape(np.zeros(50),(2,5,5))
        if paramL[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        cb_range = [-max(np.abs(np.ndarray.flatten(tmp))), max(np.abs(np.ndarray.flatten(tmp)))]
        ax[8] = sn.heatmap(tmp[0], vmin=cb_range[0], vmax=cb_range[1], cmap='vlag', cbar=False)
        ax[8].invert_yaxis()
        plt.sca(plt.subplot(gs[13]))
        ax[9] = sn.heatmap(tmp[1], vmin=cb_range[0], vmax=cb_range[1], cmap='vlag')
        ax[9].invert_yaxis()
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)

        try:
            if (p==0):
                tmp = p_d.get_means_by_cond(df, cond=[], param=[paramR[p]]) / p_d.get_means_by_cond(df_mod, cond=[], param=[paramR[p]])
            elif paramR[p]=='payoff_relative':
                tmp = p_d.get_means_by_cond(df, cond=[], param=[paramR[p]]) - p_d.get_means_by_cond(df_mod_2, cond=[], param=[paramR[p]])
            else:
                tmp = p_d.get_means_by_cond(df, cond=[], param=[paramR[p]]) - p_d.get_means_by_cond(df_mod, cond=[], param=[paramR[p]])
        except Exception:
            tmp = np.reshape(np.zeros(50),(2,5,5))
        if paramR[p]=='payoff_gross':
            tmp /= p_d.get_means_by_cond(df_mod, cond=[], param=['payoff_perfect'])
        cb_range = [-max(np.abs(np.ndarray.flatten(tmp))), max(np.abs(np.ndarray.flatten(tmp)))]
        plt.sca(plt.subplot(gs[14]))
        ax[10] = sn.heatmap(tmp[0], vmin=cb_range[0], vmax=cb_range[1], cmap='vlag', cbar=False)
        ax[10].invert_yaxis()
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
        plt.sca(plt.subplot(gs[15]))
        ax[11] = sn.heatmap(tmp[1], vmin=cb_range[0], vmax=cb_range[1], cmap='vlag')
        ax[11].invert_yaxis()
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)

        ts1=38; ts2=32; ts3=30; ts4=18;
        for i in np.arange(0,12,2):
            try:
                pts0 = ax[i].get_position().get_points()
                pos = ax[i+1].get_position()
                pts = pos.get_points()
                pts[0][0]=pts0[1][0]+.01
                pos.set_points(pts)
                ax[i+1].set_position(pos) 
            except Exception:
                print('7',i)
                pass
        pts_ = [None]*12
        for i in np.arange(0,12):
            try:
                pts = ax[i].get_position().get_points()
                pos = ax[i].get_position()
                pts[1][1]=pts[1][1]-.04
                pos.set_points(pts)
                ax[i].set_position(pos)
                pts_[i] = pts[1][1]
            except Exception:
                print('8',i)
                pass
        for i in np.arange(4,16,4):
            try:
                plt.sca(plt.subplot(gs[i]))
                plt.ylabel('Dispersion \n['+r'$\alpha^{-1}$]', fontsize=ts3)
                plt.yticks(np.arange(5)+.5, [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], va='center', rotation=0, fontsize=ts4)
            except Exception:
                print('9',i)
                pass
        for i in np.arange(12,16):
            try:
                plt.sca(plt.subplot(gs[i]))
                plt.xlabel('Cost [$\lambda$]', fontsize=ts3)
                plt.xticks(np.arange(5)+.5, ['0','1','2','4','8'], ha='center', fontsize=ts4)
            except Exception:
                print('10',i)
                pass

        l=0.28; r=0.67;
        plt.figtext(l, 1, titleL[p], ha='center', va='top', fontsize=ts1)
        plt.figtext(r, 1, titleR[p], ha='center', va='top', fontsize=ts1)
        # plt.figtext(l-.05, pts_[0]+.05, 'Stakes=75', ha='right', va='bottom', fontsize=ts3)
        # plt.figtext(l+.05, pts_[0]+.05, 'Stakes=150', ha='left', va='bottom', fontsize=ts3)
        # plt.figtext(r-.05, pts_[0]+.05, 'Stakes=75', ha='right', va='bottom', fontsize=ts3)
        # plt.figtext(r+.05, pts_[0]+.05, 'Stakes=150', ha='left', va='bottom', fontsize=ts3)
        plt.figtext(l-.05, pts_[0]+.02, 'Stakes \n[$\sigma=75$]', ha='right', va='bottom', fontsize=ts3)
        plt.figtext(l+.05, pts_[0]+.02, 'Stakes \n[$\sigma=150$]', ha='left', va='bottom', fontsize=ts3)
        plt.figtext(r-.05, pts_[0]+.02, 'Stakes \n[$\sigma=75$]', ha='right', va='bottom', fontsize=ts3)
        plt.figtext(r+.05, pts_[0]+.02, 'Stakes \n[$\sigma=150$]', ha='left', va='bottom', fontsize=ts3)
        plt.figtext(l, pts_[0], 'Model', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        plt.figtext(r, pts_[0], 'Model', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        plt.figtext(l, pts_[7], 'Participants', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        plt.figtext(r, pts_[7], 'Participants', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        plt.figtext(l, pts_[11], r'Participants $\minus$ Model', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        if (p==0):
            plt.figtext(r, pts_[11], r'Participants $\div$ Model', ha='center', va='bottom', fontsize=ts2, fontweight='bold')
        else:
            plt.figtext(r, pts_[11], r'Participants $\minus$ Model', ha='center', va='bottom', fontsize=ts2, fontweight='bold')

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ttl = out_dir+'heatmaps_'+titleL[p]+'+'+titleR[p].replace('/','')+out_suffix+'.png'
            plt.savefig(ttl.replace(' ','_'),bbox_inches='tight',pad_inches=0.05)

        plt.show()



def strategies_fillbetween(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix='', big=True, normalize=True):

    df = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]
    strategies = ['SAT-TTB+','SAT-TTB','TTB','WADD']#,'Rand','Other']

    if plot_kmeans:
        param = [i+'_strategy_click_embedding_prob_k4' for i in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']]
    else:
        param = ['TTB_SAT','SAT_TTB','TTB','WADD']#,'Rand','Other']

    if not(big):
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 5, 5]) 
        fontsize_ticks = 20
        fontsize_labels = 32
        fontsize_legend = 18
    else:
        fig = plt.figure(figsize=(32, 18))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1.25]) 
        fontsize_ticks = 32
        fontsize_labels = 42
        fontsize_legend = 36
    plt.figtext(0.5, 0.9, 'Model', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
    plt.figtext(0.5, 0.49, 'Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')    

    plt.sca(plt.subplot(gs[0]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='sigma', param=param), 0, 0, axis=1)
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.ylabel('Strategy Frequency', fontsize=fontsize_labels)
    plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
    plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
    # plt.xlim((-.2,1.2))
    plt.ylim((-.02,1.02))
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)

    plt.sca(plt.subplot(gs[1]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='alpha', param=param), 0, 0, axis=1)
    dd = np.flip(dd,axis=0)
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
    plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
    plt.ylim((-.02,1.02))
    plt.grid(True)

    plt.sca(plt.subplot(gs[2]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='cost', param=param), 0, 0, axis=1)
    # pdb.set_trace()
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
    plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
    if not(big):
        plt.legend(['SAT-TTB+','SAT-TTB','TTB','WADD'], fontsize=fontsize_legend, loc='lower right')
    else:
        ax = plt.subplot(gs[2])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(['SAT-TTB+','SAT-TTB','TTB','WADD'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))

    plt.ylim((-.02,1.02))
    plt.grid(True)

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')

    if plot_kmeans:
        param = [i+'_strategy_click_embedding_prob_k5' for i in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']]
    else:
        param = ['TTB_SAT','SAT_TTB','TTB','WADD']#,'Rand','Other']

    plt.sca(plt.subplot(gs[3]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='sigma', param=param), 0, 0, axis=1)
    # pdb.set_trace()
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
    plt.ylabel('Strategy Frequency', fontsize=fontsize_labels)
    plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
    # plt.xlim((-.2,1.2))
    plt.ylim((-.02,1.02))
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)

    plt.sca(plt.subplot(gs[4]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='alpha', param=param), 0, 0, axis=1)
    dd = np.flip(dd,axis=0)
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.xlabel('Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels)
    plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
    plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
    plt.ylim((-.02,1.02))
    plt.grid(True)

    plt.sca(plt.subplot(gs[5]))
    dd = np.insert(p_d.get_means_by_cond(df, cond='cost', param=param), 0, 0, axis=1)
    if normalize:
        dd *= np.expand_dims(1/np.sum(dd,axis=1),axis=1)
    for i in range(dd.shape[1]-1):
        plt.fill_between(np.arange(dd.shape[0]), np.sum(dd[:,:i+1],axis=1), np.sum(dd[:,:i+1],axis=1)+dd[:,i+1], lw=4)
    plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
    plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
    plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
    plt.ylim((-.02,1.02))
    plt.grid(True)
    if big:
        ax = plt.subplot(gs[5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bigstr = ''
        if big:
            bigstr = '_big'
        plt.savefig(out_dir+'strategies_fillbetween_top4'+bigstr+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)

    plt.show()   




def condition_plots_3(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix='', big=True):

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    df_mod_trials = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    # paramT = ['click_var_gamble']
    # paramB = ['click_var_outcome']
    # labelT = ['Alternative Variance']
    # labelB = ['Attribute Variance']
    # limT = [(0,.06)]
    # limB = [(0,.2)]
    # paramB2 = ['nr_clicks']
    # labelB2 = ['Information Gathered']
    # limB2 = [(2,20)]
    paramB2 = ['click_var_gamble']
    paramB = ['click_var_outcome']
    labelB2 = ['Alternative Variance']
    labelB = ['Attribute Variance']
    limB2 = [(0,.06)]
    limB = [(0,.2)]
    paramT = ['nr_clicks']
    labelT = ['Information Gathered']
    limT = [(2,20)]

    for p in range(len(paramT)):
        
        if not(big):
            fig = plt.figure(figsize=(16, 18))
            gs = gridspec.GridSpec(3, 3, width_ratios=[3, 5, 5])
            fontsize_ticks = 20
            fontsize_labels = 32
            fontsize_legend = 18
        else:
            fig = plt.figure(figsize=(32, 24))
            gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1.25])
            fontsize_ticks = 32
            fontsize_labels = 42
            fontsize_legend = 36

        plt.sca(plt.subplot(gs[0]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.ylabel(labelT[p], fontsize=fontsize_labels)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        # plt.xlim((-.2,1.2))
        plt.ylim(limT[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[3]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.xlabel('Stakes [$\sigma$]', fontsize=32)
        plt.ylabel(labelB[p], fontsize=fontsize_labels)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
        plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        # plt.xlim((-.2,1.2))
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[1]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(dispersion)), dispersion, fontsize=fontsize_ticks)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        plt.ylim(limT[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[4]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=32)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[2]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramT[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramT[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramT[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limT[p])
        plt.grid(True)
        if not(big):
            plt.legend(['Model','Participants'], fontsize=fontsize_legend, loc='upper right')
        else:
            ax = plt.subplot(gs[2])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(['Model','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))


        plt.sca(plt.subplot(gs[5]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramB[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        # plt.xlabel('Cost [$\lambda$]', fontsize=32)
        plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
        plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)
        if big:
            ax = plt.subplot(gs[5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.sca(plt.subplot(gs[6]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramB2[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB2[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB2[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
        plt.ylabel(labelB2[p], fontsize=fontsize_labels)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
        # plt.xlim((-.2,1.2))
        plt.ylim(limB2[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[7]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramB2[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB2[p])
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB2[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=fontsize_labels)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.ylim(limB2[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[8]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramB2[p]]), color='#17becf', lw=8)
        # plt.plot(p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]]), color='#1f77b4', lw=8)
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB2[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB2[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
        plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.ylim(limB2[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)
        if big:
            ax = plt.subplot(gs[8])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            bigstr = ''
            if big:
                bigstr = '_big'
            ttl = out_dir+labelT[p]+'+'+labelB[p]+'+'+labelB2[p]+bigstr+out_suffix+'.png'
            plt.savefig(ttl.replace(' ','_'),bbox_inches='tight',pad_inches=0.05)

        plt.show()

def relative_performance_only(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix='', big=True):

    df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    df_mod_trials = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    # paramT = ['nr_clicks']
    paramB = ['nr_clicks']
    # labelT = ['Information Gathered']
    labelB = ['nr_clicks']
    # limT = [(2,20)]
    limB = [(0,1.03)]
    # limB = [(-1,-.3)]
    df['tmp'] = df['payoff_gross'] / df['payoff_perfect']
    df_mod['tmp'] = df_mod['payoff_gross'] / df_mod['payoff_perfect']
    df_trials['tmp'] = df_trials['payoff_gross'] / df_trials['payoff_perfect']
    df_mod_trials['tmp'] = df_mod_trials['payoff_gross'] / df_mod_trials['payoff_perfect']

    for p in range(len(paramB)):

        if not(big):
            fig = plt.figure(figsize=(16, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[3, 5, 5])
            fontsize_ticks = 20
            fontsize_labels = 32
            fontsize_legend = 18
        else:
            fig = plt.figure(figsize=(32, 8))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25])
            fontsize_ticks = 32
            fontsize_labels = 42
            fontsize_legend = 36

        plt.sca(plt.subplot(gs[0]))
        perfect = p_d.get_means_by_cond(df_mod, cond='sigma', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[paramB[p]]), color='#17becf', lw=8)#/ perfect, 
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=paramB[p], perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='sigma', param=[paramB[p]])# / perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
        plt.ylabel(labelB[p], fontsize=fontsize_labels)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
        # plt.xlim((-.2,1.2))
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[1]))
        perfect = p_d.get_means_by_cond(df_mod, cond='alpha', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='alpha', param=[paramB[p]]), color='#17becf', lw=8)#/ perfect, 
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='alpha', param=paramB[p], perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='alpha', param=[paramB[p]])# / perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Dispersion ['+r'$\alpha$]', fontsize=fontsize_labels)
        plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False)
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)

        plt.sca(plt.subplot(gs[2]))
        perfect = p_d.get_means_by_cond(df_mod, cond='cost', param=['payoff_perfect'])
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[paramB[p]]), color='#17becf', lw=8)#, label='_nolegend_')
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=paramB[p], perfect_performance=perfect)
        y = p_d.get_means_by_cond(df, cond='cost', param=[paramB[p]])# / perfect
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
        plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
        plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        plt.ylim(limB[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)
        if not(big):
            plt.legend(['Model','Participants'], fontsize=fontsize_legend, loc='upper right')
        else:
            ax = plt.subplot(gs[2])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(['Model','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))


        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            bigstr = ''
            if big:
                bigstr = 'big_'
            ttl = out_dir+labelB[p]+'_'+bigstr+out_suffix+'.png'
            # plt.savefig(out_dir+'Reward_relative_'+bigstr+'sem.png',bbox_inches='tight',pad_inches=0.05)
            plt.savefig(ttl,bbox_inches='tight',pad_inches=0.05)

        plt.show()


def plt_conf_mat(confusion_mat, points_lost=[], save=False, savename=[], clim=[], big=True):

    if big:
        fontsize_ticks = 32 #18, 24-text
        fontsize_labels = 42 #24
        fontsize_legend = 36
        plt.figure(figsize=(16,16))
    else:
        fontsize_ticks = 20
        fontsize_labels = 32
        fontsize_legend = 18
        plt.figure(figsize=(16,16))

    strat_labels = ['SAT-TTB+','SAT-TTB','TTB','WADD','random','other']
    
    plt.imshow(confusion_mat)
    ax = plt.gca()
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            if 'pct' in savename.lower():
                ax.text(j, i, '{:.1f}'.format(points_lost[i,j])+'%',ha="center", va="center", color="w",fontsize=fontsize_ticks,fontweight='bold')
            elif 'pts' in savename.lower():
                ax.text(j, i, '{:.1f}'.format(points_lost[i,j])+'pts',ha="center", va="center", color="w",fontsize=fontsize_ticks,fontweight='bold')
            else:
                ax.text(j, i, '{:.1f}'.format(points_lost[i,j]),ha="center", va="center", color="w",fontsize=fontsize_ticks,fontweight='bold')
    plt.xticks(range(len(confusion_mat)),labels=strat_labels[:len(confusion_mat)],rotation=90,fontsize=fontsize_ticks)
    plt.yticks(range(len(confusion_mat)),labels=strat_labels[:len(confusion_mat)],fontsize=fontsize_ticks)
    plt.xlabel('Participant Strategy', fontsize=fontsize_labels)
    if savename[-7:] == 'fitcost':
        plt.ylabel('Model Strategy w/ implicit cost', fontsize=fontsize_labels)
    else:
        plt.ylabel('Model Strategy', fontsize=fontsize_labels)
    # plt.title('Strategy Selection Confusion Matrix', fontsize=fontsize_labels)
    if (savename[27:34] == 'avgPts4') | (savename[-7:] == 'avgPts4'):
        plt.title('Confusion Matrix (normalized rows)', fontsize=fontsize_labels)
    else:
        plt.title('Confusion Matrix', fontsize=fontsize_labels)
    if clim != []:
        plt.clim(0, clim)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    if save:
        plt.savefig(savename+'.png',bbox_inches='tight',pad_inches=0.05)
    plt.show()


def condition_plots_n(save=False, in_dir='', out_dir='figs/', out_suffix='', in_dir_human='data/human/', in_suffix='', params=[], labels=[], ylims=[], big=True, remove_participants=False):

    df_trials = pd.read_pickle(in_dir_human+'trials'+in_suffix+'.pkl')
    if remove_participants:
        df_trials = remove_rand_participants(df_trials)
        df = df_trials.groupby(['sigma','alpha','cost']).mean()
    else:
        df = pd.read_pickle(in_dir_human+'mean_by_condition'+in_suffix+'.pkl')
    df_mod = pd.read_csv(in_dir+'mean_by_condition_model.csv')
    df_mod_trials = pd.read_csv(in_dir+'mean_by_condition_model.csv')

    stakes = df.index.levels[0]
    dispersion = df.index.levels[1]
    costs = df.index.levels[2]

    nr_rows = len(params)

    if big:
        fig = plt.figure(figsize=(32, 8*nr_rows))
        gs = gridspec.GridSpec(nr_rows, 3, width_ratios=[1, 1, 1.25])
        fontsize_ticks = 32
        fontsize_labels = 42
        fontsize_legend = 36
    else:
        fig = plt.figure(figsize=(16, 6*nr_rows))
        gs = gridspec.GridSpec(nr_rows, 3, width_ratios=[3, 5, 5])
        fontsize_ticks = 20
        fontsize_labels = 32
        fontsize_legend = 18

    perf = []
    for p in range(nr_rows):

        if params[p]=='payoff_relative':
            df['payoff_relative'] = df['payoff_gross'] / df['payoff_perfect']
            df_trials['payoff_relative'] = df_trials['payoff_gross'] / df_trials['payoff_perfect']
            df_mod = pd.read_pickle('data/model/human_trials/mean_by_condition_model.pkl')
            df_mod['payoff_relative'] = df_mod['payoff_gross'] / df_mod['payoff_perfect']
            if remove_participants:
                df_mod_2 = pd.read_csv('data/model/human_trials_fitcost_exclude/mean_by_condition_model.csv')
            else:
                df_mod_2 = pd.read_pickle('data/model/human_trials_fitcost/mean_by_condition_model.pkl')
            df_mod_2['payoff_relative'] = df_mod_2['payoff_gross'] / df_mod_2['payoff_perfect']
            # tmp = []; tmp2 = []
            # for j, t in df_trials.iterrows():
                # tmp.append(t.payoff_matrix[t.payoff_index][np.argmax(t.EVs)]) # / t.payoff_perfect
                # tmp2.append(t.payoff_gross / t.payoff_perfect)
            # df_trials['payoff_relative_bestBet'] = tmp / df_trials.payoff_perfect
            # df_trials['payoff_relative'] = tmp2
            # df['payoff_relative_bestBet'] = df_trials.groupby(['sigma','alpha','cost']).mean()['payoff_relative_bestBet']
            # df['payoff_rÍelative'] = df_trials.groupby(['sigma','alpha','cost']).mean()['payoff_relative']

            df['payoff_relative_bestBet'] = df['payoff_gross_bestBet'] / df['payoff_perfect']
            df_trials['payoff_relative_bestBet'] = df_trials['payoff_gross_bestBet'] / df_trials['payoff_perfect']
        else:
            df_mod = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')

        plt.sca(plt.subplot(gs[3*p]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='sigma', param=[params[p]]), color='#17becf', lw=8)
        if params[p]=='payoff_relative':
            plt.plot(p_d.get_means_by_cond(df_mod_2, cond='sigma', param=[params[p]]), '--', color='#17becf', lw=8)
            sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param='payoff_relative_bestBet')
            y = p_d.get_means_by_cond(df, cond='sigma', param=['payoff_relative_bestBet'])
            plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8, ls='--')
            perf = [np.mean(p_d.get_means_by_cond(df_mod, cond='sigma', param=[params[p]])), np.mean(p_d.get_means_by_cond(df_mod_2, cond='sigma', param=[params[p]])), np.mean(p_d.get_means_by_cond(df, cond='sigma', param=['payoff_relative_bestBet'])), np.mean(p_d.get_means_by_cond(df, cond='sigma', param=[params[p]]))]
            print('model performance: ', perf[0])
            print('model performance w/ implicit cost: ', perf[1])
            print('participant perf w/ perfect execution: ', perf[2])
            print('participant performance: ', perf[3])
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='sigma', param=params[p])
        y = p_d.get_means_by_cond(df, cond='sigma', param=[params[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.ylabel(labels[p], fontsize=fontsize_labels)
        plt.xticks(np.arange(len(stakes)), stakes, fontsize=fontsize_ticks)
        plt.ylim(ylims[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)
        if (p+1)==nr_rows:
            plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
        else:
            plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[3*p+1]))
        plt.plot(np.flip(p_d.get_means_by_cond(df_mod, cond='alpha', param=[params[p]])), color='#17becf', lw=8)
        if params[p]=='payoff_relative':
            plt.plot(np.flip(p_d.get_means_by_cond(df_mod_2, cond='alpha', param=[params[p]])), '--', color='#17becf', lw=8)
            sem = np.flip(p_d.get_sem_by_cond(df_trials, cond='alpha', param='payoff_relative_bestBet')[0])
            y = np.flip(p_d.get_means_by_cond(df, cond='alpha', param=['payoff_relative_bestBet']))
            plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8, ls='--')
        sem = np.flip(p_d.get_sem_by_cond(df_trials, cond='alpha', param=params[p])[0])
        y = np.flip(p_d.get_means_by_cond(df, cond='alpha', param=[params[p]]))
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(dispersion)), dispersion, fontsize=fontsize_ticks)
        plt.ylim(ylims[p])
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True)
        if (p+1)==nr_rows:
            plt.xlabel('Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels)
            plt.xticks(np.arange(len(dispersion)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
            plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        else:
            plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        plt.sca(plt.subplot(gs[3*p+2]))
        plt.plot(p_d.get_means_by_cond(df_mod, cond='cost', param=[params[p]]), color='#17becf', lw=8)
        if params[p]=='payoff_relative':
            plt.plot(p_d.get_means_by_cond(df_mod_2, cond='cost', param=[params[p]]), '--', color='#17becf', lw=8)
            sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param='payoff_relative_bestBet')
            y = p_d.get_means_by_cond(df, cond='cost', param=['payoff_relative_bestBet'])
            plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8, ls='--')
        sem, _ = p_d.get_sem_by_cond(df_trials, cond='cost', param=params[p])
        y = p_d.get_means_by_cond(df, cond='cost', param=[params[p]])
        plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
        plt.xticks(np.arange(len(costs)), costs, fontsize=fontsize_ticks)
        plt.ylim(ylims[p])
        plt.grid(True)
        if (p+1)==nr_rows:
            plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
            plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
        else:
            plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

        if big:
            ax = plt.subplot(gs[3*p+2])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            if p==0:
                if params[p]=='payoff_relative':
                    ax.legend(['Model','Model with\nimplicit cost','Participants with\nperfect use of info.','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
                else:
                    ax.legend(['Model','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))
        elif p==0:
            plt.legend(['Model','Participants'], fontsize=fontsize_legend, loc='upper right')

    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bigstr = ''
        if big:
            bigstr = '_big'
        ttl = out_dir
        for l in labels:
            ttl += l+'+'
        randStr = ''
        if remove_participants:
            randStr = '_noRandParticipants'
        ttl = ttl[:-1]+bigstr+randStr+out_suffix+'.png'
        plt.savefig(ttl.replace(' ','_'),bbox_inches='tight',pad_inches=0.05)

        plt.show()

    return perf


def tsne(df=[], filename=[]):

    fontsize_labels = 42
    fontsize_legend = 36

    if len(df)==0:
        df = pd.read_pickle('data/human/1.0/trials.pkl')

    filenames = ['tsne_1-4','tsne_5-50','tsne_lr','tsne_centroids','tsne','tsne_centroidsMiddle','tsne_centroidsBinary','tsne_centroidsBinary2','tsne_centroidsBinaryAll','tsne_model']
    if len(filename)==0:
        filename = filenames[-1]
    with open(filename+'.pkl', 'rb') as f:
        dat = pickle.load(f)

        
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', [.8,.8,.8],'k']

    i = 0;# for i in [0]:#range(len(dat['Y'])):
    if not(filename=='tsne_model'):
        Y = dat['Y'][i]
        centroids = Y[47360:,:]
        Y = Y[:47360,:]
        perplexity = dat['perplexity'][i]
    else:
        Y = dat['Y'][i]
        centroids = Y[-4:,:]
        Y = Y[:-4,:]
        perplexity = dat['perplexity'][i]

    for label_type in ['strategy','cluster']:

        if label_type == 'strategy':
            if not(filename=='tsne_model'):
                strategies = ['TTB','WADD','SAT_TTB','TTB_SAT','Rand','Other']
                strategies_ = ['TTB','WADD','TTB_SAT','SAT_TTB','Rand','Other']
                legend_labels = ['TTB','WADD','SAT-TTB','SAT-TTB+','random','other']
                # cc = [4,5,0,2,3,1]
                cc = [0,1,3,2,4,5]
            else:
                strategies = ['TTB','WADD','SAT_TTB','TTB_SAT','Other']
                strategies_ = ['TTB','WADD','TTB_SAT','SAT_TTB','Other']
                legend_labels = ['TTB','WADD','SAT-TTB','SAT-TTB+','other']
                cc = [0,1,3,2,4]
            
        elif label_type == 'cluster':
            if not(filename=='tsne_model'):
                strategies = ['TTB','WADD','SAT_TTB','TTB_SAT','Rand']
                legend_labels = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
                cc = [0,1,2,3,4]
            else:
                strategies = ['TTB','WADD','SAT_TTB','TTB_SAT']
                legend_labels = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4']
                cc = [0,1,2,3]
            strategies_ = strategies

        # ix_pts = np.arange(0,len(Y),len(Y)/10)
        plt.figure(figsize=[9,9])
        
#         for r in range(len(ix_pts)-1):
        for j, s in enumerate(strategies_):
            if label_type == 'strategy':
                idx = df.strategy == s
            elif label_type == 'cluster':
                if not(filename=='tsne_model'):
                    idx = df.strategy_click_embedding_prob_k5 == s
                else:
                    idx = df.strategy_click_embedding_prob_k4 == s
#             ix1 = int(round(ix_pts[r])); ix2 = int(round(ix_pts[r+1]));
#             idx2 = np.zeros(idx.shape, dtype=bool)
#             idx2[ix1:ix2] = idx[ix1:ix2]
#                 plt.scatter(Y[idx2,0], Y[idx2,1], s=.8, alpha = .5, color=colors[j])
            plt.scatter(Y[idx,0], Y[idx,1], s=.2, alpha = 1, color=colors[cc[j]], marker='.', label='_nolegend_')

        # plt.title('perplexity = '+str(perplexity)+' learning rate = '+str(dat['learning_rate'][i]))

        xlim = plt.gca().get_xlim(); ylim = plt.gca().get_ylim()
        for j, s in enumerate(strategies):
            plt.scatter(-100,-100, color=colors[j], s=500)
        plt.scatter(-100, 100, s=700, alpha = 1, color='y', marker='+'); legend_labels.append('Centroids')
            
        plt.xlim(xlim); plt.ylim(ylim);
        plt.axis('off')
        plt.legend(legend_labels, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
        plt.scatter(centroids[:,0], centroids[:,1], s=200, alpha = 1, color='y', marker='+', label='_nolegend_')
        if not(filename=='tsne_model'):
            plt.title('Cluster visualization (Participants)', fontsize=fontsize_labels)
        else:
            plt.title('Cluster visualization (Model)', fontsize=fontsize_labels)

        plt.savefig('figs/no_implicit_cost/tSNE_p1'+label_type+filename[4:]+'.png',bbox_inches='tight',pad_inches=0.05)
        plt.show()



def plt_conf_mat_clusterVsKmeans(isHuman=True, save=False, df=[]):

    from statsmodels.stats.inter_rater import cohens_kappa

    if isHuman:
        strategies = ['TTB','WADD','TTB_SAT','SAT_TTB','Rand','Other']
        df = pd.read_pickle('data/human/1.0/trials.pkl')
        strat_labels = ['TTB','WADD','SAT-TTB','SAT-TTB+','random','other']
    else:
        strategies = ['TTB','WADD','TTB_SAT','SAT_TTB','Other']
        if len(df)==0:
            df = pd.read_pickle('data/model/no_implicit_cost/trials_model.pkl')
        strat_labels = ['TTB','WADD','SAT-TTB','SAT-TTB+','other']
    confusion_mat = np.zeros((len(strategies),len(strategies)))
    for i, s in enumerate(strategies):
        if isHuman:
            df_ = df[df.strategy_click_embedding_prob_k5 == s]
        else:
            df_ = df[df.strategy_click_embedding_prob_k4 == s]
        for j, s_ in enumerate(strategies):
            confusion_mat[i,j] = sum(df_.strategy == s_)
    confusion_mat = confusion_mat[:-1,:]
    pct = confusion_mat/sum(sum(confusion_mat))*100

    fontsize_ticks = 32 #18, 24-text
    fontsize_labels = 42 #24
    fontsize_legend = 36
    plt.figure(figsize=(16,16))

    clust_labels = ['1','2','3','4','5']

    plt.imshow(confusion_mat)
    ax = plt.gca()
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, '{:.1f}'.format(pct[i,j])+'%',ha="center", va="center", color="w",fontsize=fontsize_ticks,fontweight='bold')
    plt.xticks(range(confusion_mat.shape[1]),labels=strat_labels[:confusion_mat.shape[1]],rotation=90,fontsize=fontsize_ticks)
    plt.yticks(range(confusion_mat.shape[0]),labels=clust_labels[:confusion_mat.shape[0]],fontsize=fontsize_ticks)
    plt.ylabel('k-means cluster', fontsize=fontsize_labels)
    plt.xlabel('strategy', fontsize=fontsize_labels)
    plt.title('Confusion Matrix', fontsize=fontsize_labels)
    # plt.clim(0, clim)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    if save:
        if isHuman:
            plt.savefig('figs/no_implicit_cost/confusion_mat_kmeans-strategy.png',bbox_inches='tight',pad_inches=0.05)
        else:
            plt.savefig('figs/no_implicit_cost/confusion_mat_kmeans-strategy_model.png',bbox_inches='tight',pad_inches=0.05)
    plt.show()

    print(cohens_kappa(confusion_mat[:,:confusion_mat.shape[0]]))



def remove_rand_participants(df):

    nr_removed = 0
    for p in df.pid.unique():
        if df[df.pid==p]['Rand'].mean() > 0.5 :
            df = df[df['pid'] != p]
            nr_removed += 1
    print('removed ',str(nr_removed),' participants')

    return df



def lda(isHuman=False, save=False, pca=False, jitter=True, df=[], mods=[], title=''):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA

    fontsize_labels = 42
    fontsize_legend = 36

    if isHuman:
        if len(df)==0:
            df = pd.read_pickle('data/human/1.0/trials.pkl')
        km = pickle.load(open('data/human/1.0/kmeans.pkl', 'rb'))
    elif not(isHuman):
        if len(df)==0:
            df = pd.read_pickle('data/model/no_implicit_cost/trials_model.pkl')
        df = df.iloc[::10, :]
        km = pickle.load(open('data/model/no_implicit_cost/kmeans_model.pkl', 'rb'))
    if isHuman=='both':
        km1 = pickle.load(open('data/human/1.0/kmeans.pkl', 'rb'))
        km1 = km1['click_embedding_prob']['kmeans'][0].cluster_centers_
        km2 = pickle.load(open('data/model/no_implicit_cost/kmeans_model.pkl', 'rb'))
        km2 = km2['click_embedding_prob']['kmeans'][0].cluster_centers_
        km = np.append(km1, km2, axis=0)
    else:
        km = km['click_embedding_prob']['kmeans'][0].cluster_centers_
    km_bin = km >= 0.5

    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', 'k', [.8,.8,.8]]#'#e3c9a4']
    if isHuman:
        strategies_lab = ['TTB','WADD','SAT_TTB','TTB_SAT','Rand','Other']
        legend_lab_s = ['TTB','WADD','SAT-TTB','SAT-TTB+','random','other']
        order = [5,4,3,2,1,0]
        # order = [0,1,2,3,4,5]
        legend_lab_k = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
    else:
        strategies_lab = ['TTB','WADD','SAT_TTB','TTB_SAT','Other']
        legend_lab_s = ['TTB','WADD','SAT-TTB','SAT-TTB+','other']
        order = [4,3,2,1,0]
        # order = [0,1,2,3,4]
        legend_lab_k = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4']
    strategies_plt = [strategies_lab[i] for i in order]

    X = np.stack(df['click_embedding_prob'].values)

    y_km = np.ones(len(df)); y_km[:] = np.nan
    y_strat = np.ones(len(df)); y_strat[:] = np.nan
    for i, s in enumerate(strategies_plt):
        if (isHuman==True) | (isHuman=='both'):
            y_km[df.strategy_click_embedding_prob_k5 == s] = order[i]
        else:
            y_km[df.strategy_click_embedding_prob_k4 == s] = order[i]
        y_strat[df.strategy == s] = order[i]

    mods_out = []
    if pca:
        if len(mods) == 0:
            mod = PCA(n_components=2).fit(X)
            mods_out.append(mod)
        else:
            mod = mods[ii]
        X_ = mod.transform(X)
        centroids = mod.transform(km_bin.astype(float))
    for ii, y in enumerate([y_km, y_strat]):
        if not(pca):
            if len(mods) == 0:
                mod = LDA(n_components=2).fit(X, y)
                mods_out.append(mod)
            else:
                mod = mods[ii]
            X_ = mod.transform(X)
            centroids = mod.transform(km_bin.astype(float))
            # Fig3B_ex = np.zeros((4,6))
            # for i in [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(2,0)]:
            #     Fig3B_ex[i[0],i[1]] = 1
            # Fig3B_ex = mod.transform(np.ndarray.flatten(Fig3B_ex).reshape(1, -1))

        plt.figure(figsize=[9,9])

        if jitter:
            xrange = max(X_[:,0]) - min(X_[:,0]); yrange = max(X_[:,1]) - min(X_[:,1])
            varFact = 40
            jit = np.stack((np.random.normal(loc=0.0, scale=xrange/varFact, size=len(X_)), np.random.normal(loc=0.0, scale=yrange/varFact, size=len(X_)))).T
            X__ = X_ + jit
        else:
            X__ = X_
        for i, s in enumerate(np.sort(np.unique(y))):
            if ii==0:
                iii = i+1
            else:
                iii = i
                # pdb.set_trace()
            idx = y == order[iii]
            plt.scatter(X__[idx, 0], X__[idx, 1], s=50, alpha = .05, color=colors[order[iii]], marker='.', label='_nolegend_')

        xlim = plt.gca().get_xlim(); ylim = plt.gca().get_ylim()
        for j in range(len(np.unique(y))):
            plt.scatter(-100,-100, color=colors[j], s=500)
        plt.scatter(-100, 100, s=800, linewidth=4, alpha = 1, color='y', marker='+'); #legend_labels.append('Centroids')
        plt.xlim(xlim); plt.ylim(ylim);
        plt.axis('off')
        if isHuman:
            plt.title('Cluster visualization (Participants)', fontsize=fontsize_labels)
        else:
            plt.title('Cluster visualization (Model)', fontsize=fontsize_labels)
        if ii==0:
            ttl = title + 'kmeans'
            legend_lab_k.append('Centroids')
            plt.legend(legend_lab_k, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
        else:
            ttl = title + 'strategy'
            legend_lab_s.append('Centroids')
            plt.legend(legend_lab_s, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
        plt.scatter(centroids[:,0], centroids[:,1], s=300, linewidth=3, alpha = 1, color='y', marker='+', label='_nolegend_')
        # plt.scatter(Fig3B_ex[:,0], Fig3B_ex[:,1], s=100, linewidth=3, alpha = 1, color='y', marker='x', label='_nolegend_')
        if save:
            if not(isHuman):
                ttl += '_model'
            if pca:
                plt.savefig('figs/no_implicit_cost/pca_'+ttl+'.png',bbox_inches='tight',pad_inches=0.05)
            else:
                plt.savefig('figs/no_implicit_cost/lda_'+ttl+'.png',bbox_inches='tight',pad_inches=0.05)

    plt.show()

    return mods_out












# def behav_bars(save=False, in_dir='data/model/no_implicit_cost/', out_dir='figs/', out_suffix='', in_dir_human='data/human/', plot_kmeans=False, in_suffix=''):

#     df_m = pd.read_pickle(in_dir+'mean_by_condition_model.pkl')
#     # df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl')
#     #########
#     df_h = pd.read_pickle(in_dir_human+'mean_by_condition.pkl') #########
#     #########
#     df = pd.read_pickle(in_dir_human+'mean_by_condition_EVminTime.pkl')

#     behav = ['net_payoff', 'nr_clicks', 'click_var_gamble', 'click_var_outcome']
#     behav_labels = ['Reward', 'Information Gathered', 'Alternative Variance', 'Attribute Variance']

#     ind = 0
#     width = 0.2
#     offset = [-0.22,0,0.22]
#     # labels = ['Model','Exp 1','Exp 2']
#     labels = ['Model','Control','Exp']
#     colors = ['mediumturquoise','mediumaquamarine','teal']
#     for s_, s in enumerate(behav):
#         fig = plt.figure(figsize=(8, 6))
#         for i in df.index.levels[0]:
#             for j in df.index.levels[1]:
#                 for k in df.index.levels[2]:
#                     ind += 1
#                     pp = []
#                     for x, d in enumerate([df_m, df_h, df]):
#                         if x == 0:
#                             sem = 0
#                         elif x == 1:
#                             trials = pd.read_pickle(in_dir_human+'trials.pkl')
#                             sem = trials.groupby(['pid']).mean()[s].sem()
#                         elif x == 2:
#                             trials = pd.read_pickle(in_dir_human+'trials_EVminTime.pkl')
#                             sem = trials.groupby(['pid']).mean()[s].sem()
#                         dat = d.iloc[(d.T.columns.get_level_values('sigma')==i)&(d.T.columns.get_level_values('alpha')<(j+.01))&(d.T.columns.get_level_values('alpha')>(j-.01))&(d.T.columns.get_level_values('cost')==k)][s]
#                         p = plt.bar(ind+offset[x], dat, width, color=colors[x], yerr=sem)
#                         pp.append(p)

#         plt.ylabel(behav_labels[s_], fontsize=32)
#         plt.yticks(fontsize=20)
#         plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
#         plt.figtext(.15, 0, 'Alpha=$10^{-0.5}$\nCost=1',verticalalignment='bottom',horizontalalignment='left',fontsize=16)
#         plt.figtext(.35, 0, 'Alpha=$10^{-0.5}$\nCost=4',verticalalignment='bottom',horizontalalignment='left',fontsize=16)
#         plt.figtext(.55, 0, 'Alpha=$10^{0.5}$\nCost=1',verticalalignment='bottom',horizontalalignment='left',fontsize=16)
#         plt.figtext(.75, 0, 'Alpha=$10^{0.5}$\nCost=4',verticalalignment='bottom',horizontalalignment='left',fontsize=16)
#         plt.legend((pp), labels, fontsize=18)

#         if save:
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)
#             plt.savefig(out_dir+behav_labels[s_]+'Bars'+out_suffix+'.png',bbox_inches='tight',pad_inches=0.05)

#         plt.show()