import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sn;
import pdb
import pickle
import process_data as p_d
import os
from matplotlib import colors as Colors
from matplotlib import cm
from ast import literal_eval
import time
from collections import defaultdict

def centroids(model_file, human_file, fig_dir=None, save=False):
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	df1 = pd.read_csv(model_file)
	df2 = pd.read_csv(human_file)

	centers1 = eval(df1['cluster_centers'].iloc[0])
	centers2 = eval(df2['cluster_centers'].iloc[0])

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	# least to most clicks
	centroid_orders = [np.argsort(np.sum(centers1, axis=1)),\
					np.argsort(np.sum(centers2, axis=1))]

	fig = plt.figure(figsize=(32, 10))
	plt.figtext(0.5, 0.96, 'Model', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	plt.figtext(0.5, 0.49, 'Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	gs = gridspec.GridSpec(2, 5)

	sp = [centroid_orders[0][i] for i in [1,3,0,2]] # TTB, WADD, SAT-TTB, SAT-TTB+
	for i in range(np.shape(centers1)[0]):
		plt.sca(plt.subplot(gs[i]))
		plt.title('Centroid '+str(i+1),fontsize=fontsize_labels)
		plt.imshow(np.reshape(centers1[sp[i]],(4,6)), vmin=0, vmax=1)
		if i==0:
			plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
			plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
		else:
			plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
	plt.sca(plt.subplot(gs[4]))
	plt.axis('off')
	divider = make_axes_locatable(plt.gca())
	cax = divider.append_axes("right", size="20%", pad=0)
	cbar = plt.colorbar(cax=cax)
	cbar.ax.tick_params(labelsize=20)
	plt.figtext(0.86, 0.71, 'Centroid values\n(Prob. of click)', ha='center', va='center', fontsize=fontsize_legend, rotation='vertical')

	sp = [centroid_orders[1][i] for i in [2,4,1,3,0]] # TTB, WADD, SAT-TTB, SAT-TTB+, random
	for i in range(np.shape(centers2)[0]):
		plt.sca(plt.subplot(gs[i+5]))
		plt.imshow(np.reshape(centers2[sp[i]],(4,6)), vmin=0, vmax=1)
		if i==0:
			plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
			plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
		else:
			plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
			plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
		if i==4:
			plt.title('Centroid 5',fontsize=fontsize_labels)

	fig.text(0.5, 0.04, 'Total prob. of observed outcomes', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(0.087, 0.5, 'Prob. of outcome', fontsize=fontsize_labels, ha='center', va='center', rotation='vertical')

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'centroids.png', bbox_inches='tight', pad_inches=0.05, facecolor='w')
	
	plt.show()

def strategies(model_file, human_file, fig_dir=None, save=False):

	df1 = pd.read_csv(model_file)
	df2 = pd.read_csv(human_file)

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	strats = ['SAT-TTB+','SAT-TTB','TTB','WADD']
	params = ['TTB_SAT','SAT_TTB','TTB','WADD']

	for p in params:
		df1[p] = df1[p] * df1['trial_weight']

	fig = plt.figure(figsize=(32, 18))
	gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1.25]) 

	plt.figtext(0.5, 0.9, 'Model', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	plt.figtext(0.5, 0.49, 'Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')    

	plt.sca(plt.subplot(gs[0]))
	dat = np.array([df1.groupby(['sigma'])[p].apply(sum).values for p in params])
	dat /= sum(dat); dat = np.vstack([[0]*dat.shape[1], dat])
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
	plt.ylabel('Strategy Frequency', fontsize=fontsize_labels)
	plt.xticks(np.arange(dat.shape[1]), sorted(df1['sigma'].unique()), fontsize=fontsize_ticks)
	plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
	plt.yticks(fontsize=fontsize_ticks)
	plt.ylim((0,1.02))
	plt.grid(True)

	plt.sca(plt.subplot(gs[1]))
	dat = np.array([df1.groupby(['alpha'])[p].apply(sum).values for p in params])
	dat /= sum(dat); dat = np.vstack([[0]*dat.shape[1], dat]); dat = np.flip(dat, axis=1) # flip for inverse alpha
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
	plt.xticks(np.arange(dat.shape[1]), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
	plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
	plt.ylim((0,1.02))
	plt.grid(True)

	plt.sca(plt.subplot(gs[2]))
	dat = np.array([df1.groupby(['cost'])[p].apply(sum).values for p in params])
	dat /= sum(dat); dat = np.vstack([[0]*dat.shape[1], dat])
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
	plt.xticks(np.arange(dat.shape[1]), sorted(df1['cost'].unique()), fontsize=fontsize_ticks)
	plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
	plt.ylim((0,1.02))
	plt.grid(True)

	ax = plt.subplot(gs[2])
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(['SAT-TTB+','SAT-TTB','TTB','WADD'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))

	plt.sca(plt.subplot(gs[3]))
	dat = np.array([df2.groupby(['sigma'])['strategy'].apply(lambda x: x[x==p].value_counts()).values for p in params])
	dat = dat/sum(dat); dat = np.vstack([[0]*dat.shape[1], dat])
	sem = np.array([[df2[df2['sigma']==c].groupby('pid').mean()[s].sem() for c in sorted(df2['sigma'].unique())] for s in params])
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
		plt.errorbar(np.arange(dat.shape[1]), sum(dat[:i+2,:]), yerr=sem[i,:], ls='none', color='k')
	plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
	plt.ylabel('Strategy Frequency', fontsize=fontsize_labels)
	plt.xticks(np.arange(dat.shape[1]), sorted(df2['sigma'].unique()), fontsize=fontsize_ticks)
	plt.yticks(fontsize=fontsize_ticks)
	plt.ylim((0,1.02))
	plt.grid(True)

	plt.sca(plt.subplot(gs[4]))
	dat = np.array([df2.groupby(['alpha'])['strategy'].apply(lambda x: x[x==p].value_counts()).values for p in params])
	dat = dat/sum(dat); dat = np.vstack([[0]*dat.shape[1], dat]); dat = np.flip(dat, axis=1) # flip for inverse alpha
	sem = np.array([[df2[df2['alpha']==c].groupby('pid').mean()[s].sem() for c in sorted(df2['alpha'].unique())] for s in params])
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
		plt.errorbar(np.arange(dat.shape[1]), sum(dat[:i+2,:]), yerr=sem[i,:], ls='none', color='k')
	plt.xlabel('Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels)
	plt.xticks(np.arange(dat.shape[1]), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
	plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
	plt.ylim((0,1.02))
	plt.grid(True)

	plt.sca(plt.subplot(gs[5]))
	dat = np.array([df2.groupby(['cost'])['strategy'].apply(lambda x: x[x==p].value_counts()).values for p in params])
	dat = dat/sum(dat); dat = np.vstack([[0]*dat.shape[1], dat])
	sem = np.array([[df2[df2['cost']==c].groupby('pid').mean()[s].sem() for c in sorted(df2['cost'].unique())] for s in params])
	for i in range(dat.shape[0]-1):
		plt.fill_between(np.arange(dat.shape[1]), sum(dat[:i+1,:]), sum(dat[:i+2,:]), lw=4)
		plt.errorbar(np.arange(dat.shape[1]), sum(dat[:i+2,:]), yerr=sem[i,:], ls='none', color='k')
	plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
	plt.xticks(np.arange(dat.shape[1]), sorted(df2['cost'].unique()), fontsize=fontsize_ticks)
	plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
	plt.ylim((0,1.02))
	plt.grid(True)

	ax = plt.subplot(gs[5])
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	
	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'strategies.png',bbox_inches='tight',pad_inches=0.05, facecolor='w')

	plt.show()   

def heatmaps(model_file, human_file, fig_dir=None, save=False):

	paramL = ['nr_clicks','TTB','SAT_TTB','click_var_outcome','payoff_gross_relative','payoff_gross_relative']
	paramR = ['processing_pattern','WADD','TTB_SAT','click_var_gamble','payoff_gross_relative','payoff_gross_relative']
	titleL = ['Information Gathered','TTB Frequency','SAT-TTB Frequency','Attribute Variance','Relative Reward','Relative Reward']
	titleR = ['Alternative vs. Attribute','WADD Frequency','SAT-TTB+ Frequency','Alternative Variance','w/ implicit cost','w/ implicit cost']
	exclude = [False,False,False,False,False,True]

	for p in range(len(paramL)):
	
		df1 = pd.read_csv(model_file)
		df2 = pd.read_csv(human_file)
		if exclude[p]:
			df2 = p_d.remove_rand_participants(df2)
			exclude_str = '_exclude'
		else:
			exclude_str = ''
		df1[paramL[p]] = df1[paramL[p]] * df1['trial_weight'+exclude_str]
		df1[paramR[p]] = df1[paramR[p]] * df1['trial_weight'+exclude_str]

		fig = plt.figure(figsize=(16,10))
		gs = gridspec.GridSpec(4,4, width_ratios=[10,11.5,10,11.5], height_ratios=[0,100,100,100])
		ax = [None]*12

		headersize=38; titlesize=32; axislabsize = 30; ticklabsize=18;
		plt.sca(plt.subplot(gs[4]))
		dat = np.reshape(df1.groupby(['sigma','alpha','cost'])[paramL[p]].mean().values, (2,5,5))
		cb_range = [np.min(dat), np.max(dat)]
		ax[0] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
		ax[0].invert_yaxis()
		plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
		plt.sca(plt.subplot(gs[5]))
		ax[1] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
		ax[1].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		datTL = dat

		plt.sca(plt.subplot(gs[6]))
		# if titleR[p] == 'w/ implicit cost':
		# 	df1 = pd.read_csv(model_file.replace('exp1','exp1_fitcost'+exclude_str))

		# df1[paramR[p]] = df1[paramR[p]] * df1['trial_weight']
		dat = np.reshape(df1.groupby(['sigma','alpha','cost'])[paramR[p]].mean().values, (2,5,5))
		cb_range = [np.min(dat), np.max(dat)]
		ax[2] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
		ax[2].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		plt.sca(plt.subplot(gs[7]))
		ax[3] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
		ax[3].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		datTR = dat

		plt.sca(plt.subplot(gs[8]))
		dat = np.reshape(df2.groupby(['sigma','alpha','cost'])[paramL[p]].mean().values, (2,5,5))
		cb_range = [np.min(dat), np.max(dat)]
		ax[4] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
		ax[4].invert_yaxis()
		plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
		plt.sca(plt.subplot(gs[9]))
		ax[5] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
		ax[5].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		datBL = dat

		plt.sca(plt.subplot(gs[10]))
		dat = np.reshape(df2.groupby(['sigma','alpha','cost'])[paramR[p]].mean().values, (2,5,5))
		cb_range = [np.min(dat), np.max(dat)]
		ax[6] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r', cbar=False)
		ax[6].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		plt.sca(plt.subplot(gs[11]))
		ax[7] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='rocket_r')
		ax[7].invert_yaxis()
		plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
		datBR = dat

		plt.sca(plt.subplot(gs[12]))
		dat = datBL/datTL if paramL[p]=='processing_pattern' else datBL-datTL
		cb_range = [-np.max(abs(dat)), np.max(abs(dat))]
		ax[8] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='vlag', cbar=False)
		ax[8].invert_yaxis()
		plt.sca(plt.subplot(gs[13]))
		ax[9] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='vlag')
		ax[9].invert_yaxis()
		plt.tick_params(axis='y',which='both',left=False,labelleft=False)
		plt.xlabel('Cost [$\lambda$]', fontsize=30)
		plt.xticks(np.arange(5)+.5, ['0','1','2','4','8'], ha='center', fontsize=18)

		plt.sca(plt.subplot(gs[14]))
		dat = datBR/datTR if paramR[p]=='processing_pattern' else datBR-datTR
		cb_range = [-np.max(abs(dat)), np.max(abs(dat))]
		ax[10] = sn.heatmap(np.flip(dat[0],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='vlag', cbar=False)
		ax[10].invert_yaxis()
		plt.tick_params(axis='y',which='both',left=False,labelleft=False)
		plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
		plt.sca(plt.subplot(gs[15]))
		ax[11] = sn.heatmap(np.flip(dat[1],axis=0), vmin=cb_range[0], vmax=cb_range[1], cmap='vlag')
		ax[11].invert_yaxis()
		plt.tick_params(axis='y',which='both',left=False,labelleft=False)
		plt.xlabel('Cost [$\lambda$]', fontsize=30)
		plt.xticks(np.arange(5)+.5, ['0','1','2','4','8'], ha='center', fontsize=18)

		for i in range(0,12,2):
			pts0 = ax[i].get_position().get_points()
			pos = ax[i+1].get_position()
			pts = pos.get_points()
			pts[0][0]=pts0[1][0]+.01
			pos.set_points(pts)
			ax[i+1].set_position(pos) 
		pts_ = [None]*12
		for i in range(0,12):
			pts = ax[i].get_position().get_points()
			pos = ax[i].get_position()
			pts[1][1]=pts[1][1]-.04
			pos.set_points(pts)
			ax[i].set_position(pos)
			pts_[i] = pts[1][1]
		for i in range(4,16,4):
			plt.sca(plt.subplot(gs[i]))
			plt.ylabel('Dispersion \n['+r'$\alpha^{-1}$]', fontsize=axislabsize)
			plt.yticks(np.arange(5)+.5, [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], va='center', rotation=0, fontsize=ticklabsize)
		for i in range(12,16,2):
			plt.sca(plt.subplot(gs[i]))
			plt.xlabel('Cost [$\lambda$]', fontsize=axislabsize)
			plt.xticks(np.arange(5)+.5, ['0','1','2','4','8'], ha='center', fontsize=ticklabsize)

		l=0.28; r=0.67;
		plt.figtext(l, 1, titleL[p], ha='center', va='top', fontsize=headersize)
		plt.figtext(r, 1, titleR[p], ha='center', va='top', fontsize=headersize)
		plt.figtext(l-.05, pts_[0]+.02, 'Stakes \n[$\sigma=75$]', ha='right', va='bottom', fontsize=axislabsize)
		plt.figtext(l+.05, pts_[0]+.02, 'Stakes \n[$\sigma=150$]', ha='left', va='bottom', fontsize=axislabsize)
		plt.figtext(r-.05, pts_[0]+.02, 'Stakes \n[$\sigma=75$]', ha='right', va='bottom', fontsize=axislabsize)
		plt.figtext(r+.05, pts_[0]+.02, 'Stakes \n[$\sigma=150$]', ha='left', va='bottom', fontsize=axislabsize)
		plt.figtext(l, pts_[0], 'Model', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		plt.figtext(r, pts_[0], 'Model', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		plt.figtext(l, pts_[7], 'Participants', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		plt.figtext(r, pts_[7], 'Participants', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		plt.figtext(l, pts_[11], r'Participants $\minus$ Model', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		if paramR[p]=='processing_pattern':
			plt.figtext(r, pts_[11], r'Participants $\div$ Model', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')
		else:
			plt.figtext(r, pts_[11], r'Participants $\minus$ Model', ha='center', va='bottom', fontsize=titlesize, fontweight='bold')

		if save:
			if not os.path.exists(fig_dir): os.makedirs(fig_dir)
			ttl = fig_dir+'heatmaps_'+paramL[p]+'__'+paramR[p]+exclude_str+'.png'
			plt.savefig(ttl.replace(' ','_'), bbox_inches='tight', pad_inches=0.05, facecolor='w')

		plt.show()

def condition_lines(model_file, human_file, fig_dir=None, save=False):

	# params = ['nr_clicks','processing_pattern',\
	# 		'click_var_outcome','click_var_gamble',\
	# 		'payoff_gross_relative','payoff_gross_relative',\
	# 		'payoff_gross_relative','payoff_gross_relative',\
	# 		'payoff_net_relative','payoff_net_relative',\
	# 		'payoff_net_relative','payoff_net_relative']
	# labels = ['Information Gathered','Alternative vs. Attribute',\
	# 		'Attribute Variance','Alternative Variance',\
	# 		'Relative Performance','w/ implicit cost',\
	# 		'Relative Performance\n(Participants excluded)','w/ implicit cost',\
	# 		'Net Relative Performance','w/ implicit cost',\
	# 		'Net Relative Performance\n(Participants excluded)','w/ implicit cost']
	# # fitcost_and_exclude = [False,False,False,False,False,True,False,True]
	# fitcost = [False]*4 + [False,True]*4
	# exclude = [False]*4 + ([False]*2 + [True]*2)*2
	# ylims = [(2,16),(-1,-.3),(0,.2),(0,.06),(0,1.03),(0,1.03),(0,1.15),(0,1.15),(0,1.03),(0,1.03),(0,1.15),(0,1.15)]
	# idxs = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
	params = ['nr_clicks','processing_pattern',\
			'click_var_outcome','click_var_gamble',\
			'payoff_gross_relative','payoff_gross_relative',\
			'payoff_net_relative','payoff_net_relative']
	labels = ['Information Gathered','Alternative vs. Attribute',\
			'Attribute Variance','Alternative Variance',\
			'Relative\nPerformance','Relative Performance\n(Participants excluded)',\
			'Net Relative\nPerformance','Net Relative Performance\n(Participants excluded)']
	# fitcost_and_exclude = [False,False,False,False,False,True,False,True]
	# fitcost = [False]*4 + [False,True]*4
	exclude = [False]*4 + [False,True]*2
	ylims = [(2,16),(-1,-.3),(0,.2),(0,.06),(0,1.15),(0,1.15),(0,1.15),(0,1.15)]
	idxs = [[0,1],[2,3],[4,5],[6,7]]

	for idx in idxs: # seperate figures

		fig = plt.figure(figsize=(32, 8*len(idx)))
		gs = gridspec.GridSpec(len(idx), 3, width_ratios=[1, 1, 1.25])
		fontsize_ticks = 32
		fontsize_labels = 42
		fontsize_legend = 36

		perf = []
		for p_, p in enumerate(idx): # seperate rows of figure

			# if fitcost[p] and exclude[p]:
			# 	df1 = pd.read_csv(model_file.replace('exp1','exp1_fitcost_exclude'))
			# elif fitcost[p]:
			# 	df1 = pd.read_csv(model_file.replace('exp1','exp1_fitcost'))
			# else:
			# 	df1 = pd.read_csv(model_file)
			# df2 = pd.read_csv(human_file)
			# if exclude[p]:
			# 	df2 = p_d.remove_rand_participants(df2)
			# 	exclude_str = '_exclude'
			# else:
			# 	exclude_str = ''

			df1 = pd.read_csv(model_file)
			df2 = pd.read_csv(human_file)
			if exclude[p]:
				# df1_fitcost = pd.read_csv(model_file.replace('exp1','exp1_fitcost_exclude'))
				# df1_fitcost[params[p]] = df1_fitcost[params[p]] * df1_fitcost['trial_weight'] # it's implied by the directory name that the weights are for excluded participants
				df1[params[p]] = df1[params[p]] * df1['trial_weight_exclude']
				df2 = p_d.remove_rand_participants(df2)
				exclude_str = '_exclude'
			else:
				# df1_fitcost = pd.read_csv(model_file.replace('exp1','exp1_fitcost'))
				# df1_fitcost[params[p]] = df1_fitcost[params[p]] * df1_fitcost['trial_weight_exclude']
				df1[params[p]] = df1[params[p]] * df1['trial_weight']
				exclude_str = ''

			# if fitcost_and_exclude[p]:
			# 	df1 = pd.read_csv(model_file.replace('exp1','exp1_fitcost_exclude'))
			# 	df2 = pd.read_csv(human_file)
			# 	df2 = p_d.remove_rand_participants(df2)
			# else:
			# 	df1 = pd.read_csv(model_file)
			# 	df2 = pd.read_csv(human_file)
			# trial_weight_str = 'trial_weight'+exclude_str if not (fitcost[p] and exclude[p]) else 'trial_weight' # it's not in the column name for exp1_fitcost_exclude, since it's implied in the firectory name
			# df1[params[p]] = df1[params[p]] * df1[trial_weight_str]

			plt.sca(plt.subplot(gs[3*p_]))
			dat = df1.groupby('sigma')[params[p]].mean().values
			plt.plot(dat, color='#17becf', lw=8)
			dat = df2.groupby(['sigma','pid'])[params[p]].mean()
			y = [dat.loc[i].mean() for i in dat.index.levels[0]]
			sem = [dat.loc[i].sem() for i in dat.index.levels[0]]
			plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
			plt.ylabel(labels[p], fontsize=fontsize_labels)
			plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=fontsize_ticks)
			plt.ylim(ylims[p])
			plt.yticks(fontsize=fontsize_ticks)
			plt.grid(True)
			if (p_+1)==len(idx):
				plt.xlabel('Stakes [$\sigma$]', fontsize=fontsize_labels)
			else:
				plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)

			plt.sca(plt.subplot(gs[3*p_+1]))
			dat = df1.groupby('alpha')[params[p]].mean().values
			plt.plot(dat, color='#17becf', lw=8)
			dat = df2.groupby(['alpha','pid'])[params[p]].mean()
			y = [dat.loc[i].mean() for i in dat.index.levels[0]]
			sem = [dat.loc[i].sem() for i in dat.index.levels[0]]
			plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
			plt.ylim(ylims[p])
			plt.yticks(fontsize=fontsize_ticks)
			plt.grid(True)
			if (p_+1)==len(idx):
				plt.xlabel('Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels)
				plt.xticks(np.arange(len(y)), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
				plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
			else:
				plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=fontsize_ticks) 
				plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

			plt.sca(plt.subplot(gs[3*p_+2]))
			dat = df1.groupby('cost')[params[p]].mean().values
			plt.plot(dat, color='#17becf', lw=8)
			dat = df2.groupby(['cost','pid'])[params[p]].mean()
			y = [dat.loc[i].mean() for i in dat.index.levels[0]]
			sem = [dat.loc[i].sem() for i in dat.index.levels[0]]
			plt.errorbar(range(len(y)), y, yerr=sem, color='#1f77b4', lw=8)
			plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=fontsize_ticks)
			plt.ylim(ylims[p])
			plt.grid(True)
			if (p_+1)==len(idx):
				plt.xlabel('Cost [$\lambda$]', fontsize=fontsize_labels)
				plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
			else:
				plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)

			# if big:
			ax = plt.subplot(gs[3*p_+2])
			box = ax.get_position()
			ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			if p_==0:
				if params[p]=='payoff_relative':
					ax.legend(['Model','Model with\nimplicit cost','Participants with\nperfect use of info.','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
				else:
					ax.legend(['Model','Participants'], fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.5))

		if save:
			if not os.path.exists(fig_dir): os.makedirs(fig_dir)
			ttl = '__'.join(params[idx[0]:idx[-1]+1])+exclude_str+'.png'
			plt.savefig(fig_dir+ttl, bbox_inches='tight', pad_inches=0.05, facecolor='w')

		plt.show()

def strategyVsKmeans_confusion_matrix(model_file, human_file, fig_dir=None, save=False):
	from statsmodels.stats.inter_rater import cohens_kappa

	df1 = pd.read_csv(model_file)
	df2 = pd.read_csv(human_file)

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	strategies = [['TTB','WADD','SAT_TTB','TTB_SAT','Other'], # 4 vs. 5 clusters
				['TTB','WADD','SAT_TTB','TTB_SAT','Rand','Other']]
	labels = [['TTB','WADD','SAT-TTB','SAT-TTB+','other'],
			['TTB','WADD','SAT-TTB','SAT-TTB+','random','other']]

	for isHuman, df in enumerate([df1, df2]):
		strats = strategies[isHuman]
		strat_labels = labels[isHuman]

		tmp = pd.DataFrame(columns=['strategy','km_strategy'])
		if isHuman:
			tmp['strategy'] = df['strategy'].values
			tmp['km_strategy'] = df['km_strategy'].values
		else:
			trial_strategy = df['strategy'].apply(literal_eval)
			tmp['strategy'] = [s for i in range(len(df)) for s in trial_strategy[i]] # unpack samples from every trial type
			trial_km_strategy = df['km_strategy'].apply(literal_eval)
			tmp['km_strategy'] = [s for i in range(len(df)) for s in trial_km_strategy[i]]

		confusion_mat = np.zeros((len(strats),len(strats)))
		for i, s in enumerate(strats):
			idx = tmp['km_strategy'] == s
			for j, s_ in enumerate(strats):
				confusion_mat[i,j] = sum(tmp[idx]['strategy'] == s_)
		confusion_mat = confusion_mat[:-1,:]
		pct = confusion_mat/sum(sum(confusion_mat))*100

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
		cbar = plt.colorbar(fraction=0.04, pad=0.04)
		cbar.ax.tick_params(labelsize=20)

		if save:
			if not os.path.exists(fig_dir): os.makedirs(fig_dir)
			ttl_sfx = '' if isHuman else '_model'
			plt.savefig(fig_dir+'confusion_mat_kmeans-strategy'+ttl_sfx+'.png',bbox_inches='tight',pad_inches=0.05, facecolor='w')
			dirstr = (model_file.find('data/'), 'exp1' if 'exp1' in model_file else 'exp2')
			stats_dir = model_file[:dirstr[0]]+'stats/'+dirstr[1]+'/'
			res = cohens_kappa(confusion_mat[:,:confusion_mat.shape[0]])
			r1, r2, r3 = res['kappa'], res['kappa_low'], res['kappa_upp']
			with open(stats_dir+'confusion_mat_kmeans-strategy-kappa'+ttl_sfx+'.txt', 'w') as f:
				f.write(f'$\\kappa={r1:.{3}f}, 95\\% CI [{r2:.{3}f}, {r3:.{3}f}]$')
		plt.show()

def under_performance_pie(human_file1, human_file2, fig_dir=None, save=False):

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	fig = plt.figure(figsize=(32,24))

	strat_labels = ['SAT-TTB+','SAT-TTB','TTB','WADD','random','other']
	ax = fig.gca()

	for plot_ix, human_file in enumerate([human_file1, human_file2]):

		exp2_str = human_file[-8:-4] if human_file[-8:-4]!='ials' else ''
		if human_file == 'exclude':
			dat = eval(pd.read_csv(human_file1, usecols=['under_performance_exclude']).iloc[0][0])[0]
		else:
			dat = eval(pd.read_csv(human_file, usecols=['under_performance']).iloc[0][0])[0]

		# remove negative numbers to not be included in pie chart
		dat['imperfect_strat_exec_by_strat'] = [dat['imperfect_strat_exec_by_strat'][i] if dat['imperfect_strat_exec_by_strat'][i]>=0 else 0 for i in range(len(dat['imperfect_strat_exec_by_strat']))]
		dat['imperfect_strat_selec_by_strat'] = [dat['imperfect_strat_selec_by_strat'][i] if dat['imperfect_strat_selec_by_strat'][i]>=0 else 0 for i in range(len(dat['imperfect_strat_selec_by_strat']))]

		clrs = cm.get_cmap("Set2").colors[:4]
		clrs = clrs+clrs[:2]+(clrs[2],)*6+(clrs[3],)*6
		x1, x2 = dat['imperfect_strat_selec_by_strat'], dat['imperfect_strat_exec_by_strat']
		labels = ['','','','','',''] + [f'{s}: {100*x1[i]:.1f}%' if x1[i]>=.0045 else '' for i,s in enumerate(['SAT-TTB+','SAT-TTB','TTB','WADD','random','other'])]\
									+ [f'{s}: {100*x2[i]:.1f}%' if x2[i]>=.0045 else '' for i,s in enumerate(['SAT-TTB+','SAT-TTB','TTB','WADD','random','other'])]
		perf = np.hstack([0,0,0,0,dat['implicit_costs'], dat['imperfect_info_use'], dat['imperfect_strat_selec_by_strat'], dat['imperfect_strat_exec_by_strat']])
		perf = np.hstack([perf,1-sum(perf)]); clrs+=((1,1,.85),); labels+=['']

		ax_center = [-1.2, 0] if plot_ix==0 else [1.2, 0]
		wedges, texts = plt.pie(perf, center=ax_center, colors=clrs, labels=labels, startangle=90, counterclock=False, rotatelabels=False, \
						textprops={'fontsize':fontsize_ticks,'va':'center','ha':'center','linespacing':.8},\
						wedgeprops={"edgecolor":[1,1,1],'linewidth':3},labeldistance=1) # autopct=my_autopct, 
		groups = [[0,1,2,3,4],[5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18]]
		radfraction = 0.5
		perf = [dat['implicit_costs'], dat['imperfect_info_use'], dat['imperfect_strat_selec'], dat['imperfect_strat_exec']]
		perf = np.hstack([perf, 1-sum(perf)])
		for i, group in enumerate(groups):
			ang = np.deg2rad((wedges[group[0]].theta2 + wedges[group[-1]].theta1) / 2)
			radfraction = .6 if human_file2=='exclude' and i==0 else .5
			center = np.array(wedges[group[0]].center) + radfraction * np.array([np.cos(ang), np.sin(ang)])
			for j in group:
				if human_file2!='exclude':
					pos = np.array(texts[j].get_position()) 
					if plot_ix==0 and j==11:
						texts[j].set_position((pos[0], pos[1]+.035))
					if plot_ix==0 and j==13:
						texts[j].set_position((pos[0], pos[1]+.01))
					elif plot_ix==1 and j==11:
						texts[j].set_position((pos[0]+.14, pos[1]+.06))
					elif plot_ix==1 and j==13: 
						texts[j].set_position((pos[0]-.15, pos[1]+.06))
			if i<4:
				ax.text(center[0],center[1], f'{100*perf[i]:.1f}%',ha="center", va="center", color="k",fontsize=fontsize_labels,fontweight='bold')
			else:
				ax.text(center[0],center[1], f'{100*perf[i]:.1f}%\nParticipant performance',ha="center", va="center", color="k",fontsize=38)
	
		if plot_ix==0:
			ttl = 'Experimental group' if exp2_str=='_exp' else 'All participants'
		else:
			ttl = 'Control group' if exp2_str=='_con' else 'Participants excluded'
		plt.text(1.05*ax_center[0], 1.05, ttl, fontsize=fontsize_labels, va='center',ha='center')
	
	plt.text(0, 1.45, 'Sources of Participant Under-Performance\n[% of Model Net Relative Performance]', fontsize=fontsize_labels, fontweight='bold', va='center',ha='center')
	plt.legend(['Implicit costs of information gathering','Imperfect information use',\
				'Imperfect strategy selection','Imperfect strategy execution',],\
				fontsize=fontsize_legend, bbox_to_anchor=(0.02,1.05), loc='upper center')

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'performance_sources.png', bbox_inches='tight', pad_inches=0.05, facecolor='w')
	plt.show()

def under_performance_byStrat(human_file1, human_file2, fig_dir=None, save=False):

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	fig = plt.figure(figsize=(32,20))

	strat_labels = ['SAT-TTB+','SAT-TTB','TTB','WADD','random','other']

	for plot_ix, human_file in enumerate([human_file1, human_file2]):

		ax = fig.add_subplot(1,2,plot_ix+1)

		exp2_str = human_file[-8:-4] if human_file[-8:-4]!='ials' else ''
		if human_file == 'exclude':
			dat = eval(pd.read_csv(human_file1, usecols=['under_performance_exclude']).iloc[0][0])[0]
		else:
			dat = eval(pd.read_csv(human_file, usecols=['under_performance']).iloc[0][0])[0]
		
		mat = dat['trial_counts']
		plt.imshow(mat, cmap='viridis') # cividis, plasma, viridis
		ax = plt.gca()
		text0 = np.round(100*np.array(dat['imperfect_strat_selec_and_exec_by_strat']), decimals=1); text0[text0==0]=0
		for i in range(np.shape(mat)[0]):
			for j in range(np.shape(mat)[1]):
				fmt = '{:.1f}' if text0[i,j]!=0 else '{:.0f}'
				ax.text(j, i, fmt.format(text0[i,j])+'%',ha="center", va="center", color="w",fontsize=fontsize_ticks,fontweight='bold')
		plt.xticks(range(len(mat)),labels=strat_labels[:len(mat)],rotation=90,fontsize=fontsize_ticks)
		plt.yticks(range(len(mat)),labels=strat_labels[:len(mat)],fontsize=fontsize_ticks)
		plt.xlabel('Participant strategy', fontsize=fontsize_labels)
		if plot_ix==0:
			plt.ylabel('Model strategy', fontsize=fontsize_labels)
		if plot_ix==0:
			ttl = 'Experimental group' if exp2_str=='_exp' else 'All participants'
		else:
			ttl = 'Control group' if exp2_str=='_con' else 'Participants excluded'
		plt.title(ttl, fontsize=fontsize_labels)
		cbar = plt.colorbar(fraction=.045)
		cbar.ax.tick_params(labelsize=20)
		if plot_ix==1:
			cbar.set_label('Number of trials\n', rotation=270, fontsize=30, labelpad=25)
		
	plt.text(-2.1,-1.25, 'Sources of Imperfect Strategy Selection and Execution\n[% of Model Net Relative Performance]', fontsize=fontsize_labels, va='center',ha='center', fontweight='bold')
	plt.subplots_adjust(wspace=0.38)

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'performance_strategy_sources.png',bbox_inches='tight',pad_inches=0.05, facecolor='w')
	plt.show()

def centroids_exp2(model_file, human_file1, human_file2, fig_dir=None, save=False):
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	df1 = pd.read_csv(model_file)
	df2 = pd.read_csv(human_file1)
	df3 = pd.read_csv(human_file2)

	centers1 = eval(df1['cluster_centers'].iloc[0])
	centers2 = eval(df2['cluster_centers'].iloc[0])
	centers3 = eval(df3['cluster_centers'].iloc[0])

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	# least to most clicks
	centroid_orders = [np.argsort(np.sum(centers1, axis=1)),\
					np.argsort(np.sum(centers2, axis=1)),\
					np.argsort(np.sum(centers3, axis=1))]

	fig = plt.figure(figsize=(32, 15))
	plt.figtext(0.5, 0.94, 'Model', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	plt.figtext(0.5, 0.63, 'Experimental Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	plt.figtext(0.5, 0.362, 'Control Participants', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')
	gs = gridspec.GridSpec(3, 5)

	sp = [centroid_orders[0][i] for i in [1,3,0,2]] # TTB, WADD, SAT-TTB, SAT-TTB+
	for i in range(np.shape(centers1)[0]):
		plt.sca(plt.subplot(gs[i]))
		plt.title('Centroid '+str(i+1),fontsize=fontsize_labels)
		plt.imshow(np.reshape(centers1[sp[i]],(4,6)), vmin=0, vmax=1)
		if (i==0):
			# plt.ylabel('Prob. of\noutcome',fontsize=fontsize_labels)
			plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
			plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
		else:
			plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
	plt.sca(plt.subplot(gs[4]))
	plt.axis('off')

	divider = make_axes_locatable(plt.gca())
	cax = divider.append_axes("right", size="20%", pad=0)
	cbar = plt.colorbar(cax=cax)
	cbar.ax.tick_params(labelsize=20)
	plt.figtext(0.86, 0.77, 'Centroid values\n(Prob. of click)', ha='center', va='center', fontsize=fontsize_legend, rotation='vertical') #0.71

	sp = [centroid_orders[1][i] for i in [1,3,0,2]] # TTB, WADD, SAT-TTB, SAT-TTB+
	for i in range(np.shape(centers2)[0]):
		plt.sca(plt.subplot(gs[i+5]))
		plt.imshow(np.reshape(centers2[sp[i]],(4,6)), vmin=0, vmax=1)
		if (i==0):
			plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
			plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
		else:
			plt.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 
	plt.sca(plt.subplot(gs[4+5]))
	plt.axis('off')

	sp = [centroid_orders[2][i] for i in [2,4,1,3,0]] # TTB, WADD, SAT-TTB, SAT-TTB+, random
	for i in range(np.shape(centers3)[0]):
		plt.sca(plt.subplot(gs[i+5+5]))
		plt.imshow(np.reshape(centers3[sp[i]],(4,6)), vmin=0, vmax=1)
		if (i==0):
			plt.yticks([0,1,2,3], ['higher  ','','','  lower'], fontsize=fontsize_ticks, rotation='vertical', va='center')
		else:
			plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
		if i==0:
			plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
		else:
			plt.xticks([0,1,2,3,4,5], ['  higher','','','','','lower  '], fontsize=fontsize_ticks)
		if (i==4):
			plt.title('Centroid 5',fontsize=fontsize_labels)

	fig.text(0.5, 0.06, 'Total prob. of observed outcomes', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(0.087, 0.5, 'Prob. of outcome', fontsize=fontsize_labels, ha='center', va='center', rotation='vertical')

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'centroids.png', bbox_inches='tight', pad_inches=0.05, facecolor='w')
	
	plt.show()

def strategies_exp2(model_file, human_file1, human_file2, fig_dir=None, save=False):

	df1 = pd.read_csv(model_file) # model
	df2 = pd.read_csv(human_file1) # participants in experimental group
	df3 = pd.read_csv(human_file2) # participants in control group

	strategies = ['TTB_SAT','SAT_TTB','TTB','WADD']
	legend_labels = ['SAT-TTB+','SAT-TTB','TTB','WADD']
	plot_all6 = False
	if plot_all6:
		strategies = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']
		legend_labels = ['SAT-TTB+','SAT-TTB','TTB','WADD','random','other']

	for s in strategies:
		df1[s] = df1[s] * df1['trial_weight']

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	fig = plt.figure(figsize=(22, 18))

	outer = gridspec.GridSpec(1, 2, width_ratios = [1, .2]) 
	gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.25])
	plt.subplots_adjust(wspace=0.075, hspace=0.075)

	width = 0.7
	labels = ['Model','Exp.','Control']
	for i, alpha in enumerate(np.flip(sorted(df1['alpha'].unique()))): # flip for inverse alpha
		for j, cost in enumerate(sorted(df1['cost'].unique())):
			plt.sca(plt.subplot(gs[i+2*j]))
			plt.grid(axis='y')
			plt.yticks(fontsize=fontsize_ticks)
			plt.xlim([-.5,2.5])
			plt.ylim((0,1.02))
			for x, df in enumerate([df1, df2, df3]):
				plt.gca().set_prop_cycle(None)
				dat = [df[np.isclose(df['alpha'],alpha,atol=0.1) & (df['cost']==cost)].mean()[s] for s in strategies]
				sem = [df[np.isclose(df['alpha'],alpha,atol=0.1) & (df['cost']==cost)].groupby('pid').mean()[s].sem() for s in strategies] if x > 0 else [0]*len(strategies)
				for s_, s in enumerate(strategies):
					p = plt.bar(x, dat[s_]/sum(dat), width, bottom=sum(dat[:s_])/sum(dat), yerr=sem[s_])
			if i+2*j == 1 or i+2*j == 3:
				ax = plt.subplot(gs[i+2*j])
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
				if i+2*j == 1:
					ax.legend(legend_labels, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1.16, 0.4))

	plt.sca(plt.subplot(gs[0]))
	plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
	plt.yticks(fontsize=fontsize_ticks)
	plt.sca(plt.subplot(gs[1]))
	plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
	plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
	plt.sca(plt.subplot(gs[2]))
	plt.xticks([0,1,2],labels,rotation=90,fontsize=fontsize_ticks)
	plt.yticks(fontsize=fontsize_ticks)
	plt.sca(plt.subplot(gs[3]))
	plt.xticks([0,1,2],labels,rotation=90,fontsize=fontsize_ticks)
	plt.tick_params(axis='y',which='both',left=False,labelleft=False) 

	b1 = plt.subplot(gs[1]).get_position().bounds
	b2 = plt.subplot(gs[2]).get_position().bounds
	b3 = plt.subplot(gs[3]).get_position().bounds
	midx = (b3[0]+b3[2]-b2[0])/2
	midy = (b1[1]+b1[3]-b2[1])/2
	fig.text(b2[0]+midx, 0.01, 'Group', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(b2[0]-0.06, b2[1]+midy, 'Strategy Frequency', fontsize=fontsize_labels, ha='center', va='center', rotation='vertical', fontweight='bold')
	fig.text(b2[0]+midx, 0.94, 'Dispersion', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(b2[0]+midx-midx/2, 0.905, '['+r'$\alpha^{-1}=10^{-0.5}$]', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(b2[0]+midx+midx/2, 0.905, '['+r'$\alpha^{-1}=10^{0.5}$]', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(b1[0]+b1[2]+0.05, b2[1]+midy, 'Cost', fontsize=fontsize_labels, ha='center', va='center', rotation=270)
	fig.text(b1[0]+b1[2]+0.02, b2[1]+midy+midy/2, '[$\lambda=1$]', fontsize=fontsize_labels, ha='center', va='center', rotation=270)
	fig.text(b1[0]+b1[2]+0.02, b2[1]+midy-midy/2, '[$\lambda=4$]', fontsize=fontsize_labels, ha='center', va='center', rotation=270)

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		ttl_str = '' if not plot_all6 else '_all6'
		plt.savefig(fig_dir+'strategies'+ttl_str+'.png',bbox_inches='tight',pad_inches=0.05, facecolor='w')

	plt.show()

def condition_bars_exp2(model_file, human_file1, human_file2, fig_dir=None, save=False):

	df1 = pd.read_csv(model_file) # model
	df2 = pd.read_csv(human_file1) # participants in experimental group
	df3 = pd.read_csv(human_file2) # participants in control group

	params = ['nr_clicks','payoff_gross_relative','processing_pattern','click_var_outcome','click_var_gamble','nr_clicks','payoff_net_relative']
	labels = ['Information Gathered','Relative Performance','Alternative vs. Attribute','Attribute Variance','Alternative Variance','Information Gathered','Net Relative Performance']
	ylims = [(0,11),(0,1.03),(-1,0),(0,.2),(0,.06),(0,11),(-0.35,0.85)]
	idxs = [[0,1],[2,3,4],[5,6]]

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	colors = ['teal','teal','teal']#['mediumturquoise','mediumaquamarine','teal']

	width = 0.7 # bar width
	xtick_labels = ['Model','Exp.','Control']

	height1 = 16; height2 = 36
	for idx in idxs:
		if len(idx) < 3:
			fig = plt.figure(figsize=(32, height1))
			h1h2 = 1
			outer = gridspec.GridSpec(1, 2)
		else:
			fig = plt.figure(figsize=(32, height2))
			h1h2 = height1/height2
			outer = gridspec.GridSpec(2, 2)
		plt.subplots_adjust(wspace=0.45, hspace=0.38)
		
		b0 = []
		for p_, p in enumerate([params[i] for i in idx]):
			subplot_idx = p_+1 if len(idx)==3 and p_>0 else p_
			bbox = plt.subplot(outer[subplot_idx]).get_position().bounds
			b0.append(bbox)

		for p_, p in enumerate([params[i] for i in idx]):
			df1[p] = df1[p] * df1['trial_weight']

			subplot_idx = p_+1 if len(idx)==3 and p_>0 else p_
			gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outer[subplot_idx], wspace=0.1, hspace=0.07)
			for i, alpha in enumerate(np.flip(sorted(df1['alpha'].unique()))): # flip for inverse alpha
				for j, cost in enumerate(sorted(df1['cost'].unique())):
					plt.sca(plt.subplot(gs[i+2*j]))
					plt.grid(axis='y')
					plt.yticks(fontsize=fontsize_ticks)
					plt.xlim([-.5,2.5])
					plt.ylim(ylims[idx[p_]])
					for x, df in enumerate([df1, df2, df3]):
						dat = df[np.isclose(df['alpha'],alpha,atol=0.1) & (df['cost']==cost)].mean()[p]
						sem = df[np.isclose(df['alpha'],alpha,atol=0.1) & (df['cost']==cost)].groupby('pid').mean()[p].sem() if x > 0 else 0
						plt.bar(x, dat, width, color=colors[x], yerr=sem)

			plt.sca(plt.subplot(gs[0]))
			plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
			plt.yticks(fontsize=fontsize_ticks)
			plt.sca(plt.subplot(gs[1]))
			plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 
			plt.tick_params(axis='y',which='both',left=False,labelleft=False) 
			plt.sca(plt.subplot(gs[2]))
			plt.xticks([0,1,2],xtick_labels,rotation=90,fontsize=fontsize_ticks)
			plt.yticks(fontsize=fontsize_ticks)
			plt.sca(plt.subplot(gs[3]))
			plt.xticks([0,1,2],xtick_labels,rotation=90,fontsize=fontsize_ticks)
			plt.tick_params(axis='y',which='both',left=False,labelleft=False) 

			if p in ['click_var_outcome','click_var_gamble','processing_pattern']:
				dx = -0.01
			else:
				dx = 0
			bbt = plt.subplot(gs[0]).get_position().bounds
			bbb = plt.subplot(gs[2]).get_position().bounds
			fig.text(b0[p_][0]+b0[p_][2]/2, bbb[1]-.13*h1h2, 'Group', fontsize=fontsize_labels, ha='center', va='center')
			fig.text(b0[p_][0]+b0[p_][2]/2, bbt[1]+bbt[3]+.074*h1h2, 'Dispersion', fontsize=fontsize_labels, ha='center', va='center')
			fig.text(b0[p_][0]+b0[p_][2]/4, bbt[1]+bbt[3]+.028*h1h2, '['+r'$\alpha^{-1}=10^{-0.5}$]', fontsize=fontsize_labels, ha='center', va='center')
			fig.text(b0[p_][0]+b0[p_][2]*3/4, bbt[1]+bbt[3]+.028*h1h2, '['+r'$\alpha^{-1}=10^{0.5}$]', fontsize=fontsize_labels, ha='center', va='center')
			fig.text(b0[p_][0]-0.04+dx, b0[p_][1]+b0[p_][3]/2, labels[idx[p_]], fontsize=fontsize_labels, ha='center', va='center', rotation='vertical', fontweight='bold')
			fig.text(b0[p_][0]+b0[p_][2]+0.035, b0[p_][1]+b0[p_][3]/2, 'Cost', fontsize=fontsize_labels, ha='center', va='center', rotation=270)
			fig.text(b0[p_][0]+b0[p_][2]+0.015, b0[p_][1]+b0[p_][3]*3/4, '[$\lambda=1$]', fontsize=fontsize_labels, ha='center', va='center', rotation=270)
			fig.text(b0[p_][0]+b0[p_][2]+0.015, b0[p_][1]+b0[p_][3]/4, '[$\lambda=4$]', fontsize=fontsize_labels, ha='center', va='center', rotation=270)
			subplot_labs = ['A','B','C','D']
			fig.text(b0[p_][0]-.035+dx, bbt[1]+bbt[3]+.065*h1h2, subplot_labs[p_], fontsize=70, ha='center', va='center')

		if save:
			if not os.path.exists(fig_dir): os.makedirs(fig_dir)
			ttl = fig_dir+'__'.join([params[i] for i in idx])+'.png'
			plt.savefig(ttl, bbox_inches='tight', pad_inches=0.05, facecolor='w')
		plt.show()

def clicks_dispersion_cost_exp2(fig_dir=None, save=False):

	df_mod2 = pd.read_csv('../data/model/exp2/processed/trials.csv')
	df_exp = pd.read_csv('../data/human/2.3/processed/trials_exp.csv')
	df_con = pd.read_csv('../data/human/2.3/processed/trials_con.csv')
	df_mod1 = pd.read_csv('../data/model/exp1/processed/trials.csv')
	df_h1 = pd.read_csv('../data/human/1.0/processed/trials.csv')

	df_mod2['nr_clicks'] = df_mod2['nr_clicks'] * df_mod2['trial_weight']
	df_mod1['nr_clicks'] = df_mod1['nr_clicks'] * df_mod1['trial_weight_75']
	df_mod1 = df_mod1[df_mod1['sigma']==75]
	df_h1 = df_h1[df_h1['sigma']==75]


	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36
	markersize=24; linewidth=8; klinewidth=2

	fig = plt.figure(figsize=(32, 16))

	ax = fig.add_subplot(1, 2, 1); b0 = ax.get_position().bounds
	ax.set_position([b0[0],b0[1]+.02,.8*b0[2],b0[3]])
	ax1 = ax
	b0 = ax.get_position().bounds
	plt.plot([-1,10], [6,6], 'k--', linewidth=klinewidth, label='_nolegend_')
	cmap = Colors.LinearSegmentedColormap.from_list('testCmap', colors=['blue','red'], N=5)
	c1 = cmap(1); c2 = cmap(3)
	dat = np.flip(np.reshape(df_mod2.groupby(['cost','alpha']).mean()['nr_clicks'].values, (2,2)))
	plt.plot([0,1], dat[1], marker='o', markersize=markersize, linewidth=linewidth, color=c2)
	plt.plot([0,1], dat[0], marker='o', markersize=markersize, linewidth=linewidth, color=c1)
	dat = np.flip(np.reshape(df_exp.groupby(['cost','alpha']).mean()['nr_clicks'].values, (2,2)))
	sem = np.array([df_exp[(df_exp['cost']==1) & (df_exp['alpha']==a)].groupby('pid').mean()['nr_clicks'].sem() for a in np.flip(sorted(df_exp['alpha'].unique()))])
	plt.errorbar([3,4], dat[0], marker='o', markersize=markersize, linewidth=linewidth, color=c1, yerr=sem, label='_nolegend_')
	sem = np.array([df_exp[(df_exp['cost']==4) & (df_exp['alpha']==a)].groupby('pid').mean()['nr_clicks'].sem() for a in np.flip(sorted(df_exp['alpha'].unique()))])
	plt.errorbar([3,4], dat[1], marker='o', markersize=markersize, linewidth=linewidth, color=c2, yerr=sem, label='_nolegend_')
	dat = np.flip(np.reshape(df_con.groupby(['cost','alpha']).mean()['nr_clicks'].values, (2,2)))
	sem = np.array([df_con[(df_con['cost']==1) & (df_con['alpha']==a)].groupby('pid').mean()['nr_clicks'].sem() for a in np.flip(sorted(df_con['alpha'].unique()))])
	plt.errorbar([6,7], dat[0], marker='o', markersize=markersize, linewidth=linewidth, color=c1, yerr=sem, label='_nolegend_')
	sem = np.array([df_con[(df_con['cost']==4) & (df_con['alpha']==a)].groupby('pid').mean()['nr_clicks'].sem() for a in np.flip(sorted(df_con['alpha'].unique()))])
	plt.errorbar([6,7], dat[1], marker='o', markersize=markersize, linewidth=linewidth, color=c2, yerr=sem, label='_nolegend_')
	
	plt.xticks([0,1,3,4,6,7], [r'$10^{-0.5}$',r'$10^{0.5}$',r'$10^{-0.5}$',r'$10^{0.5}$',r'$10^{-0.5}$',r'$10^{0.5}$'], fontsize=fontsize_ticks-2)
	plt.yticks(fontsize=fontsize_ticks)
	plt.xlim(-.5,7.5)
	plt.ylim(0,11)
	plt.legend(['Cost=1', 'Cost=4'], fontsize=fontsize_legend)

	ax = fig.add_subplot(2, 2, 2)
	ax2 = ax
	b1 = ax.get_position().bounds
	ax.set_position([b0[0]+b0[2]+.08,b0[1],10/8*b0[2],b0[3]])
	b1 = ax.get_position().bounds
	plt.plot([-1,10], [6,6], 'k--', linewidth=klinewidth, label='_nolegend_')
	dat1 = np.flip(np.reshape(df_mod1.groupby(['cost','alpha']).mean()['nr_clicks'].values, (5,5)))
	dat2 = np.flip(np.reshape(df_h1.groupby(['cost','alpha']).mean()['nr_clicks'].values, (5,5)))
	for i, c in enumerate([8,4,2,1,0]):
		plt.plot(np.arange(5), dat1[i], marker='o', markersize=markersize, linewidth=linewidth, color=cmap(i), label='_nolegend_')
		sem = np.array([df_h1[(df_h1['cost']==c) & (np.isclose(df_h1['alpha'],a,atol=0.1))].groupby('pid').mean()['nr_clicks'].sem() for a in np.flip(sorted(df_h1['alpha'].unique()))])
		plt.errorbar(np.arange(5)+5, dat2[i], marker='o', markersize=markersize, linewidth=linewidth, yerr=sem, color=cmap(i), label='_nolegend_')
		plt.plot([100, 100], [100, 100], marker='o', markersize=markersize, linewidth=linewidth, color=cmap(4-i))
	plt.xticks(np.arange(10), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$', r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks-2)
	plt.yticks(np.arange(0,22,4), fontsize=fontsize_ticks)
	plt.xlim(-.5,9.5)
	plt.ylim(0,16)
	plt.legend(['Cost=0','Cost=1','Cost=2','Cost=4','Cost=8'], fontsize=fontsize_legend)
	# pdb.set_trace()
	ax = fig.gca()
	fig.text(b0[0]+b0[2]/8, 0.09, 'Model', fontsize=fontsize_labels-4, ha='center', va='center')
	fig.text(b0[0]+b0[2]/2, 0.09, 'Exp.', fontsize=fontsize_labels-4, ha='center', va='center')
	fig.text(b0[0]+b0[2]*7/8, 0.09, 'Control', fontsize=fontsize_labels-4, ha='center', va='center')
	fig.text(b0[0]+b0[2]/2, 0.05, 'Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(b1[0]+b1[2]/4, 0.09, 'Model', fontsize=fontsize_labels-4, ha='center', va='center')
	fig.text(b1[0]+b1[2]*3/4, 0.09, 'Participants', fontsize=fontsize_labels-4, ha='center', va='center')
	fig.text(b1[0]+b1[2]/2, 0.05, 'Dispersion ['+r'$\alpha^{-1}$]', fontsize=fontsize_labels, ha='center', va='center')
	plt.sca(ax1)
	plt.ylabel('Information Gathered', fontsize=fontsize_labels)
	plt.title('Experiment 2', fontsize=fontsize_labels, fontweight='bold')
	plt.sca(ax2)
	plt.ylabel('Information Gathered', fontsize=fontsize_labels)
	plt.title('Experiment 1', fontsize=fontsize_labels, fontweight='bold')
	fig.text(b0[0]-.03,b0[1]+b0[3]+.02, 'A', fontsize=70, ha='center', va='center')
	fig.text(b1[0]-.03,b0[1]+b0[3]+.02, 'B', fontsize=70, ha='center', va='center')

	# plt.sca(fig.gca())
	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'clicks_dispersion_cost.png',bbox_inches='tight',pad_inches=0.05, facecolor='w')
	plt.show()

def clicks_dispersion_cost_3d_exp2(fig_dir=None, save=False):

	df_mod1 = pd.read_csv('../data/model/exp1/processed/trials.csv')
	df_h1 = pd.read_csv('../data/human/1.0/processed/trials.csv')

	df_mod1['nr_clicks'] = df_mod1['nr_clicks'] * df_mod1['trial_weight_75']

	fontsize_ticks = 15
	fontsize_labels = 24
	fontsize_legend = 15
	markersize=12; linewidth=4; klinewidth=1

	fig = plt.figure(figsize=(16, 8))

	def make_trisurf(z, ticks):
		ax = fig.gca(projection='3d')
		cmap = Colors.LinearSegmentedColormap.from_list('testCmap', colors=['blue','red'], N=256)
		surf = ax.plot_trisurf(x, y, z, cmap=cmap, norm=Colors.LogNorm())#divnorm)
		cbar = fig.colorbar( surf, ticks=ticks, shrink=.6, aspect=10)#, labelsize=18)
		cbar.ax.set_yticklabels([str(i) for i in ticks]) 
		cbar.ax.tick_params(labelsize=fontsize_ticks) 
		plt.xlabel('\n\nDispersion\n['+r'$\alpha^{-1}$]', fontsize=fontsize_labels)
		plt.xticks(np.arange(5), [r'$10^{-1.0}$',r'$10^{-0.5}$',r'$10^{0.0}$',r'$10^{0.5}$',r'$10^{1.0}$'], fontsize=fontsize_ticks)
		plt.yticks(np.arange(5), ['0','1','2','4','8'], fontsize=fontsize_ticks)
		plt.ylabel('\n\nCost\n[$\lambda$]', fontsize=fontsize_labels)
		ax.set_zticks([0,5,10,15])
		ax.set_zticklabels([0,5,10,15], fontsize=fontsize_ticks)
		ax.view_init(30, 60)
	dat1 = np.rot90(np.reshape(df_mod1[df_mod1['sigma']==75].groupby(['cost','alpha']).mean()['nr_clicks'].values, (5,5)))
	dat2 = np.rot90(np.reshape(df_h1[df_h1['sigma']==75].groupby(['cost','alpha']).mean()['nr_clicks'].values, (5,5)))
	x = np.repeat(np.arange(5), 5)
	y = np.tile(np.arange(5), 5)
	z = np.ndarray.flatten(dat1)
	ax = fig.add_subplot(1, 2, 1, projection='3d'); ax1 = ax
	b2 = ax.get_position().bounds
	make_trisurf(z, [3,4,6,10,15])
	z = np.ndarray.flatten(dat2)
	ax = fig.add_subplot(1, 2, 2, projection='3d'); ax2 = ax
	b3 = ax.get_position().bounds
	make_trisurf(z, [2,3,4,6,9])
	
	ax = fig.gca()
	fig.text(.1,.5, 'Information Gathered', fontsize=fontsize_labels, ha='center', va='center', rotation=90)
	fig.text(.525,.5, 'Information Gathered', fontsize=fontsize_labels, ha='center', va='center', rotation=90)
	fig.text(b2[0]+b2[2]/2,b2[1]+b2[3]-.07, 'Experiment 1\nModel', fontsize=fontsize_labels, ha='center', va='center', fontweight='bold')
	fig.text(b3[0]+b3[2]/2,b3[1]+b3[3]-.07, 'Experiment 1\nParticipants', fontsize=fontsize_labels, ha='center', va='center', fontweight='bold')
	plt.sca(fig.gca())

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'clicks_dispersion_cost_3d.pdf',bbox_inches='tight',pad_inches=0.05, facecolor='w')

	plt.show()

def centroids_1_k(in_file, fig_dir=None, save=False, max_k=12):
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	df1 = pd.read_csv(in_file)

	fontsize_ticks = 32
	fontsize_labels = 42
	fontsize_legend = 36

	fig = plt.figure(figsize=(32, 30))
	ttl_str = 'Model' if 'model' in in_file else 'Participant'
	plt.figtext(0.5, 0.9, ttl_str+' '+r'$k$'+'-means Centroids', ha='center', va='center', fontsize=fontsize_labels, fontweight='bold')

	gs = gridspec.GridSpec(max_k, max_k)
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)

	for k in range(1,max_k+1):
		centers = eval(df1['cluster_centers_k'+str(k)].iloc[0])
		pct_labels = eval(df1['cluster_points_k'+str(k)].iloc[0])

		# least to most clicks
		centroid_orders = np.argsort(np.sum(centers, axis=1))

		sp = centroid_orders
		for i in range(np.shape(centers)[0]):
			plt.sca(plt.subplot(gs[(k-1)*(max_k)+i]))

			if k==1:
				plt.imshow(np.reshape(centers,(4,6)), vmin=0, vmax=1)
			else:		
				plt.imshow(np.reshape(centers[sp[i]],(4,6)), vmin=0, vmax=1)
			pct_points = np.round(100*pct_labels[sp[i]], decimals=1)
			plt.title(f'{pct_points}%', fontsize=fontsize_ticks)

			if i==0:
				plt.yticks([0,1,2,3], ['high','','','low'], fontsize=fontsize_ticks, rotation='vertical', va='center')
			else:
				plt.tick_params(axis='y',which='both',left=False,labelleft=False,bottom=False,labelbottom=False) 

			if k==max_k:
				plt.xticks([0,1,2,3,4,5], ['','high ','','',' low',''], fontsize=fontsize_ticks)
			else:
				plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False) 

		plt.subplot(gs[(k-1)*(max_k)+i]).annotate(\
			r'$k='+str(k)+'$', xy=(1, 0.5), xycoords='axes fraction', rotation=270, fontsize=fontsize_ticks, ha='left', va='center')

	fig.text(0.5, 0.07, 'Total prob. of observed outcomes', fontsize=fontsize_labels, ha='center', va='center')
	fig.text(0.07, 0.5, 'Prob. of outcome', fontsize=fontsize_labels, ha='center', va='center', rotation='vertical')

	plt.sca(plt.subplot(gs[round(max_k*1.5)]))
	plt.axis('off')
	divider = make_axes_locatable(plt.gca())
	cax = divider.append_axes("right", size="20%", pad=0)
	cbar = plt.colorbar(cax=cax)
	cbar.ax.tick_params(labelsize=20)
	plt.figtext(0.54, 0.8, 'Centroid values\n(Prob. of click)', ha='center', va='center', fontsize=fontsize_ticks, rotation='vertical')

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		ttl_str = '_model' if 'model' in in_file else ''
		plt.savefig(fig_dir+'centroids_1_'+str(max_k)+ttl_str+'.png', bbox_inches='tight', pad_inches=0.05, facecolor='w')
	
	plt.show()



def lda(model_file, human_file, fig_dir=None, save=False):
	# add click embeddings and labels to pkl file in p_d.append_kmeans
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

	df = pd.read_csv(human_file)

	fontsize_labels = 42
	fontsize_legend = 36

	pickle_dict = pd.read_pickle(os.path.splitext(human_file)[0]+'_click_embeddings.pkl')
	X = [pickle_dict['click_embedding'][i] for i in range(len(pickle_dict['click_embedding']))]
	y = pickle_dict['labels']
	centroids = pickle_dict['cluster_centers']
	centroids = centroids >= 0.5



	# if isHuman:
	# 	if len(df2)==0:
	# 		df2 = pd.read_pickle('data/human/1.0/trials.pkl')
	# 	km = pickle.load(open('data/human/1.0/kmeans.pkl', 'rb'))
	# elif not(isHuman):
	# 	if len(df2)==0:
	# 		df2 = pd.read_pickle('data/model/no_implicit_cost/trials_model.pkl')
	# 	df2 = df2.iloc[::10, :]
	# 	km = pickle.load(open('data/model/no_implicit_cost/kmeans_model.pkl', 'rb'))
	# if isHuman=='both':
	# 	km1 = pickle.load(open('data/human/1.0/kmeans.pkl', 'rb'))
	# 	km1 = km1['click_embedding_prob']['kmeans'][0].cluster_centers_
	# 	km2 = pickle.load(open('data/model/no_implicit_cost/kmeans_model.pkl', 'rb'))
	# 	km2 = km2['click_embedding_prob']['kmeans'][0].cluster_centers_
	# 	km = np.append(km1, km2, axis=0)
	# else:
	# 	km = km['click_embedding_prob']['kmeans'][0].cluster_centers_
	# km_bin = km >= 0.5
	isHuman = True
	colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', 'k', [.8,.8,.8]]#'#e3c9a4']
	# if isHuman:
	strategies_lab = ['TTB','WADD','SAT_TTB','TTB_SAT','Rand','Other']
	legend_lab_s = ['TTB','WADD','SAT-TTB','SAT-TTB+','random','other']
	order = [5,4,3,2,1,0]
		# order = [0,1,2,3,4,5]
	# 	legend_lab_k = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
	# else:
	# 	strategies_lab = ['TTB','WADD','SAT_TTB','TTB_SAT','Other']
	# 	legend_lab_s = ['TTB','WADD','SAT-TTB','SAT-TTB+','other']
	# 	order = [4,3,2,1,0]
	# 	# order = [0,1,2,3,4]
	# 	legend_lab_k = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4']
	# strategies_plt = [strategies_lab[i] for i in order]

	# y_km = np.ones(len(df2)); y_km[:] = np.nan
	# for i, s in enumerate(strategies_plt):
	# 	if (isHuman==True) | (isHuman=='both'):
	# 		y_km[df2.strategy_click_embedding_prob_k5 == s] = order[i]
	# 	else:
	# 		y_km[df2.strategy_click_embedding_prob_k4 == s] = order[i]

	# for ii, y in enumerate([y_km, y_strat]):
	mod = LDA(n_components=2).fit(X, y)

	X = mod.transform(X)
	centroids = mod.transform(centroids.astype(float))

	plt.figure(figsize=[9,9])

	# add jitter for visualization
	range_x = max(X[:,0]) - min(X[:,0]); yrange = max(X[:,1]) - min(X[:,1])
	varFact = 40
	jitter = np.stack((np.random.normal(loc=0.0, scale=range_x/varFact, size=len(X)), np.random.normal(loc=0.0, scale=yrange/varFact, size=len(X)))).T
	X += jitter

	for i, s in enumerate(np.sort(np.unique(y))):
		# if ii==0:
		# 	iii = i+1
		# else:
		# 	iii = i
		# idx = y == order[iii]
		idx = y == order[i]
		plt.scatter(X[idx, 0], X[idx, 1], s=50, alpha = .05, color=colors[order[i]], marker='.', label='_nolegend_')

	xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
	for j in range(len(np.unique(y))):
		plt.scatter(-100,-100, color=colors[j], s=500)
	plt.scatter(-100, 100, s=800, linewidth=4, alpha = 1, color='y', marker='+'); #legend_labels.append('Centroids')
	plt.xlim(xlim); plt.ylim(ylim);
	plt.axis('off')
	if isHuman:
		plt.title('Cluster visualization (Participants)', fontsize=fontsize_labels)
	else:
		plt.title('Cluster visualization (Model)', fontsize=fontsize_labels)
	# if ii==0:
	# 	ttl = title + 'kmeans'
	legend_lab_k.append('Centroids')
	# plt.legend(legend_lab_k, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
	plt.legend(legend_lab_s, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
	# else:
	# 	ttl = title + 'strategy'
	# 	legend_lab_s.append('Centroids')
	# 	plt.legend(legend_lab_s, fontsize=fontsize_legend, loc='lower left', bbox_to_anchor=(1, 0.2))
	plt.scatter(centroids[:,0], centroids[:,1], s=300, linewidth=3, alpha = 1, color='y', marker='+', label='_nolegend_')

	if save:
		if not os.path.exists(fig_dir): os.makedirs(fig_dir)
		plt.savefig(fig_dir+'lda.png', bbox_inches='tight', pad_inches=0.05, facecolor='w')

	plt.show()

	return
