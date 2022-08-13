import os
import json
import pickle
# import bz2
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing
import warnings
from ast import literal_eval
from sklearn.cluster import KMeans
import cfg
import pdb

def append_features(in_tuple):

	in_file, dataObj = in_tuple

	def get_click_mat(click_location_map, clicks):
		mat = np.zeros(np.shape(click_location_map))
		for c in clicks:
			mat += (c==click_location_map)
		return mat

	max_EV = pd.read_csv('../data/model/max_EV.csv')
	max_EV['alpha'] = np.round(max_EV['alpha'],decimals=1)

	if dataObj.isHuman:
		dat = pd.read_csv(dataObj.raw)
		dat = dat[dat['block']=='test']
		if dataObj.num==2 and dataObj.group=='con':
			dat = dat[dat['display_ev']==False]
		elif dataObj.num==2 and dataObj.group=='exp':
			dat = dat[dat['display_ev']==True]
		dat.loc[:,'alpha'] = dat['alpha'].round(decimals=1)
		dat = dat.to_dict('records')
		click_location_map = np.arange(24).reshape((4,6))
		same_col_idx = [np.arange(i,24,6) for i in range(6)]
		same_row_idx = [np.arange(i,i+6) for i in range(0,24,6)]
	else:
		dat = json.load(open(in_file))
		click_location_map = np.arange(24).reshape((6,4)).T
		same_col_idx = [np.arange(i,i+4) for i in range(0,24,4)]
		same_row_idx = [np.arange(i,24,4) for i in range(4)]
		cost, alpha, sigma = dat['cost'], np.round(dat['alpha'],decimals=1), dat['sigma']
		probabilities = eval(dat['probabilities'])
		payoff_matrix = eval(dat['payoff_matrix'])
		problem_id = dat['problem_id']
		payoff_perfect = float(max_EV.loc[(max_EV['sigma']==sigma) & (max_EV['alpha']==alpha), 'mean'])
		dat = dat['uncovered']

	out = defaultdict(list)

	use_EV_payoff = True
	for trial in dat:
		if dataObj.isHuman:
			cost = trial['cost']
			probabilities = eval(trial['probabilities'])
			payoff_matrix = eval(trial['payoff_matrix'])
			clicks = eval(trial['clicks'])
			choice = trial['choice_index']
			payoff_idx = trial['payoff_index']
			payoff_perfect = float(max_EV.loc[(max_EV['sigma']==trial['sigma']) & (max_EV['alpha']==trial['alpha']), 'mean'])
		else:
			clicks = [c-1 for c in trial]
			payoff_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)

		click_mat = get_click_mat(click_location_map, clicks)	
		nr_clicks = len(clicks)

		row_order = np.flip(np.argsort(probabilities))
		col_order = np.flip(np.argsort(np.matmul(probabilities,click_mat)))
		idx = click_location_map[row_order]
		click_mat_transformed = get_click_mat(idx[:,col_order], clicks)
		click_embedding = np.ndarray.flatten(click_mat_transformed)#.tolist()

		ttb_row = np.argmax(probabilities)
		payoffs = [p for pr in payoff_matrix for p in pr]
		if nr_clicks==0:
			strategy = 'Rand'
		elif nr_clicks==24:
			strategy = 'WADD'
		elif set(clicks)==set(click_location_map[ttb_row,:]):
			strategy = 'TTB'
		elif (set(clicks).issubset(set(click_location_map[ttb_row,:]))) \
		   & (np.argmax([payoffs[i] for i in clicks])==(nr_clicks-1)):
			strategy = 'SAT_TTB'
		elif (all(np.diff(np.sum(click_mat_transformed,axis=0))<=0)) \
		   & (all(np.diff(np.sum(click_mat_transformed,axis=1))<=0)):
			strategy = 'TTB_SAT'
		else:
			strategy = 'Other'

		EVs = np.matmul(probabilities, payoff_matrix*click_mat)
		EVs_perfect = np.matmul(probabilities, payoff_matrix)
		best_choice = np.argmax(EVs)
		if dataObj.isHuman:
			if use_EV_payoff:
				if strategy == 'Rand':
					# payoff_gross = 0 # for making trial-wise model comparisons, better to use actual mean than 0
					payoff_gross = np.mean(EVs_perfect)
					payoff_gross_relative_bestBet = payoff_gross / payoff_perfect
					payoff_net_relative_bestBet = payoff_gross / payoff_perfect
					payoff_gross_bestBet = payoff_gross
				else:
					payoff_gross = EVs_perfect[choice]
					payoff_gross_relative_bestBet = EVs_perfect[best_choice] / payoff_perfect
					payoff_net_relative_bestBet = (EVs_perfect[best_choice] - nr_clicks * cost) / payoff_perfect
					payoff_gross_bestBet = EVs_perfect[best_choice]
			else:
				payoff_gross = trial['payoff_value']
				payoff_gross_relative_bestBet = payoff_matrix[payoff_idx][best_choice] / payoff_perfect
				payoff_net_relative_bestBet = (payoff_matrix[payoff_idx][best_choice] - nr_clicks * cost) / payoff_perfect
				payoff_gross_bestBet = payoff_matrix[payoff_idx][best_choice]
			bad_choice = (strategy == 'Rand') or (choice != best_choice)
		else:
			if use_EV_payoff:
				payoff_gross = EVs_perfect[best_choice] if strategy!='Rand' else np.mean(EVs_perfect) # 0
			else:
				payoff_gross = payoff_matrix[payoff_idx][best_choice]
		payoff_net = payoff_gross - nr_clicks * cost
		payoff_gross_relative = payoff_gross / payoff_perfect
		payoff_net_relative = payoff_net / payoff_perfect

		type1_transitions = sum([1 for col in same_col_idx for c in range(nr_clicks-1) \
								 if (clicks[c] in col) and (clicks[c+1] in col)])
		type2_transitions = sum([1 for row in same_row_idx for c in range(nr_clicks-1) \
								 if (clicks[c] in row) and (clicks[c+1] in row)])
		processing_pattern = np.divide((type1_transitions-type2_transitions), (type1_transitions+type2_transitions))
		click_var_gamble = np.var(np.divide(np.sum(click_mat, axis=0), nr_clicks))
		click_var_outcome = np.var(np.divide(np.sum(click_mat, axis=1), nr_clicks))

		if dataObj.isHuman:
			out['problem_id'].append(trial['problem_id'])
			# out['trial'].append(trial['trial_index'])
			if 'display_ev' in trial:
				out['display_ev'].append(trial['display_ev'])
			out['pid'].append(trial['pid'])
			out['sigma'].append(trial['sigma'])
			out['alpha'].append(trial['alpha'])
			out['cost'].append(trial['cost'])
			# out['probabilities'].append(probabilities)
			# out['payoff_matrix'].append(trial['payoff_matrix'])
			# out['choice'].append(choice)
			out['best_choice'].append(choice == best_choice)
			out['bad_choice'].append(bad_choice)
			out['payoff_gross_relative_bestBet'].append(payoff_gross_relative_bestBet)
			out['payoff_net_relative_bestBet'].append(payoff_net_relative_bestBet)
			out['payoff_gross_bestBet'].append(payoff_gross_bestBet)
			out['payoff_net_bestBet'].append(payoff_gross_bestBet - nr_clicks * cost)
		out['nr_clicks'].append(nr_clicks)
		out['payoff_gross'].append(payoff_gross)
		out['payoff_net'].append(payoff_net)
		out['payoff_gross_relative'].append(payoff_gross_relative)
		out['payoff_net_relative'].append(payoff_net_relative)
		out['payoff_perfect'].append(payoff_perfect)
		out['processing_pattern'].append(processing_pattern)
		out['click_var_gamble'].append(click_var_gamble)
		out['click_var_outcome'].append(click_var_outcome)
		out['strategy'].append(strategy)
		for s in ['Rand','WADD','TTB','SAT_TTB','TTB_SAT','Other']:
			out[s].append(strategy==s)
		out['click_embedding'].append(click_embedding.astype(bool))
		strategy_ix = np.array([1 if strategy==s else 0 for s in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']])
		out['payoff_gross_relative_by_strategy'].append(payoff_gross_relative * strategy_ix)
		out['payoff_net_relative_by_strategy'].append(payoff_net_relative * strategy_ix)
		out['payoff_gross_by_strategy'].append(payoff_gross * strategy_ix)
		out['payoff_net_by_strategy'].append(payoff_net * strategy_ix)

	if not dataObj.isHuman:
		features_to_mean = [k for k in out.keys() if k not in ['click_embedding','strategy','payoff_net_relative_by_strategy',\
									'payoff_gross_relative_by_strategy','payoff_net_by_strategy','payoff_gross_by_strategy']]
		for k in features_to_mean:
			out[k] = np.nanmean(out[k])
		strategy_counts = np.array([sum([o==s for o in out['strategy']]) for s in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']])
		for i, s in enumerate(['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']):
			out[s] = strategy_counts[i] / len(dat)
		# out['payoff_matrix'] = payoff_matrix
		# out['probabilities'] = probabilities
		out['cost'] = cost
		out['alpha'] = alpha
		out['sigma'] = sigma
		out['problem_id'] = problem_id
		out['payoff_net_relative_by_strategy'] = np.nan_to_num(np.sum(out['payoff_net_relative_by_strategy'],axis=0) / strategy_counts).tolist()
		out['payoff_gross_relative_by_strategy'] = np.nan_to_num(np.sum(out['payoff_gross_relative_by_strategy'],axis=0) / strategy_counts).tolist()
		out['payoff_net_by_strategy'] = np.nan_to_num(np.sum(out['payoff_net_by_strategy'],axis=0) / strategy_counts).tolist()
		out['payoff_gross_by_strategy'] = np.nan_to_num(np.sum(out['payoff_gross_by_strategy'],axis=0) / strategy_counts).tolist()
	else:
		out = pd.DataFrame(data=out)
	
	return out

def append_R_features(df):
	for s in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']:
		# df['R_'+s] = (df[s] - df[s].mean()) / df[s].std(ddof=0)
		df['R_'+s] = df[s].astype(float)
	df['R_sigma'] = df['sigma']
	for i, s in enumerate(df['sigma'].sort_values().unique()):
		df.loc[df['R_sigma']==s, 'R_sigma'] = i
	df['R_alpha'] = df['alpha']
	for i, a in enumerate(np.flip(df['alpha'].sort_values().unique())):
		df.loc[df['R_alpha']==a, 'R_alpha'] = i
	df['R_cost'] = df['cost']
	for i, c in enumerate(np.flip(df['cost'].sort_values().unique())):
		df.loc[df['R_cost']==c, 'R_cost'] = i
	df['R_sigma'] = df['R_sigma'].astype(float)
	df['R_alpha'] = df['R_alpha'].astype(float)
	df['R_cost'] = df['R_cost'].astype(float)
	return df

def process_raw_data(dataObj_list):

	for dataObj in dataObj_list:
		with warnings.catch_warnings(): # for np.divide(0,0) and np.nanmean([ [all nans] )
			warnings.simplefilter("ignore", category=RuntimeWarning)
			if dataObj.isHuman:
				trials = append_features(([], dataObj))
				trials = append_R_features(trials)
			else:
				if os.path.isdir(dataObj.raw):
					files = [(os.path.join(dataObj.raw, f), dataObj) for f in os.listdir(dataObj.raw) if f[-5:]=='.json']
					nr_processes = multiprocessing.cpu_count()
					with multiprocessing.Pool(processes=nr_processes) as pool:
						results = pool.map(append_features, files)
					trials = pd.DataFrame()
					for res in results:
						trials = trials.append(res, ignore_index=True)
				else:
					trials = append_features(dataObj.raw)
					trials = pd.DataFrame(data=trials)

		pickle_dat = {'click_embedding':trials['click_embedding'].values, 'strategy':trials['strategy'].values}
		del trials['click_embedding']
		if not dataObj.isHuman: del trials['strategy']
		pickle_save(pickle_dat, dataObj.clicks)
		trials.to_csv(dataObj, index=False)

		print_special(f'saved {dataObj} from {dataObj.raw}, and {dataObj.clicks}.gz ({cfg.timer()})', False)

def get_trial_types(df):
	return [(p,s,a,c) for p,s,a,c in zip(df['problem_id'].values,\
										 df['sigma'].values,\
										 df['alpha'].values,\
										 df['cost'].values)]

def append_model_payoff(model_file, human_file):
	pd.options.mode.chained_assignment = None

	df1 = pd.read_csv(model_file, low_memory=False)
	df2 = pd.read_csv(human_file, low_memory=False)
	trial_types1 = get_trial_types(df1)
	trial_types2 = get_trial_types(df2)
	df1 = df1.to_dict('records')
	x1, x2, x3, x4 = [], [], [], []
	for trial_type in trial_types2:
		idx = [j for j,x in enumerate(trial_types1) if x==trial_type]
		assert(len(idx)==1)
		x1.append(df1[idx[0]]['payoff_gross'])
		x2.append(df1[idx[0]]['payoff_net'])
		x3.append(df1[idx[0]]['payoff_gross_relative'])
		x4.append(df1[idx[0]]['payoff_net_relative'])	
	df2['payoff_gross_modelDiff'] = df2['payoff_gross'] - x1
	df2['payoff_net_modelDiff'] = df2['payoff_net'] - x2
	df2['payoff_gross_relative_modelDiff'] = df2['payoff_gross_relative'] - x3
	df2['payoff_net_relative_modelDiff'] = df2['payoff_net_relative'] - x4

	df2.to_csv(human_file, index=False)
	print_special(f'appended \'payoff_*_modelDiff\' performance measures to {human_file} ({cfg.timer()})', False)

def match_human_model_trials_and_exclude(dataObjs, dataObjs_exclude):
	
	def get_good_participant_idx(df, dataObj):
		assert(dataObj.isHuman)
		if dataObj.num==1: # experiment 1 defaults
			criteria, cutoff, n = 'Rand', 0.5, None
		elif dataObj.num==2: # experiment 2 defaults
			criteria, cutoff, n = 'payoff_net_modelDiff', None, 0.27184466019417475 # fraction of exp2 control group participants with >50% random trials
		df['badTrial'] = df[criteria]
		if criteria[:6] == 'payoff': df['badTrial'] = -df['badTrial']
		mean_bad_trials_by_participant = df[['pid','badTrial']].groupby('pid').mean()['badTrial']
		df.drop(columns=['badTrial'], inplace=True)
		if n:
			cutoffs = mean_bad_trials_by_participant.sort_values().unique()
			cutoff = cutoffs[np.argmax([-abs(n - np.mean(mean_bad_trials_by_participant > c)) for c in cutoffs])]
		idx = mean_bad_trials_by_participant > cutoff
		participants_to_exclude = mean_bad_trials_by_participant[idx].index
		print_special(f'identified {sum(idx)} bad participants ({100*np.mean(idx):.1f}%) (using > {cutoff:.1f} cuttoff for participant mean \'{criteria}\')', False)
		return ~df['pid'].isin(participants_to_exclude).values

	def get_human_dat(dataObj):
		if dataObj.num==1:
			human_dat = human_dat_1
		elif dataObj.num==2 and dataObj.group=='con':
			human_dat = human_dat_con
		elif dataObj.num==2 and dataObj.group=='exp':
			human_dat = human_dat_exp
		elif dataObj.num==2 and dataObj.group=='both':
			human_dat = [human_dat_con[0] + human_dat_exp[0], 
						 np.concatenate((human_dat_con[1], human_dat_exp[1])), 
						 human_dat_con[2]+' and '+human_dat_exp[2]]
		else: raise Exception('unrecognized data object')
		return human_dat

	for dataObj in dataObjs:
		if not dataObj.isHuman: continue
		df = pd.read_csv(dataObj, low_memory=False)
		if dataObj.num==1:
			if 'human_dat' in locals(): raise Exception('expected only one human data object')
			human_dat_1 = [get_trial_types(df),
						   get_good_participant_idx(df, dataObj),
						   dataObj]
		elif dataObj.num==2 and dataObj.group=='con':
			if 'human_dat_con' in locals(): raise Exception('expected only one human data object from control group')
			human_dat_con = [get_trial_types(df),
							 get_good_participant_idx(df, dataObj),
							 dataObj]
		elif dataObj.num==2 and dataObj.group=='exp':
			if 'human_dat_exp' in locals(): raise Exception('expected only one human data object from experimental group')
			human_dat_exp = [get_trial_types(df),
							 get_good_participant_idx(df, dataObj),
							 dataObj]
	np.random.seed(123)
	for dataObj, dataObj_exclude in zip(dataObjs, dataObjs_exclude):
		assert(dataObj.num==dataObj_exclude.num and dataObj.group==dataObj_exclude.group and dataObj.isHuman==dataObj_exclude.isHuman)
		df = pd.read_csv(dataObj, low_memory=False)
		human_dat = get_human_dat(dataObj)
		pickle_in = pickle_load(dataObj.clicks)
		pickle_out = defaultdict(list)
		if not dataObj.isHuman:
			assert(len(pickle_in['click_embedding'])==len(df))
			trial_type_model = get_trial_types(df)
			dat_model = df.to_dict('records')
			df = {}
			for i, trial_type_human in enumerate(human_dat[0]):
				idx = [j for j,x in enumerate(trial_type_model) if x==trial_type_human]
				assert(len(idx)==1)
				random_samples = np.random.choice(len(pickle_in['click_embedding'][idx[0]]), dataObj.kmeans_sim_trials_per_human_trial)
				pickle_out['click_embedding'].append([pickle_in['click_embedding'][idx[0]][j] for j in random_samples])
				strategies = [pickle_in['strategy'][idx[0]][j] for j in random_samples]
				pickle_out['strategy'].append(strategies)
				tmp = dat_model[idx[0]]; tmp['strategy'] = strategies
				df[i] = tmp
			df = pd.DataFrame.from_dict(df, "index")
			df.to_csv(dataObj, index=False)
			pickle_save(pickle_out, dataObj.clicks)
			print_special(f'saved {dataObj} and {dataObj.clicks} with rows matched to {human_dat[2]} ({cfg.timer()})', False)
			pickle_in = pickle_out.copy()
		df = df[human_dat[1]]
		df.to_csv(dataObj_exclude, index=False)
		pickle_out['click_embedding'] = [pickle_in['click_embedding'][i] for i in range(len(human_dat[1])) if human_dat[1][i]]
		pickle_out['strategy'] = [pickle_in['strategy'][i] for i in range(len(human_dat[1])) if human_dat[1][i]]
		pickle_save(pickle_out, dataObj_exclude.clicks)
		print_special(f'saved {dataObj_exclude} and {dataObj_exclude.clicks} using exclusions from {human_dat[2]} ({cfg.timer()})', False)

def append_sources_of_under_performance(model_file, human_file, model_file_fitcost):
	pd.options.mode.chained_assignment = None

	df1 = pd.read_csv(model_file, low_memory=False)
	df2 = pd.read_csv(human_file, low_memory=False)
	df1_fitcost = pd.read_csv(model_file_fitcost, low_memory=False)
	# for non-fitcost model data, we store both Exp. 2 groups in one file, so need to seperate here
	if human_file.num == 2 and  human_file.group == 'con':
		df1 = df1[:len(df2)]
	if human_file.num == 2 and  human_file.group == 'exp':
		df1 = df1[-len(df2):]

	perf_metric = 'payoff_net'
	strategies = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other'] # must be in this order, based on process_data.append_features()
	trial_counts = np.zeros((len(strategies),len(strategies)))
	suboptimal_performance = np.zeros((len(strategies),len(strategies)))
	assert(len(df1)==len(df2) and len(df1)==len(df1_fitcost))
	for model_dat, human_dat in zip(df1_fitcost.to_dict('records'), df2.to_dict('records')): # use fitcost to control for implicit costs of clicking
		assert(model_dat['problem_id']==human_dat['problem_id'] and 
			   model_dat['cost']==human_dat['cost'] and 
			   model_dat['alpha']==human_dat['alpha'] and 
			   model_dat['sigma']==human_dat['sigma'])
		model_strat_freq = np.array([model_dat[s] for s in strategies])
		human_strat_idx = [x for x,s in enumerate(strategies) if human_dat[s]][0]
		model_performance_by_strategy = np.array(eval(model_dat[perf_metric+'_by_strategy']))
		human_performance = human_dat[perf_metric+'_bestBet'] # use bestBet to control for suboptimal information use
		trial_counts[:,human_strat_idx] += model_strat_freq
		suboptimal_performance[:,human_strat_idx] += model_strat_freq * (model_performance_by_strategy - human_performance)
	suboptimal_performance /= np.sum(suboptimal_performance)

	out = {}
	out['trial_counts'] = trial_counts
	out['model_performance'] = np.mean(df1[perf_metric])
	out['human_performance'] = np.mean(df2[perf_metric])
	out['human_performance_pct'] = 100 * out['human_performance'] / out['model_performance']
	out['peformance_gap_abs'] = out['model_performance'] - out['human_performance']
	out['peformance_gap_pct'] = 100 - out['human_performance_pct']
	assert(np.isclose(100*out['peformance_gap_abs']/out['model_performance'], out['peformance_gap_pct']))
	implicit_cost_diff = (out['model_performance'] - np.mean(df1_fitcost[perf_metric]))
	out['implicit_costs'] =  implicit_cost_diff / out['model_performance'] # / out['peformance_gap_abs']
	out['implicit_costs_model_fraction'] =  implicit_cost_diff / out['model_performance']
	out['imperfect_info_use'] = (np.mean(df2[perf_metric+'_bestBet']) - out['human_performance']) / out['model_performance'] # / out['peformance_gap_abs']
	out['imperfect_strat_selec_and_exec'] = 1 - out['implicit_costs'] - out['imperfect_info_use'] - out['human_performance_pct']/100 # 1 - out['implicit_costs'] - out['imperfect_info_use']
	out['imperfect_strat_selec_and_exec_by_strat'] = suboptimal_performance * out['imperfect_strat_selec_and_exec']
	out['imperfect_strat_selec'] = np.sum((-np.eye(6)+1)*out['imperfect_strat_selec_and_exec_by_strat'])
	out['imperfect_strat_exec'] = np.sum(np.eye(6)*out['imperfect_strat_selec_and_exec_by_strat'])
	out['imperfect_strat_selec_by_strat'] = np.sum((-np.eye(6)+1)*out['imperfect_strat_selec_and_exec_by_strat'], axis=0)
	out['imperfect_strat_exec_by_strat'] = np.sum(np.eye(6)*out['imperfect_strat_selec_and_exec_by_strat'], axis=0)

	out['nr_clicks_dif'] = np.mean(df1['nr_clicks']) - df2['nr_clicks'].mean()
	out['peformance_gap_points'] = np.mean(df1['payoff_gross']) - df2['payoff_gross'].mean()
	out['peformance_gap_gross_abs'] = np.mean(df1['payoff_gross_relative']) - df2['payoff_gross_relative'].mean()
	out['peformance_gap_gross_pct'] = 100 * df2['payoff_gross_relative'].mean() / np.mean(df1['payoff_gross_relative'])

	imperfect_info_use_idx = ~df2['best_choice']
	out['imperfect_info_use_points_lost'] = df2[imperfect_info_use_idx]['payoff_gross_bestBet'].mean() - df2[imperfect_info_use_idx]['payoff_gross'].mean()

	for k in out.keys():
		if isinstance(out[k], np.ndarray):
			out[k] = out[k].tolist()
	df2['under_performance'] = ''
	df2['under_performance'].iloc[0] = [out]

	df2.to_csv(human_file, index=False)
	print_special(f'appended sources of under-performance (using performance metric \'{perf_metric}\') in column \'under_performance\' to {human_file} ({cfg.timer()})', False)

def append_kmeans(in_file, k, label_cols=False):
	# label_cols is for testing many differnt values of k
	pd.options.mode.chained_assignment = None
	
	df = pd.read_csv(in_file, low_memory=False)
	pickle_dict = pickle_load(in_file.clicks)
	assert(len(pickle_dict['click_embedding'])==len(df))

	strategies = ['WADD','TTB_SAT','TTB','SAT_TTB','Rand'] # must be sorted by most to least clicks

	if in_file.isHuman:
		X = [pickle_dict['click_embedding'][i] for i in range(len(df))]
	else:
		X = [pickle_dict['click_embedding'][i][j] for i in range(len(df)) for j in range(len(pickle_dict['click_embedding'][i]))]

	with warnings.catch_warnings(): # for np.float deprecation
		warnings.simplefilter("ignore", category=DeprecationWarning)
		kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

	# make strategy labels using cluster labels
	if not label_cols and k <= len(strategies):
		km_labels_by_nr_clicks = np.flip(np.argsort(np.sum(kmeans.cluster_centers_,axis=1))) # sorted by most to least clicks
		strategies_sorted_by_km_label = [strategies[s] for s in np.argsort(km_labels_by_nr_clicks)]
		
		if in_file.isHuman:
			df['km_strategy'] = [strategies_sorted_by_km_label[l] for l in kmeans.labels_]
			for i, s in enumerate(strategies_sorted_by_km_label):
				df['km_'+s] = kmeans.labels_ == i
		else:
			x = in_file.kmeans_sim_trials_per_human_trial
			assert(len(kmeans.labels_)==x*len(df))
			df['km_strategy'] = [[strategies_sorted_by_km_label[l] for l in kmeans.labels_[i*x:i*x+x]] for i in range(len(df))]
			for i, s in enumerate(strategies_sorted_by_km_label):
				df['km_'+s] = [np.mean(kmeans.labels_[j*x:j*x+x]==i) for j in range(len(df))]
	
	k_label = '_k'+str(k) if label_cols else ''		
	df['cluster_centers'+k_label] = ''
	df['cluster_centers'+k_label].iloc[0] = [[kmeans.cluster_centers_[i][j].astype(np.float16) 
											 for j in range(len(kmeans.cluster_centers_[i]))] 
											 for i in range(len(kmeans.cluster_centers_))]
	df['cluster_points'+k_label] = ''
	df['cluster_points'+k_label].iloc[0] = [np.mean(kmeans.labels_==i).astype(np.float16) for i in range(k)]
	if not label_cols:
		pickle_dict['cluster_centers'] = df['cluster_centers'].iloc[0]
		pickle_dict['labels'] = kmeans.labels_
		pickle_save(pickle_dict, in_file.clicks)
	df.to_csv(in_file, index=False)
	print_special(f'appended k={k} cluster centers and strategy labels to {in_file} ({cfg.timer()})', False)

def run_process_data(which_experiment='both'):

	if which_experiment == 'both' or int(which_experiment) == 1:
		exp = cfg.Exp1
		
		dataObjs = [exp.human,
					exp.model, 
					exp.model_fitcost]
		process_raw_data(dataObjs)

		dataObjs_exclude = [exp.human_exclude,
							exp.model_exclude, 
							exp.model_fitcost_exclude]
		match_human_model_trials_and_exclude(dataObjs, dataObjs_exclude)

		append_sources_of_under_performance(exp.model, exp.human, exp.model_fitcost)
		append_sources_of_under_performance(exp.model_exclude, exp.human_exclude, exp.model_fitcost_exclude)

		k = 5
		append_kmeans(exp.human, k)
		append_kmeans(exp.human_exclude, k)
		for k in range(1,13):
			append_kmeans(exp.human, k, label_cols=True)

		k = 4
		append_kmeans(exp.model, k)
		append_kmeans(exp.model_exclude, k)
		for k in range(1,13):
			append_kmeans(exp.model, k, label_cols=True)

		print_special(f'finished processing Exp. 1 ({cfg.timer()})')

	if which_experiment == 'both' or int(which_experiment) == 2:
		exp = cfg.Exp2
		
		dataObjs = [exp.human_con,
					exp.human_exp,
					exp.model, 
					exp.model_fitcost_con,
					exp.model_fitcost_exp]
		process_raw_data(dataObjs)

		# this must be run after process_raw_data but before match_human_model_trials_and_exclude (exp2 exclusion criteria)
		append_model_payoff(exp.model, exp.human_con)
		append_model_payoff(exp.model, exp.human_exp)

		dataObjs_exclude = [exp.human_exclude_con,
							exp.human_exclude_exp,
							exp.model_exclude, 
							exp.model_fitcost_exclude_con,
							exp.model_fitcost_exclude_exp]
		match_human_model_trials_and_exclude(dataObjs, dataObjs_exclude)
		
		append_sources_of_under_performance(exp.model, exp.human_con, exp.model_fitcost_con)
		append_sources_of_under_performance(exp.model, exp.human_exp, exp.model_fitcost_exp)
		append_sources_of_under_performance(exp.model_exclude, exp.human_exclude_con, exp.model_fitcost_exclude_con)
		append_sources_of_under_performance(exp.model_exclude, exp.human_exclude_exp, exp.model_fitcost_exclude_exp)

		k = 5
		append_kmeans(exp.human_con, k)
		append_kmeans(exp.human_exclude_con, k)
		k = 4
		append_kmeans(exp.human_exp, k)
		append_kmeans(exp.human_exclude_exp, k)
		k = 4
		append_kmeans(exp.model, k)
		append_kmeans(exp.model_exclude, k)

		print_special(f'finished processing Exp. 2 ({cfg.timer()})')

def concat_model_runs():
	basedir = '../data/model/'
	for mod in ['exp1/','exp1_fitcost/','exp1_fitcost_exclude/',\
				'exp2/','exp2_con_fitcost/','exp2_exp_fitcost/',\
				'exp2_con_fitcost_exclude/','exp2_exp_fitcost_exclude/',\
				'exp2_con_fitcost_exclude_alt/','exp2_exp_fitcost_exclude_alt/']:
		files = [f for f in os.listdir(basedir+'A/'+mod) if f[-5:]=='.json']
		for file in files:
			dat = []
			for run in ['A/','B/','C/','D/','E/','F/','G/','H/','I/','J/']:
				dic = json.load(open(basedir+run+mod+file))
				dat += dic['uncovered']
			dic['uncovered'] = dat
			if not os.path.exists(basedir+mod): os.makedirs(basedir+mod)
			with open(basedir+mod+file, 'w') as outfile:
				json.dump(dic, outfile)

def pickle_save(data, filename):
	pickle.dump(data, gzip.open(filename+'.gz', 'wb'))

def pickle_load(filename):
	return pickle.load(gzip.open(filename+'.gz','rb'))

def cohen_d(x,y):
	nx, ny = len(x), len(y)
	dof = nx + ny - 2
	pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
	return (np.mean(x) - np.mean(y)) / pooled_std

def calc_max_EV():
	from numpy.random import dirichlet
	from numpy.random import normal
	from scipy.stats import sem
	n = int(1e7)
	sigmas = [75, 150]
	alphas = np.logspace(-1,1,5)
	dat = defaultdict(list)
	t0 = time.time()
	for s in sigmas:
		for a in alphas:
			print(s, round(a,1), time.time()-t0)
			tmp = [None]*n
			for i in range(n):
				tmp[i] = max(np.matmul(dirichlet([a]*4), normal(0,s,(4,6))))
			dat['sigma'].append(s)
			dat['alpha'].append(a)
			dat['mean'].append(np.mean(tmp))
			dat['sem'].append(sem(tmp))
	df = pd.DataFrame(dat)
	df.to_csv('../data/model/max_EV.csv', index=False)

def print_special(print_str, big=True, header=False):
	if big:
		special_str = '='*(len(print_str)+18)
		newlines = '' if not header else '\n'*3
		print(newlines+special_str+'\n'+'='*8+' '+print_str+' '+'='*8+'\n'+special_str+newlines)
	else:
		print('='*8+' '+print_str+' '+'='*8)
