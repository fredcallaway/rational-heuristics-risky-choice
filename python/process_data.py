import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing
import warnings
from ast import literal_eval
from sklearn.cluster import KMeans
import cfg


def append_features(in_file, isHuman=False, max_EV_by_condition=None):

	if isHuman:
		dat = pd.read_csv(in_file, low_memory=False)
		dat = dat[dat['block']=='test']
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

	def get_click_mat(click_location_map, clicks):
		mat = np.zeros(np.shape(click_location_map))
		for c in clicks:
			mat += (c==click_location_map)
		return mat

	if not max_EV_by_condition:
		filepath = '../data/human/1.0/processed/max_EV_by_condition.pkl' if 'exp1' in in_file else ''
		filepath = '../data/human/2.3/processed/max_EV_by_condition.pkl' if 'exp2' in in_file else filepath
		max_EV_by_condition = pd.read_pickle(filepath)

	if not(isHuman):
		cost, alpha, sigma = dat['cost'], np.round(dat['alpha'],decimals=1), dat['sigma']
		probabilities = eval(dat['probabilities'])
		payoff_matrix = eval(dat['payoff_matrix'])
		problem_id = dat['problem_id']
		payoff_perfect = max_EV_by_condition[str((sigma, alpha))]
		dat = dat['uncovered']

	out = defaultdict(list)

	use_EV_payoff = True
	for trial in dat:
		if isHuman:
			cost = trial['cost']
			probabilities = eval(trial['probabilities'])
			payoff_matrix = eval(trial['payoff_matrix'])
			clicks = eval(trial['clicks'])
			choice = trial['choice_index']
			payoff_idx = trial['payoff_index']
			payoff_perfect = max_EV_by_condition[str((trial['sigma'],trial['alpha']))] # use condition average (not max(EVs_perfect) for each trial)
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
		elif nr_clicks>=19:
			strategy = 'WADD'
		elif set(clicks)==set(click_location_map[ttb_row,:]):
			strategy = 'TTB'
		elif (set(clicks).issubset(set(click_location_map[ttb_row,:]))) & (np.argmax([payoffs[i] for i in clicks])==(nr_clicks-1)):
			strategy = 'SAT_TTB'
		elif (all(np.diff(np.sum(click_mat_transformed,axis=0))<=0)) & (all(np.diff(np.sum(click_mat_transformed,axis=1))<=0)):
			strategy = 'TTB_SAT'
		else:
			strategy = 'Other'

		EVs = np.matmul(probabilities, payoff_matrix*click_mat)
		EVs_perfect = np.matmul(probabilities, payoff_matrix)
		best_choice = np.argmax(EVs)
		assert(payoff_perfect>0)
		if isHuman:
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
			bad_choice = (strategy=='Rand') or (choice != best_choice)
		else:
			if use_EV_payoff:
				payoff_gross = EVs_perfect[best_choice] if strategy!='Rand' else 0
			else:
				payoff_gross = payoff_matrix[payoff_idx][best_choice]
		payoff_net = payoff_gross - nr_clicks * cost
		payoff_gross_relative = payoff_gross / payoff_perfect
		payoff_net_relative = payoff_net / payoff_perfect

		type1_transitions = sum([1 for col in same_col_idx for c in range(nr_clicks-1) if (clicks[c] in col) and (clicks[c+1] in col)])
		type2_transitions = sum([1 for row in same_row_idx for c in range(nr_clicks-1) if (clicks[c] in row) and (clicks[c+1] in row)])
		processing_pattern = np.divide((type1_transitions-type2_transitions), (type1_transitions+type2_transitions))
		click_var_gamble = np.var(np.divide(np.sum(click_mat, axis=0), nr_clicks))
		click_var_outcome = np.var(np.divide(np.sum(click_mat, axis=1), nr_clicks))

		if isHuman:
			out['problem_id'].append(trial['problem_id'])
			if 'display_ev' in trial:
				out['display_ev'].append(trial['display_ev'])
			out['pid'].append(trial['pid'])
			out['sigma'].append(trial['sigma'])
			out['alpha'].append(trial['alpha'])
			out['cost'].append(trial['cost'])
			out['probabilities'].append(probabilities)
			out['payoff_matrix'].append(trial['payoff_matrix'])
			out['choice'].append(choice)
			out['best_choice'].append(best_choice)
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

	if not(isHuman):
		features_to_mean = [k for k in out.keys() if k not in ['click_embedding','strategy','payoff_net_relative_by_strategy',\
									'payoff_gross_relative_by_strategy','payoff_net_by_strategy','payoff_gross_by_strategy']]
		for k in features_to_mean:
			out[k] = np.nanmean(out[k])
		strategy_counts = np.array([sum([o==s for o in out['strategy']]) for s in ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']])
		for i, s in enumerate(['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']):
			out[s] = strategy_counts[i] / len(dat)
		out['payoff_matrix'] = payoff_matrix
		out['probabilities'] = probabilities
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

	df['R_sigma'] = df['R_sigma'].astype(float)
	df['R_alpha'] = df['R_alpha'].astype(float)
	df['R_cost'] = df['cost'].astype(float)

	return df

def save_max_EV_by_condition(human_file, out_dir):
	df = pd.read_csv(human_file, low_memory=False)
	df = df[df['block']=='test']
	df.loc[:,'alpha'] = df['alpha'].round(decimals=1)
	df = df.set_index(['sigma','alpha']).sort_index()
	max_EV_by_condition = {}
	for idx in df.index.sort_values().unique():
		probabilities = df.loc[idx,'probabilities'].apply(literal_eval).values.tolist()
		payoff_matrices = df.loc[idx,'payoff_matrix'].apply(literal_eval).values.tolist()   
		max_EV_by_condition[str(idx)] = np.mean([max(np.matmul(p,r)) for p, r in zip(probabilities, payoff_matrices)])

	pickle.dump(max_EV_by_condition, open(os.path.join(out_dir,'max_EV_by_condition.pkl'),'wb'))

	print_special('saved '+os.path.join(out_dir,'max_EV_by_condition.pkl'), False)

	return max_EV_by_condition

def process_raw_data(dataObj):

	out_dir = os.path.dirname(dataObj)
	
	with warnings.catch_warnings(): # for np.divide(0,0) and np.nanmean([ [all nans] )
		warnings.simplefilter("ignore", category=RuntimeWarning)
		if dataObj.isHuman:
			max_EV_by_condition = save_max_EV_by_condition(dataObj.raw, out_dir)
			trials = append_features(dataObj.raw, True, max_EV_by_condition)
			trials = append_R_features(trials)
		else:
			if os.path.isdir(dataObj.raw):
				files = [os.path.join(dataObj.raw, f) for f in os.listdir(dataObj.raw) if f[-5:]=='.json']
				nr_processes = multiprocessing.cpu_count()
				with multiprocessing.Pool(processes=nr_processes) as pool:
					results = pool.map(append_features, files)
				trials = pd.DataFrame()
				for res in results:
					trials = trials.append(res, ignore_index=True)
			else:
				trials = append_features(dataObj.raw)
				trials = pd.DataFrame(data=trials)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if dataObj.num==2 and dataObj.isHuman:
		trials1 = trials[trials['display_ev']==True]
		trials2 = trials[trials['display_ev']==False]
		pickle_dat1 = {'click_embedding':trials1['click_embedding'].values, 'strategy':trials1['strategy'].values}
		del trials1['click_embedding']
		if not(dataObj.isHuman): del trials1['strategy']
		pickle_dat2 = {'click_embedding':trials2['click_embedding'].values, 'strategy':trials2['strategy'].values}
		del trials2['click_embedding']
		if not(dataObj.isHuman): del trials2['strategy']
		pickle.dump(pickle_dat1, open(os.path.join(out_dir,'trials_exp_click_embeddings.pkl'),'wb'))
		pickle.dump(pickle_dat2, open(os.path.join(out_dir,'trials_con_click_embeddings.pkl'),'wb'))
		trials1.to_csv(os.path.join(out_dir,'trials_exp.csv'), index=False)
		trials2.to_csv(os.path.join(out_dir,'trials_con.csv'), index=False)
	else:
		pickle_dat = {'click_embedding':trials['click_embedding'].values, 'strategy':trials['strategy'].values}
		del trials['click_embedding']
		if not(dataObj.isHuman): del trials['strategy']
		pickle.dump(pickle_dat, open(os.path.join(out_dir,'trials_click_embeddings.pkl'),'wb'))
		trials.to_csv(os.path.join(out_dir,'trials.csv'), index=False)

	print_special('saved processed trials .csv and click_embeddings .pkl in '+out_dir, False)

def append_model_trial_weight_from_human_frequencies(model_file, human_file, exclude_participants, human_group2_file=None):

	df_model = pd.read_csv(model_file, low_memory=False)

	experiment2 = True if human_file.num==2 else False
	exclude_sigma_150 = [False, True] if not experiment2 else [False]
	for exclude_sigma in exclude_sigma_150:
		for exclude in exclude_participants:
			df_human = pd.read_csv(human_file, low_memory=False)
			if exclude_sigma:
				df_human = df_human[df_human['sigma']==75]
				sig_str = '_75'
			else:
				sig_str = ''
			exc_str = '_exclude' if exclude else ''
			exclude = 2 if exclude and experiment2 else exclude
			df_human = exclude_bad_participants(df_human, exclude)

			if experiment2:
				df_human_group2 = pd.read_csv(human_group2_file, low_memory=False)
				df_human_group2 = exclude_bad_participants(df_human_group2, exclude)

			if not experiment2:
				c1, c2, c3 = df_model['problem_id'].values, df_model['cost'].values, df_model['sigma'].values
				possible_trials = [(c1[i], c2[i], c3[i]) for i in range(len(df_model))]
				trial_weight = []
				for p, c, s in possible_trials: # this is the slow part
					indices = np.logical_and(np.logical_and(df_human['problem_id']==p, df_human['cost']==c), df_human['sigma']==s) # sum(df_human['problem_id']==p) & (df_human['cost']==c) & (df_human['sigma']==s)
					trial_weight.append(sum(indices))
				assert(sum(trial_weight)==len(df_human))
				normalizing_factor = len(df_model) if not exclude_sigma else sum(df_model['sigma']==75)
				df_model['trial_weight'+exc_str+sig_str] = (np.array(trial_weight) / len(df_human) * normalizing_factor).tolist()
			else:
				c1, c2 = df_model['problem_id'].values, df_model['cost'].values
				possible_trials = [(c1[i], c2[i]) for i in range(len(df_model))]
				trial_weight1, trial_weight2, trial_weight = [], [], []
				for p, c in possible_trials:
					indices1 = np.logical_and(df_human['problem_id']==p, df_human['cost']==c) # sum(df_human['problem_id']==p) & (df_human['cost']==c)
					indices2 = np.logical_and(df_human_group2['problem_id']==p, df_human_group2['cost']==c) # sum(df_human_group2['problem_id']==p) & (df_human_group2['cost']==c)
					trial_weight1.append(sum(indices1))
					trial_weight2.append(sum(indices2))
					trial_weight.append(trial_weight1[-1]+trial_weight2[-1])
				assert(sum(trial_weight)==len(df_human)+len(df_human_group2))
				if all(df_human.display_ev.values) and all(df_human_group2.display_ev.values==False):
					df_model['trial_weight_exp'+exc_str] = (np.array(trial_weight1) / len(df_human) * len(df_model)).tolist()
					df_model['trial_weight_con'+exc_str] = (np.array(trial_weight2) / len(df_human_group2) * len(df_model)).tolist()
				elif all(df_human_group2.display_ev.values) and all(df_human.display_ev.values==False):
					df_model['trial_weight_con'+exc_str] = (np.array(trial_weight1) / len(df_human) * len(df_model)).tolist()
					df_model['trial_weight_exp'+exc_str] = (np.array(trial_weight2) / len(df_human_group2) * len(df_model)).tolist()
				else:
					Exception('control group and experimental group are''t valid')
				df_model['trial_weight'+exc_str] = (np.array(trial_weight) / (len(df_human)+len(df_human_group2)) * len(df_model)).tolist()

	df_model.to_csv(model_file, index=False)
	print_special('appended model trial_weights from human trial frequencies to '+model_file, False)

def append_model_payoff(model_file, human_file):
	pd.options.mode.chained_assignment = None

	df1 = pd.read_csv(model_file, low_memory=False)
	df2 = pd.read_csv(human_file, low_memory=False)

	df1.loc[:,'alpha'] = df1['alpha'].round(decimals=1)
	df2.loc[:,'alpha'] = df2['alpha'].round(decimals=1)

	df1_ = df1.set_index(['problem_id','sigma','alpha','cost']).sort_index()
	df2_ = df2.set_index(['problem_id','sigma','alpha','cost']).sort_index()

	df2['payoff_gross_modelDiff'] = ''
	df2['payoff_net_modelDiff'] = ''
	df2['payoff_gross_relative_modelDiff'] = ''
	df2['payoff_net_relative_modelDiff'] = ''

	for i, trial_type in enumerate(df2_.index):
		model_dat = df1_.xs(trial_type)
		df2['payoff_gross_modelDiff'].iloc[i] = model_dat['payoff_gross']
		df2['payoff_net_modelDiff'].iloc[i] = model_dat['payoff_net']
		df2['payoff_gross_relative_modelDiff'].iloc[i] = model_dat['payoff_gross_relative']
		df2['payoff_net_relative_modelDiff'].iloc[i] = model_dat['payoff_net_relative']
	df2['payoff_gross_modelDiff'] = df2['payoff_gross'] - df2['payoff_gross_modelDiff'].astype(float)
	df2['payoff_net_modelDiff'] = df2['payoff_net'] - df2['payoff_net_modelDiff'].astype(float)
	df2['payoff_gross_relative_modelDiff'] = df2['payoff_gross_relative'] - df2['payoff_gross_relative_modelDiff'].astype(float)
	df2['payoff_net_relative_modelDiff'] = df2['payoff_net_relative'] - df2['payoff_net_relative_modelDiff'].astype(float)	

	df2.to_csv(human_file, index=False)
	print_special('appended \'payoff_*_modelDiff\' performance measures to '+human_file, False)

def append_sources_of_under_performance(model_file, human_file, model_file_fitcost, exclude_participants=False):
	pd.options.mode.chained_assignment = None

	df1 = pd.read_csv(model_file, low_memory=False)
	df2_orig = pd.read_csv(human_file, low_memory=False)
	exclude_participants = 2 if exclude_participants and human_file.num==2 else exclude_participants
	df2 = exclude_bad_participants(df2_orig, exclude_participants)
	df1_fitcost = pd.read_csv(model_file_fitcost, low_memory=False)
	exclude_str = '' if not exclude_participants else '_exclude'
	exp2_str = '' if not human_file.group else '_'+human_file.group # when comparing the model to one group from exp2, use trials weights specific to that group

	perf_metric = 'payoff_net'
	strategies = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other'] # must be in this order, based on process_data.append_features()

	df1_fitcost.loc[:,'alpha'] = df1_fitcost['alpha'].round(decimals=1) # use fitcost to control for implicit costs
	df2.loc[:,'alpha'] = df2['alpha'].round(decimals=1)
	df1_ = df1_fitcost.set_index(['problem_id','sigma','alpha','cost']).sort_index()
	df2_ = df2.set_index(['problem_id','sigma','alpha','cost']).sort_index()
	trial_counts = np.zeros((len(strategies),len(strategies)))
	suboptimal_performance = np.zeros((len(strategies),len(strategies)))
	for i, trial_type in enumerate(df2_.index): #.unique():
		model_dat = df1_.xs(trial_type)
		human_dat = df2_.iloc[i]
		model_strat_freq = np.array([model_dat[s] for s in strategies])
		human_strat_idx = [x for x,s in enumerate(strategies) if human_dat[s]][0]
		model_performance_by_strategy = eval(model_dat[perf_metric+'_by_strategy'])
		human_performance = human_dat[perf_metric+'_bestBet'] # use bestBet to control for suboptimal information use
		trial_counts[:,human_strat_idx] += model_strat_freq
		suboptimal_performance[:,human_strat_idx] += model_strat_freq * (model_performance_by_strategy - human_performance)
	suboptimal_performance /= np.sum(suboptimal_performance)

	out = {}
	out['trial_counts'] = trial_counts
	out['model_performance'] = np.mean(df1[perf_metric] * df1['trial_weight'+exp2_str+exclude_str])
	out['human_performance'] = np.mean(df2[perf_metric])
	out['human_performance_pct'] = 100 * out['human_performance'] / out['model_performance']
	out['peformance_gap_abs'] = out['model_performance'] - out['human_performance']
	out['peformance_gap_pct'] = 100 - out['human_performance_pct']
	assert(np.isclose(100*out['peformance_gap_abs']/out['model_performance'], out['peformance_gap_pct']))
	implicit_cost_diff = (out['model_performance'] - np.mean(df1_fitcost[perf_metric] * df1_fitcost['trial_weight'+exp2_str+exclude_str]))
	out['implicit_costs'] =  implicit_cost_diff / out['model_performance'] # / out['peformance_gap_abs']
	out['implicit_costs_model_fraction'] =  implicit_cost_diff / out['model_performance']
	out['imperfect_info_use'] = (np.mean(df2[perf_metric+'_bestBet']) - out['human_performance']) / out['model_performance'] # / out['peformance_gap_abs']
	out['imperfect_strat_selec_and_exec'] = 1 - out['implicit_costs'] - out['imperfect_info_use'] - out['human_performance_pct']/100 # 1 - out['implicit_costs'] - out['imperfect_info_use']
	out['imperfect_strat_selec_and_exec_by_strat'] = suboptimal_performance * out['imperfect_strat_selec_and_exec']
	out['imperfect_strat_selec'] = np.sum((-np.eye(6)+1)*out['imperfect_strat_selec_and_exec_by_strat'])
	out['imperfect_strat_exec'] = np.sum(np.eye(6)*out['imperfect_strat_selec_and_exec_by_strat'])
	out['imperfect_strat_selec_by_strat'] = np.sum((-np.eye(6)+1)*out['imperfect_strat_selec_and_exec_by_strat'], axis=0)
	out['imperfect_strat_exec_by_strat'] = np.sum(np.eye(6)*out['imperfect_strat_selec_and_exec_by_strat'], axis=0)

	out['nr_clicks_dif'] = np.mean(df1['nr_clicks']*df1['trial_weight'+exp2_str+exclude_str]) - df2['nr_clicks'].mean()
	out['peformance_gap_points'] = np.mean(df1['payoff_gross'] * df1['trial_weight'+exp2_str+exclude_str]) - df2['payoff_gross'].mean()
	out['peformance_gap_gross_abs'] = np.mean(df1['payoff_gross_relative'] * df1['trial_weight'+exp2_str+exclude_str]) - df2['payoff_gross_relative'].mean()
	out['peformance_gap_gross_pct'] = 100 - 100 * df2['payoff_gross_relative'].mean() / np.mean(df1['payoff_gross_relative'] * df1['trial_weight'+exp2_str+exclude_str])

	imperfect_info_use_idx = df2['choice'] != df2['best_choice']
	out['imperfect_info_use_points_lost'] = df2[imperfect_info_use_idx]['payoff_gross_bestBet'].mean() - df2[imperfect_info_use_idx]['payoff_gross'].mean()

	for k in out.keys():
		if isinstance(out[k], np.ndarray):
			out[k] = out[k].tolist()
	df2_orig['under_performance'+exclude_str] = ''
	df2_orig['under_performance'+exclude_str].iloc[0] = [out]

	df2_orig.to_csv(human_file, index=False)
	print_special('appended sources of under-performance (using performance metric \''+perf_metric+'\') in column \'under_performance'+exclude_str+'\' to '+human_file, False)

def append_kmeans(in_file, k, sim_trials_per_human_trial=None, label_cols=False):
	# label_cols is for testing many differnt values of k
	pd.options.mode.chained_assignment = None
	
	df = pd.read_csv(in_file, low_memory=False)
	pickle_dict = pd.read_pickle(os.path.splitext(in_file)[0]+'_click_embeddings.pkl')
	assert(len(pickle_dict['click_embedding'])==len(df))

	isHuman = True if not sim_trials_per_human_trial else False

	strategies = ['WADD','TTB_SAT','TTB','SAT_TTB','Rand'] # must be sorted by most to least clicks

	if isHuman:
		X = [pickle_dict['click_embedding'][i] for i in range(len(df))]
	else:
		nr_sims = 1000
		assert(nr_sims==len(pickle_dict['click_embedding'][0]))
		X = [];
		df['strategy'] = ''
		pickle_dict['samples'] = []
		nr_samples_by_trial_type = []
		np.random.seed(123)
		for i, w in enumerate(df['trial_weight'].values):
			nr_samples = round(w*sim_trials_per_human_trial)
			random_samples = np.random.choice(np.arange(nr_sims), nr_samples) #[random.choice(np.arange(nr_sims)) for _ in range(nr_samples)]
			nr_samples_by_trial_type.append(nr_samples)
			df['strategy'].iloc[i] = [pickle_dict['strategy'][i][j] for j in random_samples]
			pickle_dict['samples'].append(random_samples)
			for j in random_samples:
				X.append(pickle_dict['click_embedding'][i][j])

	with warnings.catch_warnings(): # for np.float deprecation
		warnings.simplefilter("ignore", category=DeprecationWarning)
		kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

	# make strategy labels using cluster labels
	if not label_cols and k <= len(strategies):
		km_labels_by_nr_clicks = np.flip(np.argsort(np.sum(kmeans.cluster_centers_,axis=1))) # sorted by most to least clicks
		strategies_sorted_by_km_label = [strategies[s] for s in np.argsort(km_labels_by_nr_clicks)]
		
		if isHuman:
			df['km_strategy'] = [strategies_sorted_by_km_label[l] for l in kmeans.labels_]
			for i, s in enumerate(strategies_sorted_by_km_label):
				df['km_'+s] = kmeans.labels_ == i
		else:
			sim_idx = np.cumsum([0]+nr_samples_by_trial_type)
			df['km_strategy'] = [[strategies_sorted_by_km_label[kmeans.labels_[i]] for i in range(sim_idx[j],sim_idx[j+1])] for j in range(len(sim_idx)-1)] # km strategies from sampled sims by trial type
			for i, s in enumerate(strategies_sorted_by_km_label):
				df['km_'+s] = [np.mean(kmeans.labels_[sim_idx[j]:sim_idx[j+1]]==i) for j in range(len(sim_idx)-1)]
	
	k_label = '_k'+str(k) if label_cols else ''		
	df['cluster_centers'+k_label] = ''
	df['cluster_centers'+k_label].iloc[0] = [[kmeans.cluster_centers_[i][j].astype(np.float16) for j in range(len(kmeans.cluster_centers_[i]))] for i in range(len(kmeans.cluster_centers_))]
	df['cluster_points'+k_label] = ''
	df['cluster_points'+k_label].iloc[0] = [np.mean(kmeans.labels_==i).astype(np.float16) for i in range(k)]
	if not label_cols:
		pickle_dict['cluster_centers'] = df['cluster_centers'].iloc[0]
		pickle_dict['labels'] = kmeans.labels_
		pickle.dump(pickle_dict, open(os.path.splitext(in_file)[0]+'_click_embeddings.pkl','wb'))
	df.to_csv(in_file, index=False)
	print_special('appended k='+str(k)+' k-means cluster centers and strategy labels to '+in_file, False)

def run_process_data(which_experiment='both', process_human=True, run_kmeans=True, process_model=True):

	if which_experiment == 'both' or int(which_experiment) == 1:
		exp = cfg.Exp1()
		model_dats = [exp.model, exp.model_fitcost, exp.model_fitcost_exclude]
		exclude_participants = [[False,True], [False], [True]]
		
		if process_human:
			# this must be run prior to append_model_trial_weight_from_human_frequencies
			process_raw_data(exp.human) 
			print_special(f'finished preprocessing Exp. 1 human data from {exp.human.raw} ({cfg.timer()})')

		if process_model:
			for model_dat, exclude in zip(model_dats, exclude_participants):
				# this must be run before append_sources_of_under_performance
				process_raw_data(model_dat)
				append_model_trial_weight_from_human_frequencies(model_dat, exp.human, exclude)
				print_special(f'finished preprocessing Exp. 1 model data from {model_dat} ({cfg.timer()})')

		if process_human:
			append_sources_of_under_performance(exp.model, exp.human, exp.model_fitcost, exclude_participants=False)
			append_sources_of_under_performance(exp.model, exp.human, exp.model_fitcost_exclude, exclude_participants=True)
			print_special(f'finished appending Exp. 1 under-performance data from {exp.human} ({cfg.timer()})')

		if run_kmeans:
			if process_human:
				k = 5
				append_kmeans(exp.human, k)
				for k in range(1,13):
					append_kmeans(exp.human, k, label_cols=True)
					print_special(f'finished running k-means for Exp. 1 human data from {exp.human} with {k} clusters ({cfg.timer()})', False)
				print_special(f'finished running k-means for Exp. 1 human data from {exp.human} with {k} clusters ({cfg.timer()})')

			if process_model:
				k = 4
				sim_trials_per_human_trial = 10 # 10x human trials for kmeans
				append_kmeans(exp.model, k, sim_trials_per_human_trial)
				for k in range(1,13):
					append_kmeans(exp.model, k, sim_trials_per_human_trial, label_cols=True)
					print_special(f'finished running k-means for Exp. 1 model data from {exp.model} with {k} clusters ({cfg.timer()})', False)
				print_special(f'finished running k-means for Exp. 1 model data from {exp.model} with {k} clusters ({cfg.timer()})')

		print_special(f'finished processing Exp. 1 ({cfg.timer()})', header=True)

	if which_experiment == 'both' or int(which_experiment) == 2:
		exp = cfg.Exp2
		model_dats = [exp.model, exp.model_fitcost_con, exp.model_fitcost_exp, exp.model_fitcost_exclude_con, exp.model_fitcost_exclude_exp]
		exclude_participants = [[False,2], [False], [False], [2], [2]]
		
		if process_human:
			# this must be run prior to append_model_trial_weight_from_human_frequencies
			process_raw_data(exp.human_con) # human_con.raw = human_exp.raw, makes no difference here
			print_special(f'finished preprocessing Exp. 2 human data from {exp.human_con.raw} ({cfg.timer()})')

		if process_model:
			for i, (model_dat, exclude) in enumerate(zip(model_dats, exclude_participants)):
				# this must be run before append_model_payoff and append_sources_of_under_performance
				process_raw_data(model_dat)
				if i==0: # this must be run after process_raw_data but before append_model_trial_weight_from_human_frequencies (exp2 exclusion criteria)
					append_model_payoff(exp.model, exp.human_con)
					append_model_payoff(exp.model, exp.human_exp)
				append_model_trial_weight_from_human_frequencies(model_dat, exp.human_con, exclude, exp.human_exp)
				print_special(f'finished preprocessing Exp. 2 model data from {exp.human_con} ({cfg.timer()})')

		if process_human:
			if not process_model:
				append_model_payoff(exp.model, exp.human_con)
				append_model_payoff(exp.model, human_file_exp)
			append_sources_of_under_performance(exp.model, exp.human_con, exp.model_fitcost_con)
			append_sources_of_under_performance(exp.model, exp.human_exp, exp.model_fitcost_exp)
			append_sources_of_under_performance(exp.model, exp.human_con, exp.model_fitcost_exclude_con, exclude_participants=2)
			append_sources_of_under_performance(exp.model, exp.human_exp, exp.model_fitcost_exclude_exp, exclude_participants=2)
			print_special(f'finished appending Exp. 2 under-performance data from {exp.human_con} and {exp.human_exp} ({cfg.timer()})')

		if run_kmeans:
			if process_human:
				k = 5
				append_kmeans(exp.human_con, k)
				print_special(f'finished running k-means for Exp. 2 human data from {exp.human_con} with {k} clusters ({cfg.timer()})')

				k = 4
				append_kmeans(exp.human_exp, k)
				print_special(f'finished running k-means for Exp. 2 human data from {exp.human_exp} with {k} clusters ({cfg.timer()})')

			if process_model:
				k = 4
				sim_trials_per_human_trial = 10 # 10x human trials for kmeans
				append_kmeans(exp.model, k, sim_trials_per_human_trial)
				print_special(f'finished running k-means for Exp. 2 model data from {exp.model} with {k} clusters ({cfg.timer()})')

def cohen_d(x,y):
	nx, ny = len(x), len(y)
	dof = nx + ny - 2
	pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
	return (np.mean(x) - np.mean(y)) / pooled_std

def exclude_bad_participants(df, exclude_exp=1, **kwargs):
	if exclude_exp==False:
		return df
	elif exclude_exp==1 or exclude_exp==True: # experiment 1 defaults
		criteria, cutoff, n = 'Rand', None, None
	else: # experiment 2 defaults
		criteria, cutoff, n = 'payoff_net_modelDiff', None, 0.27184466019417475 # fraction of exp2 control group participants with >50% random trials
	if 'criteria' in kwargs: criteria = kwargs['criteria']
	if 'cutoff' in kwargs: cutoff = kwargs['cutoff']
	if 'n' in kwargs: n = kwargs['n']

	df['badTrial'] = df[criteria]
	if criteria[:6] == 'payoff':
		df['badTrial'] = -df['badTrial']

	mean_bad_trials_by_participant = df[['pid','badTrial']].groupby('pid').mean()['badTrial']
	df.drop(columns=['badTrial'], inplace=True)

	if n and cutoff:
		Exception('You can choose a number/fraction of participants to exclude, or a criteria cutoff value, but not both')
	if n:
		if type(n)==int:
			n /= len(mean_bad_trials_by_participant) # make n fraction of participants to exclude
		cutoffs = mean_bad_trials_by_participant.sort_values().unique()
		cutoff = cutoffs[np.argmax([-abs(n - np.mean(mean_bad_trials_by_participant > c)) for c in cutoffs])]
	elif not cutoff:
		cutoff = 0.5
	idx = mean_bad_trials_by_participant > cutoff
	participants_to_exclude = mean_bad_trials_by_participant[idx].index

	print_special(f'removed {sum(idx)} participants ({100*np.mean(idx):.1f}%) for current analysis (using > {cutoff:.1f} cuttoff for participant mean \'{criteria}\')', False)

	return df[~df['pid'].isin(participants_to_exclude)]

def print_special(print_str, big=True, header=False):
	if big:
		special_str = '='*(len(print_str)+18)
		newlines = '' if not header else '\n'*3
		print(newlines+special_str+'\n'+'='*8+' '+print_str+' '+'='*8+'\n'+special_str+newlines)
	else:
		print('='*8+' '+print_str+' '+'='*8)
