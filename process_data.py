import numpy as np
import pandas as pd
import pdb
from scipy.stats import sem
import ast
import os
import multiprocessing


def get_click_mat(C, clicks):

    mat = np.zeros(np.shape(C))
    for c in clicks:
        mat += (c==C)
    if (mat.shape[1]!=6):
        mat = np.hstack((mat, np.zeros((4,6-mat.shape[1]))))

    return mat



def get_payoffs(weights, reward_matrix, choice, clicks, cost, EV_subjective, payoff_idx=[]):

	weights = np.divide(weights,sum(weights))
	outcome = np.random.choice(4, p=weights)
	if payoff_idx==[]:
		gross_payoff = reward_matrix[outcome][choice]
	else:
		gross_payoff = reward_matrix[payoff_idx][choice]
	net_payoff = gross_payoff - len(clicks)*cost
	EVs = np.matmul(weights,reward_matrix)
	rand_payoff = np.mean(EVs)
	perfect_payoff = max(EVs)
	expected_gross_payoff = EVs[choice]
	relative_expected_payoff = expected_gross_payoff / perfect_payoff
	relative_payoff = gross_payoff / perfect_payoff
	relative_old = reward_matrix[outcome][choice] / perfect_payoff

	if len(payoff_idx) > 0:
		gross_payoff_bestBet = reward_matrix[payoff_idx][np.argmax(EV_subjective)]
	else:
		gross_payoff_bestBet = []
	payoff_relative_bestBet = gross_payoff_bestBet / perfect_payoff
	
	return net_payoff, gross_payoff, expected_gross_payoff, rand_payoff, perfect_payoff, relative_payoff, relative_expected_payoff, relative_old, gross_payoff_bestBet, payoff_relative_bestBet



def append_features(df, isHuman=False):

	if isHuman:
		C = np.arange(24).reshape((4, 6))
		same_col_idx = [np.arange(i,24,6) for i in range(6)]
		same_row_idx = [np.arange(i,i+6) for i in range(0,24,6)]
	else:
		if df[-8:]=='DS_Store':
			return []
		df = pd.read_json(df)
		C = np.arange(24).reshape((6,4)).T
		same_col_idx = [np.arange(i,i+4) for i in range(0,24,4)]
		same_row_idx = [np.arange(i,24,4) for i in range(4)]

	nr_clicks = []
	click_var_gamble = []
	click_var_outcome = []
	EVs = []
	EV_chosen = []
	EV_max = []
	click_embedding_seq = []
	click_embedding_prob = []
	click_embedding_EV = []
	strategy = []
	net_payoff = []
	payoff_gross = []
	payoff_expected_gross = []
	payoff_rand = []
	payoff_perfect = []
	payoff_relative = []
	payoff_relative_expected = []
	payoff_relative_old = []
	processing_pattern = []
	processing_type1 = []
	processing_type2 = []
	payoff_gross_bestBet = []
	payoff_relative_bestBet = []

	for j, trial in df.iterrows():

		if isHuman:
			clicks = trial.clicks
			click_mat = get_click_mat(C, clicks)
			weights = trial.probabilities
			payoff_matrix = trial.payoff_matrix
			choice = trial.choice_index
			payoff_idx = trial.payoff_index
		else:
			clicks = [x-1 for x in trial.uncovered]
			click_mat = get_click_mat(C, clicks)
			if -1 in clicks:
				print('unexpected numbering convention')
				break
			if 'probabilities' in df:
				weights = ast.literal_eval(trial.probabilities)
				payoff_matrix = ast.literal_eval(trial.values[5])
				choice = np.argmax(np.matmul(weights,payoff_matrix*click_mat))
			else:
				weights = trial.weights
				payoff_matrix = np.transpose(trial.values[6])######7])
				choice = trial.choice-1
			payoff_idx = []
			
		EV = np.matmul(weights,payoff_matrix*click_mat)
		EVs.append(EV)
		EV_chosen.append(EV[choice])
		EV_max.append(max(EV))

		net, gross, expected_gross, rand, perfect, relative, relative_expected, relative_old, gross_bestBet, relative_bestBet = get_payoffs(weights, payoff_matrix, choice, clicks, trial.cost, EV, payoff_idx)
		net_payoff.append(net)
		payoff_gross.append(gross)
		payoff_expected_gross.append(expected_gross)
		payoff_rand.append(rand)
		payoff_perfect.append(perfect)
		payoff_relative.append(relative)
		payoff_relative_expected.append(relative_expected)
		payoff_relative_old.append(relative_old)
		payoff_gross_bestBet.append(gross_bestBet)
		payoff_relative_bestBet.append(relative_bestBet)

		nr_clicks.append(len(clicks))

		click_var_gamble.append(np.var(np.divide(np.sum(click_mat, axis=0), np.sum(click_mat))))
		click_var_outcome.append(np.var(np.divide(np.sum(click_mat, axis=1), np.sum(click_mat))))

		row_order = np.flip(np.argsort(weights))
		col_order = [list(np.where([c in i for i in same_col_idx]))[0][0] for c in clicks]
		idx = np.unique(col_order, return_index=True)[1]
		col_order_clickSeq = [col_order[i] for i in sorted(idx)]
		col_order_prob = np.flip(np.argsort(np.matmul(weights,click_mat)))
		col_order_EV = np.flip(np.argsort(EV))
		idx = C[row_order]
		click_mat_transformedByProb = get_click_mat(idx[:,col_order_prob], clicks)
		click_embedding_seq.append(np.ndarray.flatten(get_click_mat(idx[:,col_order_clickSeq], clicks)))
		click_embedding_prob.append(np.ndarray.flatten(click_mat_transformedByProb))
		click_embedding_EV.append(np.ndarray.flatten(get_click_mat(idx[:,col_order_EV], clicks)))

		ttb_row = np.argmax(weights)
		payoffs = [p for pr in payoff_matrix for p in pr]
		if len(clicks)==0:
		    strategy.append('Rand')
		elif len(clicks)>=19:
		    strategy.append('WADD')
		elif set(clicks)==set(C[ttb_row,:]):
		    strategy.append('TTB')
		elif (set(clicks).issubset(set(C[ttb_row,:]))) & (np.argmax([payoffs[i] for i in clicks])==(len(clicks)-1)):
		    strategy.append('SAT_TTB')
		elif (all(np.diff(np.sum(click_mat_transformedByProb,axis=0))<=0)) & (all(np.diff(np.sum(click_mat_transformedByProb,axis=1))<=0)): # remove second condition? o/w -> "...but never more cells from a less probable row than from a more probable row" + " and never more cells from a column with less total probability of revealed cells than more"
		    strategy.append('TTB_SAT')
		else:
		    strategy.append('Other')

		type1_transitions = 0
		type2_transitions = 0
		for c in range(len(clicks)-1):
			for col in same_col_idx:
				if (clicks[c] in col) & (clicks[c+1] in col):
					type1_transitions += 1
			for row in same_row_idx:
				if (clicks[c] in row) & (clicks[c+1] in row):
					type2_transitions += 1
		processing_pattern.append(np.divide((type1_transitions-type2_transitions), (type1_transitions+type2_transitions)))
		processing_type1.append(type1_transitions)
		processing_type2.append(type2_transitions)

	df['nr_clicks'] = nr_clicks
	df['click_var_gamble'] = click_var_gamble
	df['click_var_outcome'] = click_var_outcome
	df['processing_pattern'] = processing_pattern
	df['processing_type1'] = processing_type1
	df['processing_type2'] = processing_type2
	df['EVs'] = EVs
	df['EV_chosen'] = EV_chosen
	df['EV_max'] = EV_max
	df['click_embedding_seq'] = click_embedding_seq
	df['click_embedding_prob'] = click_embedding_prob
	df['click_embedding_EV'] = click_embedding_EV
	df['strategy'] = strategy
	for s in ['Rand','WADD','TTB','SAT_TTB','TTB_SAT','Other']:
		df[s] = df.strategy == s
	if not(isHuman):
		df['net_payoff'] = net_payoff
	df['payoff_gross'] = payoff_gross
	df['payoff_expected_gross'] = payoff_expected_gross
	df['payoff_rand'] = payoff_rand
	df['payoff_perfect'] = payoff_perfect
	df['payoff_relative_expected'] = payoff_relative_expected
	df['payoff_relative'] = payoff_relative
	df['payoff_relative_old'] = payoff_relative_old
	df['payoff_gross_bestBet'] = payoff_gross_bestBet
	df['payoff_relative_bestBet'] = payoff_relative_bestBet

	return df



def process_data(isHuman=True, in_file='', out_dir='', two_groups=False):
	
	if isHuman:
		trials = pd.read_csv(in_file)
		trials = trials[trials.block=='test']
		trials.probabilities = trials.probabilities.apply(ast.literal_eval)
		trials.clicks = trials.clicks.apply(ast.literal_eval)
		trials.payoff_matrix = trials.payoff_matrix.apply(ast.literal_eval)
		trials.click_times = trials.click_times.apply(ast.literal_eval)
		trials = append_features(trials, isHuman=True)
	else:
		if os.path.isdir(in_file):
			filenames = [in_file + s for s in os.listdir(in_file)]
			nr_processes = multiprocessing.cpu_count()
			with multiprocessing.Pool(processes=nr_processes) as pool:
				results = pool.map(append_features, filenames)
			trials = pd.DataFrame()
			for r in range(len(results)):
			    trials = trials.append(results[r])
		else:
			trials = append_features(in_file)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	s = ''
	if not(isHuman):
		s = '_model'

	if two_groups:
		trials1 = trials[trials.display_ev==True]
		trials2 = trials[trials.display_ev==False]
		trials1.groupby(['sigma','alpha','cost']).mean().to_pickle(out_dir+'mean_by_condition_EVminTime'+s+'.pkl')
		trials1.to_pickle(out_dir+'trials_EVminTime'+s+'.pkl')
		trials2.groupby(['sigma','alpha','cost']).mean().to_pickle(out_dir+'mean_by_condition'+s+'.pkl')
		trials2.to_pickle(out_dir+'trials'+s+'.pkl')
	else:
		mean_by_condition = trials.groupby(['sigma','alpha','cost']).mean()
		mean_by_condition.to_pickle(out_dir+'mean_by_condition'+s+'.pkl',protocol=4)
		trials.to_pickle(out_dir+'trials'+s+'.pkl',protocol=4)
		mean_by_condition.to_csv(out_dir+'mean_by_condition'+s+'.csv')
		trials.to_csv(out_dir+'trials'+s+'.csv')
		trials_short = trials[['problem_id','cost','strategy']]
		trials_short.to_pickle(out_dir+'trials_short'+s+'.pkl',protocol=4)
		trials_short.to_csv(out_dir+'trials_short'+s+'.csv')



def run_kmeans(in_file='', k_range=[4], df=[]):

	import pickle
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score

	out_file = in_file.replace('trials','kmeans')

	if len(df)==0:
		df = pd.read_pickle(in_file)
	kmean_strategies = ['WADD','TTB_SAT','TTB','SAT_TTB','Rand'] # sorted by most to least clicks

	K_means = dict()
	for embed in ['click_embedding_prob']: #['click_embedding_seq','click_embedding_prob','click_embedding_EV']:
		X = [df[embed].iloc[i] for i in range(len(df))]

		sse = []
		silhouette = []
		Kmeans = []
		list_k = list(k_range)
		for k in list_k:
			kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
			Kmeans.append(kmeans)
		    # silhouette.append(silhouette_score(X, kmeans.labels_, sample_size=10000))
		    
		    # make strategy labels using cluster labels
			if (k <= len(kmean_strategies)):
				col_str = ['strategy_'+embed+'_k'+str(k)][0]
				df[col_str] = ''
				idx_s = np.flip(np.argsort(np.sum(kmeans.cluster_centers_,axis=1)))
				for st in range(len(idx_s)):
					idx_n = kmeans.labels_ == idx_s[st]
					df[col_str].iloc[idx_n] = kmean_strategies[st]
				for st in ['Rand','WADD','TTB','SAT_TTB','TTB_SAT','Other']:
					df[[st+'_'+col_str][0]] = df[col_str] == st
				df[['mkeansLabels_'+col_str][0]] = kmeans.labels_

		K_means[embed] = dict()
		K_means[embed]['kmeans'] = Kmeans
		# K_means[embed]['silhouette'] = silhouette

	pickle.dump(K_means, open(out_file, 'wb'))
	df.to_pickle(in_file)#, protocol=4)


	mean_by_condition = df.groupby(['sigma','alpha','cost']).mean()
	out_file = in_file.replace('trials','mean_by_condition')
	mean_by_condition.to_pickle(out_file)



def get_means_by_cond(df, cond=[], param=['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']):

	'''this function assumes df is ordered using df = trials.groupby(['sigma','alpha','cost']).mean() '''
	
	means_by_cond = []

	for s in param:
		if not(any(df.keys()==s)):
			means_by_cond.append(np.zeros(len(means_by_cond[0])))
		else:
			if (len(df[s])<50):
				idx = pd.MultiIndex.from_product([[75,150], [.1,10**(-.5),1,10**(.5),10], [0,1,2,4,8]], names=['sigma','alpha','cost'])
				tmp = pd.Series(np.zeros(50), index=idx)
				tmp.iloc[[6,8,16,18]] = df[s].values
				tmp = np.reshape(np.array(tmp),(2,5,5))
			else:
				tmp = np.reshape(np.array(df[s]),(2,5,5))
			if (cond=='sigma'):
				means_by_cond.append([np.mean(tmp[i,:,:]) for i in range(tmp.shape[0])])
			elif (cond=='alpha'):
				means_by_cond.append([np.mean(tmp[:,i,:]) for i in range(tmp.shape[1])])
			elif (cond=='cost'):
				means_by_cond.append([np.mean(tmp[:,:,i]) for i in range(tmp.shape[2])])
			else:
				means_by_cond = np.transpose(tmp)

	return np.transpose(means_by_cond)



def get_sem_by_cond(df, cond, param, perfect_performance=[]):

	from scipy.stats import sem

	sem_by_cond = []
	dat = []

	conds = np.sort(df[cond].unique())
	for i, cc in enumerate(conds):
		dd = df[df[cond]==cc]
		if perfect_performance==[]:
			y = dd.groupby('pid').mean()[param].values
		else:
			y = dd.groupby('pid').mean()[param].values / perfect_performance[i]
		dat.append(y)
		sem_by_cond.append(sem(y))

	return sem_by_cond, dat



def calc_centroid_similarity():

	from sklearn.metrics.pairwise import cosine_similarity
	import random
	import pickle
	from sklearn.utils import shuffle
	from scipy.stats import pearsonr

	kmeans = pickle.load(open('kmeans.pkl', 'rb'))
	km = kmeans['click_embedding_prob']['kmeans'][0]
	kmeans = pickle.load(open('kmeans_model_small.pkl', 'rb')) # 'kmeans_model.pkl'
	km_mod = kmeans['click_embedding_prob']['kmeans'][0]
	print('using ',km.n_clusters,' participant clusters, ',km_mod.n_clusters,' model clusters')

	sp = [0,1,4,3]
	for i in range(km_mod.n_clusters):
		r = pearsonr(km.cluster_centers_[sp[i]], km_mod.cluster_centers_[i])
		v1 = km.cluster_centers_[sp[i]].reshape(1, -1)
		v2 = km_mod.cluster_centers_[i].reshape(1, -1)
		css = cosine_similarity(v1,v2)
		v1, v2 = shuffle(v1, v2)
		css2 = cosine_similarity(v1,v2)
		bootstrap = []
		for j in range(1000):
		    random.shuffle(v1[0])
		    random.shuffle(v2[0])
		    bootstrap.append(cosine_similarity(v1,v2))
		print('CSS: ',css[0][0],' significance: ',np.mean([css > i for i in bootstrap]),'\n CSS shuffled: ',css2[0][0],'\n Pearson: ',r)



def calc_cluster_strategies():

	df = pd.read_pickle('trials.pkl')
	df_mod = pd.read_pickle('trials_model_small.pkl')

	kmeans = pickle.load(open('kmeans.pkl', 'rb'))
	km = kmeans['click_embedding_prob']['kmeans'][0]
	kmeans = pickle.load(open('kmeans_model_small.pkl', 'rb')) # 'kmeans_model.pkl'
	km_mod = kmeans['click_embedding_prob']['kmeans'][0]

	df['kmeans5_labels'] = km.labels_
	df_mod['kmeans4_labels'] = km_mod.labels_

	sp = [0,1,4,3,2]
	most_freq_strategy = ['TTB_SAT','SAT_TTB','WADD','TTB','Rand','Other']
	for i in range(4):
	    print('CLUSTER ',i)
	    for s in most_freq_strategy:
	        tmp = df[df.kmeans5_labels==sp[i]].strategy
	        strat_freq = [j==s for j in tmp]
	        tmp = df_mod[df_mod.kmeans4_labels==i].strategy
	        strat_freq_mod = [j==s for j in tmp]
	        print(s, np.mean(strat_freq), np.mean(strat_freq_mod))



def get_3d_correlations(df, df_mod, param):

	a = np.reshape(np.array(df[param]),(2,5,5))
	b = np.reshape(np.array(df_mod[param]),(2,5,5))
	dimensions = ['sigma','alpha','cost']

	for d in range(3):
		az = a - a.mean(axis=d, keepdims=True)
		bz = b - b.mean(axis=d, keepdims=True)
		a2 = az**2;
		b2 = bz**2;
		ab = az * bz;
		r = np.sum(ab, axis=d) / np.sqrt(np.sum(a2, axis=d) * np.sum(b2, axis=d));
		print(param,' Correlation along ',dimensions[d],r)



def remove_participants(in_file='data/human/1.0/trials.pkl', out_dir='data/human/1.0/'):

	trials = pd.read_pickle(in_file)
	df = trials[['pid','nr_clicks']].groupby('pid').mean()
	pids = df[df.nr_clicks > 0].index
	df = trials[trials['pid'].isin(pids)]
	mean_by_condition = df.groupby(['sigma','alpha','cost']).mean()
	mean_by_condition.to_pickle(out_dir+'mean_by_condition_noRandom.pkl')
	df.to_pickle(out_dir+'trials_noRandom.pkl')



def calc_confusion_mat(df, df_mod):

	import time
	t=time.time()

	df['payoff_relative'] = df['payoff_gross'] / df['payoff_perfect']
	df['payoff_relative_bestBet'] = df['payoff_gross_bestBet'] / df['payoff_perfect']
	df_mod['payoff_relative'] = df_mod['payoff_gross'] / df_mod['payoff_perfect']

	strategies = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']
	confusion_mat = np.zeros((len(strategies),len(strategies)))
	points_lost = np.zeros((len(strategies),len(strategies)))
	relative_perf_lost = np.zeros((len(strategies),len(strategies)))
	count_tot = np.zeros((len(strategies),len(strategies)))
	kappa_list = []; kappa_list_m = []
	for prob_id in df.problem_id.unique():
	    dat_ = df[(df.problem_id==prob_id)]
	    dat_m_ = df_mod[(df_mod.problem_id==prob_id)]
	    for cost in dat_.cost.unique():
	        dat = dat_[(dat_.cost==cost)]
	        dat_m = dat_m_[(dat_m_.cost==cost)]
	        kappa_list_m.extend(np.resize(dat_m.strategy.values, len(dat)))
	        kappa_list.extend(dat.strategy.values)
	        count = []; count_m = []
	        points = []; points_m = []
	        rel_perf = []; rel_perf_m = []
	        for s in strategies:
	            idx = dat.strategy==s
	            idx_m = dat_m.strategy==s
	            count.append(sum(idx))
	            count_m.append(sum(idx_m)/len(dat_m))
	            points.append(dat[idx].net_payoff.mean())
	            points_m.append(dat_m[idx_m].net_payoff.mean())
	            rel_perf.append(dat[idx].payoff_relative_bestBet.mean())
	            rel_perf_m.append(dat_m[idx_m].payoff_relative.mean())
	        for i in range(len(strategies)):
	            for j in range(len(strategies)):
	                confusion_mat[i,j] += count_m[i]*count[j]
	                points_lost[i,j] += np.nansum([points[j],-1*points_m[i]])*count[j]*count_m[i] # ! you can't count points if model (or human) didn't use strategy!
	                relative_perf_lost[i,j] += np.nansum([rel_perf[j],-1*rel_perf_m[i]])*count[j]*count_m[i]
	                count_tot[i,j] += count[j]*count_m[i]
	kappa_lists = [kappa_list, kappa_list_m]              

	print('completed in ', round((time.time()-t)/60,1), 'min.')

	return confusion_mat, points_lost, relative_perf_lost, count_tot, kappa_lists



def run_mds(df=[]):

	from sklearn.manifold import MDS
	import time

	if df == []:
		df = pd.read_pickle('data/human/1.0/trials.pkl')

	t=time.time()
	X = np.stack(df['click_embedding_prob'].values)
	mds = MDS(n_components=2, metric=True, eps=1e-9, dissimilarity="euclidean")
	mds = mds.fit(X)
	points = mds.embedding_
	
	print('time: ',time.time()-t)

	return mds


def run_tsne(isHuman=False, isLocal=False):

	import numpy as np
	import pandas as pd
	from sklearn import manifold
	import pickle
	import time
	import csv

	if isHuman:
		if isLocal:
			df = pd.read_pickle('data/human/1.0/trials.pkl')
			km = pickle.load(open('data/human/1.0/kmeans.pkl', 'rb'))
		else:
			df = pd.read_pickle('trials.pkl')
			km = pickle.load(open('kmeans.pkl', 'rb'))
	else:
		if isLocal:
			df = pd.read_pickle('data/model/no_implicit_cost/trials_model.pkl')
			km = pickle.load(open('data/model/no_implicit_cost/kmeans_model.pkl', 'rb'))
		else:
			df = pd.read_pickle('trials_model.pkl')
			df = df.iloc[::10, :]
			km = pickle.load(open('kmeans_model.pkl', 'rb'))

	data = {"Y":[],"perplexity":[],"learning_rate":[]}

	t = time.time()
	for p in [1]:#[1,5,15]:
		# for lr in [10, 50, 100, 500, 1000, 'auto']:

		print(p, time.time()-t)
		X = np.stack(df['click_embedding_prob'].values)
		km = km['click_embedding_prob']['kmeans'][0].cluster_centers_
		for k in [.5]:#[.1,.2,.3,.4,.5,.6,.7,.8,.9]:
			km_bin = km >= k
			X = np.append(X,km_bin.astype(float),axis=0)
		tsne = manifold.TSNE(n_components=2, init="random", random_state=0, perplexity=p, n_jobs=8)#, learning_rate=lr)

		Y = tsne.fit_transform(X)
		data["Y"].append(Y)
		data["perplexity"].append(p)
		data["learning_rate"].append('default')

		print(time.time()-t)

	with open('tmp.pkl', 'wb') as f:
		pickle.dump(data, f, 4)



def calc_confusion_mat_strategyVsKmeans():

	df = pd.read_pickle('data/human/1.0/trials.pkl')

	strategies = ['SAT_TTB','TTB_SAT','TTB','WADD','Rand','Other']
	            
	confusion_mat = np.zeros((len(strategies),len(strategies)))

	for i, s in enumerate(strategies):
	    idx = df.strategy == s
	    df_ = df[df.strategy == s]
	    for j, s_ in enumerate(strategies):
	        idx = df_.strategy_click_embedding_prob_k5 == s

























