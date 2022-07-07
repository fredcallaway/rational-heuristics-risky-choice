import numpy as np
import pandas as pd
import process_data as p_d
import os
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats import multitest
from itertools import combinations
from statsmodels.stats.proportion import proportion_effectsize
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def under_performance(human_file, stats_dir, exclude_participants=False):

	exclude_str = '' if not exclude_participants else '_exclude'
	dat = eval(pd.read_csv(human_file, columns=['under_performance'+exclude_str]).iloc[0])[0]
	exp2_str = human_file[-8:-4] if 'exp2' in stats_dir else ''

	# model clicks - human clicks
	x = dat['nr_clicks_dif']
	with open(stats_dir+'perf-reduc_nr_clicks'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}$')

	# overall perforamcne gap in units of gross reward
	x = dat['peformance_gap_points']
	with open(stats_dir+'perf-reduc_points'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}$')

	# overall perforamcne gap in units of gross relative reward
	x = dat['peformance_gap_gross_abs']
	with open(stats_dir+'perf-reduc_overall_gross-abs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in percentage of model gross relative reward
	x = dat['peformance_gap_gross_pct']
	with open(stats_dir+'perf-reduc_overall_gross-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}\\%$')

	# overall perforamcne in percentage of model net relative reward
	x = dat['human_performance_pct']
	with open(stats_dir+'perf-overall-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in units of net relative reward
	x = dat['peformance_gap_abs']
	with open(stats_dir+'perf-reduc_overall-abs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in percentage of model net relative reward
	x = dat['peformance_gap_pct']
	with open(stats_dir+'perf-reduc_overall-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}\\%$')

	# reduction in performance from implicit costs, as a fraction of model performance
	x = dat['implicit_costs_model_fraction']
	with open(stats_dir+'perf-reduc_implicit-costs_model'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from implicit costs, as a fraction of model-human performance gap
	x = dat['implicit_costs']
	with open(stats_dir+'perf-reduc_implicit-costs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect use of information, as a fraction of model-human performance gap
	x = dat['imperfect_info_use']
	with open(stats_dir+'perf-reduc_info-use'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect strategy selection, as a fraction of model-human performance gap
	x = dat['imperfect_strat_selec']
	with open(stats_dir+'perf-reduc_strat-selec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# fraction of previous value from random gambling
	x = dat['imperfect_strat_selec_by_strat'][4] / dat['imperfect_strat_selec']
	with open(stats_dir+'perf-reduc_strat-selec-from-rand'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# and random gambling as fraction of total under-performance
	x = dat['imperfect_strat_selec_by_strat'][4]
	with open(stats_dir+'perf-reduc_strat-selec-rand'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect strategy selection, as a fraction of model-human performance gap
	x = dat['imperfect_strat_exec']
	with open(stats_dir+'perf-reduc_strat-exec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	# reduction in performance from imperfect strategy selection and execution, as a fraction of model-human performance gap
	x = dat['imperfect_strat_selec_and_exec']
	with open(stats_dir+'perf-reduc_strat-selec-exec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	# reduction in performance from imperfect strategy selection and execution, as a fraction of model-human performance gap
	x = dat['imperfect_info_use_points_lost']
	with open(stats_dir+'perf-reduc_info-use-pointsPerTrial'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	print('saved under-performance files to '+stats_dir+'perf-reduc_....txt')


def strategy_postHoc_cohenD(human_file, stats_dir):

	df = pd.read_csv(human_file)

	if not os.path.exists(stats_dir): os.makedirs(stats_dir)

	strategies = ['SAT_TTB','TTB_SAT','Other','TTB','Rand','WADD']

	sigmas = np.sort(df.sigma.unique())
	alphas = np.flip(np.sort(df.alpha.unique()))
	costs = np.sort(df.cost.unique())

	for g in strategies:
		x = proportion_effectsize(np.mean(df[df['sigma']==sigmas[0]][g]), \
			np.mean(df[df['sigma']==sigmas[1]][g]))
		with open(stats_dir+g+'-sigma_postHoc-cohenD.txt', 'w') as f:
			f.write('n/a & '+f'${x:.{2}}$')
	
	for g in strategies:
		p_vals = []
		for i in list(combinations([1,2,3,4],2)):
			chi_table = [[sum(df[df['alpha']==alphas[i[0]]][g].values==True), \
			sum(df[df['alpha']==alphas[i[0]]][g].values==False)], \
			[sum(df[df['alpha']==alphas[i[1]]][g].values==True), \
			sum(df[df['alpha']==alphas[i[1]]][g].values==False)]]
			ch, p, _, _ = chi2_contingency(chi_table)
			p_vals.append(p)
		reject_list, corrected_p_vals, _, _ = multipletests(p_vals, method='fdr_bh', alpha=0.05)
		if all(reject_list):
			x = 'all pairs & $'
		else:
			x = ''
			for pair in list(combinations([0,1,2,3,4],2)):
				x += str(pair[0])+' \& '+str(pair[1])+', '
			x = x[:-2]+' & $'
		for i in range(4):
			res = proportion_effectsize(np.mean(df[df['alpha']==alphas[i]][g]), \
				np.mean(df[df['alpha']==alphas[i+1]][g]))
			x += f'{res:.{2}}, '
		x = x[:-2]+'$'
		with open(stats_dir+g+'-alpha_postHoc-cohenD.txt', 'w') as f:
			f.write(x)

	for g in strategies:
		p_vals = []
		for i in list(combinations([1,2,3,4],2)):
			chi_table = [[sum(df[df['cost']==costs[i[0]]][g].values==True), \
			sum(df[df['cost']==costs[i[0]]][g].values==False)], \
			[sum(df[df['cost']==costs[i[1]]][g].values==True), \
			sum(df[df['cost']==costs[i[1]]][g].values==False)]]
			ch, p, _, _ = chi2_contingency(chi_table)
			p_vals.append(p)
		reject_list, corrected_p_vals, _, _ = multipletests(p_vals, method='fdr_bh', alpha=0.05)
		if all(reject_list):
			x = 'all pairs & $'
		else:
			x = ''
			for pair in list(combinations([0,1,2,3,4],2)):
				x += str(pair[0])+' \& '+str(pair[1])+', '
			x = x[:-2]+' & $'
		for i in range(4):
			res = proportion_effectsize(np.mean(df[df['cost']==costs[i]][g]), \
				np.mean(df[df['cost']==costs[i+1]][g]))
			x += f'{res:.{2}}, '
		x = x[:-2]+'$'
		with open(stats_dir+g+'-cost_postHoc-cohenD.txt', 'w') as f:
			f.write(x)


	print('saved stats files to '+stats_dir+' post-hoc_cohen-d_....txt')

	filenames = [stats_dir + f for f in[\
				'SAT_TTB-sigma_postHoc-cohenD.txt',\
			  	'TTB_SAT-sigma_postHoc-cohenD.txt',\
			  	'TTB-alpha_postHoc-cohenD.txt',\
			  	'Rand-alpha_postHoc-cohenD.txt',\
			  	'TTB_SAT-cost_postHoc-cohenD.txt',\
			  	'TTB-cost_postHoc-cohenD.txt',\
			  	'SAT_TTB-cost_postHoc-cohenD.txt']]

	table_string = \
	'\\begin{tabular}{llccc}\n'+\
	'Strategy frequency & \\begin{tabular}{@{}c@{}}Independent\\\\variable\\end{tabular} & \\begin{tabular}{@{}c@{}}'+\
	'significant\\\\post-hoc comparisons\\end{tabular} & \\begin{tabular}{@{}c@{}}effect sizes (Cohen\'s $d$)\\end{tabular}\\\\\n'+\
	'\\midrule\n'+\
	'SAT-TTB & stakes & '+open(filenames[0],'r').read()+'\\\\\n'+\
	'SAT-TTB+ & stakes & '+open(filenames[1],'r').read()+'\\\\\n'+\
	'TTB & dispersion & '+open(filenames[2],'r').read()+'\\\\\n'+\
	'random & dispersion & '+open(filenames[3],'r').read()+'\\\\\n'+\
	'SAT-TTB+ & cost & '+open(filenames[4],'r').read()+'\\\\\n'+\
	'TTB & cost & '+open(filenames[5],'r').read()+'\\\\\n'+\
	'SAT-TTB & cost & '+open(filenames[6],'r').read()+'\\\\\n'+\
	'\\bottomrule\n'+\
	'\\end{tabular}'

	with open(stats_dir+'table_strategies.tex', 'w') as f:
		f.write(table_string)

	print('saved latex table to '+stats_dir+'table_strategies.tex')


def behavioral_features(human_file, stats_dir, print_summary=True):

	for exclude in [False, True]:
		df = pd.read_csv(human_file)
		if exclude:
			df = p_d.remove_rand_participants(df)
		exclude_str = '_exclude' if exclude else ''

		cond_str = {'alpha': ['10^{-1.0}','10^{-0.5}','10^{.0}','10^{0.5}','10^{1.0}'],
					'cost': ['0','1','2','4','8']}

		df['alpha'] = 1 / df['alpha'] # dispersion is alpha^-1
		# make sigma and alpha levels unit and linear
		for i, s in enumerate(df['sigma'].sort_values().unique()):
			df[df['sigma']==s]['sigma'] = i
		for i, a in enumerate(df['alpha'].sort_values().unique()):
			df[df['alpha']==a]['alpha'] = i
		for p in ['nr_clicks','payoff_gross_relative','processing_pattern','click_var_gamble','click_var_outcome']:
			for c in ['sigma','alpha','cost']:
				
				# normalize for standardized regression coefficients
				df_z = df.dropna(subset=[p]); df_z[p] = (df_z[p] - df_z[p].mean())/df_z[p].std(ddof=0)
				
				# mixed-effects linear regression of behavioral features on environmental parameters
				res = smf.mixedlm(p+'~'+c, df_z, groups=df_z['pid']).fit()
				lm_str1, lm_str2 = f'$B={res.params[1]:.{2}}, p', f'= {res.pvalues[1]:.{2}}$' if res.pvalues[1]>=0.001 else '< 0.001$'
				
				dat = [df_z[df_z[c]==i].groupby('pid').mean()[p].values for i in np.sort(df_z[c].unique())]
			
				# main effects, post-hoc comparisons, and effect sizes
				if len(dat)==2:
					F, P = ttest_ind(dat[0],dat[1])
					dof = len(dat[0]) + len(dat[1]) - 2
					me_str1, me_str2 = f'$t$({dof}) = {F:.2f},\\\\$p$ ',f'= {P:.2}' if P>=0.001 else '<0.001'
					ph_str = 'n/a'
				else:
					F, P = f_oneway(dat[0],dat[1],dat[2],dat[3],dat[4])
					dof = (4, sum([len(dat[i]) for i in range(len(dat))]) - 5)
					me_str1, me_str2 = f'$F$({dof[0]},{dof[1]}) = {F:.2f},\\\\$p$ ',f'= {P:.2}' if P>=0.001 else '<0.001'
					if P < 0.1:
						tukey = pairwise_tukeyhsd(\
								endog = np.concatenate(dat),
								groups = np.concatenate([np.repeat(i,repeats=len(dat[i])) for i in range(len(dat))]),
								alpha = 0.05)
						pairwise_sig = [x[-1] for x in tukey.summary().data[1:] if abs(x[0]-x[1])==1]
						if all(pairwise_sig):
							ph_str = 'all pairs'
						elif all([not b for b in pairwise_sig]):
							ph_str = 'n/a'
						else:
							ph_str = '$'+', '.join([cond_str[c][i]+'\&'+cond_str[c][i+1] for i in range(len(pairwise_sig)) if pairwise_sig[i]])+'$'
					else:
						tukey = ''
						ph_str = 'n/a'
						
				cd_str = '$'+', '.join([f'{p_d.cohen_d(dat[i],dat[i+1]):.2}' for i in range(len(dat)-1)])+'$'
				
				if not os.path.exists(stats_dir): os.makedirs(stats_dir)

				with open(stats_dir+p+exclude_str+'-'+c+'_mixedlm.txt', 'w') as f:
					f.write(lm_str1+lm_str2)
					
				with open(stats_dir+p+exclude_str+'-'+c+'_mainEffect.txt', 'w') as f:
					f.write(me_str1+me_str2)
					
				with open(stats_dir+p+exclude_str+'-'+c+'_postHoc.txt', 'w') as f:
					f.write(ph_str)
					
				with open(stats_dir+p+exclude_str+'-'+c+'_cohenD.txt', 'w') as f:
					f.write(cd_str)
						
				if print_summary:
					p_d.print_special('IV: '+p+', DV: '+c)
					print(res.summary())
					print('main effect:\n',me_str1.replace('$','').replace('\\','\n'), me_str2.replace('$',''),'\n')
					if len(dat) > 2: print(tukey,'\n')
					print('effect sizes (Cohen\'s d):\n',cd_str.replace('$',''),'\n'*3)

	print('saved .txt stats files with regression results, main effects, post-hoc comparisons, and effect sizes to '+stats_dir)

	# filenames = [stats_dir+''.join(x) for x in zip(\
	# 											['nr_clicks-']*9+\
	# 											['processing_pattern-']*9+\
	# 											['click_var_outcome-']*9+\
	# 											['click_var_gamble-']*9+\
	# 											['payoff_gross_relative-']*9, \
	# 											np.tile(['sigma_']*3+['alpha_']*3+['cost_']*3,5).tolist(), \
	# 											np.tile(['mainEffect.txt']+['postHoc.txt']+['cohenD.txt'],15).tolist())\
	# 			]

	# table_string = \
	# '\\begin{tabular}{llccc}\n'+\
	# 'Dependent variable & \\begin{tabular}{@{}c@{}}Independent\\\\variable\\end{tabular}& '+\
	# 'main effect & significant post-hoc comparisons & effect sizes (Cohen\'s $d$)\\\\\n'+\
	# '\\midrule\n'+\
	# '\\\\\n'.join([''.join(x) for x in zip(\
	# 									['Information gathering']*3 + \
	# 									['Processing pattern']*3 + \
	# 									['Attribute variance']*3 + \
	# 									['Alternative variance']*3 + \
	# 									['Relative performance']*3, \
	# 									[' & ']*15,\
	# 									['stakes','dispersion','cost']*5, \
	# 									[' & ']*15,\
	# 									[' \\begin{tabular}{@{}c@{}} '+open(filenames[i],'r').read()+' \\end{tabular} ' for i in range(0,len(filenames),3)],\
	# 									[' & ']*15,\
	# 									[open(filenames[i],'r').read() for i in range(1,len(filenames),3)],\
	# 									[' & ']*15,\
	# 									[open(filenames[i],'r').read() for i in range(2,len(filenames),3)])\
	# 				])+\
	# '\\\\\n'+\
	# '\\bottomrule'+\
	# '\\end{tabular}'

	def make_latex_table(varnames, tablenames, exclude=None):
		varnames = [x+'-' for y in [[vn]*9 for vn in varnames] for x in y]
		n_vars = len(varnames)
		filenames = [stats_dir+''.join(x) for x in zip(varnames, \
													np.tile(['sigma_']*3+['alpha_']*3+['cost_']*3,n_vars).tolist(), \
													np.tile(['mainEffect.txt']+['postHoc.txt']+['cohenD.txt'],3*n_vars).tolist())\
					]

		exclude = [False]*n_vars if not exclude else exclude
		tablenames = [[tn]*3 if not exclude[i] else ['\\begin{tabular}{@{}c@{}}'+tn+'\\\\(Participants excluded)'+' \\end{tabular} ']*3 for i,tn in enumerate(tablenames)]
		tablenames = [x for y in tablenames for x in y]
		n_rows = len(tablenames)
		table_string = \
		'\\begin{tabular}{llccc}\n'+ \
		'Dependent variable & \\begin{tabular}{@{}c@{}}Independent\\\\variable\\end{tabular}& '+ \
		'main effect & significant post-hoc comparisons & effect sizes (Cohen\'s $d$)\\\\\n'+ \
		'\\midrule\n'+ \
		'\\\\\n'.join([''.join(x) for x in zip(tablenames, \
											[' & ']*n_rows, \
											['stakes','dispersion','cost']*n_vars, \
											[' & ']*n_rows, \
											['\\begin{tabular}{@{}c@{}} '+open(filenames[i],'r').read()+' \\end{tabular} ' for i in range(0,len(filenames),3)],\
											[' & ']*n_rows, \
											[open(filenames[i],'r').read() for i in range(1,len(filenames),3)],\
											[' & ']*n_rows, \
											[open(filenames[i],'r').read() for i in range(2,len(filenames),3)])\
						])+\
		'\\\\\n'+\
		'\\bottomrule'+\
		'\\end{tabular}'
		return table_string

	table = make_latex_table(['nr_clicks','processing_pattern','click_var_outcome','click_var_gamble'], \
							['Information gathering','Alternative vs. Attribute','Attribute variance','Alternative variance'])
	table_perf = make_latex_table(['payoff_gross_relative','payoff_gross_relative_exclude'], \
								['Relative performance','Relative performance'],\
								exclude = [False, True])

	with open(stats_dir+'table_behavior.tex', 'w') as f:
		f.write(table)
	print('saved latex table to '+stats_dir+'table_behavior.tex')

	with open(stats_dir+'table_performance.tex', 'w') as f:
		f.write(table_perf)
	print('saved latex table to '+stats_dir+'table_performance.tex')


def participant_demographics(human_file):

	participants = pd.read_csv(human_file)

	print('Participants: ',len(participants),
		 '\nFemales: ',sum(participants.gender=='female'),
		 '\nAge ',np.mean(participants.age),' +/- ',np.std(participants.age),' ',min(participants.age),'-',max(participants.age),
		 '\nBonus: ',np.mean(participants.bonus),' +/- ',np.std(participants.bonus),' ',min(participants.bonus),'-',max(participants.bonus),
		 '\nExperiment length (minutes): ',np.mean(participants.total_time)/60000,' +/- ',np.std(participants.total_time)/60000,' ',min(participants.total_time)/60000,'-',max(participants.total_time)/60000)


def exp2_strategy(human_file1, human_file2, stats_dir, print_summary=True):

	df1 = pd.read_csv(human_file1)
	df2 = pd.read_csv(human_file2)

	alphas = np.flip(np.sort(df1['alpha'].unique()))
	costs = np.sort(df1['cost'].unique())

	dof1, dof2 = 1, min(len(df1),len(df2))
	for s in ['TTB_SAT','SAT_TTB','TTB','WADD']:
	    for i, a in enumerate(alphas):
	        for j, c in enumerate(costs):
	            chi_table = [
	                        [sum(df1[(df1['alpha']==a)&(df1['cost']==c)][s]==True),\
	                            sum(df1[(df1['alpha']==a)&(df1['cost']==c)][s]==False)],\
	                        [sum(df2[(df2['alpha']==a)&(df2['cost']==c)][s]==True),\
	                            sum(df2[(df2['alpha']==a)&(df2['cost']==c)][s]==False)]\
	                        ]
	            chi, p, _, _ = chi2_contingency(chi_table)

	            d = proportion_effectsize(df1[df1['cost']==c][s].mean(),\
	                                      df2[df2['cost']==c][s].mean())

	            p_str = f'={p:.2}' if p >= 0.001 else '<0.001'
	            latex_str = f'$\chi^2({dof1},{dof2})={chi:.1f}, p{p_str}, d={d:.2f}$'
	            with open(f'{stats_dir}chi2-{s}-alpha{i}-cost{c}.txt', 'w') as f:
	                f.write(latex_str)

	            if print_summary:
	                print(f'{s}-alpha{i}-cost{c}')
	                print(latex_str.replace('$','').replace('\\',''))


def exp2_behavior(human_file1, human_file2, stats_dir, print_summary=True):

	df1 = pd.read_csv(human_file1)
	df2 = pd.read_csv(human_file2)

	alphas = np.flip(np.sort(df1['alpha'].unique()))
	costs = np.sort(df1['cost'].unique())

	for s in ['processing_pattern','click_var_outcome','click_var_gamble','nr_clicks','payoff_net_relative']:
	    for i, a in enumerate(alphas):
	        for j, c in enumerate(costs):
	            df1_ = df1.dropna(subset=[s])
	            df2_ = df2.dropna(subset=[s])
	            
	            x1 = df1_[(df1_['alpha']==a)&(df1_['cost']==c)].groupby('pid').mean()[s]
	            x2 = df2_[(df2_['alpha']==a)&(df2_['cost']==c)].groupby('pid').mean()[s]
	            
	            t, p = ttest_ind(x1, x2)
	            d = p_d.cohen_d(x1, x2)
	                
	            dof = len(x1) + len(x2) - 2
	            p_str = f'={p:.2}' if p >= 0.001 else '<0.001'
	            # table formatting vs. in-text
	            sep = '$ & $' if s in ['processing_pattern','click_var_outcome','click_var_gamble'] else ', '
	            latex_str = f'$t({dof})={t:.2f}{sep}p{p_str}{sep}d={d:.2f}$'
	            with open(f'{stats_dir}ttest-{s}-alpha{i}-cost{c}.txt', 'w') as f:
	                f.write(latex_str)
	                
	            if print_summary:
	                print(f'{s}-alpha{i}-cost{c}')
	                print(latex_str.replace('$','').replace('\\','').replace(' &',','))

	print('saved .txt stats files with t-tests, p-values, and effect sizes to '+stats_dir)

	filenames = [stats_dir+''.join(x) for x in zip(\
	                                            ['ttest-']*12,\
	                                            ['processing_pattern-']*4+\
	                                            ['click_var_outcome-']*4+\
	                                            ['click_var_gamble-']*4,\
	                                            np.tile(['alpha0-','alpha1-'],6).tolist(),
	                                            np.tile(['cost1.txt']*2+['cost4.txt']*2,3).tolist())]

	table_string = \
	'\\begin{tabular}{llccc}\n'+\
	'Dependent variable & '+\
	'\\begin{tabular}{@{}c@{}}Condition\\\\(dispersion, cost)\\end{tabular}& '+\
	'$t$-statistic & $p$-value & '+\
	'\\begin{tabular}{@{}c@{}}Effect size\\\\(Cohen\'s $d$)\\end{tabular}& '+\
	'\\midrule\n'+\
	'\\\\\n'.join([''.join(x) for x in zip(\
	                                    ['Processing pattern']*4 + \
	                                    ['Attribute variance']*4 + \
	                                    ['Alternative variance']*4, \
	                                    [' & ']*12,\
										[' \\begin{tabular}{@{}c@{}} '+y+' \\end{tabular} ' for y in \
										['\\\\'.join(z) for z in zip(np.tile(['$\\alpha^{-1}=10^{-0.5}$','$\\alpha^{-1}=10^{0.5}$'],6).tolist(),
																	 np.tile(['$\\lambda=1$']*2+['$\\lambda=4$']*2,3).tolist())]], \
	                                    [' & ']*12,\
	                                    [open(filenames[i],'r').read() for i in range(len(filenames))])\
	                ])+\
	'\\\\\n'+\
	'\\bottomrule'+\
	'\\end{tabular}'

	with open(stats_dir+'table_behavior.tex', 'w') as f:
		f.write(table_string)

	print('saved latex table to '+stats_dir+'table_behavior.tex')



