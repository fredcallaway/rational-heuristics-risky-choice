import os
import numpy as np
import pandas as pd
import subprocess
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations
from statsmodels.stats.proportion import proportion_effectsize
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import process_data as p_d
import cfg

def exp1_strategy_logistic_regression(exp=cfg.exp1):
	dump_dir, latex_dir = exp.stats+'dump/1/', exp.stats+'1/'
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	def R_output_to_df(data):
		columns = ['param','beta','std_error','z_value','p']
		df = pd.DataFrame(columns=columns)
		for row in range(1,len(data)):
			if data[row]=='': continue
			i,j,substr = 0,0,[]
			while i < len(data[row]):
				while data[row][j] == ' ': j+=1
				i=j
				while i < len(data[row]) and data[row][i] != ' ': i+=1
				substr.append(data[row][j:i])
				j=i
			df.loc[row,columns[0]] = substr[0].replace('R_','')
			for i,s in enumerate(substr[1:]):
				df.loc[row,columns[i+1]] = eval(s)
		return df

	try:
		call_str = '/usr/bin/Rscript --vanilla '+os.path.dirname(os.getcwd())+'/R/logistic_regression.R'# +' '+dump_dir+' '+exp.human+' '+os.getcwd()
		subprocess.call(call_str, shell=True)
	except:
		try:
			subprocess.call (call_str, shell=True)
		except:
			p_d.print_special('!!! WARNING: you may need to run ../R/logistic_regression.R !!!')

	if exp.stats.print_summary: p_d.print_special('Results for Exp. 1 logistic regression of strategy frequencies on environment conditions', header=True)

	strategies = ['TTB_SAT','SAT_TTB','TTB','WADD','Rand','Other']
	for strategy in strategies:
		for param in ['sigma','alpha','cost']:
			try:
				with open(dump_dir+'R_'+strategy+'-R_'+param+'.txt', 'r') as f:
					data = f.read().split('\n\n')
				data = [i.split('\n') for i in data][0]
			except:
				p_d.print_special('!!! ABORTING exp1_logistic_regression(): you need to run ../R/logistic_regression.R !!!')
				return
			df = R_output_to_df(data)

			# for param in ['sigma','alpha','cost']:
			idx = np.where(df['param']==param)[0][0]
			B, p = df.loc[idx,'beta'], df.loc[idx,'p']
			p_str = f'= {p:.2}' if p >= 0.001 else '< 0.001'
			latex_str = f'$B = {B:.2}, p {p_str}$'
			with open(latex_dir+strategy+'-'+param+'.txt', 'w') as f:
				f.write(latex_str)

			if exp.stats.print_summary:
				p_d.print_special('IV: '+strategy+', DV: '+param)
				print(df, '\n')
	p_d.print_special('saved latex strategy logistic regression stats to '+latex_dir, False)

def exp1_strategy_table(exp=cfg.exp1):
	dump_dir, latex_dir = exp.stats+'dump/1/', exp.stats+'1/'
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	df = pd.read_csv(exp.human, low_memory=False)

	strategies = ['SAT_TTB','TTB_SAT','Other','TTB','Rand','WADD']

	cond_str = {'alpha': ['10^{-1.0}','10^{-0.5}','10^{.0}','10^{0.5}','10^{1.0}'],
				'cost': ['0','1','2','4','8']}

	df['alpha'] = 1/df['alpha']

	if exp.stats.print_summary: p_d.print_special('Results for Exp. 1 strategy frequency post-hoc comparisons, and effect sizes', header=True)
	if exp.stats.print_summary: p_d.print_special('sigma')
	for g in strategies:
		x = proportion_effectsize(np.mean(df[df['sigma']==75][g]), \
			np.mean(df[df['sigma']==150][g]))
		with open(dump_dir+g+'-sigma_postHoc-cohenD.txt', 'w') as f:
			f.write('n/a & '+f'${x:.{2}}$')
		if exp.stats.print_summary:
			print(f'{g} effect size: {x:.{2}}')
	
	for c in ['alpha','cost']:
		if exp.stats.print_summary: p_d.print_special(c)
		cond_levels = df[c].sort_values().unique()
		for g in strategies:
			p_vals = []
			for i,j in list(combinations([0,1,2,3,4],2)):
				chi_table = [\
							[sum(df[df[c]==cond_levels[i]][g].values==True), \
								sum(df[df[c]==cond_levels[i]][g].values==False)], \
							[sum(df[df[c]==cond_levels[j]][g].values==True), \
								sum(df[df[c]==cond_levels[j]][g].values==False)]\
							]
				ch, p, _, _ = chi2_contingency(chi_table)
				p_vals.append(p)
			reject_list, _, _, _ = multipletests(p_vals, method='fdr_bh', alpha=0.05)
			ph_str, ph_str_ = make_posthoc_str(reject_list, list(combinations([0,1,2,3,4],2)), cond_str[c])
			cd_str = '$'
			for i in range(4):
				res = proportion_effectsize(\
						df[df[c]==cond_levels[i]][g].mean(), \
						df[df[c]==cond_levels[i+1]][g].mean())
				cd_str += f'{res:.{2}}, '
			cd_str = cd_str[:-2]+'$'
			with open(dump_dir+g+'-'+c+'_postHoc-cohenD.txt', 'w') as f:
				f.write(ph_str+' & '+cd_str)
			if exp.stats.print_summary:
				print(g, 'significant post-hoc pairs: '+ph_str_, \
						'effect sizes (adjacent pairs): '+cd_str.replace('$',''), sep='\n')

	filenames = [dump_dir + f for f in[\
				'SAT_TTB-sigma_postHoc-cohenD.txt',\
				'TTB_SAT-sigma_postHoc-cohenD.txt',\
				'TTB-alpha_postHoc-cohenD.txt',\
				'Rand-alpha_postHoc-cohenD.txt',\
				'TTB_SAT-cost_postHoc-cohenD.txt',\
				'TTB-cost_postHoc-cohenD.txt',\
				'SAT_TTB-cost_postHoc-cohenD.txt']]

	table_string = \
	'\\begin{tabular}{llccc}\n'+\
	'Strategy & \\begin{tabular}{@{}c@{}}Independent\\\\variable\\end{tabular} & \\begin{tabular}{@{}c@{}}'+\
	'significant\\\\post-hoc comparisons\\end{tabular} & \\begin{tabular}{@{}c@{}}effect sizes\\\\(Cohen\'s $d$)\\end{tabular}\\\\\n'+\
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

	with open(latex_dir+'table_strategies.tex', 'w') as f:
		f.write(table_string)

	p_d.print_special('saved latex strategy table to '+latex_dir+'table_strategies.tex', False)

def exp1_behavioral_features(exp=cfg.exp1):
	pd.options.mode.chained_assignment = None
	dump_dir, latex_dir = exp.stats+'dump/2/', exp.stats+'2/'
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	for exclude in [False, True]:
		if exp.stats.print_summary:
			exclude_str = ' (participants excluded)' if exclude else ''
			p_d.print_special('Results for Exp. 1 linear regression of behavioral features on environment conditions, '+\
				'main effects, post-hoc corrections for multiple comparisions, and effect sizes'+exclude_str, header=True)

		df = pd.read_csv(exp.human, low_memory=False)
		if exclude:
			df = p_d.exclude_bad_participants(df)
		exclude_str = '_exclude' if exclude else ''

		cond_str = {'alpha': ['10^{-1.0}','10^{-0.5}','10^{.0}','10^{0.5}','10^{1.0}'],
					'cost': ['0','1','2','4','8']}

		df['alpha'] = 1 / df['alpha'] # dispersion is alpha^-1
		# make sigma, alpha, cost levels linear unit increments
		for i, s in enumerate(df['sigma'].sort_values().unique()):
			df.loc[df['sigma']==s, 'sigma'] = i
		df['alpha'] += 0.1 # avoid assigning integer alphas twice
		for i, a in enumerate(df['alpha'].sort_values().unique()):
			df.loc[df['alpha']==a, 'alpha'] = i
		for i, c in enumerate(df['cost'].sort_values().unique()):
			df.loc[df['cost']==c, 'cost'] = i
		for p in ['nr_clicks','payoff_gross_relative','processing_pattern','click_var_gamble','click_var_outcome']:
			for c in ['sigma','alpha','cost']:
				
				# normalize for standardized regression coefficients
				df_z = df.dropna(subset=[p])
				# df_z.loc[:, p] = (df_z[p] - df_z[p].mean())/df_z[p].std(ddof=0)
				
				# mixed-effects linear regression of behavioral features on environmental parameters
				res = smf.mixedlm(p+'~'+c, df_z, groups=df_z['pid']).fit()
				lm_str1, lm_str2 = f'$B={res.params[1]:.{2}}, p', f'= {res.pvalues[1]:.{2}}$' if res.pvalues[1]>=0.001 else '< 0.001$'
				
				dat = [df_z[df_z[c]==i].groupby('pid').mean()[p].values for i in np.sort(df_z[c].unique())]
			
				# main effects, post-hoc comparisons, and effect sizes
				if len(dat)==2:
					F, P = ttest_ind(dat[0],dat[1])
					dof = len(dat[0]) + len(dat[1]) - 2
					me_str1, me_str2 = f'$t$({dof}) = {F:.2f},\\\\$p$ ',f'= {P:.2}' if P>=0.001 else '<0.001'
					ph_str, tukey = 'n/a', ''
				else:
					F, P = f_oneway(dat[0],dat[1],dat[2],dat[3],dat[4])
					dof = (4, sum([len(dat[i]) for i in range(len(dat))]) - 5)
					me_str1, me_str2 = f'$F$({dof[0]},{dof[1]}) = {F:.2f},\\\\$p$ ',f'= {P:.2}' if P>=0.001 else '<0.001'
					if P < 0.1:
						tukey = pairwise_tukeyhsd(\
								endog = np.concatenate(dat),
								groups = np.concatenate([np.repeat(i,repeats=len(dat[i])) for i in range(len(dat))]),
								alpha = 0.05)
						# reject_list = [x[:2]+[x[-1]] for x in tukey.summary().data[1:] if abs(x[0]-x[1])==1] # report adjacent pairs only
						reject = [x[-1] for x in tukey.summary().data[1:]] # report all pairs
						pairs = [x[:2] for x in tukey.summary().data[1:]] # report all pairs
						ph_str, ph_str_ = make_posthoc_str(reject, pairs, cond_str[c])
					else:
						tukey = ''
						ph_str = 'n/a'
						
				cd_str = '$'+', '.join([f'{p_d.cohen_d(dat[i],dat[i+1]):.2}' for i in range(len(dat)-1)])+'$'
				
				with open(latex_dir+p+exclude_str+'-'+c+'_mixedlm.txt', 'w') as f:
					f.write(lm_str1+lm_str2)
					
				with open(dump_dir+p+exclude_str+'-'+c+'_mainEffect.txt', 'w') as f:
					f.write(me_str1+me_str2)
					
				with open(dump_dir+p+exclude_str+'-'+c+'_postHoc.txt', 'w') as f:
					f.write(ph_str)
					
				with open(dump_dir+p+exclude_str+'-'+c+'_cohenD.txt', 'w') as f:
					f.write(cd_str)
						
				if exp.stats.print_summary:
					p_d.print_special('IV: '+c+', DV: '+p)
					print('\n', \
						res.summary(), \
						'main effect: '+me_str1.replace('$','').replace('\\',' ') + me_str2.replace('$',''), \
						tukey, \
						'effect sizes (adjacent pairs): '+cd_str.replace('$',''), '\n', sep='\n')

	p_d.print_special('saved latex behavioral feature regression stats to '+latex_dir, False)

	def make_latex_table(varnames, tablenames, exclude=None):
		varnames = [x+'-' for y in [[vn]*9 for vn in varnames] for x in y]
		n_vars = len(varnames)
		filenames = [dump_dir+''.join(x) for x in zip(varnames, \
													np.tile(['sigma_']*3+['alpha_']*3+['cost_']*3,n_vars).tolist(), \
													np.tile(['mainEffect.txt']+['postHoc.txt']+['cohenD.txt'],3*n_vars).tolist())\
					]

		exclude = [False]*n_vars if not exclude else exclude
		tablenames = [[tn]*3 if not exclude[i] else ['\\begin{tabular}{@{}c@{}}'+tn+'\\\\(Participants excluded)'+' \\end{tabular} ']*3 for i,tn in enumerate(tablenames)]
		tablenames = [x for y in tablenames for x in y]
		n_rows = len(tablenames)
		table_string = \
		'\\begin{tabular}{llccc}\n'+ \
		'Behavioral feature & \\begin{tabular}{@{}c@{}}Independent\\\\variable\\end{tabular}& '+ \
		'main effect & \\begin{tabular}{@{}c@{}}significant\\\\post-hoc comparisons\\end{tabular} & \\begin{tabular}{@{}c@{}}effect sizes\\\\(Cohen\'s $d$)\\end{tabular}\\\\\n'+ \
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

	with open(latex_dir+'table_behavior.tex', 'w') as f:
		f.write(table)
	p_d.print_special('saved latex table to '+latex_dir+'table_behavior.tex', False)

	with open(latex_dir+'table_performance.tex', 'w') as f:
		f.write(table_perf)
	p_d.print_special('saved latex  table to '+latex_dir+'table_performance.tex', False)

def exp2_strategies(exp=cfg.exp2):
	dump_dir, latex_dir = exp.stats+'dump/1/', exp.stats+'1/'
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	df1 = pd.read_csv(exp.human_exp, low_memory=False)
	df2 = pd.read_csv(exp.human_con, low_memory=False)

	alphas = np.flip(np.sort(df1['alpha'].unique()))
	costs = np.sort(df1['cost'].unique())

	dof1, dof2 = 1, min(len(df1),len(df2))
	if exp.stats.print_summary: p_d.print_special('Results for Exp. 2 group comparisons of strategy frequencies and effect sizes', header=True)
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

				d = proportion_effectsize(df1[(df1['alpha']==a)&(df1['cost']==c)][s].mean(),\
										  df2[(df2['alpha']==a)&(df2['cost']==c)][s].mean())

				p_str = f'={p:.2}' if p >= 0.001 else '<0.001'
				latex_str = f'$\chi^2({dof1},{dof2})={chi:.1f}, p{p_str}, d={d:.2f}$'
				with open(f'{latex_dir}chi2-{s}-alpha{i}-cost{c}.txt', 'w') as f:
					f.write(latex_str)

				if exp.stats.print_summary:
					print(f'{s}-alpha{i}-cost{c}')
					print(latex_str.replace('$','').replace('\\',''))

	p_d.print_special('saved latex strategy chi-square stats to '+latex_dir, False)

def exp2_behavioral_features(exp=cfg.exp2):
	dump_dir, latex_dir = exp.stats+'dump/2/', exp.stats+'2/'
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	df1 = pd.read_csv(exp.human_exp, low_memory=False)
	df2 = pd.read_csv(exp.human_con, low_memory=False)

	alphas = np.flip(np.sort(df1['alpha'].unique()))
	costs = np.sort(df1['cost'].unique())

	if exp.stats.print_summary: p_d.print_special('Results for Exp. 2 group comparisons of behavioral features and effect sizes', header=True)
	for s in ['processing_pattern','click_var_outcome','click_var_gamble','nr_clicks','payoff_net_relative']:
		dump_file = s in ['processing_pattern','click_var_outcome','click_var_gamble'] # results formated for a table, not in-text
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
				sep = '$ & $' if dump_file else ', '
				out_dir = dump_dir if dump_file else latex_dir
				out_str = f'$t({dof})={t:.2f}{sep}p{p_str}{sep}d={d:.2f}$'
				with open(f'{out_dir}ttest-{s}-alpha{i}-cost{c}.txt', 'w') as f:
					f.write(out_str)
					
				if exp.stats.print_summary:
					print(f'{s}-alpha{i}-cost{c}')
					print(out_str.replace('$','').replace('\\','').replace(' &',','))

	p_d.print_special('saved latex behavioral stats to '+latex_dir, False)

	filenames = [dump_dir+''.join(x) for x in zip(\
												['ttest-']*12,\
												['processing_pattern-']*4+\
												['click_var_outcome-']*4+\
												['click_var_gamble-']*4,\
												np.tile(['alpha0-','alpha1-'],6).tolist(),
												np.tile(['cost1.txt']*2+['cost4.txt']*2,3).tolist())]

	table_string = \
	'\\begin{tabular}{lclll}\n'+\
	'Behavioral feature & '+\
	'\\begin{tabular}{@{}c@{}}Condition\\\\(dispersion, cost)\\end{tabular}& '+\
	'$t$-statistic & $p$-value & '+\
	'\\begin{tabular}{@{}c@{}}effect size\\\\(Cohen\'s $d$)\\end{tabular}\\\\\n'+\
	'\\midrule\n'+\
	'\\\\\n'.join([''.join(x) for x in zip(\
										['Processing pattern']*4 + \
										['Attribute variance']*4 + \
										['Alternative variance']*4, \
										[' & ']*12,\
										[' \\begin{tabular}{@{}c@{}} '+y+' \\end{tabular}' for y in \
											['\\\\'.join(z) for z in zip(np.tile(['$\\alpha^{-1}=10^{-0.5}$','$\\alpha^{-1}=10^{0.5}$'],6).tolist(),
																		 np.tile(['$\\lambda=1$']*2+['$\\lambda=4$']*2,3).tolist())]], \
										[' & ']*12,\
										[open(filenames[i],'r').read() for i in range(len(filenames))])\
					])+\
	'\\\\\n'+\
	'\\bottomrule'+\
	'\\end{tabular}'

	with open(latex_dir+'table_behavior.tex', 'w') as f:
		f.write(table_string)

	p_d.print_special('saved latex table to '+latex_dir+'table_behavior.tex', False)

def under_performance(exp=cfg.exp1.human, exclude=False):
	dump_dir, latex_dir = exp.stats+'dump/3/', exp.stats+'3/'
	latex_dir = exp.stats+'3b/' if exclude and exp.num==2 else latex_dir
	if not os.path.exists(dump_dir): os.makedirs(dump_dir)
	if not os.path.exists(latex_dir): os.makedirs(latex_dir)

	exclude_str = '_exclude' if exclude else ''
	dat = eval(pd.read_csv(exp, usecols=['under_performance'+exclude_str], low_memory=False).iloc[0][0])[0]
	exp2_str = '' if exp.num==1 else '_'+exp.group

	# model clicks - human clicks
	x = dat['nr_clicks_dif']
	with open(latex_dir+'perf-reduc_nr_clicks'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}$')

	# overall perforamcne gap in units of gross reward
	x = dat['peformance_gap_points']
	with open(latex_dir+'perf-reduc_points'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}$')

	# overall perforamcne gap in units of gross relative reward
	x = dat['peformance_gap_gross_abs']
	with open(latex_dir+'perf-reduc_overall_gross-abs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in percentage of model gross relative reward
	x = dat['peformance_gap_gross_pct']
	with open(latex_dir+'perf-reduc_overall_gross-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}\\%$')

	# overall perforamcne in percentage of model net relative reward
	x = dat['human_performance_pct']
	with open(latex_dir+'perf-overall-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in units of net relative reward
	x = dat['peformance_gap_abs']
	with open(latex_dir+'perf-reduc_overall-abs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{2}f}$')

	# overall perforamcne gap in percentage of model net relative reward
	x = dat['peformance_gap_pct']
	with open(latex_dir+'perf-reduc_overall-pct'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${x:.{1}f}\\%$')

	# reduction in performance from implicit costs, as a fraction of model performance
	x = dat['implicit_costs_model_fraction']
	with open(latex_dir+'perf-reduc_implicit-costs_model'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from implicit costs, as a fraction of model-human performance gap
	x = dat['implicit_costs']
	with open(latex_dir+'perf-reduc_implicit-costs'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect use of information, as a fraction of model-human performance gap
	x = dat['imperfect_info_use']
	with open(latex_dir+'perf-reduc_info-use'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect strategy selection, as a fraction of model-human performance gap
	x = dat['imperfect_strat_selec']
	with open(latex_dir+'perf-reduc_strat-selec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# fraction of previous value from random gambling
	x = dat['imperfect_strat_selec_by_strat'][4] / dat['imperfect_strat_selec']
	with open(latex_dir+'perf-reduc_strat-selec-from-rand'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# and random gambling as fraction of total under-performance
	x = dat['imperfect_strat_selec_by_strat'][4]
	with open(latex_dir+'perf-reduc_strat-selec-rand'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')
		
	# reduction in performance from imperfect strategy selection, as a fraction of model-human performance gap
	x = dat['imperfect_strat_exec']
	with open(latex_dir+'perf-reduc_strat-exec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	# reduction in performance from imperfect strategy selection and execution, as a fraction of model-human performance gap
	x = dat['imperfect_strat_selec_and_exec']
	with open(latex_dir+'perf-reduc_strat-selec-exec'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	# reduction in performance from imperfect strategy selection and execution, as a fraction of model-human performance gap
	x = dat['imperfect_info_use_points_lost']
	with open(latex_dir+'perf-reduc_info-use-pointsPerTrial'+exp2_str+exclude_str+'.txt', 'w') as f:
		f.write(f'${100*x:.{1}f}\\%$')

	p_d.print_special('saved under-performance files to '+latex_dir+'perf-reduc_....txt', False)

def participant_demographics(exp=cfg.exp1):

	participants = pd.read_csv(exp.participants, low_memory=False)

	p_d.print_special('Participant demographics from '+exp.participants, header=True)

	nr_female = sum(participants['gender']=='female')
	print(f'Participants: {len(participants)}',
		 f'Females: {nr_female}',
		 f'Age {participants.age.mean():.2f}, \
			std: {participants.age.std():.2f}, \
			range: {participants.age.min():.2f}-{participants.age.max():.2f}',
		 f'Bonus: {participants.bonus.mean():.2f}, \
			std: {participants.bonus.std():.2f}, \
			range: {participants.bonus.min():.2f}-{participants.bonus.max():.2f}',
		 f'Experiment length (min.): {participants.total_time.mean()/60000:.2f}, \
			std: {participants.total_time.std()/60000:.2f}, \
			range: {participants.total_time.min()/60000:.2f}-{participants.total_time.max()/60000:.2f}', \
		 sep='\n')

def make_posthoc_str(reject, pairs, names):
	reject = np.array(reject)
	if all(reject):
		return 'all pairs', 'all pairs'
	elif all(~reject):
		return 'n/a', 'n/a'
	if np.mean(reject) > 0.6:
		posthoc_str = '\\begin{tabular}{@{}c@{}}all pairs except\\\\'
		reject = ~reject
	else:
		posthoc_str = '\\begin{tabular}{@{}c@{}}'
	for i,pair in enumerate(pairs):
		if reject[i]:
			if sum(reject[:i+1]) == 3: posthoc_str += '\\\\'
			posthoc_str += '$'+str(names[pair[0]])+\
					' \& '+str(names[pair[1]])+'$, '
	posthoc_str = posthoc_str[:-2] + '\\end{tabular}'
	print_str = posthoc_str.replace('\\begin{tabular}{@{}c@{}}','').replace('\\end{tabular}','').replace('\\\\',' ').replace('\\&','&')
	return posthoc_str, print_str
