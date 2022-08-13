import os
from time import time
import pdb

t0 = time()
def timer():
	s = time() - t0
	m = int(s/60)
	return f'{m:.0f}m:{s-m*60:02.0f}s'

if 'basedir' not in locals():
	def basedir():
		basedir.figs = '../figs/'
		basedir.stats = '../stats/'
		basedir.model = '../data/model/'
		basedir.human = '../data/human/'
	basedir()

class ExpSpecs(str):
	def data_specs(self, num, isHuman, exclude, raw, group, figs, stats):
		self.num = num
		self.isHuman = isHuman
		self.exclude = exclude
		self.raw = raw
		self.group = group
		self.figs = figs
		self.stats = stats
		exclude_str = '_exclude' if exclude else ''
		group_str = '' if num==1 else '_'+group
		human_dir = basedir.human+'1.0/processed' if num==1 else basedir.human+'2.3/processed'
		self.clicks = os.path.join(os.path.dirname(self), 'click_embeddings'+group_str+exclude_str+'.pkl')
		if not isHuman:
			self.kmeans_sim_trials_per_human_trial = 10

	def fig_specs(self, save, show):
		self.save = save
		self.show = show

	def stat_specs(self, print_summary):
		self.print_summary = print_summary

class Exp1():
	num = 1
	figs = ExpSpecs(basedir.figs+'exp1/')
	figs.fig_specs(True, True)
	stats = ExpSpecs(basedir.stats+'exp1/')
	stats.stat_specs(True)
	model = ExpSpecs(basedir.model+'exp1/processed/trials.csv')
	model.data_specs(num, False, False, basedir.model+'exp1/', None, figs, stats)
	model_exclude = ExpSpecs(basedir.model+'exp1/processed/trials_exclude.csv')
	model_exclude.data_specs(num, False, True, basedir.model+'exp1/', None, figs, stats)
	model_fitcost = ExpSpecs(basedir.model+'exp1_fitcost/processed/trials.csv')
	model_fitcost.data_specs(num, False, False, basedir.model+'exp1_fitcost/', None, figs, stats)
	model_fitcost_exclude = ExpSpecs(basedir.model+'exp1_fitcost_exclude/processed/trials_exclude.csv')
	model_fitcost_exclude.data_specs(num, False, True, basedir.model+'exp1_fitcost_exclude/', None, figs, stats)
	human = ExpSpecs(basedir.human+'1.0/processed/trials.csv')
	human.data_specs(num, True, False, basedir.human+'1.0/trials.csv', None, figs, stats)
	human_exclude = ExpSpecs(basedir.human+'1.0/processed/trials_exclude.csv')
	human_exclude.data_specs(num, True, True, basedir.human+'1.0/trials.csv', None, figs, stats)
	participants = basedir.human+'1.0/participants.csv'

	if not os.path.exists(figs): os.makedirs(figs)
	if not os.path.exists(stats): os.makedirs(stats)
	if not os.path.exists(os.path.dirname(model)): os.makedirs(os.path.dirname(model))
	if not os.path.exists(os.path.dirname(model_exclude)): os.makedirs(os.path.dirname(model_exclude))
	if not os.path.exists(os.path.dirname(model_fitcost)): os.makedirs(os.path.dirname(model_fitcost))
	if not os.path.exists(os.path.dirname(model_fitcost_exclude)): os.makedirs(os.path.dirname(model_fitcost_exclude))
	if not os.path.exists(os.path.dirname(human)): os.makedirs(os.path.dirname(human))
	if not os.path.exists(os.path.dirname(human_exclude)): os.makedirs(os.path.dirname(human_exclude))

class Exp2():
	num = 2
	figs = ExpSpecs(basedir.figs+'exp2/')
	figs.fig_specs(True, True)
	stats = ExpSpecs(basedir.stats+'exp2/')
	stats.stat_specs(True)
	model = ExpSpecs(basedir.model+'exp2/processed/trials.csv')
	model.data_specs(num, False, False, basedir.model+'exp2/', 'both', figs, stats)
	model_exclude = ExpSpecs(basedir.model+'exp2/processed/trials_exclude.csv')
	model_exclude.data_specs(num, False, True, basedir.model+'exp2/', 'both', figs, stats)
	model_fitcost_con = ExpSpecs(basedir.model+'exp2_con_fitcost/processed/trials.csv')
	model_fitcost_con.data_specs(num, False, False, basedir.model+'exp2_con_fitcost/', 'con', figs, stats)
	model_fitcost_exp = ExpSpecs(basedir.model+'exp2_exp_fitcost/processed/trials.csv')
	model_fitcost_exp.data_specs(num, False, False, basedir.model+'exp2_exp_fitcost/', 'exp', figs, stats)
	model_fitcost_exclude_con = ExpSpecs(basedir.model+'exp2_con_fitcost_exclude_alt/processed/trials_exclude.csv')
	model_fitcost_exclude_con.data_specs(num, False, True, basedir.model+'exp2_con_fitcost_exclude_alt/', 'con', figs, stats)
	model_fitcost_exclude_exp = ExpSpecs(basedir.model+'exp2_exp_fitcost_exclude_alt/processed/trials_exclude.csv')
	model_fitcost_exclude_exp.data_specs(num, False, True, basedir.model+'exp2_exp_fitcost_exclude_alt/', 'exp', figs, stats)
	human_con = ExpSpecs(basedir.human+'2.3/processed/trials_con.csv')
	human_con.data_specs(num, True, False, basedir.human+'2.3/trials.csv', 'con', figs, stats)
	human_exp = ExpSpecs(basedir.human+'2.3/processed/trials_exp.csv')
	human_exp.data_specs(num, True, False, basedir.human+'2.3/trials.csv', 'exp', figs, stats)
	human_exclude_con = ExpSpecs(basedir.human+'2.3/processed/trials_exclude_con.csv')
	human_exclude_con.data_specs(num, True, True, basedir.human+'2.3/trials.csv', 'con', figs, stats)
	human_exclude_exp = ExpSpecs(basedir.human+'2.3/processed/trials_exclude_exp.csv')
	human_exclude_exp.data_specs(num, True, True, basedir.human+'2.3/trials.csv', 'exp', figs, stats)
	participants = basedir.human+'2.3/participants.csv'

	if not os.path.exists(figs): os.makedirs(figs)
	if not os.path.exists(stats): os.makedirs(stats)
	if not os.path.exists(os.path.dirname(model)): os.makedirs(os.path.dirname(model))
	if not os.path.exists(os.path.dirname(model_exclude)): os.makedirs(os.path.dirname(model_exclude))
	if not os.path.exists(os.path.dirname(model_fitcost_con)): os.makedirs(os.path.dirname(model_fitcost_con))
	if not os.path.exists(os.path.dirname(model_fitcost_exp)): os.makedirs(os.path.dirname(model_fitcost_exp))
	if not os.path.exists(os.path.dirname(model_fitcost_exclude_con)): os.makedirs(os.path.dirname(model_fitcost_exclude_con))
	if not os.path.exists(os.path.dirname(model_fitcost_exclude_exp)): os.makedirs(os.path.dirname(model_fitcost_exclude_exp))
	if not os.path.exists(os.path.dirname(human_con)): os.makedirs(os.path.dirname(human_con))
	if not os.path.exists(os.path.dirname(human_exp)): os.makedirs(os.path.dirname(human_exp))
	if not os.path.exists(os.path.dirname(human_exclude_con)): os.makedirs(os.path.dirname(human_exclude_con))
	if not os.path.exists(os.path.dirname(human_exclude_exp)): os.makedirs(os.path.dirname(human_exclude_exp))

def basedir(subdir=''):
	basedir.figs = '../figs/'+subdir
	basedir.stats = '../stats/'+subdir
	basedir.model = '../data/model/'+subdir
	basedir.human = '../data/human/'

exp1 = Exp1()
exp2 = Exp2()

# fitcost useful attribute?
# _exclude_exp vs _exp_exclude, etc.




