#!/usr/bin/env python3

import os, sys, re, io, math, argparse
import matplotlib, shap, xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import *
import scipy
import statsmodels.stats.api as sms
from math import isnan, nan
from matplotlib.widgets import Slider
from glob import glob
from ipywidgets import *
from datetime import datetime, timedelta
from dateutil.tz import tzlocal
import ipywidgets as widgets
from tqdm import tqdm

from pandas_serializer import *


def p_test(pop1, pop2, n_perm=10000):
	"""
	Run a permutation test. Null hypothesis is that pop1 and pop2 have the same means. Returns p-value.

	:param pop1:
	:param pop2:
	:param n_perm: The number of permutations.
	:return:
	"""
	pop1, pop2 = pop1[pop1.notna()].to_numpy(), pop2[pop2.notna()].to_numpy()
	n1_ = len(pop1)
	true_diff = np.abs(np.mean(pop1) - np.mean(pop2))
	all_ = np.concatenate([pop1, pop2])
	pval = 0
	for _ in range(n_perm):
		np.random.shuffle(all_)
		samp1, samp2 = all_[:n1_], all_[n1_:]
		samp_diff = np.abs(samp1.mean() - samp2.mean())
		pval += int(samp_diff >= true_diff)
	return '%.5f'%(pval/n_perm)


def t_test(pop1, pop2):
	pop1, pop2 = pop1[pop1.notna()].to_numpy(), pop2[pop2.notna()].to_numpy()
	return '%.5f'%(scipy.stats.ttest_ind(pop1, pop2).pvalue)


def ttest_rel(X1, X2):
	return '%.5f'%scipy.stats.ttest_rel(X1, X2)[1]

def wilcoxon(X1, X2):
	return '%.5f'%scipy.stats.wilcoxon(X1, X2)[1]

def CI_ttest(X1, X2):
	cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
	out = cm.tconfint_diff(usevar='unequal')
	return '[%.2f, %.2f]'%(out[0], out[1])


def get_stats(df, funcs=['max']):
	return [eval('df.%s()' % f) for f in funcs]


def compare_stats(df_negative, df_positive):
	return pd.DataFrame({'neg_mean': df_negative.mean(), 'pos_mean': df_positive.mean(),
	                     'neg_std': df_negative.std(), 'pos_std': df_positive.std(),
	                     'neg_max': df_negative.max(), 'pos_max': df_positive.max(),
	                     'neg_min': df_negative.min(), 'pos_min': df_positive.min(),
	                     'neg_median': df_negative.median(), 'pos_median': df_positive.median()})


def summarize(df):
	col_groups = defaultdict(lambda: [])
	for col in df.columns:
		if re.search('_[0-9][0-9]h$', col):
			col_groups[col[:-4]] += [col]
	ret = df[[col for col in df.columns if col[:-4] not in col_groups]].copy()
	for grp, cols in col_groups.items():
		ret[grp] = df[cols].mean(axis=1)
	return ret, col_groups


def get_compare(neg, pos):
	df_compare = compare_stats(neg, pos)
	[df_compare.insert(2, col, nan) for col in ['t-test', 'p-test']]
	for col in tqdm(pos.columns):
		df_compare.loc[col, 'p-test'] = p_test(neg[col], pos[col])
		df_compare.loc[col, 't-test'] = t_test(neg[col], pos[col])
	return df_compare


def get_shap(df_before, df_after, figwidth=0, **kwargs):
	X = pd.concat([df_before, df_after])
	Y = pd.Series([-1] * len(df_before.index) + [1] * len(df_after.index))

	# Build the model
	if True:
		# use xgboost algorithm
		model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y), 100)
	else:
		# use random forest regression algorithm
		np.random.seed(0)
		from sklearn.ensemble import RandomForestRegressor
		model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
		model.fit(X, Y)

	shap_values = shap.TreeExplainer(model).shap_values(X)
	f = plt.figure()
	shap.summary_plot(shap_values, X, plot_size=[figwidth, figwidth*X.shape[1]/40] if figwidth else 'auto', **kwargs)
	return f, shap_values, X


def split_pos_neg(DP_data, drug_data, criterion):
	pos, neg = [], []
	n_pos, n_neg = [], []
	for PID, df in DP_data.items():
		try:
			cls, n_cls = (pos, n_pos) if eval(criterion) else (neg, n_neg)
			cls += [summarize(df)[0]]
			n_cls += [PID]
		except:
			pass
	print('POSITIVE class has %s, NEGATIVE class has %s'%(n_pos, n_neg), file=sys.stderr)
	return pd.concat(pos), pd.concat(neg)


## MAIN
if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage='$0 [options] <input 1>output 2>progress',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	                                 description='compare data before and after circuit breaker')
	parser.add_argument('--input-file', '-i', help='input file, default="-" (STDIN)', type=str, default='-')
	parser.add_argument('--drug-file', '-id', help='input drug file, default="-" (STDIN)', type=str, default='-')
	parser.add_argument('--output-file', '-o', help='output file, default="-" (STDOUT)', type=str, default='-')
	parser.add_argument('--save-shap', '-s', help='save Shap plot to file', type=str, default='')
	parser.add_argument('--shap-width', '-w', help='width of Shap plot, default=0: auto', type=int, default=0)
	parser.add_argument('--dpi', '-dpi', help='save figure dpi', type=int, default=900)
	parser.add_argument('--max-display', '-N', help='max number of features in Shap plot', type=int, default=9999)
	opt = parser.parse_args()
	globals().update(vars(opt))

	# load both DP and drug data, converting email ID to participant ID, e.g. S001
	DP_data = {('S%s'%(k[10:13])):v for k,v in pandas_load(input_file).items()}
	drug_data = pandas_load(drug_file)

	# split into positive and negative class
	pos, neg = split_pos_neg(DP_data, drug_data, "8 in drug_data['drugs_antipsychotic']['antipsychotic_name_'][PID].values")

	# perform the comparison between the positive and negative class
	get_compare(neg, pos).to_csv(Open(output_file, 'w'))

	if save_shap:
		f, shap_values, X = get_shap(pos, neg, figwidth=shap_width, max_display=max_display)
		f.savefig(save_shap, bbox_inches='tight', dpi=dpi)