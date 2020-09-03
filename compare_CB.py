#!/usr/bin/env python3

import os, sys, re, io, math, argparse
import matplotlib, shap, xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import *
from tqdm import tqdm
import scipy
import statsmodels.stats.api as sms
from math import isnan, nan
from matplotlib.widgets import Slider
from glob import glob
from ipywidgets import *
from datetime import datetime, timedelta
from dateutil.tz import tzlocal
import ipywidgets as widgets

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


def compare_stats(df, DateRangeA, DateRangeB):
	df_before = df[DateRangeA[0]:DateRangeA[1]]
	df_after = df[DateRangeB[0]:DateRangeB[1]]
	return pd.DataFrame({'before_mean': df_before.mean(), 'after_mean': df_after.mean(),
	                     'before_std': df_before.std(), 'after_std': df_after.std(),
	                     'before_max': df_before.max(), 'after_max': df_after.max(),
	                     'before_min': df_before.min(), 'after_min': df_after.min(),
	                     'before_median': df_before.median(), 'after_median': df_after.median()})


def summarize(df):
	col_groups = defaultdict(lambda: [])
	for col in df.columns:
		if re.search('_\d\dh$', col):
			col_groups[col[:-4]] += [col]
	ret = df[[col for col in df.columns if col[:-4] not in col_groups]].copy()
	for grp, cols in col_groups.items():
		ret[grp] = df[cols].mean(axis=1)
	return ret, col_groups


def get_compare(all_data, DateRangeA, DateRangeB):
	DateRangeA, DateRangeB = [str(i) for i in DateRangeA], [str(i) for i in DateRangeB]
	dfs = [summarize(df)[0] for p, df in all_data.items()]
	dfs = [df for df in dfs if not df[DateRangeA[0]:DateRangeA[1]].empty and not df[DateRangeB[0]:DateRangeB[1]].empty]
	DF = pd.concat([compare_stats(df, DateRangeA, DateRangeB) for df in dfs])
	df_compare = DF.groupby(DF.index).mean()
	df = pd.concat(dfs)
	df_before = df[DateRangeA[0]:DateRangeA[1]]
	df_after = df[DateRangeB[0]:DateRangeB[1]]
	for col in tqdm(df.columns):
		for func in [ttest_rel, wilcoxon, CI_ttest]:
			inp = DF.loc[col, ['before_mean', 'after_mean']].dropna()
			df_compare.loc[col, func.__name__] = func(inp.iloc[:,0], inp.iloc[:,1]) + ' (%d)'%inp.shape[0]
		df_compare.loc[col, 'p-test'] = p_test(df_before[col], df_after[col])
		df_compare.loc[col, 't-test'] = t_test(df_before[col], df_after[col])
	cols = list(df_compare.columns)
	ret = df_compare[cols[:2]+cols[-5:]+cols[2:-5]]
	return ret


def get_shap(all_data, DateRangeA, DateRangeB, figwidth=0, **kwargs):
	DateRangeA, DateRangeB = [str(i) for i in DateRangeA], [str(i) for i in DateRangeB]
	dfs = [summarize(df)[0] for p, df in all_data.items()]
	df = pd.concat([df for df in dfs if not df[DateRangeA[0]:DateRangeA[1]].empty and not df[DateRangeB[0]:DateRangeB[1]].empty])
	df_before = df[DateRangeA[0]:DateRangeA[1]]
	df_after = df[DateRangeB[0]:DateRangeB[1]]
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


## MAIN
if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage='$0 [options] <input 1>output 2>progress',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	                                 description='compare data before and after circuit breaker')
	parser.add_argument('--input-file', '-i', help='input file, default="-" (STDIN)', type=str, default='-')
	parser.add_argument('--output-file', '-o', help='output file, default="-" (STDOUT)', type=str, default='-')
	parser.add_argument('--CB-start-date', '-d', help='date when circuit breaker occurred', type=str, default='2020-04-07')
	parser.add_argument('--CB-boundary-gap', '-g', help='gap in days before and after CB-start-date', type=str, default='7D')
	parser.add_argument('--CB-boundary-end', '-e', help='maximum number of days before and after CB-start-date', type=str, default='1Y')
	parser.add_argument('--save-shap', '-s', help='save Shap plot to file', type=str, default='')
	parser.add_argument('--shap-width', '-w', help='width of Shap plot, default=0: auto', type=int, default=0)
	parser.add_argument('--dpi', '-dpi', help='save figure dpi', type=int, default=900)
	parser.add_argument('--max-display', '-N', help='max number of features in Shap plot', type=int, default=9999)
	opt = parser.parse_args()
	globals().update(vars(opt))

	CB_start_date = pd.Timestamp(CB_start_date, tz='tzlocal()')
	CB_boundary_gap = pd.to_timedelta(CB_boundary_gap)
	CB_boundary_end = pd.to_timedelta(CB_boundary_end)
	DateRangeA = [CB_start_date - CB_boundary_end, CB_start_date - CB_boundary_gap]
	DateRangeB = [CB_start_date + CB_boundary_gap, CB_start_date + CB_boundary_end]

	all_data = pandas_load(input_file)

	get_compare(all_data, DateRangeA, DateRangeB).to_csv(Open(output_file, 'w'))

	if save_shap:
		f, shap_values, X = get_shap(all_data, DateRangeA, DateRangeB, figwidth=shap_width, max_display=max_display)
		f.savefig(save_shap, bbox_inches='tight', dpi=dpi)
