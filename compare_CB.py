#!/usr/bin/env python3

import os, sys, re, io, math, argparse
import matplotlib, shap, xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import *
from scipy.stats import ttest_ind
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
	return pval / n_perm


def t_test(pop1, pop2):
	pop1, pop2 = pop1[pop1.notna()].to_numpy(), pop2[pop2.notna()].to_numpy()
	return ttest_ind(pop1, pop2).pvalue


def get_stats(df, funcs=['max']):
	return [eval('df.%s()' % f) for f in funcs]


def compare_stats(df, CB_start_date, CB_boundary_gap, CB_boundary_end):
	df_before = df[(df.index < CB_start_date - CB_boundary_gap) & (df.index > CB_start_date - CB_boundary_end)]
	df_after = df[(df.index > CB_start_date + CB_boundary_gap) & (df.index < CB_start_date + CB_boundary_end)]
	return pd.DataFrame({'before_mean': df_before.mean(), 'after_mean': df_after.mean(),
	                     'before_std': df_before.std(), 'after_std': df_after.std(),
	                     'before_max': df_before.max(), 'after_max': df_after.max(),
	                     'before_min': df_before.min(), 'after_min': df_after.min(),
	                     'before_median': df_before.median(), 'after_median': df_after.median()})


def summarize(df):
	col_groups = defaultdict(lambda: [])
	for col in df.columns:
		if re.search('_[0-9][0-9]h$', col):
			col_groups[col[:-4]] += [col]
	ret = df[[col for col in df.columns if col[:-4] not in col_groups]]
	for grp, cols in col_groups.items():
		ret[grp] = df[cols].mean(axis=1)
	return ret, col_groups


def get_compare(all_data, CB_start_date, CB_boundary_gap, CB_boundary_end):
	dfs = [summarize(df)[0] for p, df in all_data.items()]
	dfs = [df for df in dfs if ((df.index < CB_start_date - CB_boundary_gap)&(df.index > CB_start_date - CB_boundary_end)).sum()
	       and ((df.index > CB_start_date + CB_boundary_gap)&(df.index < CB_start_date + CB_boundary_end)).sum()]
	df = pd.concat([compare_stats(df, CB_start_date, CB_boundary_gap, CB_boundary_end) for df in dfs])
	df_compare = df.groupby(df.index).mean()
	df = pd.concat(dfs)
	df_before = df[(df.index < CB_start_date - CB_boundary_gap) & (df.index > CB_start_date - CB_boundary_end)]
	df_after = df[(df.index > CB_start_date + CB_boundary_gap) & (df.index < CB_start_date + CB_boundary_end)]
	for col in df.columns:
		df_compare.loc[col, 'p-test'] = p_test(df_before[col], df_after[col])
		df_compare.loc[col, 't-test'] = t_test(df_before[col], df_after[col])
	cols = list(df_compare.columns)
	ret = df_compare[cols[:2]+cols[-2:]+cols[2:-2]]
	return ret


def get_shap(all_data, CB_start_date, CB_boundary_gap, CB_boundary_end, figwidth=0, **kwargs):
	dfs = [summarize(df)[0] for p, df in all_data.items()]
	df = pd.concat([df for df in dfs if ((df.index < CB_start_date - CB_boundary_gap)&(df.index > CB_start_date - CB_boundary_end)).sum()
	       and ((df.index > CB_start_date + CB_boundary_gap)&(df.index < CB_start_date + CB_boundary_end)).sum()])
	df_before = df[(df.index < CB_start_date - CB_boundary_gap) & (df.index > CB_start_date - CB_boundary_end)]
	df_after = df[(df.index > CB_start_date + CB_boundary_gap) & (df.index < CB_start_date + CB_boundary_end)]
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

	all_data = pandas_load(input_file)

	get_compare(all_data, CB_start_date, CB_boundary_gap, CB_boundary_end).to_csv(Open(output_file, 'w'))

	if save_shap:
		f, shap_values, X = get_shap(all_data, CB_start_date, CB_boundary_gap, CB_boundary_end, figwidth=shap_width, max_display=max_display)
		f.savefig(save_shap, bbox_inches='tight', dpi=dpi)