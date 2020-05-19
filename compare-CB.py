#!/usr/bin/env python3

import os, sys, re, io, math, argparse
import matplotlib, shap, xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import *
from math import isnan, nan
from matplotlib.widgets import Slider
from glob import glob
from ipywidgets import *
from datetime import datetime, timedelta
from dateutil.tz import tzlocal
import ipywidgets as widgets

from pandas_serializer import *


def get_stats(df, funcs=['max']):
	return [eval('df.%s()' % f) for f in funcs]


def compare_stats(df):
	df_before = df[df.index < CB_start_date - CB_boundary_gap]
	df_after = df[df.index > CB_start_date + CB_boundary_gap]
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


## MAIN
if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage='$0 [options] <input 1>output 2>progress',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	                                 description='compare data before and after circuit breaker')
	parser.add_argument('--input-file', '-i', help='input file, default="-" (STDIN)', type=str, default='-')
	parser.add_argument('--output-file', '-o', help='output file, default="-" (STDOUT)', type=str, default='-')
	parser.add_argument('--CB-start-date', '-d', help='date when circuit breaker occurred', type=str, default='2020-04-07')
	parser.add_argument('--CB-boundary-gap', '-g', help='gap in days before and after CB-start-date', type=str, default='7D')
	parser.add_argument('--save-shap', '-s', help='save Shap plot to file', type=str, default='')
	opt = parser.parse_args()
	globals().update(vars(opt))

	CB_start_date = pd.Timestamp(CB_start_date, tz='tzlocal()')
	CB_boundary_gap = pd.to_timedelta(CB_boundary_gap)

	all_data = pandas_load(input_file)
	dfs = pd.concat([compare_stats(summarize(df)[0]) for p, df in all_data.items()])
	df_compare = dfs.groupby(dfs.index).mean()
	df_compare.to_csv(Open(output_file, 'w'))

	if save_shap:
		df = pd.concat([summarize(df)[0] for p, df in all_data.items()])
		df_before = df[df.index < CB_start_date - CB_boundary_gap]
		df_after = df[df.index > CB_start_date + CB_boundary_gap]
		X = pd.concat([df_before, df_after])
		Y = pd.Series([-1] * len(df_before.index) + [1] * len(df_after.index))

		# Build the model
		if True:
			# use xgboost algorithm
			model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y), 100)
		else:
			# use random forest regression algorithm
			model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
			model.fit(X, Y)

		shap_values = shap.TreeExplainer(model).shap_values(X)
		f = plt.figure()
		shap.summary_plot(shap_values, X, max_display=9999)
		f.savefig(save_shap, bbox_inches='tight', dpi=900)