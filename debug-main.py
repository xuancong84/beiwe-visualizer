#!/usr/bin/env python3

import os, sys
os.main_path = os.getenv('HOME')+'/projects/beiwe-gitlab/beiwe-visualizer/3.decrypted/'

from core import *
import datetime
from pandas_serializer import *
from compare_CB import *


def group_dates(data, labels, figsize):
	periods = [(s + 'D' if s.isdigit() else s) for s in re.split('[ ,;]+', os.DateGroup)]
	index = data.index if type(data.index) == pd.DatetimeIndex else data.datetime
	max_date = index.max().floor('D')
	cut_points = [max_date]
	for period in periods[::-1]:
		max_date -= pd.to_timedelta(period)
		cut_points = [max_date] + cut_points
	dfg = data.groupby(pd.cut(index, cut_points, right=False))
	figsize[0] = len(periods) * 0.4
	if index.nunique() == len(data):
		ret = dfg.mean()
		ret_index = ret.index
	else:
		ret = pd.concat([set_index_to_value(v.reset_index(drop=True), k, 'datetime') for k, v in dfg],
		                ignore_index=True)
		ret_index = [k for k, v in dfg]
	return ret, ret_index, figsize


def summarize(df):
	col_groups = defaultdict(lambda: [])
	for col in df.columns:
		if re.search('_\d\dh$', col):
			col_groups[col[:-4]] += [col]
	ret = df[[col for col in df.columns if col[:-4] not in col_groups]].copy()
	for grp, cols in col_groups.items():
		ret[grp] = df[cols].mean(axis=1)
	return ret, col_groups

def compare_stats1(df, DateRangeA, DateRangeB):
	df_before = df[DateRangeA[0]:DateRangeA[1]]
	df_after = df[DateRangeB[0]:DateRangeB[1]]
	return pd.DataFrame({'before': df_before.mean(), 'after': df_after.mean()})

def get_compare1(all_data, DateRangeA, DateRangeB):
	DateRangeA, DateRangeB = [str(i) for i in DateRangeA], [str(i) for i in DateRangeB]
	dfs = [summarize(df)[0] for p, df in all_data.items()]
	dfs = [df for df in dfs if not df[DateRangeA[0]:DateRangeA[1]].empty and not df[DateRangeB[0]:DateRangeB[1]].empty]
	DF = pd.concat([compare_stats1(df, DateRangeA, DateRangeB) for df in dfs])
	DFG = DF.groupby(DF.index)
	ret = pd.DataFrame({'before_mean': DFG['before'].mean(), 'before_std': DFG['before'].std(),
	              'after_mean': DFG['after'].mean(), 'after_std': DFG['after'].std()})
	return ret


def f1():
	CB_start_date = pd.Timestamp('2020-4-7', tz = 'tzlocal()')
	CB_boundary_gap = pd.to_timedelta('3D')  # at least N days from CB start date
	CB_boundary_end = pd.to_timedelta('45D')  # at most N days from CB start date

	# Singapore COVID dates: 2020-4-7 2020-6-2 2020-6-19
	DateRangeA = [CB_start_date - CB_boundary_end, CB_start_date - CB_boundary_gap]
	DateRangeB = [CB_start_date + CB_boundary_gap, CB_start_date + CB_boundary_end]
	DateRanges = [DateRangeA, DateRangeB]

	N_user = 25
	f_getN = lambda s: int(re.search('[0-9]+@', s).group(0)[:-1])
	f = '5.decrypted/izedAa85XXrDS85XlwrOsIDU/all-data.pson.gz'
	if os.path.exists(f):
		print('Loading files ...', end = '')
		all_data = {k: v for k, v in pandas_load(f).items() if f_getN(k) < N_user}
		all_cols = all_data[list(all_data.keys())[0]].columns.tolist()
		print('Finished loading')
		compare_res = get_compare(all_data, DateRangeA, DateRangeB)
		print('Done')

if __name__ == '__main__':
	# 1.
	# draw('moht.dsth.150@moht.com.sg_e3fb5e097f2b', None, None, 0.0, False, 'heart.csv.gz', 'grouped values by each interval', '1D', 0.0, 21,
	# 	 'time chart grouped box plot', 'HR', 'mean', '<entry-count>', False, 'no sort', False, False, False, False, True)

	# 2.
	# draw('moht.dsth.150@moht.com.sg_e3fb5e097f2b', None, None, 0.0, False, 'light.csv.gz', '# of readings in each interval', '1D', 0.0, 0,
	# 	 'time chart (bar)', 'value', 'mean', '<entry-count>', False, 'no sort', False, False, False, False, True)

	# 3.
	f1()

	# 4.
	os.DateGroup = '30,7,7'
	PP = group_dates
	Username = 'moht.dsth.150@moht.com.sg_e3fb5e097f2b'
	CyclePeriod = 21
	file_suffix = '.csv.gz'

	# arrange into horizontal grid
	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[12, 4])

	# debug plot code
	df1 = load_df(Username, 'gps-mobility' + file_suffix, verbose=-1)
	df1 = filter_by_date(df1[['Hometime', 'RoG']], None, None).sort_index()
	df = (24 - df1[['Hometime']] / 60).rename(columns={'Hometime': 'TAFH'})
	# prepare RoG
	RoG, _, _ = group_dates(np.log10(df1[['RoG']].clip(0, np.inf) + 1), None, [16, 9])
	RoG = [10 ** v for v in RoG.RoG]


	def cmap(val):
		if val < 100:
			return 'red'
		elif val > 1000:
			return 'green'
		return 'orange'


	# plot time-away-from-home together with RoG
	os.plot3 = plot3 = draw(df, None, None, 0.0, False, None, 'mean value in each interval', '1D', 0, 0,
	                        'time chart (bar)', 'TAFH', 10,
	                        '<entry-count>', True, 'no sort', False, False, False, False, True, verbose=-1, ax=axs[0],
	                        post_processor=PP, plot_options={'color': [cmap(v) for v in RoG]},
	                        set_title='Mobility', set_ylabel='Daily Time Away From Home (hours)')
	plot3.legend(handles=[Patch(color='green', label='>1km'), Patch(color='orange', label='100m-1km'),
	                      Patch(color='red', label='<100m')], title='Radius of Gyration')

	x=3
