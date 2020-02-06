#!/usr/bin/env python3

import os, sys
os.main_path = os.getenv('HOME')+'/projects/beiwe-gitlab/beiwe-visualizer/3.decrypted/'

from core import *
import datetime


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


if __name__ == '__main__':
	# draw('moht.dsth.150@moht.com.sg_e3fb5e097f2b', None, None, 0.0, False, 'heart.csv.gz', 'grouped values by each interval', '1D', 0.0, 21,
	# 	 'time chart grouped box plot', 'HR', 'mean', '<entry-count>', False, 'no sort', False, False, False, False, True)

	# init parameters
	plt.switch_backend('nbAgg')
	os.DateGroup = '30,7,7'
	PP = group_dates
	Username = 'moht.dsth.150@moht.com.sg_e3fb5e097f2b'
	CyclePeriod = 21
	file_suffix = '.csv.gz'

	# arrange into horizontal grid
	plt.switch_backend('nbAgg')
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
	                        '<entry-count>', True, 'no sort', False, False, False, False, True, verbose=-1,
	                        post_processor=PP, plot_options={'color': [cmap(v) for v in RoG]},
	                        set_title='Mobility', set_ylabel='Daily Time Away From Home (hours)')
	plot3.legend(handles=[Patch(color='green', label='>1km'), Patch(color='orange', label='100m-1km'),
	                      Patch(color='red', label='<100m')], title='Radius of Gyration')

	x=3
