#!/usr/bin/env python3

from core import *


if __name__ == '__main__':
	# init parameters
	Username = 'moht.dsth.150@moht.com.sg_e3fb5e097f2b'
	CyclePeriod = 21
	file_suffix = '.csv.gz'

	# sleep stage
	dbg_df = df = load_df(Username, 'sleep' + file_suffix).copy()
	df = df[df.Level != 'main']
	plot3 = draw(df, None, None, 0.0, False, 'sleep' + file_suffix, 'grouped values by each interval', '1D', 0.5,
				 CyclePeriod, 'time chart stacked bar', 'Level', 10,
				 'Seconds', 'no sort', False, False, False, None, size_ratio=1, plot_title='Sleep Stage')

	# sleep efficiency
	plot4 = draw(Username, None, None, 0.0, False, 'sleep' + file_suffix, 'mean value in each interval', '1D', 0.5,
				 CyclePeriod, 'time chart (bar)', 'Efficiency', 10,
				 '<entry-count>', 'no sort', False, False, False, None, size_ratio=0.7, plot_title='Sleep Efficiency')

