#!/usr/bin/env python3

from core import *
import datetime

if __name__ == '__main__':
	draw('moht.dsth.150@moht.com.sg_e3fb5e097f2b', None, None, 0.0, False, 'heart.csv.gz', 'grouped values by each interval', '1D', 0.0, 21,
		 'time chart grouped box plot', 'HR', 'mean', '<entry-count>', False, 'no sort', False, False, False, True)

	# init parameters
	Username = 'moht.dsth.150@moht.com.sg_e3fb5e097f2b'
	CyclePeriod = 21
	file_suffix = '.csv.gz'

	# sociability messages
	df = load_df(Username, 'sociabilityLog' + file_suffix).copy()
	df.orientation = df.orientation.apply(lambda t: ('outgoing' if t == 0 else 'incoming'))
	plot1a = draw(df, None, None, 0.0, False, 'sociabilityLog' + file_suffix, 'grouped values by each interval', '1D',
				  0.0, CyclePeriod, 'time chart stacked bar', 'orientation', 10,
				  '<entry-count>', True, 'no sort', False, False, False, None, size_ratio=1, set_title='WhatsApp Messages', ylabel='log(N)')


