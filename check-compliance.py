#!/usr/bin/env python3
# coding=utf-8

import os, sys, argparse, re, gzip
import pandas as pd
from glob import glob
from utils1 import *
from tqdm import tqdm


file_list = {'accel.csv.gz':'hourly', 'ambientLight.csv.gz':'hourly', 'gps.csv.gz':'daily', 'powerState.csv.gz':'daily', 'smsLog.csv.gz':'daily',
             'sociabilityMsgLog.csv.gz':'daily', 'tapsLog.csv.gz':'daily', 'accessibilityLog.csv.gz':'daily', 'callLog.csv.gz':'daily',
             'heart.csv.gz':'hourly', 'sleep.csv.gz':'daily', 'sociabilityCallLog.csv.gz':'daily', 'steps.csv.gz':'daily'}


def comp_compliance(dir_bn):
	bn = os.path.basename(dir_bn)
	un = bn.split('_')[0]
	dt = pd.to_timedelta('1s')
	if un not in df_master.index:
		return 0.0
	enrol_date = df_master.loc[un, 'enrol_date']
	start_date = pd.to_datetime(enrol_date).tz_localize('tzlocal()') + pd.to_timedelta('1D')
	end_date = min(start_date+pd.to_timedelta(duration), pd.Timestamp.now('tzlocal()').floor('D')) - dt
	ret = {}
	for bn1, type in file_list.items():
		interval = '1H' if type=='hourly' else '1D'
		denom = (end_date+dt-start_date)/pd.to_timedelta(interval)
		fn1 = dir_bn+'/'+bn1
		if not os.path.exists(fn1):
			sc1 = 0.0
		else:
			df = load_and_preprocess(fn1)
			df1 = df.iloc[:, 0]
			sr1 = (df1.groupby(pd.Grouper(freq=interval)).count()>0)[start_date:end_date]
			sc1 = sr1.sum()/denom
		ret[bn1.split('.')[0]+f'({type})'] = sc1

	return pd.Series(ret)


if __name__=='__main__':
	parser = argparse.ArgumentParser(usage='$0 arg1 1>output 2>progress', description='compute compliance rate by looking at hourly presence of ambient light and heart rate',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('master_csv', help="the master CSV file")
	parser.add_argument('in_dirs', help="wildcard of patients' directories", nargs = '+')
	parser.add_argument('--duration', '-dur', help='optional argument', default = '7D')
	parser.add_argument('-optional', help='optional argument')
	#nargs='?': optional positional argument; action='append': multiple instances of the arg; type=; default=
	opt=parser.parse_args()
	globals().update(vars(opt))

	df_master = pd.read_csv(master_csv).set_index('user_id')

	out = {}
	paths = [path for patn in in_dirs for path in glob(patn)]
	for path in tqdm(paths):
		dir_bn = path.rstrip('/')
		out[os.path.basename(dir_bn).split('@')[0]] = comp_compliance(dir_bn)

	out_df = pd.DataFrame(out).transpose()

	print(out_df.sort_index().to_csv())