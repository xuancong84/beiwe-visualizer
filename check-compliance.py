#!/usr/bin/env python3
# coding=utf-8

import os, sys, argparse, re, gzip
import pandas as pd
from glob import glob
from utils1 import *
from tqdm import tqdm


def comp_compliance(dir_bn):
	bn = os.path.basename(dir_bn)
	un = bn.split('_')[0]
	if un not in df_master.index:
		return 0.0
	enrol_date = df_master.loc[un, 'enrol_date']
	start_date = pd.to_datetime(enrol_date).tz_localize('tzlocal()') + pd.to_timedelta('1D')
	end_date = min(start_date+pd.to_timedelta(duration), pd.Timestamp.now('tzlocal()').floor('D'))
	denom = (end_date-start_date)/pd.to_timedelta('1H')
	end_date -= pd.to_timedelta('1s')
	if not os.path.exists(dir_bn+'/ambientLight.csv.gz'):
		sc1 = 0.0
	else:
		df1 = load_and_preprocess(dir_bn+'/ambientLight.csv.gz')['value']
		sr1 = (df1.groupby(pd.Grouper(freq='1H')).count()>0)[start_date:end_date]
		sc1 = sr1.sum()/denom
	if not os.path.exists(dir_bn+'/heart.csv.gz'):
		sc2 = 0.0
	else:
		df2 = load_and_preprocess(dir_bn+'/heart.csv.gz')['HR']
		sr2 = (df2.groupby(pd.Grouper(freq='1H')).count()>0)[start_date:end_date]
		sc2 = sr2.sum()/denom
	return (sc1+sc2)*.5


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

	print(pd.Series(out).sort_index().to_csv())