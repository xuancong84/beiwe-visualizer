#!/usr/bin/env python3

import os, sys, matplotlib, re, io, traceback, gzip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as trf
from math import isnan, nan, inf
from matplotlib.widgets import Slider
from glob import glob
from ipywidgets import *
from datetime import datetime, timedelta
from dateutil.tz import tzlocal
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML, Javascript
from termcolor import colored
from collections import *
from matplotlib import cm, colors
from matplotlib.patches import Patch


# disable the annoying cell scrolling (browser scroll is enough)
disable_js = """IPython.OutputArea.prototype._should_scroll = function(lines) {return false;} """
display(Javascript(disable_js))

# use full browser width
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.options.display.width = 1000
pd.options.display.max_columns = 1001
pd.options.display.max_rows = 10000
MAX_CHART_WIDTH = 400
MAX_XTICK_LABELS = 1200
CYCLE_COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'pink', 'brown', 'black'] # this also controls the max number of cycles to display
DAYOFWEEK_COLOR = {5:'purple', 6:'red'}

# set grid style
sns.set_style('darkgrid')

# initializations
# Setup all paths and sources
feature_list = ['accel', 'callLog', 'tapsLog', 'usage', 'accessibilityLog', 'gps', 'light', 'powerState', 'textsLog']
os.manual_file_data = {}
study_name_map = defaultdict(lambda:{}, {
	'GxXEPM08ZK0GS1gIaLe9YhEn' : {'Nikolas old ZTE':'16kbga47', 'Nikolas':'1s3g19f7', 'IMH1-Judy':'33kr56tx', 'IMH2-Amirah':'1tfan3jn',
								  'Nikolas old Nokia':'8e3ukdwy', 'Praveen':'d35pt9m4', 'Faye':'drdlfo5c', 'Robert':'gqmrnhvv', 'Xuancong':'hcy9th57'},
	'ow13XVde2Tj41dRypUyGf4ML' : {'Karthik':'uonp271d', 'John':'bwf2k411', 'MOHT':'4dt2cn7r', 'Robert':'3cjkxdnw', 'Alan':'1fbyhqo7', 'Nikola':'oodfytk3',
								  'Thisum':'xiu7qkje', 'Xuancong':'tzm4kfa3', 'Faye':'nbhv5uxm', 'Sung':'reorodmm', 'Amirah':'uylrw2aj'},
	'mxeicoqioghzxhrkdenwpmba' : {'Staff-test':'moht.dsth.140@moht.com.sg_7676561546b2', 'N':'moht.dsth.141@moht.com.sg_c314878c78ec', 'K':'moht.dsth.142@moht.com.sg_2470c70541b5',
								  'R':'moht.dsth.143@moht.com.sg_d75817f841a2', 'F':'moht.dsth.144@moht.com.sg_07842b0f0aba', 'P':'moht.dsth.145@moht.com.sg_fe7d87d448b3',
								  'X':'moht.dsth.146@moht.com.sg_929e9c909aaa', 'T':'moht.dsth.147@moht.com.sg_b42e85c44950', 'A':'moht.dsth.148@moht.com.sg_29508afb5d99', 'QR-test':'moht.dsth.149@moht.com.sg_92df56d445a6'}
})

# specify source folder directory
main_path = os.main_path if hasattr(os, 'main_path') else os.path.dirname(os.path.abspath(__file__))+'/2.decrypted/'
dropdown_studies = Dropdown(options=[(d,d) for d in os.listdir(main_path) if os.path.isdir(main_path+d)]
									+[('<manually selected files>',None)], description = 'Select Study')
dropdown_userlist = Dropdown(options=[])
def on_change_study(changes):
	global data_path, user_map0, user_map1, user_map, user_list, df_all, cols_all
	study = changes['new']
	if study is None:
		dropdown_userlist.options = user_list = ['']
		dropdown_userlist.value = ''
	else:
		data_path = main_path + study + '/'
		user_map0 = study_name_map[study]
		user_map1 = {v:k for k,v in user_map0.items()}
		user_map = lambda t:(user_map0[t] if t in user_map0 else t)
		user_list = [u for u in os.listdir(data_path) if os.path.isdir(data_path+'/'+u)]
		user_list = sorted([user_map1[t] for t in user_list if t in user_map1])+sorted([t for t in user_list if t not in user_map1])
		dropdown_userlist.options = user_list
		dropdown_userlist.value = user_list[0] if user_list else None
	df_all, cols_all = {}, {}
dropdown_studies.observe(on_change_study, names='value')

fileupload = FileUpload(accept='*', multiple=True, layout=Layout(width='300px'), description='Manually Select Files')
def on_change_files(changes):
	os.manual_file_data = {fn:(gzip.decompress(dct['content']) if fn.endswith('.gz') else dct['content']) for fn,dct in changes['new'].items()}
	dropdown_studies.value = None
	on_change_study({'new':None}) if dropdown_studies.value is None else None
fileupload.observe(on_change_files, names='value')

on_change_study({'new':dropdown_studies.options[0][1]})



# utility functions
def Open(fn, mode='r'):
	return gzip.open(fn, mode) if fn.lower().endswith('.gz') else open(fn, mode)

def parse_csv(L, repair=False, **kwargs):
	Ls = L.replace(b'\r', b' ').decode('utf8', 'ignore').splitlines()

	if repair:
		last_good_line = 0

		# fix span over multiple lines
		for i,L in enumerate(Ls):
			if i>0 and not L[0:13].isdigit() and not (len(L)>88 and ',' not in L[0:88]):
				Ls[last_good_line] = ''.join(Ls[last_good_line:i+1])
				Ls[i] = ''
			else:
				last_good_line = i

		# fix long field with comma inside square brackets
		nc = Ls[0].count(',')
		for i,L in enumerate(Ls):
			if L.count(',') in [nc, 0]:
				continue
			L1 = re.sub(r'(\[.*),(.*\])', '\\1\\2', L)
			while L1!=L:
				L1, L = re.sub(r'(\[.*),(.*\])', '\\1\\2', L1), L1
			Ls[i] = L1 if L1.count(',')==nc else ''

	txt = '\n'.join([L for L in Ls if L])
	return pd.read_csv(io.StringIO(txt), **kwargs) if txt.strip() else pd.DataFrame()

def load_csv(fn, repair=False, **kwargs):
	try:
		return parse_csv(Open(fn, 'rb').read(), repair, **kwargs)
	except:
		traceback.print_exc()
		print('CSV error: in File %s ...'%fn[len(data_path):])
		print('CSV content after processing:\n%s'%txt)
		return pd.DataFrame()

def load_df(user, feature, verbose=1):
	if isinstance(user, pd.DataFrame): return user
	if user:
		key = user + ' : ' + feature
		if key in df_all:
			if verbose>0:
				print('Loading data from cache ... [Username=%s, Feature=%s]'%(user, feature), flush=True)
			return df_all[key]

		if verbose>0:
			print('Loading data from files ... [Username=%s, Feature=%s]'%(user, feature), flush=True)

		fea_path = user if os.path.isfile(user) else os.path.join(data_path, user_map(user), feature)
		if os.path.isfile(fea_path):
			df = load_csv(fea_path, error_bad_lines=True, low_memory=False)
		else:
			fns = sorted(glob(fea_path+'/*.csv'))
			if fns:
				df = pd.concat([load_csv(fn, error_bad_lines=True) for fn in fns])
			else:
				if verbose>-1:
					display('Warning: %s is empty or does not exist!'%fea_path)
				df = pd.DataFrame()
	else:
		df = pd.concat([parse_csv(L, error_bad_lines=True) for L in os.manual_file_data.values()]) \
			if feature.startswith('<all ') else parse_csv(os.manual_file_data[feature], error_bad_lines=True)

	if 'timestamp' in df.columns:
		dt = pd.to_datetime(df['timestamp'], unit='ms', origin='unix', utc=True)
		df['datetime'] = pd.DatetimeIndex(dt).tz_convert(tzlocal())
		df = df.set_index('datetime', drop=True)

	if 'key' in locals():
		df_all[key] = df

	return df

def filter_by_date(df, StartDate=None, LastDate=None, DateOffset=0, details=None):
	if len(df)==0: return df
	earliest_date, latest_date = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
	start_date = earliest_date if StartDate == None else date2datetime(StartDate)
	end_date = latest_date if LastDate == None else date2datetime(LastDate) + timedelta(days=1)
	if DateOffset != 0:
		dateoffset = timedelta(days=1) * DateOffset
		start_date += dateoffset
		end_date += dateoffset
	if details is not None:
		details.start_date, details.end_date = start_date, end_date
	return df[(df.index >= start_date) & (df.index < end_date)]

def load_fea(Username):
	if isinstance(Username, pd.DataFrame): return [None]
	if not Username:
		return list(os.manual_file_data.keys()) + ['<all %d files>'%len(os.manual_file_data)]
	user_path = data_path+'/'+user_map(Username)
	return [fn for fn in sorted(os.listdir(user_path)) if (os.path.isdir(user_path+'/'+fn) or fn.endswith(".csv.gz") or fn.endswith(".csv"))]

def load_col(user, feature):
	if isinstance(user, pd.DataFrame): return user.columns
	if feature in cols_all: return cols_all[feature]
	if feature is None: return [None]
	if not user:
		df = parse_csv(os.manual_file_data[feature] if feature in os.manual_file_data else list(os.manual_file_data.values())[0], error_bad_lines=False)
	else:
		fea_path = os.path.join(data_path, user_map(user), feature)
		df = load_csv(glob(dir_path+'/*.csv')[0] if os.path.isdir(fea_path) else fea_path, error_bad_lines=False)
	return list(df.columns)

def draw_arrows(axes, df, TH=0.01):
	data = df[['timestamp', 'longitude', 'latitude']]
	_, x_span, y_span = data.max()-data.min()
	th = np.sqrt(x_span**2+y_span**2)*TH
	HW = np.sqrt(x_span**2+y_span**2)*.005
	HL = np.sqrt(x_span**2+y_span**2)*.01
	lx = ly = None
	for r in data.itertuples():
		x, y = r.longitude, r.latitude
		if lx != None:
			delta = np.sqrt((x-lx)**2+(y-ly)**2)
			if delta > th:
				axes.arrow((x+lx)*0.5, (y+ly)*0.5, (x-lx)*HW/delta, (y-ly)*HW/delta,
						   width=0, head_width=HW, head_length=HL)
		lx, ly = x,y

def date2datetime(d):
	return datetime(d.year, d.month, d.day, tzinfo=tzlocal())

def generate_colormap(N):
	S = 7
	arr = np.arange(N)/N
	N_up = int(np.ceil(N/S)*S)
	arr.resize(N_up, refcheck=False)
	arr = arr.reshape(S,N_up//S).T.reshape(-1)
	ret = matplotlib.cm.hsv(arr)
	n = ret[:,3].size
	a = n//2
	b = n-a
	for i in range(3):
		ret[0:n//2, i] *= np.arange(0.2+0.8/a, 1+0.8/a, 0.8/a).clip(0,1)
	ret[n//2:, 3] *= np.arange(1, 0.1, -0.9/b)
	return ret

def safe_display(df):
	if df.shape[0]>pd.options.display.max_rows:
		display(HTML('<font color=red>Displaying the head-and-tail %d out of %d rows</font>'%(pd.options.display.max_rows, df.shape[0])))
		N = pd.options.display.max_rows//2
		df = df.iloc[:N].append(df.iloc[-N:])
	display(df)

# every data point one tick, but labels must be sufficiently far apart
def calc_figsize_xticks(data, scale=1, width_mul=1):
	mul = max(1, width_mul//2+0.5)
	chart_width = min(MAX_CHART_WIDTH, len(data)*0.4*scale)
	figsize = [chart_width*mul, 3*scale]
	index = data.index if type(data)==pd.core.frame.DataFrame else [_[0] for _ in data]
	min_value_interval = (index[-1] - index[0])*MAX_CHART_WIDTH/(chart_width*MAX_XTICK_LABELS)/mul
	xticks = [str(g) for g in index]
	labels = [g for g in index]
	for ii,t in enumerate(labels):
		if ii==0 or t-prev_t>min_value_interval:
			prev_t = t
		else:
			labels[ii] = ''
	labels = list(map(str, labels))
	return figsize, xticks, labels

def split_by_cycle(df, cycle_in_days):
	ret = []
	cur_end_datetime = df.index.max()
	while len(df[df.index<=cur_end_datetime]):
		cur_start_datetime = cur_end_datetime-pd.to_timedelta('1D')*cycle_in_days
		ret += [df[(df.index<=cur_end_datetime) & (df.index>cur_start_datetime)]]
		cur_end_datetime = cur_start_datetime
	return ret[::-1]

def set_index_to_value(df, value, col=None):
	ret = df.copy()
	if col: ret[col] = [value]*len(ret)
	else: ret.index = [value]*len(ret)
	return ret

def sample(df, SelCol, N=256):
	inc = round(len(df)/N)
	if len(df)<=N or inc==1:
		return df
	ret = df.sort_values(SelCol)
	return ret.iloc[0:-1:inc].append(ret.iloc[-1])

def add_cycle_mean(df, Interval, cycle_in_days, SelCol=True):
	if SelCol is True:
		# compute cycle mean for each column
		col_cycle_means = [add_cycle_mean(df, Interval, cycle_in_days, col) for col in df.columns]
		# combine all columns for each cycle, the 1st N-1 columns are named current_cycle, previous_cycle1/2/..., must rename to the last column
		return [pd.concat([col_cycle_mean[[cycle]].rename(columns={cycle:col_cycle_mean.columns[-1]}) for col_cycle_mean in col_cycle_means], axis=1)
				for cycle in col_cycle_means[0].columns[:-1]] + [pd.concat([col_cycle_mean[col_cycle_mean.columns[-1:]] for col_cycle_mean in col_cycle_means], axis=1)]

	if 'DataFrameGroupBy' in str(type(df)):
		data_map = OrderedDict([[k, sample(v.reset_index(drop=True), SelCol)] for k,v in df])
		cycles = split_by_cycle(pd.DataFrame(index=data_map.keys()), cycle_in_days)
		if len(cycles)>1 and len(cycles[0])<7:	# merge the 1st 2 cycles if the 1st cycle is less than 1 week
			cycles = [cycles[0].append(cycles[1])] + cycles[2:]

		# create hues for current_value
		curr_cycle_dates = cycles[-1].index
		ret = pd.concat([set_index_to_value(data_map[i], i, col='datetime') for i in curr_cycle_dates], ignore_index=True)
		ret['hues'] = 'current_value'

		# create hues for each cycle
		for ii, cycle in enumerate(cycles[::-1]):
			ret1 = pd.concat([set_index_to_value(data_map[cycle_tms], curr_tms, col='datetime') for curr_tms in curr_cycle_dates \
							  for cycle_tms in cycle.index[cycle.index.dayofweek==curr_tms.dayofweek]], ignore_index=True)
			ret1['hues'] = 'current_cycle' if ii==0 else ('previous_cycle%d'%ii)
			ret = ret.append(ret1, ignore_index=True)

		# return hued DataFrame instead of DataFrameGroupBy
		return ret

	dfs = split_by_cycle(df, cycle_in_days)
	intv_in_sec = pd.to_timedelta(Interval).total_seconds()
	n_intv_per_week = round(pd.to_timedelta('1W').total_seconds()/intv_in_sec)
	cycle_means = [df1[[SelCol]].groupby(np.floor((df1.index.dayofweek*24*3600+df1.index.hour*3600+df1.index.minute*60+df1.index.second)/intv_in_sec)).mean()
				  for df1 in dfs[-1:-len(CYCLE_COLORS):-1]] # latest-first order
	if len(cycle_means[-1])<n_intv_per_week: # merge the last 2 cycles if the last cycle has not enough data for 1 week
		if len(cycle_means)<2:
			return dfs[-1]
		cycle_means = cycle_means[:-2]+[cycle_means[-1].append(cycle_means[-2])]
	ret = dfs[-1][[SelCol]]
	for ii,cm in enumerate(cycle_means):
		prev = cm.iloc[np.floor((ret.index.dayofweek*24*3600+ret.index.hour*3600+ret.index.minute*60+ret.index.second)/intv_in_sec)]
		ret = ret.join(prev.rename(columns={SelCol:('current_cycle' if ii==0 else 'previous_cycle%d'%ii)}).set_index(ret.index))
	return ret[ret.columns[::-1]]

def FFT(data, labels, figsize):
	tms_total = abs(data.index.max()-data.index.min())
	ret = pd.DataFrame(abs(np.fft.fft(data.fillna(0), axis=0)), columns=data.columns, index=[(tms_total/i if i else np.inf) for i in range(len(data))])
	return ret, ret.index, figsize

def calc_bar_width_posi(N):
	width = (0.8-(0.2 if N>1 else 0))/N
	posi = [-i*1.2+N*0.6 for i in range(N)]
	return width, posi

def convert_index_to_string(df, col=None):
	if col:
		df[col] = df[col].apply(lambda x:str(x))
	else:
		df.index = df.index.map(lambda x:str(x))
	return df

def safe_log(data, SelCol, TakeLog):
	if TakeLog:
		try:
			data.loc[:,SelCol] = np.log(data[SelCol]+1)
			return data, False
		except:
			return data, True
	return data, False

def stacked_log(df):
	df_sum = df.sum(axis=1)
	ret = df.mul(np.log(df_sum+1), axis=0).div(df_sum, axis=0).fillna(0)
	return ret




### MAIN
topNmax = 100
F1 = {'# of readings in each interval':'.count()', 'max value in each interval':'.max()', 'min value in each interval':'.min()', 'median value in each interval':'.mean()',
	  'mean value in each interval':'.mean()', 'std in each interval':'.std()', 'sum in each interval':'.sum()', 'grouped values by each interval':''}
daysofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
select_column0 = Dropdown(options=['value'], description='Select Column')
select_durCol0 = Dropdown(options=['<entry-count>', '<timestamp-diff>'], description='Duration Column')
sort_column = Dropdown(options=['no sort'], description='Sort by Col.')
drop1 = Dropdown(options=['mean', 'max', 'min', 'median', 'std'], value='mean', description='Agg. Func', layout={'visibility':'hidden'})
drop1_M = {'value heatmap':['Agg. Func', ['mean', 'max', 'min', 'median', 'std'], 'mean'],
		   'time chart stacked bar':['top N', list(range(1,topNmax+1))+[inf], 10],
		   'time chart stacked area':['top N', list(range(1,topNmax+1))+[inf], 10],
		   'frequency distribution (h-bar)':['top N', list(range(1,topNmax+1))+[inf], 10],
		   'frequency distribution (pie)':['top N', list(range(1,topNmax+1))+[inf], 10],
		   'histogram of values':['# of bins', list(range(2,201)), 20]}
feature_list0 = Dropdown(options=load_fea(user_list[0]) if user_list else feature_list, value=None)
function_list0 = Dropdown(options=list(F1.keys()) + ['value range in each interval', 'pass through selected', 'pass through all'])
intv_shift0 = FloatSlider(min=0, max=1, step=0.01, value=0, continuous_update=False, description='Interval Shift')
cycle_period0 = IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Cycle (days)')
dateoffset0 = widgets.BoundedFloatText(value=0, step=1, description='Date Offset')
interval0 = Dropdown(options=['1min', '5min', '15min', '30min', '1H', '2H', '3H', '6H', '12H', '1D', '2D', '1W', '1M'], value='1D', description='Bin Interval')
fig, axes, g_prevPlotType, g_lock = None, None, None, False
def draw(Username, StartDate, LastDate, DateOffset, ContOffset, Feature, Function, Interval, IntvShift, CyclePeriod, PlotType, SelCol, Extra, DurCol, ForwardFill,
		   SortByCol, TakeLog, TakeFFT, DrawArrow, SpreadXYaxis, DoPlot, verbose=1, size_ratio=1, post_processor=None, ax=None, plot_options={}, **kwargs):
	global fig, axes, g_prevPlotType

	if DoPlot!=None and not isinstance(Username, pd.DataFrame) and verbose:
		print([Username, StartDate, LastDate, DateOffset, ContOffset, Feature, Function, Interval, IntvShift, CyclePeriod,
			   PlotType, SelCol, Extra, DurCol, ForwardFill, SortByCol, TakeLog, TakeFFT, DrawArrow, SpreadXYaxis, DoPlot])

	if type(DoPlot) is bool:
		## Prepare control items
		dateoffset0.step = 0.01 if ContOffset else 1
		interval0.layout.visibility = 'hidden' if (Function.startswith('pass through') or PlotType.startswith('display')) else 'visible'
		function_list0.layout.visibility = 'hidden' if PlotType.startswith('display') else 'visible'

		# Set feature list
		feature_list0.options = load_fea(Username)
		feature_list0.value = Feature if Feature in feature_list0.options else feature_list0.options[0]

		# Set select columns => select_column0
		if Feature is not None:
			cols = load_col(Username, Feature)
			select_column0.layout.visibility = 'hidden' if PlotType=='XY path' or PlotType.startswith('display') else 'visible'
			select_column0.options = cols = [c for c in cols if c not in ['timestamp', 'datetime']]+[None]
			select_column0.value = SelCol = SelCol if SelCol in cols else ('value' if 'value' in cols else cols[0])
			select_durCol0.options = list(select_durCol0.options[:2])+cols
			sort_column.options = ['no sort'] + cols

		if SortByCol not in sort_column.options:
			SortByCol = sort_column.options[0]

		# Set Extra
		if PlotType != g_prevPlotType:
			if PlotType in drop1_M:
				drop1.description, drop1.options, drop1.value = drop1_M[PlotType]
				drop1.layout.visibility = 'visible'
			else:
				drop1.layout.visibility = 'hidden'
		g_prevPlotType = PlotType

		if not DoPlot:
			clear_output()
			return

	# Switch matplot backend
	if PlotType == 'XY path' and matplotlib.get_backend()!='nbAgg':
		plt.switch_backend('nbAgg')
	elif PlotType != 'XY path' and matplotlib.get_backend()=='nbAgg':
		plt.switch_backend('module://ipykernel.pylab.backend_inline')

	## Execute draw function
	dfa = load_df(Username, Feature, verbose=verbose)
	if DurCol=='<timestamp-diff>' and '<timestamp-diff>' not in dfa.columns and 'timestamp' in dfa.columns:
		dfa['<timestamp-diff>'] = dfa['timestamp'].diff().iloc[1:].append(pd.Series(), ignore_index=True)
	if len(dfa) == 0:
		display(HTML('<font color=red>Warning: the whole data is empty</font>'))
		return

	if DoPlot is not None and verbose:
		print('Processing data ...', flush=True)
	dfc = dfa.sort_values(SortByCol) if SortByCol!='no sort' else dfa.copy()
	df = dfc = dfc.ffill() if ForwardFill else dfc
	if StartDate!=None or LastDate!=None:
		df = filter_by_date(df, StartDate, LastDate, DateOffset, os)
		if verbose:
			print(colored('Specified Start Date: %.10s ; End Date: %.10s ;'%(os.start_date, os.end_date), 'red', attrs=['bold']), end=' ')

	# Warn and return if empty
	os.df = df
	if df.shape[0] == 0:
		display(HTML('<font color=red>Warning: selected data is empty</font>'))
		return

	if 'DatetimeIndex' in str(type(df.index)) and verbose:
		print(colored('Data Start Date: %s ; End Date: %s'%(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()), 'red', attrs=['bold']))

	scale = (0.9 if matplotlib.get_backend()=='nbAgg' else 1.0)*size_ratio
	figsize = [16*scale, 9*scale]

	# Transform values
	if PlotType.startswith('display'):
		data = df
	elif Function in F1:
		if not F1[Function]: df, TakeLog = safe_log(df, SelCol, TakeLog)
		data = (df[[SelCol,DurCol]] if DurCol!='<entry-count>' else df[[SelCol]]).groupby(pd.Grouper(freq=Interval, base=IntvShift))
		data = eval('data'+F1[Function])
	elif Function == 'value range in each interval':
		data1 = df[[SelCol]].groupby(pd.Grouper(freq=Interval, base=IntvShift))
		data = data1.min().rename(columns={SelCol:'min'})
		data['max'] = data1.max()[SelCol]-data['min']
	elif Function == 'pass through selected':
		data = df[[SelCol]]
	elif Function == 'pass through all':
		data = df

	data, TakeLog = safe_log(data, SelCol, TakeLog)


	### Start plotting
	os.data = data
	agg_fn = (lambda data, c : data[[c, DurCol]].groupby(c).sum().sort_values(DurCol, ascending=False)[DurCol]) \
		if DurCol!='<entry-count>' else (lambda data,c:data[c].value_counts())
	if hasattr(data,'shape') and data.shape[0] == 0:
		display(HTML('<font color=red>Warning: processed data is empty</font>'))
		return
	if PlotType.startswith('time chart stacked'):
		if SelCol is None: SelCol = list(data.columns)
		if type(SelCol)==list:
			selected_cls = SelCol
			N_cls = len(selected_cls)
		else:
			data0 = data.filter(lambda t:True) if 'DataFrameGroupBy' in str(type(data)) else data
			stats = agg_fn(data0, SelCol)
			N_cls_present = stats.size
			N_cls = min(Extra, N_cls_present)
			selected_cls = stats.index.tolist()[:N_cls]
			if N_cls_present > N_cls:
				data0[SelCol] = data0[SelCol].apply(lambda t:t if t in selected_cls else '<Others>')
				selected_cls += ['<Others>']
				N_cls += 1
			data = data0.groupby(pd.Grouper(freq=Interval, base=IntvShift))
			map_null = lambda t:t if len(t) else {selected_cls[0]:0}
			data = pd.DataFrame.from_dict({g[0]:map_null(agg_fn(g[1],SelCol)) for g in data}, orient='index').fillna(0).sort_index()
		if TakeLog: data = stacked_log(data)
		figsize, xticks, labels = calc_figsize_xticks(data, scale)
		if post_processor is not None:
			data, labels, figsize  = post_processor(data, labels, figsize)
		if PlotType.endswith('bar'):
			if CyclePeriod:
				datas = add_cycle_mean(data, Interval, CyclePeriod)
				N = len(datas)
				figsize, xticks, labels = calc_figsize_xticks(datas[0], scale, N)
				width, posi = calc_bar_width_posi(N)
				for i, data in enumerate(datas):
					xy_plot = data[selected_cls[::-1]].plot.bar(stacked=True, rot=45, position=posi[i], width=width,
																ax=xy_plot if i else ax, figsize=figsize, color=generate_colormap(N_cls))
			else:
				xy_plot = data[selected_cls[::-1]].plot.bar(stacked=True, rot=45, figsize=None if ax else figsize, ax=ax, color=generate_colormap(N_cls))
		elif PlotType.endswith('area'):
			xy_plot = data[selected_cls[::-1]].plot.area(stacked=True, rot=45, figsize=None if ax else figsize, color=generate_colormap(N_cls))
		xy_plot.set_xticklabels(labels, ha='right')
		if ax is None:
			xy_plot.get_figure().subplots_adjust(right=0.8)
			lhs, lls = xy_plot.get_legend_handles_labels()
			lhs, lls = lhs[::-1], lls[::-1]
			if 'datas' in locals():
				lhs = lhs[:len(lhs)//N]+[matplotlib.patches.Rectangle((0,0), 1, 1, edgecolor='none', visible=False)]*N
				lls = lls[:len(lls)//N]+[('bar%d : '%(N-1-i))+('previous_cycle%d'%i if i else 'current_cycle') for i in range(N-1)][::-1]+['bar%d : current_value'%N]
				xy_plot.legend(lhs, lls, loc='center left', prop={'size': 10}, bbox_to_anchor=(1,0,0.2,1))
			else:
				xy_plot.legend(lhs, lls, loc='center left', prop={'size': 10}, bbox_to_anchor=(1,0,0.2,1))
	elif PlotType == 'time chart grouped box plot':
		if CyclePeriod:
			data = add_cycle_mean(data, Interval, CyclePeriod, SelCol)
			figsize, xticks, labels = calc_figsize_xticks(pd.DataFrame(index=sorted(data.datetime.unique())), scale, data.hues.nunique())
			if ax is None:
				fig, ax = plt.subplots(figsize=figsize)
			data = convert_index_to_string(data, 'datetime')
			xy_plot = sns.boxplot(x=data.datetime, y=SelCol, hue="hues", data=data, ax=ax,
								  hue_order=[('previous_cycle%d'%i if i else 'current_cycle') for i in range(data.hues.nunique()-1)]+['current_value'])
			xy_plot.legend(loc='center left', prop={'size': 10}, bbox_to_anchor=(1, 0, 0.2, 1))
		else:
			figsize, xticks, labels = calc_figsize_xticks(pd.DataFrame(index=[i for i, j in data]), scale)
			data = data.apply(lambda x:x.reset_index(drop=True)).droplevel(1).reset_index()
			if post_processor is not None:
				data, labels, figsize = post_processor(data, labels, figsize)
			xy_plot = data.boxplot(by='datetime', figsize=figsize, ax=ax)
			xy_plot.get_figure().suptitle('')
		os.data = data
		xy_plot.set_xticklabels(labels, ha='right', rotation=45)
	elif PlotType.startswith('time chart'):
		if TakeLog: data = safe_log(data)[0]
		if CyclePeriod: data = add_cycle_mean(data, Interval, CyclePeriod, SelCol)
		figsize, xticks, labels = calc_figsize_xticks(data, scale, len(data.columns) if CyclePeriod else 1)
		if TakeFFT:
			data, labels, figsize = FFT(data, labels, figsize)
		if post_processor is not None:
			data, labels, figsize = post_processor(data, labels, figsize)
		if 'bar' in PlotType:
			xy_plot = data.plot.bar(figsize=None if ax else figsize, rot=45, ax=ax, **plot_options)
		elif 'scatter' in PlotType:
			data['tms'] = data.index.astype(int)
			if CyclePeriod:
				xy_plot = data.plot.scatter(x='tms', y=SelCol, figsize=None if ax else figsize, ax=ax, color=CYCLE_COLORS[0], label=SelCol,
											xticks=data.tms, rot=45, xlim=(data.tms[0], data.tms[-1]))
				for cycle_n in range(len(data.columns)):
					cycle_name = 'current_cycle' if cycle_n==0 else ('previous_cycle%d'%cycle_n)
					if cycle_name not in data.columns: break
					data.plot.scatter(x='tms', y=cycle_name, figsize=figsize, ax=xy_plot, color=CYCLE_COLORS[cycle_n+1], label=cycle_name,
									  xticks=data.tms, rot=45, xlim=(data.tms[0], data.tms[-1]))
			else:
				xy_plot = data.plot.scatter(x='tms', y=SelCol, figsize=None if ax else figsize, xticks=data.tms, ax=ax, rot=45, xlim=(data.tms[0], data.tms[-1]))
		elif 'line' in PlotType:
			xy_plot = data.plot.line(figsize=None if ax else figsize, xticks=xticks, rot=45, ax=ax)
		xy_plot.set_xticklabels(labels, ha='right')
		if ax is None:
			xy_plot.legend(loc='center left', prop={'size': 10}, bbox_to_anchor=(1,0,0.2,1))
	elif PlotType == 'value heatmap':
		fig, ax = plt.subplots(figsize=[v*0.8 for v in figsize])
		data.loc[:,'day_of_week'] = data.index.dayofweek
		data.loc[:,'hour'] = data.index.hour
		piv = pd.pivot_table(data, values=SelCol, index='hour', columns='day_of_week', fill_value=0, aggfunc=eval('np.'+Extra))
		xy_plot = sns.heatmap(piv, annot=True, cmap="plasma", fmt='.5g', linewidths=1, xticklabels=daysofweek)
		xy_plot.invert_yaxis()
		plt.tight_layout()
		plt.title(SelCol)
	elif PlotType == 'XY path':
		dfs = df if SpreadXYaxis else dfc
		x_max, x_min = float(dfs[['longitude']].max()), float(dfs[['longitude']].min())
		y_max, y_min = float(dfs[['latitude']].max()), float(dfs[['latitude']].min())
		D = max((x_max-x_min), (y_max-y_min))*0.05
		xy_plot = df.plot.line(x='longitude', y='latitude', xlim=[x_min-D, x_max+D], ylim=[y_min-D, y_max+D], figsize=figsize)
		xy_plot.set_aspect(1)
		if DrawArrow:
			draw_arrows(xy_plot, df)
	elif PlotType == 'histogram of values':
		xy_plot = data.plot.hist(figsize=figsize, bins=Extra)
	elif PlotType.startswith('frequency distribution'):
		data = agg_fn(data, SelCol)
		N_classes = min(Extra, len(data))
		sel_classes = data[:N_classes]
		if len(data)>N_classes:
			sel_classes['<Others>'] = data[N_classes:].sum()
		data = sel_classes[::-1]
		N_cls_total = len(data)
		if verbose:
			print('Total number of categories (including [other]) = %d'%N_cls_total)
		if 'pie' in PlotType:
			fig_sz = np.clip(N_cls_total,8,24)
			xy_plot = data.plot.pie(figsize=[fig_sz,fig_sz], title='[%s]'%SelCol, colors=generate_colormap(N_cls_total))
		else:
			figsize[1] = max(4, len(data)*9/40)
			xy_plot = data.plot.barh(figsize=figsize, title='[%s]'%SelCol)
	elif PlotType == 'show pipeline processed data':
		safe_display(data)
	elif PlotType == 'display selected/pre-computed data':
		safe_display(df)
	elif PlotType == 'display raw unprocessed data':
		safe_display(dfa)
	else:
		xy_plot = data.plot()

	if DoPlot is not None and verbose:
		print('Loading finished! Plotting ...', flush=True)
	if 'xy_plot' in locals():
		# set Sunday xlabels to red
		for xlab in xy_plot.get_xticklabels():
			try:
				dow = pd.Timestamp(xlab._text).dayofweek
				if dow in DAYOFWEEK_COLOR:
					xlab.set_color(DAYOFWEEK_COLOR[dow])
			except:
				pass

		# execute custom SubplotAxes options
		for k,v in kwargs.items():
			try:
				eval("xy_plot.%s"%k)(v)
			except:
				traceback.print_exc()
		return xy_plot

def update(**kwargs):
	global g_lock
	if not g_lock:
		g_lock = True
		try:
			return draw(**kwargs)
		except:
			traceback.print_exc()
		finally:
			g_lock = False

W = interactive(update,
	Username = dropdown_userlist, StartDate = DatePicker(value=None), LastDate = DatePicker(value=None), DateOffset = dateoffset0, ContOffset = Checkbox(value=False, description="Continuous Date Offset"), # 0-4
	Feature = feature_list0, Function = function_list0, Interval = interval0, IntvShift = intv_shift0, CyclePeriod = cycle_period0, # 5-9
	PlotType = ['time chart (bar)', 'time chart (line)', 'time chart (scatter)', 'time chart stacked bar', 'time chart stacked area', 'time chart grouped box plot',
				'value heatmap', 'XY path', 'frequency distribution (h-bar)', 'frequency distribution (pie)', 'histogram of values', 'show pipeline processed data',
				'display selected/pre-computed data', 'display raw unprocessed data'], Extra = drop1, SelCol = select_column0, DurCol = select_durCol0, ForwardFill = False, # 10-14
	SortByCol = sort_column, TakeLog = False, TakeFFT = False, DrawArrow = False, SpreadXYaxis = False,
	DoPlot = ToggleButton(value=False, description='Update Plot') # -2
	)

