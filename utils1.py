import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, gzip


def Open(fn, mode = 'r', **kwargs):
	if fn == '-':
		return sys.stdin if mode.startswith('r') else sys.stdout
	fn = os.path.expanduser(fn)
	return gzip.open(fn, mode, **kwargs) if fn.lower().endswith('.gz') else open(fn, mode, **kwargs)


def convert_timestamp(timestamp, zone = 'UTC'):
	datetime = pd.to_datetime(timestamp, unit = 'ms', origin = 'unix')
	if zone == 'SG':
		datetime = datetime.tz_localize('UTC').tz_convert('Asia/Singapore')
	return datetime


def localize_date(date):
	datetime = pd.to_datetime(date)

	return datetime.tz_localize('UTC').tz_convert('Asia/Singapore')


def create_df(path, directory):
	df_raw = pd.DataFrame({})
	dir_path = os.path.join(path, directory)
	for file in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file)
		df_raw = pd.concat([df_raw, pd.read_csv(file_path, error_bad_lines = False)])

	df = pd.DataFrame({})
	if 'timestamp' in df_raw.columns:
		df = df_raw[df_raw.timestamp != 'timestamp']
		df['timestamp'] = df['timestamp'].astype('str')
		#         df = df[df['timestamp'].str.len() == 13]
		df['datetime'] = df.timestamp.apply(convert_timestamp, zone = 'SG')
		df['day_of_week'] = df.datetime.dt.dayofweek
		df['weekday_name'] = df.datetime.dt.weekday_name
		df['hour'] = df.datetime.dt.hour

	if directory == 'tapsLog':
		df.fillna(method = 'ffill', inplace = True)

	return df


def plot_measurement_heatmap(df, cat, norm = False):
	plot_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name = 'count')
	if norm:
		plot_data['count'] = (plot_data['count'] - plot_data['count'].mean()) / plot_data['count'].std()

	fig, ax = plt.subplots(figsize = (10, 10))

	piv = pd.pivot_table(plot_data, values = 'count', index = ['hour'], columns = ['day_of_week'], fill_value = 0)
	ax = sns.heatmap(piv, square = True, cmap = "BuPu")
	ax.invert_yaxis()
	plt.tight_layout()
	plt.title(cat)


def plot_apps_usage(taps_data, start_date, end_date):
	app_taps_activity = taps_data['in_app_name'].value_counts()
	android_apps = app_taps_activity.index.str.contains('android')
	no_android_apps = app_taps_activity[~android_apps]
	no_android_apps.index.name = 'Apps'
	no_android_apps = no_android_apps.reset_index(name = 'Taps')

	f, ax = plt.subplots(figsize = (9, 8))
	sns.barplot(x = 'Taps', y = 'Apps', data = no_android_apps, orient = 'h')
	plt.yticks(fontsize = 14)
	plt.title(f'{start_date.date()} to {end_date.date()}')


def gps_heatmap(x, y):
	'''
	The point intensity corresponds to the times a place is visited.
	'''
	from scipy.stats import gaussian_kde
	import numpy as np

	xy = np.vstack([x, y])
	z = gaussian_kde(xy)(xy)

	fig, ax = plt.subplots()
	fig.set_size_inches(10, 7)

	ax.scatter(x, y, c = z, s = z * 0.5, edgecolor = '')

	plt.plot(x, y, c = 'purple', linewidth = 0.8)
	plt.show()


def preprocess_df(df):
	# convert timestamp to datetime and set as index
	dt = pd.to_datetime(df['timestamp'], unit = 'ms', origin = 'unix', utc = True)
	df = df.set_index(pd.DatetimeIndex(dt).tz_convert('tzlocal()')).sort_index()

	# remove rows with duplicate timestamp
	df = df.loc[~df.index.duplicated(keep = 'last')]
	df.index.name = 'datetime'
	return df


def load_and_preprocess(fn):
	df = pd.read_csv(Open(fn))
	return preprocess_df(df)
