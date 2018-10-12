#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from functionForTrain import *

def generateTrainData(RawDataDir, ifReturnDF=False):
	RawData = pd.read_csv(RawDataDir, encoding='big5')
	mapping = {'NR': 0}
	for i in range(24):
		RawData.replace({str(i): mapping}, inplace=True)
	PartialTitle = RawData['測項'][0:18].tolist()

	Title = [PT + '-' + str(i + 1) for i in range(9) for PT in PartialTitle]
	Title.insert(0, 'PM2.5-0')

	TrainData = pd.DataFrame(columns = Title)

	# one group contains 20 days
	numDay = len(RawData['日期'].unique().tolist())
	hourTag = [str(i) for i in range(24)]

	for m in range(numDay // 20):
		print('Processing: %i' % (m + 1))
		MonthlyData = \
			RawData.loc[range(m * 360, m * 360 + 18), '測項'].reset_index(drop=True)
		for d in range(20):
			NewDay = \
				RawData.loc[range(m * 360 + d * 18, m * 360 + d * 18 + 18), hourTag]
			NewDay = NewDay.reset_index(drop=True)
			NewDay.set_axis([str(i + 24 * d) for i in range(24)], \
				axis=1, inplace=True)
			MonthlyData = pd.concat([MonthlyData, NewDay], axis= 1)
		MonthlyData.set_index('測項', inplace=True)
		SmoothList = ['PM2.5', 'PM10', 'NO2', 'WIND_SPEED', 'O3', 'RH', 'SO2', 'NO']
		for s in SmoothList:
			TBSmoothed = np.array(list(map(float, MonthlyData.loc[s, :].tolist())))
			TBSmoothed = smooth(TBSmoothed, 3, 'hanning').tolist()
			MonthlyData.loc[s, :] = TBSmoothed

		for hr in range(9, 20 * 24):
			inputData = {}
			inputData['PM2.5-0'] = MonthlyData[str(hr)]['PM2.5']
			for pa in range(1, 10):
				for r in range(len(PartialTitle)):
					inputData[PartialTitle[r] + '-' + str(pa)] = \
						MonthlyData[str(hr - pa)][PartialTitle[r]]
			TrainData = pd.concat([TrainData, \
				pd.DataFrame([inputData], columns=inputData.keys())], \
							ignore_index=True)

	if ifReturnDF:
		return TrainData
	else:
		TrainData.to_csv('./data/TrainData.csv')

if __name__ == '__main__':
	print('Don''t run this py file!!')
