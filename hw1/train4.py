#-*- coding=utf-8 -*-
'''
Directly test total SSE

'''

import pandas as pd
import numpy as np
from copy import deepcopy
from functionForTrain import *

if __name__ == '__main__':
	TrainData = pd.read_csv('./data/TrainData.csv', index_col=0)
	TrainData['1Const'] = 1
	
	SetZero = {}
	Attr = []
	ConsideredDim = [1]
	OriAttr = list(TrainData)
	OriAttr.remove('PM2.5-0')
	OriTrainData = TrainData
	
	y_STD = np.std(TrainData['PM2.5-0'].tolist())
	y_mean = np.mean(TrainData['PM2.5-0'].tolist())
	
	for key in OriAttr:
		print('Processing %s' % key)
		if key == '1Const':
			Attr.append(key)
			SetZero[key] = 0
		else:
			for d in ConsideredDim:
				Attr.append(str(d) + key)
				SetZero[str(d) + key] = 0
				if d == 1:
					TrainData.rename(columns={key: str(d) + key}, inplace=True)
				else:
					TrainData[str(d) + key] = \
						TrainData.apply(lambda r: r[key] ** d, axis=1)
	
	for i in range(len(TrainData['PM2.5-0']) - 1, 0, -1):
		if abs(TrainData['PM2.5-0'][i] - y_mean) > 2.5 * y_STD:
			print('Delete ID-%i: %f' % (i, float(TrainData['PM2.5-0'][i])))
			TrainData.drop(TrainData.index[i])
	TrainData.reset_index(drop=True)

	BestAttr = deepcopy(SetZero)
	
	TryCombin = [ \
		['1PM10-1', '1PM10-2', '1PM10-5', '1PM2.5-1', '1PM2.5-2', '1PM2.5-4', '1PM2.5-5', \
			'1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']]

# 3051817.544056: 14 (3046147.078797)
#['1PM10-2', '1PM10-6', '1PM2.5-1', '1PM2.5-2', '1PM2.5-5', '1PM2.5-6', '1NO2-2', \
#'1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4', '1RH-1', '1SO2-1', '1NO-4']

# 3053085.671163: 11
#['1PM10-2', '1PM2.5-1', '1PM2.5-2', '1PM2.5-5', '1NO2-2', '1NO2-4', \
#'1WIND_SPEED-3', '1O3-2', '1O3-4', '1SO2-1', '1NO-4']

# 3053215.980394: 10
#['1PM10-2', '1PM2.5-1', '1PM2.5-2', '1PM2.5-5', '1NO2-2', '1NO2-4', \
#'1WIND_SPEED-3', '1O3-2', '1O3-4', '1SO2-1']

# 3039065.221336: 13
#['1PM10-2', '1PM2.5-1', '1PM2.5-2', '1PM2.5-3', '1PM2.5-4', '1PM2.5-5', '1PM2.5-6',\
#'1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4', '1SO2-1']

# 3039759.245667: 12
#['1PM10-2', '1PM2.5-1', '1PM2.5-2', '1PM2.5-3', '1PM2.5-4', '1PM2.5-5', \
#'1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

# 3039840.772900: 11
#['1PM10-2', '1PM2.5-1', '1PM2.5-2', '1PM2.5-4', '1PM2.5-5', \
#'1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

# 3038975.332694: 14
#['1PM10-1', '1PM10-2', '1PM10-3', '1PM10-4', '1PM2.5-1', '1PM2.5-2', \
#'1PM2.5-4', '1PM2.5-5', '1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

# 3037450.069252: 13
#['1PM10-1', '1PM10-2', '1PM10-4', '1PM2.5-1', '1PM2.5-2', \
#'1PM2.5-4', '1PM2.5-5', '1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

# 3035615.042241: 13
#['1PM10-1', '1PM10-2', '1PM10-5', '1PM2.5-1', '1PM2.5-2', \
#'1PM2.5-4', '1PM2.5-5', '1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

# 3034718.549238: 12 (3029904.482520) -> overfit
#['1PM10-1', '1PM10-2', '1PM10-5', '1PM2.5-1', '1PM2.5-2', \
#'1PM2.5-4', '1PM2.5-5', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', '1O3-4']

	AddedAttr = 0
		
	TotalSSE = {}
	AttrTest = {}
	for t in range(len(TryCombin)):
		print('\nStart a new try!: %i' % (t + 1))
		print(TryCombin[t])
		TotalSSE[t] = 0
		AttrTest[t] = deepcopy(BestAttr)
		
		for item in TryCombin[t]:
			AttrTest[t][item] = 1

		wtest = minSSE(TrainData, Attr, AttrTest[t], 100, 1e-4, 1e6, False)
		TotalSSE[t] = evalLoss(TrainData, wtest)
		print('SSE: %f' % TotalSSE[t])
		print('Weight:')
		for key in TryCombin[t]:
			print('\t%s: %f' % (key, wtest[key]))
	
	CurrBest = min(TotalSSE, key=TotalSSE.get)
	SetZero = deepcopy(AttrTest[CurrBest])
	BestSSE = TotalSSE[CurrBest]

	Weight = minSSE(TrainData, Attr, SetZero, 100, 1e-9, 1e8, False)
	FinalSSE = evalLoss(TrainData, Weight)
	print('\nFinal SSE: %f' % FinalSSE)

	print(TryCombin[CurrBest])
	print('Weight:')
	for key in TryCombin[CurrBest]:
		print('\t%s: %f' % (key, Weight[key]))
	ResultDF = pd.DataFrame(Weight, index=[0])
	ResultDF.to_csv('./Coefficient.csv')
