#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
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
	
	includedAttr = ['1PM10-2', '1PM10-6', '1PM2.5-1', '1PM2.5-2', '1PM2.5-5', \
					'1PM2.5-6', '1NO2-2', '1NO2-4', '1WIND_SPEED-3', '1O3-2', \
					'1O3-4', '1RH-1', '1SO2-1', '1NO-4']

	for item in includedAttr:
		SetZero[item] = 1
	
	REG = 1000000
	
	Weight = \
		minSSE(TrainData, Attr, SetZero, 1, 1e-9, 1e3, True, False, True, REG)
	FinalSSE = evalLoss(TrainData, Weight)
	print('\nFinal SSE: %f' % FinalSSE)
	print('L2 norm: %f' % L2Norm(Weight))
	print('Weight:')
	for key in includedAttr:
		print('\t%s: %f' % (key, Weight[key]))
	ResultDF = pd.DataFrame(Weight, index=[0])
	ResultDF.to_csv('./Coefficient.csv')
