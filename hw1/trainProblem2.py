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
	
	# include all features
#	includedAttr = Attr
	
	# include only PM2.5
	includedAttr = ['1PM2.5-' + str(i) for i in range(9, 0, -1)]

	includedAttr.append('1Const')

	for item in includedAttr:
		SetZero[item] = 1
	
	Weight = minSSE(TrainData, Attr, SetZero, 0.1, 1e-9, 1e3, True, False)
	FinalSSE = evalLoss(TrainData, Weight)
	print('\nFinal SSE: %f' % FinalSSE)

	print(includedAttr)
	print('Weight:')
	for key in includedAttr:
		print('\t%s: %f' % (key, Weight[key]))
	ResultDF = pd.DataFrame(Weight, index=[0])
	ResultDF.to_csv('./Coefficient.csv')
