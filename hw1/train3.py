#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from functionForTrain import *

if __name__ == '__main__':
	TrainData = pd.read_csv('./data/TrainData.csv', index_col=0)
	
	DataL = len(TrainData['PM2.5-0'])
	NumGroup = mp.cpu_count()
	TrainData['1Const'] = 1
	
	SetZero = {}
	t_ratio = {}
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
			t_ratio[key] = 0
		else:
			for d in ConsideredDim:
				Attr.append(str(d) + key)
				SetZero[str(d) + key] = 0
				t_ratio[str(d) + key] = 0
				if d == 1:
					TrainData.rename(columns={key: str(d) + key}, inplace=True)
				else:
					TrainData[str(d) + key] = \
						TrainData.apply(lambda r: r[key] ** d, axis=1)
	
	for i in range(len(TrainData['PM2.5-0']) - 1, 0, -1):
		if abs(TrainData['PM2.5-0'][i] - y_mean) > 2 * y_STD:
			print('Delete ID-%i: %f' % (i, float(TrainData['PM2.5-0'][i])))
			TrainData.drop(TrainData.index[i])
	TrainData.reset_index(drop=True)
	TrainData = TrainData.assign(GroupID = lambda x: x.index % NumGroup)

	# Choose candidates of columns: corr > 0.3
	BestAttr = deepcopy(SetZero)
	
	TryCombin = [['1PM10-2', '1PM2.5-1']]
	
	AddedAttr = 0
		
	TotalSSE = {}
	AttrTest = {}
	wtest = {}
	for t in range(len(TryCombin)):
		print('Start a new try!')
		print(TryCombin[t])
		TotalSSE[str(t)] = 0
		AttrTest[str(t)] = deepcopy(BestAttr)
		
		for item in TryCombin[t]:
			AttrTest[str(t)][item] = 1

		wtest[str(t)] = [[] for _ in range(NumGroup)]
		pool = mp.Pool()
		print('MT start~')
		for i in range(NumGroup):
			wtest[str(t)][i] = pool.apply_async(minSSE, \
				args=(TrainData[TrainData['GroupID'] != i], Attr, \
				AttrTest[str(t)], 10000, False, 1e-6, 1e5, NumGroup, False))
		pool.close()
		pool.join()
		print('MT stop~')

		for i in range(NumGroup):
			TotalSSE[str(t)] += evalLoss(TrainData[TrainData['GroupID'] == i], \
							wtest[str(t)][i].get())
		print('SSE: %f' % TotalSSE[str(t)])
	
	CurrBest = min(TotalSSE, key=TotalSSE.get)
	SetZero = deepcopy(AttrTest[CurrBest])
	BestSSE = TotalSSE[CurrBest]

	Weight = minSSE(TrainData, Attr, SetZero, 10000, False, 1e-8, 1e8, NumGroup, False)
	FinalSSE = evalLoss(TrainData, Weight)

#	wtest = [[] for _ in range(NumGroup)]
#	pool = mp.Pool()
#	print('MT start~')
#	for i in range(NumGroup):
#		wtest[i] = pool.apply_async(minSSE, \
#			args=(TrainData[TrainData['GroupID'] != i], Attr, \
#			AttrTest[CurrBest], 10000, False, 1e-8, 1e8, NumGroup, False))
#	pool.close()
#	pool.join()
#	print('MT stop~')
#
#	TotalSSE = {}
#	for i in range(NumGroup):
#		TotalSSE[str(i)] = evalLoss(TrainData, wtest[i].get())
#	
#	BestSSE = min(TotalSSE, key=TotalSSE.get)
#	Weight = deepcopy(wtest[int(BestSSE)].get())
	print('Final SSE: %f' % FinalSSE)

	print(TryCombin[int(CurrBest)])
	ResultDF = pd.DataFrame(Weight, index=[0])
	
	print(ResultDF)
	ResultDF.to_csv('./Coefficient.csv')

	#CovMat = TrainData.cov()
	#for key in t_ratio:
		#if SetZero[key] == 0:
			#t_ratio[key] = 1e100
		#else:
			#t_ratio[key] = abs(Weight[key] / np.sqrt(CovMat[key[1:]][key[1:]]))
