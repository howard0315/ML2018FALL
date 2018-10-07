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
			SetZero[key] = 1
			t_ratio[key] = 0
		else:
			for d in ConsideredDim:
				Attr.append(str(d) + key)
				SetZero[str(d) + key] = 0
				t_ratio[str(d) + key] = 0
				TrainData[str(d) + key] = \
					TrainData.apply(lambda r: r[key] ** d, axis=1)
	
	for i in range(len(TrainData['PM2.5-0']) - 1, 0, -1):
		if abs(TrainData['PM2.5-0'][i] - y_mean) > 2 * y_STD:
			print('Delete ID-%i: %f' % (i, float(TrainData['PM2.5-0'][i])))
			TrainData.drop(TrainData.index[i])
	TrainData.reset_index(drop=True)
	TrainData = TrainData.assign(GroupID = lambda x: x.index % NumGroup)

	# Choose candidates of columns: corr > 0.3
	Candidate = []
	BestAttr = deepcopy(SetZero)
	BestSSE = 1e100
	for key in Attr:
		if abs(TrainData[key].corr(TrainData['PM2.5-0'])) > 0.4:
			print('Find a candidate: %s (corr = %f)' % \
					(key,TrainData[key].corr(TrainData['PM2.5-0'])))
			Candidate.append(key)
	
	AddedAttr = 0
	while True:
		print('Start a new epoch!')
		TotalSSE = {}
		AttrTest = {}
		wtest = {}
		for key in Candidate:
			print('Start testing %s' % key)
			TotalSSE[key] = 0
			AttrTest[key] = deepcopy(BestAttr)
			
			AttrTest[key][key] = 1
	
			wtest[key] = [[] for _ in range(NumGroup)]
			pool = mp.Pool()
			print('MT start~')
			for i in range(NumGroup):
				wtest[key][i] = pool.apply_async(minSSE, \
					args=(TrainData[TrainData['GroupID'] != i], Attr, \
					AttrTest[key], 10000, False, 1e-6, 1e4, NumGroup, False))
			pool.close()
			pool.join()
			print('MT stop~')
	
			for i in range(NumGroup):
				TotalSSE[key] += \
					evalLoss(TrainData[TrainData['GroupID'] == i], \
								wtest[key][i].get())
		
		CurrBest = min(TotalSSE, key=TotalSSE.get)
		if TotalSSE[CurrBest] < BestSSE:
			BestAttr = deepcopy(AttrTest[CurrBest])
			BestSSE = TotalSSE[CurrBest]
			Candidate.remove(CurrBest)
			print('Add %s to the model!' % CurrBest)
			AddedAttr += 1
		else:
			print('Stop searching, %i variables added!' % AddedAttr)
			SetZero = deepcopy(BestAttr)
			break

	Weight = \
		minSSE(TrainData, Attr, SetZero, 10000, False, 1e-7, 1e10, NumGroup, True)
	
	ResultDF = pd.DataFrame(Weight, index=[0])
	
	print(ResultDF)
	ResultDF.to_csv('./Coefficient.csv')

	#CovMat = TrainData.cov()
	#for key in t_ratio:
		#if SetZero[key] == 0:
			#t_ratio[key] = 1e100
		#else:
			#t_ratio[key] = abs(Weight[key] / np.sqrt(CovMat[key[1:]][key[1:]]))
