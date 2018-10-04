#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functionForTrain import *

if __name__ == '__main__':
	TrainData = pd.read_csv('./data/TrainData.csv', index_col=0)
	
	DataL = len(TrainData['PM2.5-0'])
	NumGroup = 10
	TrainData = TrainData.assign(GroupID = lambda x: x.index % NumGroup)
	TrainData['Const'] = 1

	SetZero = {}
	t_ratio = {}
	x_STD = {}
	Attr = []
	SSE = 1e100
	ConsideredDim = [1]
	OriAttr = list(TrainData)
	OriAttr.remove('PM2.5-0')
	OriAttr.remove('GroupID')
	OriTrainData = TrainData
	
	#y_STD = np.std(TrainData['PM2.5-0'].tolist())
	#TrainData['PM2.5-0'] /= y_STD
	
	for key in OriAttr:
		if key is not 'Const':
			x_STD[key] = np.std(TrainData[key].tolist())
			TrainData[key] /= x_STD[key]
		else:
			x_STD[key] = 1
		for d in ConsideredDim:
			Attr.append(str(d) + key)
			SetZero[str(d) + key] = 1
			t_ratio[str(d) + key] = 0
			
	CovMat = TrainData.cov()
	
	print(TrainData)
	

	StopCond = 1e-1
	IfPrint = False
	'''
	while True:
		OldZero = SetZero
		OldSSE = SSE
		Weight, SSE = \
			minSSE(TrainData, Attr, SetZero, 100, False, StopCond, NumGroup, IfPrint)
		
		for key in t_ratio:
			if SetZero[key] == 0:
				t_ratio[key] = 1e100
			else:
				t_ratio[key] = abs(Weight[key] / np.sqrt(CovMat[key[1:]][key[1:]]))
		
		minVari = min(t_ratio, key=t_ratio.get)
		if t_ratio[minVari] < 0.68:
			print('Delete %s, t-ratio = %f' % (minVari, t_ratio[minVari]))
			SetZero[minVari] = 0
		else:
			StopCond = 1e-3
			IfPrint = True
			if SSE > OldSSE * (1 - StopCond):
				print('No improve')
				SetZero = OldZero
				break
			else:
				minVari = min(t_ratio, key=t_ratio.get)
		
				if t_ratio[minVari] < 1.3:
					print('Delete %s, t-ratio = %f' % (minVari, t_ratio[minVari]))
					SetZero[minVari] = 0
				else:
					print('All significant')
					break
	'''
	Weight, SSE = \
		minSSE(TrainData, Attr, SetZero, 10000, False, 1e-4/2, NumGroup, True)
	
	for key in SetZero:
		Weight[key] = Weight[key] * SetZero[key] / x_STD[key[1:]]
	
	ResultDF = pd.DataFrame(Weight, index=[0])
	
	print(ResultDF)
	print(t_ratio)
	ResultDF.to_csv('./Coefficient.csv')
