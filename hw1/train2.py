#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np

def gradDece(df, weight, SqSig, LearningRate=10, time=1):
	DeceRate = {}
	NewWeight = {}
	CurGrad = evalGrad(df, weight)
	for key in weight:
		SqSig[key] += CurGrad[key] ** 2
		DeceRate[key] = LearningRate / np.sqrt(SqSig[key])
		NewWeight[key] = weight[key] - DeceRate[key] * CurGrad[key]
	
	return NewWeight, SqSig

def evalLoss(df, weight):
	Output = pd.DataFrame(np.zeros((len(df['PM2.5-0']), 1)), columns=['Proj'])
	for key in weight:
		Output['Proj'] += (df[key[1:]] ** int(key[0])) * weight[key]
	return ((df['PM2.5-0'] - Output['Proj']) ** 2).sum()

def evalGrad(df, weight):
	Diff = pd.DataFrame(np.zeros((len(df['PM2.5-0']), 1)), columns=['Delta'])
	Diff['Delta'] = df['PM2.5-0']
	
	for key in weight:
		Diff['Delta'] -= weight[key] * (df[key[1:]] ** int(key[0]))
	
	Grad = {}
	for key in weight:
		Grad[key] = (-2 * Diff['Delta'] * (df[key[1:]] ** int(key[0]))).sum()
	return Grad

if __name__ == '__main__':
	TrainData = pd.read_csv('./data/TrainData.csv', index_col=0)
	
	DataL = len(TrainData['PM2.5-0'])

	TrainData = TrainData.assign(GroupID = \
		pd.Series(np.random.randint(0, 3, DataL)).values)
	TrainData['Const'] = 1

	Weight = {}
	CurSqSig = {}
	Attr = list(TrainData)
	Attr.remove('PM2.5-0')
	
	for key in Attr:
		Weight['1' + key] = 1
		CurSqSig['1' + key] = 0

		#Weight['3' + key] = 1
		#CurSqSig['3' + key] = 0
	
	SSE = evalLoss(TrainData, Weight)
	print('%i: %f' % (0, SSE))
	
	for i in range(500):
		Weight, CurSqSig = gradDece(TrainData, Weight, CurSqSig, 1000, i)
		SSE = evalLoss(TrainData, Weight)
		print('%i: %f' % (i + 1, SSE))
	
	ResultDF = pd.DataFrame(Weight, index=[0])
	
	print(ResultDF)
	ResultDF.to_csv('./Coefficient.csv')
