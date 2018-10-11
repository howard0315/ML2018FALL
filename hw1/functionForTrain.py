#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from copy import deepcopy

def minSSE(DF, Attr, Zero, LR, StopCondition=1e-5, \
			StopIter = 10000, printSSE=True):
	Weight = {}
	CurSqSig = {}
	for key in Attr:
		Weight[key] = 0.1 * Zero[key]
		CurSqSig[key] = 0

	SSE = evalLoss(DF, Weight)
	OldSSE = 1e100
	if printSSE:
		print('%i: %f' % (0, SSE))
	i = 0
	while abs(OldSSE - SSE) / SSE > StopCondition and i <= StopIter:
		i += 1
		Weight, CurSqSig = \
			gradDece(DF, Zero, Weight, CurSqSig, LR)
		OldSSE = SSE
		SSE = evalLoss(DF, Weight)
		if printSSE:
			print('%i: %f' % (i, SSE))
	return Weight

def gradDece(df, Zero, weight, SqSig, LearningRate=10):
	DeceRate = {}
	NewWeight = {}
	CurGrad = evalGrad(df, weight, Zero)
	for key in weight:
		if Zero[key] != 0:
			SqSig[key] += CurGrad[key] ** 2
			DeceRate[key] = LearningRate / np.sqrt(SqSig[key])
			NewWeight[key] = weight[key] - DeceRate[key] * CurGrad[key]
		else:
			NewWeight[key] = 0
			SqSig[key] = 1e10

	return NewWeight, SqSig

def evalLoss(df, weight):
	df.reset_index(drop=True)
	Output = pd.DataFrame(np.zeros((len(df['PM2.5-0']), 1)), columns=['Proj'])
	for key in weight:
		if weight[key] != 0:
			Output['Proj'] += df[key] * weight[key]
	return np.mean(((df['PM2.5-0'] - Output['Proj']) ** 2))

def evalGrad(df, weight, zero):
	Diff = pd.DataFrame(np.zeros((len(df['PM2.5-0']), 1)), columns=['Delta'])
	Diff['Delta'] = df['PM2.5-0']
	df = df.reset_index(drop=True)
	for key in weight:
		if zero[key] != 0:
			Diff['Delta'] -= weight[key] * df[key]

	Grad = {}
	for key in weight:
		if zero[key] != 0:
			Grad[key] = (-2 * Diff['Delta'] * df[key]).sum()
		else:
			Grad[key] = 1e10
	return Grad

def cleanData(df, attrList, lenWind):
	for attr in attrList:
		for i in range(len(df['1PM2.5-1'])):
			data = []
			attrName = [attr + '-' + str(t) for t in range(9, 0, -1)]
			data = list(map(float, df.loc[i, attrName].tolist()))
			data = smooth(np.array(data), lenWind, 'hanning').tolist()
			df.loc[i, attrName] = data
		print('%s smoothed!' % attr)
	return df

# from https://stackoverflow.com/questions/5515720/python-smooth-time-series-data
def smooth(x,window_len=11,window='hanning'):
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:  
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='same')
	return y[window_len:-window_len+1]

if __name__ == '__main__':
	print('Don''t run this py file!!')
