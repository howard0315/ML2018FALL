#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from copy import deepcopy

def minSSE(DF, Attr, Zero, LR, IfSto=False, StopCondition=1e-5, \
			StopIter = 10000, NGroup=10, printSSE=True):
	Weight = {}
	CurSqSig = {}
	for key in Attr:
		Weight[key] = 0
		CurSqSig[key] = 0

	SSE = evalLoss(DF, Weight)
	OldSSE = 1e100
	if printSSE:
		print('%i: %f' % (0, SSE))
	i = 0
	while abs(OldSSE - SSE) / SSE > StopCondition and i <= StopIter:
		i += 1
		Weight, CurSqSig = \
			gradDece(DF, Zero, Weight, CurSqSig, LR, NGroup, IfSto)
		OldSSE = SSE
		SSE = evalLoss(DF, Weight)
		if OldSSE - SSE < 0.01 * SSE and IfSto:
			print('Change Method')
			IfSto = False
			for key in CurSqSig:
				CurSqSig[key] = 0
			LR /= NGroup * 5
		if printSSE:
			print('%i: %f' % (i, SSE))
	return Weight

def gradDece(df, Zero, weight, SqSig, LearningRate=10, nGroup=1, IfSto=False):
	DeceRate = {}
	NewWeight = {}
	if IfSto:
		for g in range(nGroup):
			CurGrad = evalGrad(df[df['GroupID'] == g], weight, Zero)
			for key in weight:
				if Zero[key] != 0:
					SqSig[key] += CurGrad[key] ** 2
					DeceRate[key] = LearningRate / np.sqrt(SqSig[key])
					NewWeight[key] = weight[key] - DeceRate[key] * CurGrad[key]
				else:
					NewWeight[key] = 0
					SqSig[key] = 1e10
			weight = deepcopy(NewWeight)
	else:
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
	return ((df['PM2.5-0'] - Output['Proj']) ** 2).sum()

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

if __name__ == '__main__':
	print('Don''t run this py file!!')
