#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import sys

class logistic():
	def __init__(self):
		pass
	
	def optimize(self, Ydata, Xdata, scalingMethod, \
					consideredFeat, LR, N_epoch, ifAddB):
		weight = self.generateInitialWeight(consideredFeat)
		b = 0
		Xdata, dataAdjustment = \
			manageData().featureScaling(scalingMethod, Xdata, consideredFeat)
		EpochText = 'New epoch: {:>' + str(int(np.log10(N_epoch)) + 1) + 'd}'
		print('Start training', end='')
		for i in range(N_epoch):
			estimatedY = calculation().estimate(Xdata, weight, ifAddB, b)
			print('\t-lnL: {:f}'.format( \
				calculation().negativelnL(Ydata, estimatedY)))
			print(EpochText.format(i + 1), end='')
			weight, b = self.calcNewWeight(Ydata, estimatedY, Xdata, \
										consideredFeat, weight, LR, ifAddB, b)
		estimatedY = calculation().estimate(Xdata, weight, ifAddB, b)
		self.printWeight(weight, Ydata, estimatedY, ifAddB, b)
		return weight, b, dataAdjustment
	
	def generateInitialWeight(self, featName):
		weight = {}
		for e in featName:
			weight[e] = 0
		return weight
	
	def calcNewWeight(self, Ydata, estiY, Xdata, \
							consideredFeat, weight, LR, ifB, b):
		estiDiff = Ydata['Y'] - estiY['Y']
		if ifB:
			b += LR * estiDiff.sum()
		XtimesDiff = Xdata[consideredFeat].multiply(estiDiff, axis='index')
		XtimesDiffSUM = XtimesDiff.sum()
		for i in weight:
			weight[i] += LR * XtimesDiffSUM[i]
		return weight, b

	def printWeight(self, weight, Ydata, Yesti, ifB, b):
		print('\t-lnL: {:f}'.format( \
				calculation().negativelnL(Ydata, Yesti)))
		print('End of training')
		print('\nResulting weights:')
		print('\n{:>24s}\t{:s}'.format('Feature', 'Weight'))
		print('{:>24s}\t{:s}'.format('-------', '------'))
		if ifB:
			print('{:>24s}\t{:=9.6f}'.format('CONST', b))
		for k in weight:
			print('{:>24s}\t{:=9.6f}'.format(k, weight[k]))

class manageData():
	def __init__(self):
		pass

	def importTrain(self, trainXDir, trainYDir):
		Xtrain = pd.read_csv(trainXDir)
		Ytrain = pd.read_csv(trainYDir)
		return Xtrain, Ytrain
	
	def oneHotEncoding(self, Xdata, titleDict):
		for t in titleDict:
			for k in titleDict[t]:
				Xdata[t + '_' + titleDict[t][k]] = 0
				Xdata.loc[Xdata[t] == k, t + '_' + titleDict[t][k]] = 1
		return Xdata
	
	def splitData(self, Xdata, Ydata, factor):
		pass
	
	def featureScaling(self, method, Xdata, featureName):
		if method == 'minmax':
			scalingSpec = \
				pd.DataFrame(index=featureName, columns=['min', 'max'])
			for ft in featureName:
				minX = Xdata[ft].min()
				maxX = Xdata[ft].max()
				Xdata[ft] = (Xdata[ft] - minX) / (maxX - minX)
				scalingSpec['min'][ft] = minX
				scalingSpec['max'][ft] = maxX
		elif method == 'standardization':
			scalingSpec = \
				pd.DataFrame(index=featureName, columns=['mean', 'std'])
			for ft in featureName:
				meanX = Xdata[ft].mean()
				stdX = Xdata[ft].std()
				Xdata[ft] = (Xdata[ft] - meanX) / stdX
				scalingSpec['mean'][ft] = meanX
				scalingSpec['std'][ft] = stdX
		return Xdata, scalingSpec

class calculation():
	def __init__(self):
		pass
	
	def estimate(self, Xdata, weight, ifB, b):
		esti = pd.DataFrame(np.zeros((len(Xdata['LIMIT_BAL']), 1)), \
			columns=['Y'])
		if ifB:
			esti['Y'] += b
		for key in weight:
			esti['Y'] += weight[key] * Xdata[key]
		esti = self.sigmoid(esti)
		return esti

	def sigmoid(self, Ydata):
		Ydata.Y = (1 + np.exp(-1 * Ydata.Y)).rdiv(1)
		return Ydata
	
	def negativelnL(self, Ydata, estiY):
		estilnL = \
			-(Ydata.Y * np.log(estiY.Y) + (1 - Ydata.Y) * np.log(1 - estiY.Y))
		return estilnL.sum()

	def avgError(self, Ydata, estimatedY, weight):
		err = Ydata['Y'] - estimatedY['Y']
		absErr = list(map(abs, err['Y'].tolist()))
		return sum(absErr) / len(absErr)

if __name__ == '__main__':
	
	XDataF = sys.argv[1]
	YDataF = sys.argv[2]
	ModelDir = sys.argv[3]
	
	XD, YD = manageData().importTrain(XDataF, YDataF)
	
	oneHotEncodingMap = {}
	oneHotEncodingMap['SEX'] = {1: 'male', 2: 'female'}
	oneHotEncodingMap['EDUCATION'] = {1: 'graduate', 2: 'university', \
										3: 'highSchool', 4: 'others'}
	oneHotEncodingMap['MARRIAGE'] = {1: 'married', 2: 'single', 3: 'others'}
	
	XD = manageData().oneHotEncoding(XD, oneHotEncodingMap)
	
	scalingMethod = 'minmax'
	consideredFeat = list(XD)
	for k in oneHotEncodingMap:
		consideredFeat.remove(k)
	
	dropList = ['BILL_AMT6', 'PAY_4', 'PAY_6', 'BILL_AMT4', 'BILL_AMT2', 'EDUCATION_others', 'AGE', 'MARRIAGE_single']
	for k in dropList:
		consideredFeat.remove(k)


	LR = 1e-4 * 1.5
	NumEpoch = 10000
	AddConst = True
	
	finalWeight, finalB, scalingSpec = logistic().optimize(YD, XD, \
		scalingMethod, consideredFeat, LR, NumEpoch, AddConst)
	
	estimatedY = calculation().estimate(XD, finalWeight, AddConst, finalB)
	nlnL = calculation().negativelnL(YD, estimatedY)
	
	np.save(ModelDir, [finalWeight, finalB, scalingMethod, oneHotEncodingMap])
