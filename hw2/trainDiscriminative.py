#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import sys

class logistic():
	def __init__(self):
		pass
	
	def optimize(self, Ydata, Xdata, scalingMethod, \
					consideredFeat, LR, N_epoch, ifAddB):
		weight, SqSiqW = self.generateInitialWeight(consideredFeat)
		b = 0
		SqSiqB = 0
		Xdata, dataAdjustment = \
			manageData().featureScaling(scalingMethod, Xdata, consideredFeat)
		EpochText = 'New epoch: {:>' + str(int(np.log10(N_epoch)) + 1) + 'd}'
		print('Start training')
		
		for i in range(N_epoch):
			estimatedY = calculation().estimate(Xdata, weight, b)
			
			weight, b, SqSiqW, SqSiqB = \
				self.calcNewWeight(Ydata.Y - estimatedY.Y, Xdata, LR, \
					weight, ifAddB, b, SqSiqW, SqSiqB)
			
			if i % 200 == 0:
				print(EpochText.format(i), end='')
				print('\t-lnL: {:f}'.format( \
					calculation().crossEntropy(Ydata, estimatedY)))
			
		estimatedY = calculation().estimate(Xdata, weight, b)
		self.printWeight(weight, consideredFeat, Ydata, estimatedY, ifAddB, b)
		weight = self.weight2Dict(weight, consideredFeat)
		return weight, b, dataAdjustment
	
	def generateInitialWeight(self, featName):
		weight = np.zeros((len(featName), 1))
		SqSiqW = np.zeros((len(featName), 1))
		return weight, SqSiqW
	
	def calcNewWeight(self, estiDiff, Xdata, LR, weight, \
							ifB, b, SqSiqW, SqSiqB):
		if ifB:
			SqSiqB += estiDiff.sum() ** 2
			b += LR / np.sqrt(SqSiqB + 1e-6) * estiDiff.sum()
		
		grad = np.matmul(estiDiff.transpose(), Xdata).reshape(weight.shape)
		SqSiqW = np.add(SqSiqW, np.power(grad, 2))
		weight = weight + LR * np.multiply(1 / np.sqrt(SqSiqW + 1e-6), grad)
		
		return weight, b, SqSiqW, SqSiqB
	
	def weight2Dict(self, weight, consideredFeat):
		newWeight = {}
		for i in range(len(consideredFeat)):
			newWeight[consideredFeat[i]] = float(weight[i])
		return newWeight
	
	def printWeight(self, weight, featName, Ydata, Yesti, ifB, b):
		print('End of training\t\t', end='')
		print('-lnL: {:f}'.format(calculation().crossEntropy(Ydata, Yesti)))
		print('\nResulting weights:')
		print('\n{:>24s}\t{:s}'.format('Feature', 'Weight'))
		print('{:>24s}\t{:s}'.format('-------', '------'))
		if ifB:
			print('{:>24s}\t{:=9.6f}'.format('CONST', b))
		for k in range(len(featName)):
			print('{:>24s}\t{:=9.6f}'.format(featName[k], float(weight[k])))

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
	
	def estimate(self, Xdata, weight, b):
		esti = np.dot(Xdata, weight) + b
		esti = pd.DataFrame(esti, columns=['Y'])
		return self.sigmoid(esti)

	def sigmoid(self, Ydata):
		Ydata.Y = (1 + np.exp(-1 * Ydata.Y)).rdiv(1)
		return Ydata
	
	def crossEntropy(self, Ydata, estiY):
		return -(np.dot(Ydata.Y, np.log(estiY.Y)) \
			+ np.dot(1 - Ydata.Y, np.log(1 - estiY.Y)))

	def avgError(self, Ydata, estimatedY):
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
	
	consideredFeat = ['PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2']
	for i in list(XD):
		if i not in consideredFeat:
			XD = XD.drop(i, 1)
	
#	consideredFeat = list(XD)
#	for k in oneHotEncodingMap:
#		consideredFeat.remove(k)
#		XD = XD.drop(k, 1)
#	
#	dropList = ['BILL_AMT6', 'PAY_4']
#	for k in dropList:
#		consideredFeat.remove(k)
#		XD = XD.drop(k, 1)

	LR = 10
	MaxEpoch = 10000
	AddConst = True
	
	finalWeight, finalB, scalingSpec = logistic().optimize(YD, XD, \
		scalingMethod, consideredFeat, LR, MaxEpoch, AddConst)
	
	np.save(ModelDir, [finalWeight, finalB, scalingMethod, oneHotEncodingMap])
