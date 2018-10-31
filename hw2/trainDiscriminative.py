#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys

class logistic():
	def __init__(self):
		pass
	
	def optimize(self, Ydata, Xdata, \
				scalingMethod, consideredFeat, LR, miniBatchSize, N_epoch):
		
		weight, SqSiqW = self.generateInitialWeight(consideredFeat)
		b = 0
		SqSiqB = 0
		
		Xdata, _ = \
			manageData().featureScaling(scalingMethod, Xdata, consideredFeat)
		
		EpochText = 'New epoch: {:>' + str(int(np.log10(N_epoch)) + 1) + 'd}'
		
		if miniBatchSize == -1:
			numBatch = 1
		else:
			numBatch = math.floor(len(Ydata.Y) / miniBatchSize)
		BatchSize = []
		miniXD = []
		miniYD = []
		for i in range(numBatch):
			BatchSize.append(len(Ydata.Y[i::numBatch]))
			miniXD.append(Xdata[i::numBatch])
			miniYD.append(Ydata[i::numBatch])
			
		print('\nStart training')
		
		for i in range(N_epoch):
			estimatedY = calculation().estimate(Xdata, weight, b)
			for m in range(numBatch):
				miniYE = estimatedY[m::numBatch]
				weight, b, SqSiqW, SqSiqB = \
					self.calcNewWeight(miniYD[m].Y - miniYE.Y, miniXD[m], LR, \
										weight, b, SqSiqW, SqSiqB, BatchSize[m])
			
			if i % 1000 == 0:
				print(EpochText.format(i), end='')
				print('\t-lnL: {:f}'.format( \
					calculation().crossEntropy(Ydata, estimatedY)))
			
		estimatedY = calculation().estimate(Xdata, weight, b)
		self.printWeight(weight, consideredFeat, b, Ydata, estimatedY, Xdata)
		return weight, b
	
	def generateInitialWeight(self, featName):
		weight = np.zeros((len(featName), 1))
		SqSiqW = np.zeros((len(featName), 1))
		return weight, SqSiqW
	
	def calcNewWeight(self, estiDiff, Xdata, LR, \
						weight, b, SqSiqW, SqSiqB, BatchSize):
		SqSiqB += estiDiff.sum() ** 2
		b += LR / np.sqrt(SqSiqB) * estiDiff.sum()
		grad = np.matmul(estiDiff.T, Xdata).reshape(weight.shape)
		SqSiqW = np.add(SqSiqW, np.power(grad, 2))
		weight = weight + LR * np.multiply(1 / np.sqrt(SqSiqW), grad)
		return weight, b, SqSiqW, SqSiqB
	
	def weight2Dict(self, weight, consideredFeat):
		newWeight = {}
		for i in range(len(consideredFeat)):
			newWeight[consideredFeat[i]] = float(weight[i])
		return newWeight
	
	def printWeight(self, weight, featName, b, Ydata, Yesti, Xdata):
		Xmean = Xdata.mean()
		print('End of training\t\t', end='')
		print('-lnL: {:f}'.format(calculation().crossEntropy(Ydata, Yesti)))
		print('\nResulting weights:')
		print('\n{:>24s}\t{:9s}\t{:6s}'.format('Feature', 'Weight', 'Mean'))
		print('{:>24s}\t{:9s}\t{:6s}'.format('-------', '------', '----'))
		print('{:>24s}\t{:=9.6f}\t{:=6.4f}'.format('CONST', b, 1))
		for k in range(len(featName)):
			print('{:>24s}\t{:=9.6f}\t{:=6.4f}\t{:=9.5f}'.format(\
					featName[k], float(weight[k]), Xmean[featName[k]], \
					float(weight[k]) * Xmean[featName[k]]))

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
		numData = len(Ydata.Y)
		dist = int(1 / factor)
		Xtest = Xdata.iloc[list(range(0, numData, dist)), :]
		Ytest = Ydata.iloc[list(range(0, numData, dist)), :]
		Xdata = Xdata.drop(list(range(0, numData, dist)), axis=0)
		Ydata = Ydata.drop(list(range(0, numData, dist)), axis=0)
		Xtest = Xtest.reset_index(drop=True)
		Ytest = Ytest.reset_index(drop=True)
		Xdata = Xdata.reset_index(drop=True)
		Ydata = Ydata.reset_index(drop=True)
		return Xdata, Ydata, Xtest, Ytest
	
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
	
	def testScore(self, YD, Ye):
		numData = len(YD.Y)
		Ye = self.toBinary(Ye)
		Diff = Ye - YD
		count = Diff.Y.value_counts()
		score = count[0] / numData
		print('\nExpected Score: {:f}'.format(score))

	def toBinary(self, Ydata):
		Ydata.Y[Ydata.Y >= 0.5] = 1
		Ydata.Y[Ydata.Y < 0.5] = 0
		Ydata.Y = Ydata.Y.astype(int)
		return Ydata

if __name__ == '__main__':
	
	XDataF = sys.argv[1]
	YDataF = sys.argv[2]
	ModelDir = sys.argv[3]
	
	XD, YD = manageData().importTrain(XDataF, YDataF)
	
	oneHotEncodingMap = {
		'SEX':		 	{1: 'male', 2: 'female'}, \
		'EDUCATION': 	{1: 'graduate', 2: 'university', \
						 3: 'highSchool', 4: 'others'}, \
		'MARRIAGE': 	{1: 'married', 2: 'single', 3: 'others'}}
	
	XD = manageData().oneHotEncoding(XD, oneHotEncodingMap)
	
	scalingMethod = 'minmax'
	
#	consideredFeat = ['PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2']
#	consideredFeat = ['LIMIT_BAL', 'PAY_0', 'BILL_AMT1',  'PAY_AMT1', \
#						'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4']
	consideredFeat = list(XD)
	dropList = []
#	dropList = ['PAY_4', 'PAY_AMT2', 'SEX_female', 'MARRIAGE_single', 'EDUCATION_others', 'MARRIAGE_others']
#	dropList = ['BILL_AMT6', 'PAY_4']
#	dropList = ['SEX', 'EDUCATION', 'MARRIAGE', 'SEX_female', 'MARRIAGE_single']
#	dropList = ['EDUCATION', 'PAY_AMT3', 'EDUCATION_others', 'MARRIAGE_others']
	for i in dropList:
		consideredFeat.remove(i)

	for i in list(XD):
		if i not in consideredFeat:
			XD = XD.drop(i, 1)
	
	Xdata, _ = manageData().featureScaling(scalingMethod, XD, consideredFeat)
	
	XCOV = Xdata.corr()
	print('Feature pair with high correlation (>0.7)')
	for i in XCOV.index.tolist():
		for j in list(XCOV):
			if abs(XCOV.loc[i, j]) > 0.7 and i is not j and i < j:
				print('{:24s} {:24s} {:9.6f}'.format(i, j, XCOV.loc[i, j]))
	
	testfactor = 0.1
	
	XD, YD, Xtest, Ytest = manageData().splitData(XD, YD, testfactor)

	LR = 10
	miniBatchSize = -1
	MaxEpoch = 10000
	
	Weight, B = logistic().optimize(YD, XD, \
					scalingMethod, consideredFeat, LR, miniBatchSize, MaxEpoch)
	
	Yesti = calculation().estimate(Xtest, Weight, B)
	calculation().testScore(Ytest, Yesti)
	Weight = logistic().weight2Dict(Weight, consideredFeat)
	np.save(ModelDir, [Weight, B, scalingMethod, oneHotEncodingMap])
