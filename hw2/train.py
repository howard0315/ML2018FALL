#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import sys

class generative():
	def __init__(self):
		pass

	def optimize(self, Ydata, Xdata, scalingMethod, consideredFeat):
		Xdata, _ = \
			manageData().featureScaling(scalingMethod, Xdata, consideredFeat)
		N1 = len(Ydata.Y[Ydata.Y == 1])
		N2 = len(Ydata.Y[Ydata.Y == 0])
		mu1 = calculation().mu(Xdata, Ydata, 1)
		mu2 = calculation().mu(Xdata, Ydata, 0)
		sigma1 = calculation().sigma(Xdata, Ydata, 1)
		sigma2 = calculation().sigma(Xdata, Ydata, 0)
		sigma = (N1 * sigma1 + N2 * sigma2) / (N1 + N2)
		
		weightT = np.matmul((mu1 - mu2).T, np.linalg.inv(sigma))
		weight = weightT.transpose()

		b = -0.5 * np.matmul(np.matmul(mu1.T, np.linalg.inv(sigma1)), mu1) + \
			0.5 * np.matmul(np.matmul(mu2.T, np.linalg.inv(sigma2)), mu2) + \
			np.log(N1 / N2)
		
		Yesti = calculation().estimate(Xdata, weight, b)
		self.printWeight(weight, b, consideredFeat, Ydata, Yesti)
		weight = self.weight2Dict(weight, consideredFeat)
		return weight, b

	def weight2Dict(self, weight, consideredFeat):
		newWeight = {}
		for i in range(len(consideredFeat)):
			newWeight[consideredFeat[i]] = float(weight[i])
		return newWeight
	
	def printWeight(self, weight, b, featName, Ydata, Yesti):
		print('Resulting Cross Entropy -lnL = {:f}'.format( \
			calculation().crossEntropy(Ydata, Yesti)))
		print('\nResulting weights:')
		print('\n{:>24s}\t{:s}'.format('Feature', 'Weight'))
		print('{:>24s}\t{:s}'.format('-------', '------'))
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
	
	def mu(self, Xdata, Ydata, classNumber):
		Xdata = Xdata[Ydata.Y == classNumber]
		return Xdata.mean(axis=0)
	
	def sigma(self, Xdata, Ydata, classNumber):
		Xdata = Xdata[Ydata.Y == classNumber]
		return Xdata.cov()
		
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

if __name__ == '__main__':
	
	XDataF = sys.argv[1]
	YDataF = sys.argv[2]
	
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
	
	finalWeight, finalB = \
		generative().optimize(YD, XD, scalingMethod, consideredFeat)
	
	np.save('./modelGenerative.npy', \
		[finalWeight, finalB, scalingMethod, oneHotEncodingMap])
