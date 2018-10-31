#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import sys

class outputData():
	def __init__(self):
		pass
	
	def DF2CSV(self, estimatedY):
		pass

class manageData():
	def __init__(self):
		pass
	
	def oneHotEncoding(self, Xdata, titleDict):
		for t in titleDict:
			for k in titleDict[t]:
				Xdata[t + '_' + titleDict[t][k]] = 0
				Xdata.loc[Xdata[t] == k, t + '_' + titleDict[t][k]] = 1
		return Xdata
	
	def scaleXTest(self, method, Xtrain, Xtest):
		if method == 'minmax':
			for i in list(Xtrain):
				minX = Xtrain[i].min()
				maxX = Xtrain[i].max()
				Xtest[i] = (Xtest[i] - minX) / (maxX - minX)
		elif method == 'standardization':
			for ft in list(Xtrain):
				meanX = Xtrain[ft].mean()
				stdX = Xtrain[ft].std()
				Xtest[ft] = (Xtest[ft] - meanX) / stdX + 1
		return Xtest
	
	def outputData(self, Ydata):
		Ydata = calculation().toBinary(Ydata)
		ID = ['id_' + str(i) for i in range(len(Ydata.Y))]
		Ydata.insert(loc=0, column='id', value=ID)
		Ydata.columns = ['id', 'value']
		Ydata = Ydata.set_index('id')
		Ydata.to_csv(OutputF)
		print('done!!')

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
		Ydata.Y = 1 + np.exp(-1 * Ydata.Y)
		Ydata.Y = Ydata.Y.rdiv(1)
		return Ydata
	
	def toBinary(self, Ydata):
		Ydata.Y[Ydata.Y >= 0.5] = 1
		Ydata.Y[Ydata.Y < 0.5] = 0
		Ydata.Y = Ydata.Y.astype(int)
		return Ydata

if __name__ == '__main__':
	
	XTrainF = sys.argv[1]
	YTrainF = sys.argv[2]
	XTestF = sys.argv[3]
	OutputF = sys.argv[4]
	
	Spec = np.load('./modelGenerative.npy')
	Weight = Spec[0]
	B = Spec[1]
	ScalingMethod = Spec[2]
	EncodingMap = Spec[3]
	
	Xtrain = pd.read_csv(XTrainF)
	Ytrain = pd.read_csv(YTrainF)
	Xtest = pd.read_csv(XTestF)
	
	Xtest = manageData().oneHotEncoding(Xtest, EncodingMap)
	Xtest = manageData().scaleXTest(ScalingMethod, Xtrain, Xtest)
	
	Yesti = calculation().estimate(Xtest, Weight, True, B)
	manageData().outputData(Yesti)
