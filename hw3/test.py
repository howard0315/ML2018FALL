#-*- coding=utf-8 -*-

from keras import Sequential, regularizers, optimizers
from keras.models import load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout, Activation, LeakyReLU
from keras.utils import to_categorical
from keras.preprocessing.image  import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
import sys
import os
	
class manageData():
	def __init__(self):
		pass
	
	def splitData(self, data, factor):
		print('Start Splitting Data!!')
		numData = len(data.label)
		Test = data.iloc[:int(numData*factor), :]
		Train = data.drop(range(int(numData*factor)), axis=0)
		Test = Test.reset_index(drop=True)
		Train = Train.reset_index(drop=True)
		TrainFeat = self.splitString(Train.feature.values)
		TrainFeat, imageWidth = self.reshapeImage(TrainFeat)
		TrainLabl = to_categorical(Train.label.values)
		TestFeat = self.splitString(Test.feature.values)
		TestFeat, imageWidth = self.reshapeImage(Test.feature.values)
		TestLabl = to_categorical(Test.label.values)
		print('Finish Splitting Data!!')
		return TrainFeat, TrainLabl, TestFeat, TestLabl, imageWidth

	def splitString(self, Image):
		for i in range(len(Image)):
			Image[i] = list(map(int, Image[i].split()))
		return Image

	def reshapeImage(self, Image):
		NumData = len(Image)
		TotalPx = len(Image[0])
		Width = int(np.sqrt(TotalPx))
		Image = np.array([np.array(i) / 255. for i in Image])
		Image = Image.reshape(NumData, Width, Width, 1)
		return np.array(Image), Width

if __name__ == '__main__':
	
	os.environ["THEANO_FLAGS"] = "device=gpu0"
	
	model = load_model('CurrentModel.h5')
	
	TestF = sys.argv[1]
	OutputF = sys.argv[2]
	
	TestData = pd.read_csv(TestF)
	TestFeat = manageData().splitString(TestData.feature.values)
	TestFeat, _ = manageData().reshapeImage(TestFeat)
	result = model.predict_classes(TestFeat)
	TestData['label'] = pd.Series(result, index=TestData.index)
	TestData = TestData.drop(columns=['feature'])
	TestData.to_csv(OutputF, index=False)
