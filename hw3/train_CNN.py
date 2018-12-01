#-*- coding=utf-8 -*-

from keras import Sequential, regularizers, optimizers
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
	
	TrainF = sys.argv[1]
	
	TrainData = pd.read_csv(TrainF)

	SplitFactor = 0.25
	Nfilters = 64
	kernelS = 3
	poolS = (2, 2)

	TrainFeat, TrainLabl, ValiFeat, ValiLabl, imageWidth = \
		manageData().splitData(TrainData, SplitFactor)
	
	datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		fill_mode='constant',
		horizontal_flip=True)

	datagen.fit(TrainFeat)
	
	print(TrainFeat.shape)
	print('Start fitting!!')
	
	model = Sequential()
	
	model.add(Conv2D(Nfilters, kernel_size=kernelS, \
		input_shape=(imageWidth, imageWidth, 1)))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	
	#model.add(Conv2D(Nfilters, kernel_size=kernelS, \
	#	input_shape=(imageWidth, imageWidth, 1)))
	#model.add(LeakyReLU(alpha=0.05))
	#model.add(BatchNormalization())
	
	model.add(MaxPooling2D(pool_size=poolS))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(Nfilters * 2, kernel_size=kernelS, padding='same'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	
	#model.add(Conv2D(Nfilters * 2, kernel_size=kernelS, padding='same'))
	#model.add(LeakyReLU(alpha=0.05))
	#model.add(BatchNormalization())
	
	model.add(MaxPooling2D(pool_size=poolS))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(Nfilters * 4, kernel_size=kernelS, padding='same'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	
	#model.add(Conv2D(Nfilters * 4, kernel_size=kernelS, padding='same'))
	#model.add(LeakyReLU(alpha=0.05))
	#model.add(BatchNormalization())
	
	model.add(MaxPooling2D(pool_size=poolS))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(Nfilters * 8, kernel_size=kernelS, padding='same'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	
	#model.add(Conv2D(Nfilters * 8, kernel_size=kernelS, padding='same'))
	#model.add(LeakyReLU(alpha=0.05))
	#model.add(BatchNormalization())
	
	model.add(MaxPooling2D(pool_size=poolS))
	model.add(Dropout(0.35))

	model.add(Flatten())
	
	model.add(Dense(units=500, \
		kernel_regularizer=regularizers.l2(1e-4)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	
	model.add(Dense(units=500, \
		kernel_regularizer=regularizers.l2(1e-4)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	
	model.add(Dense(units=len(TrainLabl[0])))
	model.add(Activation('softmax'))
	
	ADAM = optimizers.adam(lr=0.001)

	model.compile(loss='categorical_crossentropy', \
				optimizer=ADAM, metrics=['accuracy'])
	
	history = model.fit_generator(
		datagen.flow(TrainFeat, TrainLabl, batch_size=128), \
		validation_data=(ValiFeat, ValiLabl), \
		steps_per_epoch=(len(TrainFeat) * 10 / 128), \
		epochs=40, verbose=1)
	
	print('End of fitting!!')

	score = model.evaluate(ValiFeat, ValiLabl)
	print('Total loss on Testing Set: ', score[0])
	print('Accuracy of Testing Set: ', score[1])

	model.summary()
	model.save('CNN.h5')

