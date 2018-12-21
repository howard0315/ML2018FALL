#-*- coding=utf-8 -*-

from keras import Sequential, regularizers, optimizers, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, GRU, Dense, Input, \
						Flatten, Dropout, Activation
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import jieba as jb
import pandas as pd
import numpy as np
import re
import sys
import os
import pickle

def ReadData():
	print('Start Reading!')
	
	TrainX = pd.read_csv(TrainXF, sep='\n', engine='python')
	
	TrainX = pd.DataFrame(
		TrainX.loc[:, 'id,comment'].str.split(',', 1).tolist(), 
		columns=['id', 'comment']).drop(columns=['id'])
	
	print('End of Reading!')
	print('====================')
	return TrainX

def SplitSentence():
	print('Start Splitting!')
	
	jb.set_dictionary(DictF)
	
	TrainXCut = []
	for ind in TrainX.index:
		WordList = jb.lcut(TrainX.loc[ind, 'comment'])
		WordList = [wd for wd in WordList 
			if wd not in StopWord if not re.match(r"B[0-9]+", wd)]
		TrainXCut.append(WordList)
	
	print('End of Splitting!')
	print('====================')
	return TrainXCut

def LoadStopWord():
	with open('./StopWordList/Chinese.txt', 'r') as cf:
		CHList = cf.read().split('\n')
	CHList.pop()
	
	with open('./StopWordList/English.txt', 'r') as ef:
		ENList = ef.read().split(',')
	
	StopWord = []
	StopWord.append(CHList)
	StopWord.append(ENList)
	StopWord = [wd for dt in StopWord for wd in dt]
	StopWord.append(' ')
	return StopWord

def Text2Index(corpus):
	new_corpus = []
	for doc in corpus:
		new_doc = []
		for word in doc:
			try:
				new_doc.append(word2idx[word])
			except:
				new_doc.append(0)
		new_corpus.append(new_doc)
	return np.array(new_corpus)

def MainModel():
	
	init = Input(shape=(len(word2idx) + 1,))
	x = BatchNormalization()(init)
	x = Dropout(0.5)(x)
	x = Dense(64,
	 	kernel_regularizer=regularizers.l2(1e-2), activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(1, activation='sigmoid')(x)
	
	model = Model(inputs=init, outputs=x)
	
	ADAM = optimizers.adam(lr=0.002)

	model.compile(loss='binary_crossentropy', \
				optimizer=ADAM, metrics=['accuracy'])
	
	model.summary()
	
	return model

if __name__ == '__main__':
	
	os.environ["THEANO_FLAGS"] = "device=gpu0"
	
	TrainXF = sys.argv[1]
	TrainYF = sys.argv[2]
	TestXF = sys.argv[3]
	DictF = sys.argv[4]
	
	StopWord = LoadStopWord()
	
	TrainX = ReadData()
	TrainY = pd.read_csv(TrainYF)
	TrainX = SplitSentence()
	
	with open('./WordEmbedding/embedding_matrix_BOW.pickle', 'rb') as file:
		embedding_matrix = pickle.load(file)
	
	with open('./WordEmbedding/word2idx_BOW.pickle', 'rb') as file:
		word2idx = pickle.load(file)
	
	PaddingLength = 60
	
	TrainX = Text2Index(TrainX)
	TrainX = pad_sequences(TrainX, maxlen=PaddingLength)
	
	print('Start Index->BOW! (len of dict: %i)' % (len(word2idx) + 1))
	BOW = np.zeros((len(TrainX), len(word2idx) + 1))
	for v in range(len(TrainX)):
		for wd in TrainX[v]:
			BOW[v, wd] = BOW[v, wd] + 1
	TrainX = BOW
	print('End of Index->BOW!')
	print('====================')
	
	TrainYCat = np.array(TrainY.label)
	
	model = MainModel()
	
	# Setting callback functions
	csv_logger = CSVLogger('./model/log/training_BOW.log')
	checkpoint = ModelCheckpoint(filepath='./model/best_BOW.h5',
		verbose=1, save_best_only=True, monitor='val_acc', mode='max')
	earlystopping = EarlyStopping(monitor='val_acc',
		patience=9, verbose=1, mode='max')
	
	history = model.fit(x=TrainX, y=TrainYCat, 
		batch_size=512, validation_split=0.3, epochs=100, verbose=1,
		callbacks=[earlystopping, checkpoint, csv_logger])
	
	print('End of fitting!!')

	model.save('./model/final_BOW.h5')
	
