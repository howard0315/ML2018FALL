#-*- coding=utf-8 -*-

from keras import Sequential, regularizers, optimizers, Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, GRU, Dense, Input, \
						Flatten, Dropout, Activation
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import jieba as jb
from gensim.models import word2vec
import pandas as pd
import numpy as np
import re
import sys
import os
import pickle

def ReadData():
	global TestXF
	print('Start Reading!')
	
	TestX = pd.read_csv(TestXF, sep='\n', engine='python')
	
	TestX = pd.DataFrame(
		TestX.loc[:, 'id,comment'].str.split(',', 1).tolist(), 
		columns=['id', 'comment']).drop(columns=['id'])
	
	print('End of Reading!')
	print('====================')
	return TestX

def SplitSentence():
	global TestX, StopWord, DictF
	print('Start Splitting!')
	
	jb.set_dictionary(DictF)
	
	TestXCut = []
	for ind in TestX.index:
		WordList = jb.lcut(TestX.loc[ind, 'comment'])
		WordList = [wd for wd in WordList 
			if wd not in StopWord if not re.match(r"B[0-9]+", wd)]
		TestXCut.append(WordList)
	
	print('End of Splitting!')
	print('====================')
	return TestXCut

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

def Pred2Cate(p):
	if p[0] >= 0.5:
		return 1
	else:
		return 0

if __name__ == '__main__':
	
	TestXF = sys.argv[1]
	DictF = sys.argv[2]
	OutputF = sys.argv[3]
	StopWord = LoadStopWord()
	
	with open('./WordEmbedding/word2idx_BOW.pickle', 'rb') as file:
		word2idx = pickle.load(file)
	
	TestX = ReadData()
	TestX = SplitSentence()
	
	PaddingLength = 60
	
	TestX = Text2Index(TestX)
	TestX = pad_sequences(TestX, maxlen=PaddingLength)
	
	print('Start Index->BOW! (len of dict: %i)' % (len(word2idx) + 1))
	BOW = np.zeros((len(TestX), len(word2idx) + 1))
	for v in range(len(TestX)):
		for wd in TestX[v]:
			BOW[v, wd] = BOW[v, wd] + 1
	TestX = BOW
	print('End of Index->BOW!')
	print('====================')
	
	model = load_model('./model/best_BOW.h5')
	model.summary()
	
	result = model.predict(TestX, verbose=1)
	result = list(map(Pred2Cate, result))
	TestData = pd.DataFrame(result, columns=['label'])
	TestData.index.name = 'id'
	TestData.to_csv(OutputF, index=True)
	
