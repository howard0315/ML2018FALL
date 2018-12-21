#-*- coding=utf-8 -*-

import jieba as jb
from gensim.models import word2vec
import itertools
import pandas as pd
import numpy as np
import re
import sys
import os
import pickle

def ReadData():
	print('Start Reading!')
	
	TrainX = pd.read_csv(TrainXF, sep='\n', engine='python')
	
	TestX = pd.read_csv(TestXF, sep='\n', engine='python')
	
	Test2X = pd.read_csv(Test2XF, sep='\n', engine='python')
	
	TrainX = pd.DataFrame(
		TrainX.loc[:, 'id,comment'].str.split(',', 1).tolist(), 
		columns=['id', 'comment']).drop(columns=['id'])
	
	TestX = pd.DataFrame(
		TestX.loc[:, 'id,comment'].str.split(',', 1).tolist(), 
		columns=['id', 'comment']).drop(columns=['id'])
	
	Test2X = pd.DataFrame(
		Test2X.loc[:, 'id,comment'].str.split(',', 1).tolist(), 
		columns=['id', 'comment']).drop(columns=['id'])
	
	print('End of Reading!')
	print('====================')
	return TrainX, TestX, Test2X

def SplitSentence():
	print('Start Splitting!')
	
	jb.set_dictionary(DictF)
	
	TrainXCut = []
	for ind in TrainX.index:
		WordList = jb.lcut(TrainX.loc[ind, 'comment'])
		WordList = [wd for wd in WordList 
			if wd not in StopWord if not re.match(r"B[0-9]+", wd)]
		TrainXCut.append(WordList)
	
	TestXCut = []
	for ind in TestX.index:
		WordList = jb.lcut(TestX.loc[ind, 'comment'])
		WordList = [wd for wd in WordList 
			if wd not in StopWord if not re.match(r"B[0-9]+", wd)]
		TestXCut.append(WordList)
	
	Test2XCut = []
	for ind in Test2X.index:
		WordList = jb.lcut(Test2X.loc[ind, 'comment'])
		WordList = [wd for wd in WordList 
			if wd not in StopWord if not re.match(r"B[0-9]+", wd)]
		Test2XCut.append(WordList)
	
	FullDict = []
	FullDict.append(TrainXCut)
	FullDict.append(TestXCut)
	FullDict.append(Test2XCut)
	FullDict = [' '.join(st) for dt in FullDict for st in dt]
	
	print('End of Splitting!')
	print('====================')
	return FullDict

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

def TrainSaveW2V(LSF):
	print('Start Training Word2Vec!')
	sentences = word2vec.LineSentence(LSF)
	W2Vmodel = word2vec.Word2Vec(sentences, size=256, sg=1, workers=10)
	print('End of Training Word2Vec!')
	print('====================')
	W2Vmodel.save('./W2V/W2V.model')
	return W2Vmodel

if __name__ == '__main__':
	
	os.environ["THEANO_FLAGS"] = "device=gpu0"
	
	TrainXF = sys.argv[1]
	TestXF = sys.argv[2]
	DictF = sys.argv[3]
	
	LSF = './Corpus/SegSentence.txt'
	Test2XF = 'test2_x.csv'
	
	StopWord = LoadStopWord()
	
	TrainX, TestX, Test2X = ReadData()
	
	FullDict = SplitSentence()
	
	del TrainX, TestX, Test2X
	
	output = open(LSF, 'w', encoding='utf-8')
	for sen in FullDict:
		output.write(sen + '\n')
	output.close()
	
	del FullDict
	
	W2Vmodel = TrainSaveW2V(LSF)
	
	embedding_matrix = np.zeros(
		(len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))
	word2idx = {}
	
	vocab_list = [(word, W2Vmodel.wv[word])
		for word, _ in W2Vmodel.wv.vocab.items()]
	for i, vocab in enumerate(vocab_list):
		word, vec = vocab
		embedding_matrix[i + 1] = vec
		word2idx[word] = i + 1
	
	file = open('./WordEmbedding/embedding_matrix.pickle', 'wb')
	pickle.dump(embedding_matrix, file)
	file.close()
	
	file = open('./WordEmbedding/word2idx.pickle', 'wb')
	pickle.dump(word2idx, file)
	file.close()
	
