#-*- coding=utf-8 -*-

import pandas as pd
import sys
import csv
from functionForTrain import *

if __name__ == '__main__':

	TestDataF = sys.argv[1]
	OutputDir = sys.argv[2]

	TestHeader = ['ID', 'x']
	for i in range(9, 0, -1):
		TestHeader.append('-' + str(i))

	RawTestData = pd.read_csv(TestDataF, header=None)
	RawTestData.columns = TestHeader
	Coef = pd.read_csv('./Coefficient.csv', index_col=0)
	for i in range(9, 0, -1):
		for j in range(len(RawTestData['-' + str(i)])):
			if RawTestData['-' + str(i)][j] == 'NR':
				RawTestData['-' + str(i)][j] = 0

	IDList = RawTestData['ID'].unique().tolist()
	xList = RawTestData['x'][0 : 18].tolist()
	AvalDim = [1]
	title = [str(i) + x + '-' + str(t) \
		for i in AvalDim for x in xList for t in range(9, 0, -1)]
	title.append('1Const')
	TestData = pd.DataFrame(columns = title)

	for s in range(len(IDList)):
		inputData = {}
		inputData['1Const'] = 1
		for i in AvalDim:
			for x in range(len(xList)):
				for t in range(9, 0, -1):
					inputData[str(i) + xList[x] + '-' + str(t)] = \
						float(RawTestData['-' + str(t)][x + s * 18]) ** i
		TestData = pd.concat([TestData, \
			pd.DataFrame([inputData], columns=inputData.keys())], \
						ignore_index=True, sort=True)
	
	SmoothList = ['1PM2.5', '1PM10', '1NO2', '1WIND_SPEED', '1O3', '1RH', \
					'1SO2', '1NO']
	TestData = cleanData(TestData, SmoothList, 3)

	Projection = {}

	for s in range(len(IDList)):
		Projection[IDList[s]] = 0
		for key in list(TestData):
			Projection[IDList[s]] += \
				float(Coef[key][0]) * float(TestData[key][s])

	with open(OutputDir, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'value'])
		for row in Projection.items():
			writer.writerow(row)
