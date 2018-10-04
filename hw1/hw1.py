#-*- coding=utf-8 -*-

import pandas as pd
import sys
import csv

TestDataF = sys.argv[1]
OutputDir = sys.argv[2]

TestHeader = ['ID', 'x']
for i in range(9, 0, -1):
	TestHeader.append('-' + str(i))

TestData = pd.read_csv(TestDataF, header=None)
TestData.columns = TestHeader
Coef = pd.read_csv('./Coefficient.csv', index_col=0)

for i in range(9, 0, -1):
	for j in range(len(TestData['-' + str(i)])):
		if TestData['-' + str(i)][j] == 'NR':
			TestData['-' + str(i)][j] = 0

IDList = TestData['ID'].unique().tolist()
xList = TestData['x'][0 : 18].tolist()
Projection = {}

AvalDim = [1]

for s in range(len(IDList)):
	for i in AvalDim:
		Projection[IDList[s]] = float(Coef[str(i) + 'Const'][0]) * 1
	for t in range(9, 0, -1):
		for d in range(len(xList)):
			for i in AvalDim:
				Projection[IDList[s]] += \
					(float(TestData['-' + str(t)][d + s * 18]) ** i) * \
					float(Coef[str(i) + xList[d] + '-' + str(t)])

with open(OutputDir, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['id', 'value'])
	for row in Projection.items():
		writer.writerow(row)
