#-*- coding=utf-8 -*-

import pandas as pd

RawData = pd.read_csv('./data/train.csv', encoding='big5')

PartialTitle = RawData['測項'][0:18].tolist()

Title = [PT + '-' + str(i + 1) for i in range(9) for PT in PartialTitle]
Title.insert(0, 'PM2.5-0')

TrainData = pd.DataFrame(columns = Title)

# one group contains 20 days

numDay = len(RawData['日期'].unique().tolist())

numHour = numDay * 24

for m in range(numDay // 20):
	print(m)
	for hr in range(9, 20 * 24):
		inputData = {}
		inputData['PM2.5-0'] = \
			RawData[str(hr % 24)][9 + m * 20 * 18 + hr // 24 * 18]
		for pa in range(1, 10):
			for r in range(len(PartialTitle)):
				inputData[PartialTitle[r] + '-' + str(pa)] = RawData[\
					str((hr - pa) % 24)][r + m * 20 * 18 + (hr - pa) // 24 * 18]
				if inputData[PartialTitle[r] + '-' + str(pa)] == 'NR':
					inputData[PartialTitle[r] + '-' + str(pa)] = 0
		TrainData = pd.concat([TrainData, \
			pd.DataFrame([inputData], columns=inputData.keys())], \
						ignore_index=True)

TrainData.to_csv('./data/TrainData.csv')

print(TrainData)
