#-*- coding=utf-8 -*-

import numpy as np
from scipy import linalg as la

samples = np.array([[1., 2., 3.], [4., 8., 5.], [3., 12., 9.], [1., 8., 5.], 
					[5., 14., 2.], [7., 4., 1.], [9., 8., 9.], [3., 8., 1.], 
					[11., 5., 6.], [10., 11., 7.]])

M, N = samples.shape

samples -= samples.mean(axis=0)

sigma = np.cov(samples, rowvar=False)

evals, evecs = la.eigh(sigma)

idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]

twoD = evecs[:, :2]

PCSamples = np.dot(evecs.T, samples.T).T

newSamples = np.dot(twoD.T, samples.T).T

recSamples = np.dot(la.pinv(twoD.T), newSamples.T).T

L2norm = la.norm(samples - recSamples)

print(samples)
print(PCSamples)
print(newSamples)
print(recSamples)
print(L2norm)
print(idx)
print(evecs)
print(twoD)
