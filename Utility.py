'''
Created on 19.07.2014

@author: Jimbo
'''

import numpy as np

def getSampleCovariance(samples):
    samples = samples.transpose()
    sampleMean = np.matrix(np.mean(samples,axis=1)).transpose()
    ones = np.matrix(np.ones((1,samples.shape[1])))
    sampleCov = 1./(samples.size-1) * np.dot( np.dot(sampleMean, ones) , np.dot(sampleMean, ones).transpose())

    return sampleCov