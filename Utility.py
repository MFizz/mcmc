'''
Created on 19.07.2014

@author: Jimbo
'''

import numpy as np

def getSampleCovariance(samples):
    samples = samples.transpose()
    sampleMean = np.matrix(np.mean(samples,axis=1)).transpose()
    ones = np.matrix(np.ones((1,samples.shape[1])))
    sampleCov = 1./(samples.shape[0]-1) * np.dot( np.dot(sampleMean, ones) , np.dot(sampleMean, ones).transpose())

    return sampleCov

class covarianceCalculator(object):
    
    def __init__(self, sampleVector):
        self.mean = np.matrix(np.zeros_like(sampleVector)).transpose()
        self.covariance = np.identity(self.mean.shape[0])
    
    def getSampleCovariance(self, samples):
        lastSample = np.matrix(samples[-1]).transpose()
        self.mean = self.mean + 1./(samples.shape[0]-1) * (lastSample - self.mean)
        self.covariance = self.covariance + 1./(samples.shape[0]-1) * ( np.dot((lastSample - self.mean) , (lastSample - self.mean).transpose()) - self.covariance)
        return self.covariance