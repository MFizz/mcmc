'''
Created on 19.07.2014

@author: Jimbo
'''

import numpy as np
import scipy as sp

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
    
def getSuboptimality(covarianceMatrix, realCovarianceMatrix):
    if realCovarianceMatrix == None:
        return 0
    dimensions = covarianceMatrix.shape[0]
    sum = 0
    print sp.linalg.sqrtm(covarianceMatrix)
    eigenvalues, evectors = np.linalg.eig(sp.linalg.sqrtm(np.dot(np.matrix(covarianceMatrix),np.matrix(realCovarianceMatrix)**(-1))))
    bs = []
    for d in xrange(dimensions):
        print [l**(-1) for l in eigenvalues[:d+1]]
        b = (d+1) * np.sum([l**(-2) for l in eigenvalues[:d+1]]) / np.sum([l**(-1) for l in eigenvalues[:d+1]])**2
        bs.append(b)
        print b
    return np.mean(bs)