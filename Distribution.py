'''
Created on 16.07.2014

@author: Jimbo
'''

import numpy as np 
import scipy.stats as stats
import math 

class UnivariateNormal(object):

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
    
    def getSample(self, currentValue):
        if currentValue == None:
            return np.random.normal(self.mean, self.variance)
        else:
            return np.random.normal(currentValue, self.variance)
    
    def getPDF(self, value, currentValue):
        if currentValue==None:
            return stats.norm.pdf(value, loc=self.mean, scale=self.variance)
        else:
            return stats.norm.pdf(value, loc=currentValue, scale=self.variance)
    
    def getStartPoint(self):
        return self.mean
    
    def adjust(self, adjustParameter):
        self.variance = self.variance
        




class MultivariateNormal(object):

    def __init__(self, meanVector, covarianceMatrix, beta=0.05):
        self.mean = meanVector
        self.covarianceMatrix = covarianceMatrix
        self.beta = beta
    
    def getSample(self, currentValue, sampleCovariance=None):
        if sampleCovariance==None:
            if currentValue == None:
                return np.random.multivariate_normal(self.mean, self.covarianceMatrix)
            else:
                return np.random.multivariate_normal(currentValue, self.covarianceMatrix)
        else:
            #adjustedDistribution = MultivariateNormal(self.mean, 2.38**2 * sampleCovariance) # sampleCovariance is already divided by dimensionality in the MH method
          #  return (1-self.beta) * adjustedDistribution.getSample(currentValue) + self.beta * self.getSample(currentValue)
            return (1-self.beta) * np.random.multivariate_normal(currentValue, sampleCovariance) + self.beta * self.getSample(currentValue)
    
    def getPDF(self, value, currentValue):
        if currentValue==None:
            return 1/( (2*np.pi)**(self.mean.shape[0]/2) * np.linalg.det(self.covarianceMatrix)**0.5 ) * np.e**(np.dot(np.dot( -0.5*np.transpose((value-self.mean)), np.linalg.inv(self.covarianceMatrix)), (value-self.mean)) )
        else:
            return 1/( (2*np.pi)**(currentValue.shape[0]/2) * np.linalg.det(self.covarianceMatrix)**0.5 ) * np.e**(np.dot(np.dot( -0.5*np.transpose((value-currentValue)), np.linalg.inv(self.covarianceMatrix)), (value-currentValue)) )
            '''
    def getPDF(self, r, currentValue):
        dim  = r.shape[-1]
        dev  = r - self.mean
        maha = np.einsum('...k,...kl,...l->...', dev, np.linalg.pinv(self.covarianceMatrix), dev)
        return (2 * np.pi)**(-0.5 * dim) * np.linalg.det(self.covarianceMatrix)**(0.5) * np.exp(-0.5 * maha)'''
    def getStartPoint(self):
        return np.random.multivariate_normal(self.mean, self.covarianceMatrix)
    
class RegionalMultivariateNorm(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def getPDF(self, x):
        size = len(x)
        if size == len(self.mu) and (size, size) == self.sigma.shape:
            det = np.linalg.det(self.sigma)
            if det == 0:
                raise NameError("The covariance matrix can't be singular")
            print self.sigma
            print det
            norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
            x_mu = np.matrix(x -self.mu)
            inv = np.linalg.inv(self.sigma)      
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")
    

def arbitraryPDF(x):
    if 0<x[1]<4: 
        return stats.uniform.pdf(x[0]) * 0.1*x[1]
    else:
        return 0.0000000000000000000000000000001

        
class AdaptiveMHProposal(object):

    def __init__(self):
        pass
    
    def getSample(self):
        pass
    
    def getPDF(self, value):
        pass
    
    def getStartPoint(self):
        pass
    
    def adjust(self):
        pass
    
    
    
class AdaptiveGibbsProposal(object):

    def __init__(self, dimensionality):
        self.logVariance = np.zeros(dimensionality) 
    
    def getSample(self, currentValue, index):
        sample = currentValue.copy()
        np.put(sample, index, np.random.normal(sample[index], 10**self.logVariance[index]))
        return sample
    
    def getStartPoint(self):
        return np.array([np.random.normal(0, 1) for var in self.logVariance])
    
    def adjust(self, list, n):
        delta = min(0.01, n**(-0.5))
        for i in xrange(self.logVariance.size):
            if list[i][1] > 0:
                acceptance = list[i][0] / list[i][1]
            else:
                acceptance = 0
           # print i, self.logVariance[i], acceptance
            if acceptance > 0.44:
                self.logVariance[i] += delta
            elif acceptance < 0.44:
                self.logVariance[i] -= delta
            
           # print "to", self.logVariance[i]
    
    
class CoercedAcceptanceProposal(object):

    def __init__(self):
        pass
    
    def getSample(self):
        pass
    
    def getPDF(self, value):
        pass
    
    def getStartPoint(self):
        pass
    
    def adjust(self):
        pass
        
  