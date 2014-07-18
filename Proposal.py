'''
Created on 16.07.2014

@author: Jimbo
'''

import numpy as np 
import scipy.stats as stats

class MHProposal(object):

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
        return np.random.normal(self.mean, self.variance)
    
    def adjust(self, adjustParameter):
        self.variance = self.variance
        
    
        
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
        
  