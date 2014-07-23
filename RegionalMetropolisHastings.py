import Name
import math
import numpy as np
import matplotlib.pyplot as plt
import Animation

class RegionalMetropolisHastings():


    def __init__(self, algorithm, desired):
        self.algorithm = algorithm      # name of the used algorithm
        self.desired = desired          # the function to be sampled from (a pdf)

    
    def delta(self, x, n, acceptancerate):
        
        # ideal acceptance rate
        ideal = 0.234
        boundary = 100
        threshold = 0.01
        delta = min(0.01, math.pow(n,-0.5))
        if acceptancerate - threshold > ideal:
            return max(-boundary,min(boundary,x + delta))
        elif acceptancerate + threshold < ideal:
            return max(-boundary,min(boundary,x - delta))
        return x
    
    def check(self,n):
        if abs(n) < 0.0001:
            return 0.0001
        return n
    
    def start(self, noOfSamples, animate, stepSize, dimensionality):
        acceptedA = 0 
        acceptedB = 0
        propA = False
        propB = False
        acceptanceRateA = 0
        acceptanceRateB = 0
        aSamples = np.array([0])
        bSamples = np.array([0])
        # adaptive params a and b for two regions
        a=-0.3
        b=-0.13
        # number of 100 samples
        n=0
        # get a start point
        x = np.random.multivariate_normal(np.zeros(dimensionality), np.identity(dimensionality))
        samplesA = np.array([x])
        samplesB = np.array([x])
        
        if animate:
            plt.ion()

        for i in xrange(noOfSamples):
            
            acceptance = 0
            
            if np.linalg.norm(x) <= dimensionality:
                propA = True
                x_new = np.random.multivariate_normal(x, np.identity(dimensionality)*np.exp(2*a))
                if np.linalg.norm(x_new) <= dimensionality:
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) 
                else:
                    
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) * np.exp(dimensionality*(a-b)-0.5*np.dot(x,x_new)*(np.exp(-2*b)-np.exp(-2*a)))
            else: 
                propB = True
                x_new = np.random.multivariate_normal(x, np.identity(dimensionality)*np.exp(2*b))
                if np.linalg.norm(x_new) > dimensionality:
                    
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) 
                else:
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) * np.exp(dimensionality*(b-a)-0.5*np.dot(x,x_new)*(np.exp(-2*a)-np.exp(-2*b)))
                
            acceptance = min(1,acceptance);
            
            # accept the new proposal or stick with the old one
            if acceptance > np.random.random():
                x = x_new
            if propA:
                if acceptance > np.random.random():
                    acceptedA+=1
                if dimensionality==1:
                    samplesA = np.append(samplesA, x)
                else:
                    samplesA = np.append(samplesA, [x], axis=0)
                acceptanceRateA = float(acceptedA)/samplesA.shape[0]
            elif propB:
                if acceptance > np.random.random():
                    acceptedB+=1
                if dimensionality==1:
                    samplesB = np.append(samplesB, x)
                else:
                    samplesB = np.append(samplesB, [x], axis=0)
                acceptanceRateB = float(acceptedB)/samplesB.shape[0]
                        
            if i % 100 == 0:
                n += 1  
                a = self.delta(a,n,acceptanceRateA)
                b = self.delta(b,n,acceptanceRateB)
                aSamples = np.append(aSamples, a)
                bSamples = np.append(bSamples, b)
                
            propA = False
            propB = False             
            
            if animate and (i+2)%stepSize==0:
                #Animation.animateParams(aSamples, bSamples, acceptedA, acceptedB, a, b)
                if dimensionality==2:
                    Animation.regionalAnimate2D(samplesA, samplesB, acceptanceRateA, acceptanceRateB, acceptedA, acceptedB, a, b)
            
    
        
        