import Name
import math
import numpy as np
import matplotlib.pyplot as plt
import Animation
import Utility

class RegionalMetropolisHastings():


    def __init__(self, algorithm, desired):
        self.algorithm = algorithm      # name of the used algorithm
        self.desired = desired          # the function to be sampled from (a pdf)

    
    def delta(self, x, n, acceptancerate):
        
        # ideal acceptance rate
        ideal = 0.234
        boundary = 100
        threshold = 0.01
        delta = min(0.005, math.pow(n,-0.5))
        if acceptancerate - threshold > ideal:
            return max(-boundary,min(boundary,x + delta))
        elif acceptancerate + threshold < ideal:
            return max(-boundary,min(boundary,x - delta))
        return x
    
    def check(self,n):
        if abs(n) < 0.0001:
            return 0.0001
        return n
    
    def start(self, noOfSamples, stepSize, dimensionality, animateStatistics=False, animateDistribution=False, desiredCovarianceMatrix=None):
        acceptedA = 0 
        acceptedB = 0
        propA = False
        propB = False
        propABatch = False
        propBBatch = False
        acceptanceRateA = 0
        acceptanceRateB = 0
        aSamples = np.array([0])
        bSamples = np.array([0])
        # adaptive params a and b for two regions
        a=0
        b=0
        # number of 100 samples
        n=0
        # get a start point
        x = np.random.multivariate_normal(np.zeros(dimensionality), np.identity(dimensionality))
        samplesA = np.array([x])
        samplesB = np.array([x])
        samples = np.array([x])
        
        # stats variables
        acceptanceWindow = 1000
        covCalc = Utility.covarianceCalculator(x)
        suboptimality = []
        act = []
        asjdList = []
        asjd = 0
        sampleX = []
        acceptanceRates = []

        if animateDistribution or animateStatistics:
            animationAx = None
            acceptanceRateAx = None
            plt.ion()
            if animateDistribution and not animateStatistics:
                fig = plt.figure(figsize=(10,8))
                animationAx = plt.subplot(111)
            if animateDistribution and dimensionality==1:
                xDesired = np.arange(-10, 10, 0.1)
                pDesired = self.desired(xDesired)
                binSize = 0.25
                binBoundaries = np.arange(-10,10,binSize)
            if animateStatistics:
                if animateDistribution:
                    fig = plt.figure(figsize=(18,10))
                    animationAx = plt.subplot(231)
                    if dimensionality>1:
                        animationAx2 = plt.subplot(232)
                        Animation.animate2DReal(self.desired, animationAx2)
                    acceptanceRateAx = plt.subplot(233)
                    suboptimalityAx = plt.subplot(234)
                    actAx = plt.subplot(235)
                    asjdAx = plt.subplot(236)
                else:
                    fig = plt.figure(figsize=(10,10))
                    acceptanceRateAx = plt.subplot(221)
                    suboptimalityAx = plt.subplot(222)
                    actAx = plt.subplot(223)
                    asjdAx = plt.subplot(224)

        for i in xrange(noOfSamples):
            
            acceptance = 0
            
            #if self.desired.one.getPDF(x) <= self.desired.two.getPDF(x):
            if np.linalg.norm(x) <= dimensionality:
                propA = True
                propABatch = True
                x_new = np.random.multivariate_normal(x, np.identity(dimensionality)*np.exp(2*a))
                #if self.desired.one.getPDF(x_new) <= self.desired.two.getPDF(x_new): 
                if np.linalg.norm(x_new) <= dimensionality:
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) 
                else:
                    
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) * np.exp(dimensionality*(a-b)-0.5*np.dot(x,x_new)*(np.exp(-2*b)-np.exp(-2*a)))
            else: 
                propB = True
                propBBatch = True
                x_new = np.random.multivariate_normal(x, np.identity(dimensionality)*np.exp(2*b))
                #if self.desired.one.getPDF(x_new) > self.desired.two.getPDF(x_new): 
                if np.linalg.norm(x_new) > dimensionality:
                    
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) 
                else:
                    acceptance = self.desired(x_new)/self.check(self.desired(x)) * np.exp(dimensionality*(b-a)-0.5*np.dot(x,x_new)*(np.exp(-2*a)-np.exp(-2*b)))
                
            acceptance = min(1,acceptance);
            
            
            # accept the new proposal or stick with the old one
            if acceptance > np.random.random():
                asjd = (asjd*(samples.shape[0]-1) + np.linalg.norm(x-x_new)) / samples.shape[0]
                x = x_new
                
            if dimensionality==1:
                samples = np.append(samples, x)
            else:
                samples = np.append(samples, [x], axis=0)
            if propA:
                if acceptance > np.random.random():
                    acceptedA+=1
                samplesA = np.append(samplesA, [x], axis=0)
                acceptanceRateA = float(acceptedA)/samplesA.shape[0]
            elif propB:
                if acceptance > np.random.random():
                    acceptedB+=1
                samplesB = np.append(samplesB, [x], axis=0)
                acceptanceRateB = float(acceptedB)/samplesB.shape[0]
                        
            if i % 100 == 0:
                
                n += 1 
                if propABatch: 
                    a = self.delta(a,n,acceptanceRateA)
                if propBBatch:
                    b = self.delta(b,n,acceptanceRateB)
                aSamples = np.append(aSamples, a)
                bSamples = np.append(bSamples, b)
                
            propA = False
            propB = False             
            
            # calculate stats
            if samplesA.shape[0]%acceptanceWindow !=0:
                acceptanceRateA = float(acceptedA)/(samplesA.shape[0]%acceptanceWindow)
                #acceptanceRateB = float(acceptedB)/(samples.shape[0]%acceptanceWindow)
            else:
                print i, acceptanceRateA
                acceptanceRates.append(acceptanceRateA)
                accepted = 0
                if animateStatistics:
                    sampleX.append(samples.shape[0])
                    suboptimality.append(Utility.getSuboptimality(covCalc.getSampleCovariance(samples), desiredCovarianceMatrix))
                    act.append(Utility.getACT(samples[-5000:]))
                    
                    asjdList.append(asjd)
                
            
            if (animateDistribution or animateStatistics) and (i+2)%(stepSize)==0:

                if animateDistribution and dimensionality==1:
                    Animation.animate1D(samples, binBoundaries, binSize, xDesired, pDesired, animationAx)
                if animateDistribution and dimensionality==2:
                    Animation.animate2D(samples, animationAx)
                        
                if animateStatistics:
                    Animation.animateStats(sampleX, acceptanceRates, acceptanceRateAx, suboptimality, suboptimalityAx, act, actAx, asjdList, asjdAx)
                        
                plt.pause(0.00001)
            
        if dimensionality > 2:
            acceptanceRate = acceptanceRates[-1]
            act = Utility.getACT(samples)
            suboptimality = Utility.getSuboptimality(covCalc.getSampleCovariance(samples), desiredCovarianceMatrix)
            
            asjd = asjd
            print self.algorithm
            print "acc: ", acceptanceRate
            print "mean acc: ", np.mean(acceptanceRates)
            print "subopt: ", suboptimality
            print "act: ", act
            print "asjd", asjd
        
        