'''
Created on 16.07.2014

@author: Jimbo
'''

import Name
import numpy as np
import matplotlib.pyplot as plt
import Animation
import Utility

class MetropolisHastings():


    def __init__(self, algorithm, desired, proposal, randomWalk):
        self.algorithm = algorithm      # name of the used algorithm
        self.desired = desired          # the function to be sampled from (a pdf)
        self.proposal = proposal        # the proposal function
        self.randomWalk = randomWalk    # boolean, indicates whether we use Random Walk MH or not ( is the proposal distribution dependent on the current value or not )
    
    
    
    def start(self, noOfSamples, stepSize, dimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=1, desiredCovarianceMatrix=None, ACT=True):
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        samples = np.array([x])
        
        # stats variables
        acceptanceWindow = 1000
        covCalc = Utility.covarianceCalculator(x)
        suboptimality = []
        act = []
        asjdList = []
        asjd = 0
        
        gibbsFactor = 1
        if self.algorithm == Name.ADAPTIVE_GIBBS:
            gibbsList = [[0,0] for dimension in x]
            gibbsFactor = gibbsBatchSize * x.size
        
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
                sampleX = []
                acceptanceRates = []
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
                    
                

        # here we go
        for i in xrange(noOfSamples):
            # get a proposal and calulate the acceptance rate
            if self.randomWalk:
                if self.algorithm == Name.ADAPTIVE_METROPOLIS_HASTINGS and samples.shape[0] > dimensionality/2.:
                    x_new = self.proposal.getSample(x, sampleCovariance=covCalc.getSampleCovariance(samples)* 2.38**2 * 1./dimensionality)
                elif self.algorithm == Name.ADAPTIVE_GIBBS:
                    x_new = self.proposal.getSample(x, samples.shape[0]%x.size)
                else:
                    x_new = self.proposal.getSample(x)
                acceptance = ( self.desired(x_new)/self.desired(x) )
            else:
                if self.algorithm == Name.ADAPTIVE_GIBBS:
                    x_new = self.proposal.getSample(x, samples.shape[0]%x.size)
                    acceptance = ( self.desired(x_new)/self.desired(x) )
                else:
                    x_new = self.proposal.getSample(None)
                    acceptance = ( self.desired(x_new)/self.desired(x) * self.proposal.getPDF(x,None)/self.proposal.getPDF(x_new,None) )
            
            # accept the new proposal or stick with the old one
            if acceptance > np.random.random():
                asjd = (asjd*(samples.shape[0]-1) + np.linalg.norm(x-x_new)) / samples.shape[0]
                x = x_new
                accepted+=1.
                if self.algorithm == Name.ADAPTIVE_GIBBS:
                    gibbsList[samples.shape[0]%x.size][0] += 1.
            else:
                x = x
                asjd = (asjd*(samples.shape[0]-1)) / samples.shape[0]
            if self.algorithm == Name.ADAPTIVE_GIBBS:
                gibbsList[samples.shape[0]%x.size][1] += 1.
            
            # do not take it as a sample when it is absolutely impossible to generate from the desired distribution
            if acceptance>0.0:
                if dimensionality==1:
                    samples = np.append(samples, x)
                else:
                    samples = np.append(samples, [x], axis=0)
            
            
            # calculate stats
            if samples.shape[0]%acceptanceWindow !=0:
                acceptanceRate = float(accepted)/(samples.shape[0]%acceptanceWindow)
            elif animateStatistics:
                sampleX.append(samples.shape[0])
                acceptanceRates.append(acceptanceRate)
                accepted = 0
                suboptimality.append(Utility.getSuboptimality(covCalc.getSampleCovariance(samples), desiredCovarianceMatrix))
                if ACT:
                    act.append(Utility.getACT(samples[-5000:]))
                else:
                    act.append(0)
                asjdList.append(asjd)
               # act.append(69)
                
            if self.algorithm==Name.ADAPTIVE_GIBBS and samples.shape[0]%gibbsFactor == 0:
                self.proposal.adjust(gibbsList, samples.shape[0]/float(gibbsFactor))
                gibbsList = [[0,0] for dimension in x]
            
            
            if (animateDistribution or animateStatistics) and (i+2)%(stepSize)==0:

                if animateDistribution and dimensionality==1:
                    Animation.animate1D(samples, binBoundaries, binSize, xDesired, pDesired, animationAx)
                if animateDistribution and dimensionality==2:
                    Animation.animate2D(samples, animationAx)
                        
                if animateStatistics:
                    Animation.animateStats(sampleX, acceptanceRates, acceptanceRateAx, suboptimality, suboptimalityAx, act, actAx, asjdList, asjdAx)
                        
                plt.pause(0.00001)
        plt.ioff()
        plt.show()