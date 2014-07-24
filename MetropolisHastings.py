'''
Created on 16.07.2014

@author: Jimbo
'''

import Name
import numpy as np
import matplotlib.pyplot as plt
import Animation
import Utility
import math

class MetropolisHastings():


    def __init__(self, algorithm, desired, proposal, randomWalk):
        self.algorithm = algorithm      # name of the used algorithm
        self.desired = desired          # the function to be sampled from (a pdf)
        self.proposal = proposal        # the proposal function
        self.randomWalk = randomWalk    # boolean, indicates whether we use Random Walk MH or not ( is the proposal distribution dependent on the current value or not )
    
    
    
    def start(self, noOfSamples, stepSize, dimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=1, desiredCovarianceMatrix=None, ACT=True, subopt=True):
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        while not self.desired(x) > 0:
            x = self.proposal.getStartPoint()
            
        samples = np.array([x])
        
        # stats variables
        acceptanceWindow = 1000
        covCalc = Utility.covarianceCalculator(x)
        suboptimality = []
        act = []
        asjdList = []
        asjd = 0
        acceptanceRates = []
        sampleX = []
        
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
                div1 = self.desired(x_new)
                div2 = self.desired(x)
                if not div1 >= 0 or math.isnan(div1):
                    div1 = 0.
                if not div2 >0 or math.isnan(div2):
                    div2 = 0.0001
               # print div1, div2
                acceptance = ( float(self.desired(x_new))/float(self.desired(x)) )
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
            #if dimensionality<3:
            if samples.shape[0]%acceptanceWindow !=0:
                acceptanceRate = float(accepted)/(samples.shape[0]%acceptanceWindow)
               # print i,acceptanceRate
            else:
                print i, acceptanceRate
                acceptanceRates.append(acceptanceRate)
                accepted = 0
                if animateStatistics:
                    sampleX.append(samples.shape[0])
                    if subopt:
                        suboptimality.append(Utility.getSuboptimality(covCalc.getSampleCovariance(samples), desiredCovarianceMatrix))
                    else:
                        suboptimality.append(0)
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
        
        if dimensionality > 2:
            acceptanceRate = acceptanceRates[-1]
            if ACT:
                act = Utility.getACT(samples)
            else:
                act = 0
            if subopt:
                suboptimality = Utility.getSuboptimality(covCalc.getSampleCovariance(samples), desiredCovarianceMatrix)
            else:
                suboptimality = 0
            asjd = asjd
            print self.algorithm
            print "acc: ", acceptanceRate
            print "mean acc: ", np.mean(acceptanceRates)
            print "subopt: ", suboptimality
            print "act: ", act
            print "asjd", asjd
        
        plt.ioff()
        plt.show()