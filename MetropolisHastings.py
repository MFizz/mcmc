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
    
    
    
    def start(self, noOfSamples, stepSize, dimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=1):
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        gibbsList = [[0,0] for dimension in x]
        samples = np.array([x])
        acceptanceWindow = 1000
        covCalc = Utility.covarianceCalculator(x)
        
        gibbsFactor = 1
        if self.algorithm == Name.ADAPTIVE_GIBBS:
            gibbsFactor = gibbsBatchSize * x.size
        
        if animateDistribution or animateStatistics:
            animationAx = None
            statAx = None
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
                    fig = plt.figure(figsize=(10,10))
                    animationAx = plt.subplot(221)
                    if dimensionality>1:
                        animationAx2 = plt.subplot(222)
                        Animation.animate2DReal(self.desired, animationAx2)
                    statAx = plt.subplot(223)
                else:
                    fig = plt.figure(figsize=(10,8))
                    statAx = plt.subplot(111)
                    
                

        for i in xrange(noOfSamples):
            # get a proposal and calulate the acceptance rate
            if self.randomWalk:
                if self.algorithm == Name.ADAPTIVE_METROPOLIS_HASTINGS and samples.shape[0] > dimensionality/2.:
                    x_new = self.proposal.getSample(x, sampleCovariance=covCalc.getSampleCovariance(samples)* 1.38**2 * 1./dimensionality)
                elif self.algorithm == Name.ADAPTIVE_GIBBS:
                    x_new = self.proposal.getSample(x, samples.shape[0]%x.size)
                else:
                    x_new = self.proposal.getSample(x)
                acceptance = ( self.desired(x_new)/self.desired(x) )
            else:
                if self.algorithm == Name.ADAPTIVE_GIBBS:
                    x_new = self.proposal.getSample(x)
                    acceptance = ( self.desired(x_new)/self.desired(x) )
                else:
                    x_new = self.proposal.getSample(None)
                    acceptance = ( self.desired(x_new)/self.desired(x) * self.proposal.getPDF(x,None)/self.proposal.getPDF(x_new,None) )
            
            # accept the new proposal or stick with the old one
            if acceptance > np.random.random():
                x = x_new
                accepted+=1.
                gibbsList[samples.shape[0]%x.size][0] += 1.
            else:
                x = x
            gibbsList[samples.shape[0]%x.size][1] += 1.

            if acceptance>0.0:
                if dimensionality==1:
                    samples = np.append(samples, x)
                else:
                    samples = np.append(samples, [x], axis=0)
            
            if samples.shape[0]%acceptanceWindow !=0:
                acceptanceRate = float(accepted)/(samples.shape[0]%acceptanceWindow)
            else:
                sampleX.append(samples.shape[0])
                acceptanceRates.append(acceptanceRate)
                accepted = 0
                
            if self.algorithm==Name.ADAPTIVE_GIBBS and samples.shape[0]%gibbsFactor == 0:
                self.proposal.adjust(gibbsList, samples.shape[0]/float(gibbsFactor))
                gibbsList = [[0,0] for dimension in x]
            
            
            if (animateDistribution or animateStatistics) and (i+2)%(stepSize)==0:

                if animateDistribution and dimensionality==1:
                    Animation.animate1D(samples, binBoundaries, binSize, xDesired, pDesired, animationAx)
                if animateDistribution and dimensionality==2:
                    if self.algorithm == Name.ADAPTIVE_GIBBS:
                        Animation.animate2D(samples, animationAx)
                    else:
                        Animation.animate2D(samples, animationAx)
                        
                if animateStatistics:
                    if self.algorithm == Name.ADAPTIVE_GIBBS:
                        Animation.animateStats(sampleX[::gibbsBatchSize], acceptanceRates[::gibbsBatchSize], statAx)
                    else:
                        Animation.animateStats(sampleX, acceptanceRates, statAx)
                        
                plt.pause(0.00001)
            
        