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
    
    
    
    def start(self, noOfSamples, stepSize, dimensionality, animateStatistics=False, animateDistribution=False):
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        samples = np.array([x])
        acceptanceWindow = 1000
        
        
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
                    fig = plt.figure(figsize=(20,8))
                    animationAx = plt.subplot(121)
                    statAx = plt.subplot(122)
                else:
                    fig = plt.figure(figsize=(10,8))
                    statAx = plt.subplot(111)
                    
                

        for i in xrange(noOfSamples):
            # get a proposal and calulate the acceptance rate
            if self.randomWalk:
                if self.algorithm == Name.ADAPTIVE_METROPOLIS_HASTINGS and self.samples > dimensionality/2.:
                    x_new = self.proposal.getSample(x, sampleCovariance=Utility.getSampleCovariance(samples)*1./dimensionality)
                else:
                    x_new = self.proposal.getSample(x)
                acceptance = ( self.desired(x_new)/self.desired(x) )
            else:
                x_new = self.proposal.getSample(None)
                acceptance = ( self.desired(x_new)/self.desired(x) * self.proposal.getPDF(x,None)/self.proposal.getPDF(x_new,None) )
            
            # accept the new proposal or stick with the old one
            if acceptance > np.random.random():
                x = x_new
                accepted+=1
            else:
                x = x

            if dimensionality==1:
                samples = np.append(samples, x)
            else:
                samples = np.append(samples, [x], axis=0)
            
            if samples.size%acceptanceWindow !=0:
                acceptanceRate = float(accepted)/(samples.size%acceptanceWindow)
            else:
                accepted = 0
            
            
            if (animateDistribution or animateStatistics) and (i+2)%stepSize==0:
                if animateDistribution and dimensionality==1:
                    Animation.animate1D(samples, binBoundaries, binSize, xDesired, pDesired, acceptanceRate, animationAx)
                if animateDistribution and dimensionality==2:
                    Animation.animate2D(samples, acceptanceRate, animationAx)
                if animateStatistics:
                    sampleX.append(samples.size)
                    acceptanceRates.append(acceptanceRate)
                    Animation.animateStats(sampleX, acceptanceRates, statAx)
            
        