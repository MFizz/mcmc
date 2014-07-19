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
    
    
    
    def start(self, noOfSamples, animate, stepSize, dimensionality):
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        samples = np.array([x])
        
        if animate:
            plt.ion()
            if dimensionality==1:
                xDesired = np.arange(-10, 10, 0.1)
                pDesired = self.desired(xDesired)
                binSize = 0.25
                binBoundaries = np.arange(-10,10,binSize)


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
                
            acceptanceRate = float(accepted)/samples.size
            
            
            if animate and (i+2)%stepSize==0:
                if dimensionality==1:
                    Animation.animate1D(samples, binBoundaries, binSize, xDesired, pDesired, acceptanceRate)
                if dimensionality==2:
                    Animation.animate2D(samples, acceptanceRate)
            
        