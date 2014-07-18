'''
Created on 16.07.2014

@author: Jimbo
'''

import Name
import numpy as np
import matplotlib.pyplot as plt

class MetropolisHastings():


    def __init__(self, algorithm, desired, proposal, randomWalk):
        self.algorithm = algorithm      # name of the used algorithm
        self.desired = desired          # the function to be sampled from (a pdf)
        self.proposal = proposal        # the proposal function
        self.randomWalk = randomWalk    # boolean, indicates whether we use Random Walk MH or not ( is the proposal distribution dependent on the current value or not )
    
    
    
    def start(self, noOfSamples, animate, stepSize):
        samples = np.ndarray([])
        accepted = 0
        # get a start point
        x = self.proposal.getStartPoint()
        
        if animate:
            plt.ion()
            xDesired = np.arange(-10, 10, 0.1)
            pDesired = self.desired(xDesired)
            binSize = 0.25
            binBoundaries = np.arange(-10,10,binSize)


        for i in xrange(noOfSamples):
            # get a proposal and calulate the acceptance rate
            if self.randomWalk:
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

            samples = np.append(samples, x)
            
            
            if animate and (i+2)%stepSize==0:
                plt.clf()
                plt.title("Approximation")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.hist(samples, bins=binBoundaries, weights=np.zeros_like(samples) + (1. / samples.size / binSize))
                plt.plot(xDesired, pDesired)
                plt.xlim([-10,10])
                plt.text(5, np.max(pDesired-0.05), "Samples: %.0f" %samples.size)
                plt.text(5, np.max(pDesired), "Acceptance: %.2f" %(float(accepted)/samples.size))
                plt.draw()
                plt.pause(0.00001)
            
        