'''
Created on 16.07.2014

@author: Jimbo
'''
import MetropolisHastings as MH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

dimensionality = 2
desiredMean = np.array([-5,-2])
desiredCov = np.array([[6,-1],[-1,2]])
stepSize = 500

if __name__ == '__main__':
    desired = Distribution.MultivariateNormal(desiredMean, desiredCov)
    problem0 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, stats.cauchy.pdf, Distribution.UnivariateNormal(0,1), True)
    problem0b = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.array([0,0]), np.identity(dimensionality)/5.), randomWalk=True)
    problem1 = MH.MetropolisHastings(Name.ADAPTIVE_METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.array([0,0]), 0.1 * np.identity(dimensionality) * 1./dimensionality), randomWalk=True)
    problem2 = MH.MetropolisHastings(Name.ADAPTIVE_GIBBS, lambda x: desired.getPDF(x, None), Distribution.AdaptiveGibbsProposal(dimensionality), randomWalk=True)
    
  #  problem0.start(noOfSamples=10000000, stepSize=1000, dimensionality=dimensionality, animateStatistics=False, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
    problem0b.start(noOfSamples=10000000, stepSize=1000, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem1.start(noOfSamples=10000000, stepSize=stepSize, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov )
    #problem2.start(noOfSamples=10000000, stepSize=1000, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)