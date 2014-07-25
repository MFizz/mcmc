'''
Created on 16.07.2014

@author: Jimbo
'''
import MetropolisHastings as MH
import RegionalMetropolisHastings as RMH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

desiredDimensionality = 2
desiredMean = np.array([-5,-2])
desiredCov = np.array([[6,-1],[-1,2]])
stepSize = 1000
noOfSamples = 100000

# high dim
highDimensionality = 12
highDimMean = (np.random.rand(highDimensionality)-0.5) * 0
mat = (np.random.rand(highDimensionality, highDimensionality)-0.5)
highDimCov = (mat + mat.transpose()) * 0.01
highDimCov = highDimCov - np.diag(np.diag(highDimCov)) + np.diag(np.diag(np.random.rand(highDimensionality, highDimensionality))) * 4
#highDimCov = np.diag(np.ones_like(highDimMean))
print highDimMean
print highDimCov


if __name__ == '__main__':
    desired = Distribution.MultivariateNormal(desiredMean, desiredCov)
    highdimensional = Distribution.MultivariateNormal(highDimMean, highDimCov)
    problem0 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, stats.cauchy.pdf, Distribution.UnivariateNormal(0,1), True)
    
    problem1 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.zeros((desiredDimensionality,)), np.identity(desiredDimensionality)/5.), randomWalk=True)
    
    problem2 = MH.MetropolisHastings(Name.ADAPTIVE_METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.zeros((desiredDimensionality,)), 0.1**2 * np.identity(desiredDimensionality) * 1./desiredDimensionality), randomWalk=True)
    problem2b = MH.MetropolisHastings(Name.ADAPTIVE_METROPOLIS_HASTINGS, lambda x: highdimensional.getPDF(x,None), Distribution.MultivariateNormal(np.zeros((highDimensionality,)), 0.1**2 * np.identity(highDimensionality) * 1./highDimensionality), randomWalk=True)
    
    problem3 = MH.MetropolisHastings(Name.ADAPTIVE_GIBBS, lambda x: desired.getPDF(x, None), Distribution.AdaptiveGibbsProposal(desiredDimensionality), randomWalk=True)
    problem3b = MH.MetropolisHastings(Name.ADAPTIVE_GIBBS, lambda x: highdimensional.getPDF(x, None), Distribution.AdaptiveGibbsProposal(highDimensionality), randomWalk=True)
    
    problem4 = RMH.RegionalMetropolisHastings(Name.REGIONAL_ADAPTIVE_METROPOLIS_HASTINGS,lambda x: desired.getPDF(x,None))
    problem4b = RMH.RegionalMetropolisHastings(Name.REGIONAL_ADAPTIVE_METROPOLIS_HASTINGS,lambda x: highdimensional.getPDF(x,None))
  #  problem0.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=desiredDimensionality, animateStatistics=False, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem1.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=desiredDimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem2.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=desiredDimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov )
   # problem2b.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=highDimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=50, desiredCovarianceMatrix=highDimCov )
    #problem3.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=desiredDimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem3b.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=highDimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=50, desiredCovarianceMatrix=highDimCov )
    problem4.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=desiredDimensionality, animateStatistics=True, animateDistribution=True,desiredCovarianceMatrix=desiredCov)
   # problem4b.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=highDimensionality, animateStatistics=False, animateDistribution=False,desiredCovarianceMatrix=highDimCov)
    
    
    
    
    