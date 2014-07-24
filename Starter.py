'''
Created on 16.07.2014

@author: Jimbo
'''
import MetropolisHastings as MH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

dimensionality = 20
desiredMean = np.array([-5,-2])
desiredCov = np.array([[6,-1],[-1,2]])
stepSize = 1000
noOfSamples = 100000

# high dim
highDimMean = (np.random.rand(dimensionality)-0.5) * 0
mat = (np.random.rand(dimensionality, dimensionality)-0.5)
highDimCov = (mat + mat.transpose()) * 0.01
highDimCov = highDimCov - np.diag(np.diag(highDimCov)) + np.diag(np.diag(np.random.rand(dimensionality, dimensionality))) * 4
#highDimCov = np.diag(np.ones_like(highDimMean))
print highDimMean
print highDimCov


if __name__ == '__main__':
    desired = Distribution.MultivariateNormal(desiredMean, desiredCov)
    highdimensional = Distribution.MultivariateNormal(highDimMean, highDimCov)
    problem0 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, stats.cauchy.pdf, Distribution.UnivariateNormal(0,1), True)
    
    problem1 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.zeros((dimensionality,)), np.identity(dimensionality)/5.), randomWalk=True)
    
    problem2 = MH.MetropolisHastings(Name.ADAPTIVE_METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, None), Distribution.MultivariateNormal(np.zeros((dimensionality,)), 0.1**2 * np.identity(dimensionality) * 1./dimensionality), randomWalk=True)
    problem2b = MH.MetropolisHastings(Name.ADAPTIVE_METROPOLIS_HASTINGS, lambda x: highdimensional.getPDF(x,None), Distribution.MultivariateNormal(np.zeros((dimensionality,)), 0.1**2 * np.identity(dimensionality) * 1./dimensionality), randomWalk=True)
    
    problem3 = MH.MetropolisHastings(Name.ADAPTIVE_GIBBS, lambda x: desired.getPDF(x, None), Distribution.AdaptiveGibbsProposal(dimensionality), randomWalk=True)
    problem3b = MH.MetropolisHastings(Name.ADAPTIVE_GIBBS, lambda x: highdimensional.getPDF(x, None), Distribution.AdaptiveGibbsProposal(dimensionality), randomWalk=True)
    
  #  problem0.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=dimensionality, animateStatistics=False, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem1.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)
   # problem2.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov )
    problem2b.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=dimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=50, desiredCovarianceMatrix=highDimCov )
    problem3b.start(noOfSamples=noOfSamples, stepSize=stepSize, dimensionality=dimensionality, animateStatistics=False, animateDistribution=False, gibbsBatchSize=50, desiredCovarianceMatrix=highDimCov )
    #problem3.start(noOfSamples=noOfSamples, stepSize=1000, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True, gibbsBatchSize=50, desiredCovarianceMatrix=desiredCov)