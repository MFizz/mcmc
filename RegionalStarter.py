import MetropolisHastings as MH
import RegionalMetropolisHastings as RMH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

dimensionality = 2
desiredMeanOne = np.array(np.zeros(dimensionality))
desiredMeanTwo = np.array(np.zeros(dimensionality))
desiredCovOne = np.identity(dimensionality)
desiredCovTwo = np.identity(dimensionality)

if __name__ == '__main__':
    
    desired = Distribution.GaussianMixture(desiredMeanOne, desiredCovOne, desiredMeanTwo, desiredCovTwo)
    problem = RMH.RegionalMetropolisHastings(Name.REGIONAL_ADAPTIVE_METROPOLIS_HASTINGS,lambda x: desired.getPDF(x))
    problem.start(noOfSamples=10000000, stepSize=1000, dimensionality=dimensionality, animateStatistics=True, animateDistribution=True)