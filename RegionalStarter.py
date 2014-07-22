import MetropolisHastings as MH
import RegionalMetropolisHastings as RMH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

dimensionality = 2

if __name__ == '__main__':
    
    desired = Distribution.RegionalMultivariateNorm(np.array(np.zeros(dimensionality)), np.identity(dimensionality))
    problem = RMH.RegionalMetropolisHastings(Name.REGIONAL_ADAPTIVE_METROPOLIS_HASTINGS, lambda x: desired.getPDF(x))
    problem.start(noOfSamples=10000000, animate=True, stepSize=100, dimensionality=dimensionality)