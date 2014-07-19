'''
Created on 16.07.2014

@author: Jimbo
'''
import MetropolisHastings as MH
import scipy.stats as stats
import Name
import Distribution
import numpy as np

if __name__ == '__main__':
    desired = Distribution.MultivariateNormal(np.array([3,-2]), np.array([[3,0.5],[0.5,2]]))
    problem1 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, lambda x: desired.getPDF(x, np.array([0,0])), Distribution.MultivariateNormal(np.array([0,0]), np.array([[5,0],[0,5]])), True)
   # problem1 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, stats.cauchy.pdf, Distribution.UnivariateNormal(0,1), True)
    problem1.start(noOfSamples=10000000, animate=True, stepSize=100, dimensionality=1)