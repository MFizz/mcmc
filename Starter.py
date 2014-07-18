'''
Created on 16.07.2014

@author: Jimbo
'''
import MetropolisHastings as MH
import scipy.stats as stats
import Name
import Proposal


if __name__ == '__main__':
    
    problem1 = MH.MetropolisHastings(Name.METROPOLIS_HASTINGS, stats.cauchy.pdf, Proposal.MHProposal(10,3), True)
    problem1.start(noOfSamples=10000000, animate=True, stepSize=100)