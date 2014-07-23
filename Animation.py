'''
Created on 19.07.2014

@author: Jimbo
'''

import matplotlib.pyplot as plt
import numpy as np

def animate1D(samples, binBoundaries, binSize, xDesired, pDesired, acceptanceRate):
    plt.clf()
    plt.title("Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.hist(samples, bins=binBoundaries, weights=np.zeros_like(samples) + (1. / samples.size / binSize))
    plt.plot(xDesired, pDesired)
    plt.xlim([-10,10])
    plt.text(5, np.max(pDesired-0.05), "Samples: %.0f" %samples.size)
    plt.text(5, np.max(pDesired), "Acceptance: %.2f" %acceptanceRate)
    plt.draw()
    plt.pause(0.00001)
    
    

def animate2D(samples, acceptanceRate):
    plt.clf()
    plt.title("Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    xItems = [x[0] for x in samples]
    yItems = [x[1] for x in samples]
    plt.text(np.max(xItems)*0.75, np.max(yItems), "Samples: %.0f" %samples.size)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.7, "Acceptance: %.2f" %acceptanceRate)
    plt.axis('equal')
    plt.plot(xItems, yItems, linestyle="None", marker=".")
    plt.draw()
    plt.pause(0.00001)
    
def regionalAnimate2D(samplesA, samplesB, acceptanceRateA, acceptanceRateB, acceptedA, acceptedB, a, b):
    plt.clf()
    plt.title("Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    xItems = [x[0] for x in samplesA]
    yItems = [x[1] for x in samplesA]
    plt.text(np.max(xItems)*0.75, np.max(yItems), "Samples: %.0f" %(samplesA.shape[0]+samplesB.shape[0]))
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.1, "Acceptance for a: %.2f" %acceptanceRateA)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.2, "Acceptance for b: %.2f" %acceptanceRateB)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.3, "Param a: %.2f" %a)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.4, "Param b: %.2f" %b)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.5, "akzeptierte a: %.2f" %acceptedA)
    plt.text(np.max(xItems)*0.75, np.max(yItems)*0.6, "akzeptierte b: %.2f" %acceptedB)
    plt.axis('equal')
    plt.plot(xItems, yItems, linestyle="None", marker=".")
    xItems = [x[0] for x in samplesB]
    yItems = [x[1] for x in samplesB]
    plt.plot(xItems, yItems, linestyle="None", marker=".")
    plt.draw()
    plt.pause(0.00001)
