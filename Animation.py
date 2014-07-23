'''
Created on 19.07.2014

@author: Jimbo
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def animate1D(samples, binBoundaries, binSize, xDesired, pDesired, animationAx):
    animationAx.cla()
    animationAx.set_title("Approximation")
    animationAx.set_xlabel("x")
    animationAx.set_ylabel("f(x)")
    animationAx.hist(samples, bins=binBoundaries, weights=np.zeros_like(samples) + (1. / samples.size / binSize))
    animationAx.plot(xDesired, pDesired)
    animationAx.set_xlim([-10,10])
    plt.draw()
    
    

def animate2D(samples, animationAx):
    animationAx.cla()
    animationAx.set_title("Approximation")
    animationAx.set_xlabel("x")
    animationAx.set_ylabel("y")
    xItems = [x[0] for x in samples]
    yItems = [x[1] for x in samples]
    animationAx.axis('equal')
   # animationAx.plot(xItems, yItems, linestyle="None", marker=".")
    animationAx.hist2d(xItems, yItems,bins=50, norm=colors.LogNorm())
    animationAx.set_xlim(-10,10)
    animationAx.set_ylim(-10,10)
    
    plt.draw()
    

def animate2DReal(function, ax):
    xs = np.arange(-10,10.4,0.4)
    ys = np.arange(-10,10.4,0.4)
    densities = []
    for x in xs:
        for y in ys:
            for i in xrange(int(10000*function([x,y]))):
                densities.append([x,y])
    xItems = [x[0] for x in densities]
    yItems = [x[1] for x in densities]
    ax.hist2d(xItems, yItems,bins=(xs,ys),  norm=colors.LogNorm())
    
    
def animateStats(x, acceptanceRates, statAx):
    statAx.cla()
    statAx.set_title("Acceptance Rate")
    statAx.set_xlabel("sample")
    statAx.set_ylabel("acceptance")
    statAx.set_ylim([0.,1.])
    statAx.plot(x,acceptanceRates)   

    
    
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
