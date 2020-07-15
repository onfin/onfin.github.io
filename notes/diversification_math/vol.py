#!/usr/bin/python3

import sys
import numpy as np
import numpy.random as rnd
from scipy.optimize import fsolve

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


N = 10000
sm = 0.09
logp = np.array([0.0]*N)
stocks = np.array([0.0]*N)

def model(t, logp):
    for k in range(1,t+1):
        p = np.exp(logp)
        d = rnd.normal(0, sm, N)
        logp += d


model(120, logp)

caps = np.sort(np.exp(logp))

plt.figure(figsize=(8,4))
plt.plot(np.arange(0,N)/N*100, caps)
ax = plt.gca()
ax.set_ylabel('цена')
ax.set_xlabel('доля акций, %')
ax.set_ylim([1e-2,1e2])
ax.set_yscale('log')
ax.set_yticks([1e-2,1e-1,1,10,1e2])
ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
ax.add_line(Line2D([50, 50], plt.ylim(), color='gray', linewidth=0.3))
ax.add_line(Line2D(plt.xlim(), [1,1], color='gray', linewidth=0.3))

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.tick_params(top=False, bottom=True, labeltop=False,labelbottom=True)
ax2.set_xticks([70,90])
ax2.set_xticklabels(['Ax','Bx'])

ax3 = ax.twinx()
ax3.set_ylim(ax.get_ylim())
ax3.set_yscale('log')
ax3.tick_params(right=False, left=False, labelright=False,labelleft=True)
ax3.set_yticks([caps[7000], caps[9000]])
ax3.set_yticklabels(['Ay', 'By'])


ax.add_line(Line2D([70, 70], [plt.ylim()[0], caps[7000]], color='black', linewidth=0.5))
ax.add_line(Line2D([plt.xlim()[0], 70], [caps[7000], caps[7000]], color='black', linewidth=0.5))
ax.add_line(Line2D([90, 90], [plt.ylim()[0], caps[9000]], color='black', linewidth=0.5))
ax.add_line(Line2D([plt.xlim()[0], 90], [caps[9000], caps[9000]], color='black', linewidth=0.5))

plt.savefig("price_dist_10y.svg")
plt.clf()

y = caps**(1/10)-1
plt.plot(np.arange(0,N)/N*100, y*100) #, label=f"{k/12:.0f} лет")
ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('доля акций, %')
ax.add_line(Line2D([50, 50], plt.ylim(), color='gray', linewidth=0.3))
ax.add_line(Line2D(plt.xlim(), [0,0], color='gray', linewidth=0.3))

plt.savefig("yield_dist_10y.svg")
plt.clf()


plt.plot(np.arange(0,N)/N*100, y*100, label=f"10 лет")
ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('доля акций, %')

model(240, logp)

y = caps**(1/30)-1
plt.plot(np.arange(0,N)/N*100, y*100, label=f"30 лет")

ax.add_line(Line2D([50, 50], plt.ylim(), color='gray', linewidth=0.3))
ax.add_line(Line2D(plt.xlim(), [0,0], color='gray', linewidth=0.3))

plt.legend()
plt.savefig("yield_dist_30y.svg")
plt.clf()


logp = np.array([0.0]*N)
model(120, logp)
caps = np.sort(np.exp(logp))

plt.plot(np.arange(0,N)/N*100, caps) #, label=f"{k/12:.0f} лет")
ax = plt.gca()
ax.set_ylabel('цена')
ax.set_xlabel('доля акций, %')
ax.set_yscale('log')
ax.set_yticks([1e-3,1e-2,1e-1,1,10,1e2,1e3])
ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100', '1000'])

plt.plot(np.arange(0,N)/N*100, caps, label=f"10 лет")
model(240, logp)
caps = np.sort(np.exp(logp))

plt.plot(np.arange(0,N)/N*100, caps, label=f"30 лет")

ax.add_line(Line2D([50, 50], plt.ylim(), color='gray', linewidth=0.3))
ax.add_line(Line2D(plt.xlim(), [1,1], color='gray', linewidth=0.3))
plt.legend()
plt.savefig("price_dist_30y.svg")
plt.clf()


caps = np.exp(logp)
y = np.sort(caps)**(1/30)-1
plt.plot(np.arange(0,N)/N*100, y*100, label=f"отдельные акции")
s = np.std(y)
for n in (10, 50, 500):
    p = (np.convolve(caps, np.ones(n), 'valid')/n)**(1/30)-1
    y = np.sort(p)
    plt.plot(np.arange(0,len(y))/len(y)*100, y*100, label=f"{n} акций")

ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('доля акций/портфелей, %')
ax.add_line(Line2D([50, 50], plt.ylim(), color='gray', linewidth=0.3))
ax.add_line(Line2D(plt.xlim(), [0,0], color='gray', linewidth=0.3))

plt.legend()
plt.savefig("portfolio_yield_dist_30y.svg")
plt.clf()