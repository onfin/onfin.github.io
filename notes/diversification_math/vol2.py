#!/usr/bin/python3

import sys
import numpy as np
import numpy.random as rnd
from scipy.optimize import fsolve

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.figure(figsize=(8,4))

def res(rr, s, k, cap):
    r = rr[0]
    return [s*(1-r**k)/(1-r) - cap if r != 1 else s*k-cap]

def find_r(s, k, cap):
    return fsolve(res, [1.01], (s,k,cap))[0]-1


N = 100000
Ns = 50
sm = 0.09
f = 0.000

def year(logp, fate, stocks, invest):
    for m in range(12):
        d = rnd.normal(0, sm, N) + fate
        logp += d

    price = np.exp(logp)
    for i in range(0, N, Ns):
        s = stocks[i:i+Ns]
        p = price[i:i+Ns]
        invest(s, p)
    return price

def model(t, invest):
    logp = np.array([0.0]*N)
    stocks = np.array([1.0]*N)
    cap0 = np.array(stocks)
    fate = rnd.normal(0, f, N)

    for i in range(t):
        p = year(logp, fate, stocks, invest)

    cap = p * stocks
    pcap = []
    for i in range(0, N, Ns):
        pcap.append(np.sum(cap[i:i+Ns])/np.sum(cap0[i:i+Ns]))
    return pcap

t = 30
pcap = model(t, lambda s, p: 0)
y = np.sort(pcap)**(1/t)-1
plt.plot(np.arange(0,len(y))/len(y)*100, y*100, label=f"без довложений и ребалансировок")
print(np.average(pcap))

def invest_equal(stocks, price):
    stocks += 1.0/price

pcap = model(t, invest_equal)
print(np.average(pcap))
y = []
for p in pcap:
    y.append(find_r(1, t, p))
y = np.sort(np.array(y))

plt.plot(np.arange(0,len(y))/len(y)*100, y*100, label=f"докупка на равную сумму")


def invest_cap(stocks, price):
    stocks += price/sum(price) * Ns/price

pcap = model(t, invest_cap)
print(np.average(pcap))
y = []
for p in pcap:
    y.append(find_r(1, t, p))
y = np.sort(np.array(y))

plt.plot(np.arange(0,len(y))/len(y)*100, y*100, label=f"докупка пропорционально капитализации")


def invest_rebal(stocks, price):
    stocks += 1.0/price
    stocks[:] = np.sum(price*stocks)/price/Ns

pcap = model(t, invest_rebal)
print(np.average(pcap))
y = []
for p in pcap:
    y.append(find_r(1, t, p))
y = np.sort(np.array(y))

plt.plot(np.arange(0,len(y))/len(y)*100, y*100, label=f"довложение с ребалансировкой")

ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('доля портфелей, %')

plt.legend()
plt.savefig("portfolio_yields.svg")