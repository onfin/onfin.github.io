#!/usr/bin/python3

import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import datetime
from calendar import monthrange
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline, BSpline

full_size = (8, 4)
plt.rcParams['font.size'] = 14
plt.rcParams['font.sans-serif'] = ['Liberation Sans Narrow']

def setup_ax(ax, log=True):
    ax.grid(linestyle="--", linewidth=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)

    if log:
        ax.set_yscale("log")
        ax.get_yaxis().set_major_locator(plt.NullLocator())
        ax.get_yaxis().set_major_formatter(plt.NullFormatter())
        ax.get_yaxis().set_minor_locator(plt.NullLocator())
        ax.get_yaxis().set_minor_formatter(plt.NullFormatter())


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = int(len(xdata)/2)-1 #np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1


    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )



data = pd.read_excel("stats.xls", [0], index_col=0)
dt = data[0]
snp = dt.iloc[:,2]
bills = dt.iloc[:,4]
bonds = dt.iloc[:,6]
gold = dt.iloc[:,8]

def portfolio(f, ds1, ds2):
    res = pd.DataFrame(index=ds1.index, columns=["val"])
    snp_v = f
    bonds_v = 1-f
    for d in ds1.index[1:]:
        snp_v *= ds2[d]/ds2[d-1]
        bonds_v *= ds1[d]/ds1[d-1]
        dff = (snp_v/(snp_v+bonds_v) - f)*snp_v
        snp_v -= dff
        bonds_v += dff
        res.loc[d, "val"] = snp_v+bonds_v
    return res.iloc[:,0]



# def plot_instr_price(ds, interval, label):
#     rt = ds.rolling(interval).apply(lambda d: d[-1]/d[0], raw=True)
#     rt = (rt[~ pd.isnull(rt)]-1)*100
#     rt = sorted(rt.values)

#     plt.plot(np.arange(0,len(rt))/len(rt), rt, label=label)

# def plot_instr_yield(ds, interval, label):
#     rt = ds.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#     rt = (rt[~ pd.isnull(rt)]-1)*100
#     rt = sorted(rt.values)

#     plt.plot(np.arange(0,len(rt))/len(rt), rt, label=label)


# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_instr_yield(snp, 10, 'акции')
# plot_instr_yield(bonds, 10, 'облигации')
# plot_instr_yield(bills, 10, 'кэш')
# plot_instr_yield(gold, 10, 'золото')
# plt.legend()

# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('доля временных периодов, %')

# fig.tight_layout()
# plt.savefig('yield_distr.svg')
# plt.clf()


# def plot_risk(ds, label):
#     interval = [2,5,10,15,20,25,30]
#     rsk = []
#     for i in interval:
#         rt = ds.rolling(i).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         rt = sorted(rt.values)
#         rsk.append((rt[int(len(rt)/10*9)] - rt[int(len(rt)/10)])/2)
#         #rsk.append(rt.std())
#     ln = plt.plot(interval, rsk, label=label)[0]

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_risk(snp, 'акции')
# plot_risk(bonds, 'облигации')
# plot_risk(bills, 'кэш')
# plot_risk(gold, 'золото')

# ax = plt.gca()
# ax.set_ylabel('риск, ±%г')
# ax.set_xlabel('горизонт инвестирования, лет')

# plt.legend()

# fig.tight_layout()
# plt.savefig('risk_pure_std.svg')
# plt.clf()


# def plot_yield(ds, label):
#     interval = [2,5,10,15,20,25,30]
#     yld = []
#     for i in interval:
#         rt = ds.rolling(i).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         yld.append(rt.median())
#     ln = plt.plot(interval, yld, label=label)[0]

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_yield(snp, 'акции')
# plot_yield(bonds, 'облигации')
# plot_yield(bills, 'кэш')
# plot_yield(gold, 'золото')

# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('горизонт инвестирования, лет')

# plt.legend()
# fig.tight_layout()
# plt.savefig('yield_pure_std.svg')
# plt.clf()


# def plot_portfolio(ds1, ds2, interval,label):
#     yld = []
#     rsk = []
#     fr = []
#     for f in np.linspace(0, 1, 11):
#         pf = portfolio(f, ds1, ds2)
#         rt = pf.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         yld.append(rt.median())
#         rt = sorted(rt.values)
#         rsk.append((rt[int(len(rt)/10*9)] - rt[int(len(rt)/10)])/2)
#         fr.append(f)
#     ln = ax.plot(rsk, yld, label=label, marker=".")[0]
#     add_arrow(ln)

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_portfolio(snp, bonds, 25, 'акции+облигации')
# plot_portfolio(snp, gold, 25, 'акции+золото')
# plot_portfolio(bonds, gold, 25, 'облигации+золото')

# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('риск, ±%г')

# plt.legend()
# fig.tight_layout()
# plt.savefig('portfolios_ry.svg')
# plt.clf()


# def max_drawdown(d):
#     prev_max = d[0]
#     max_dd = 2
#     for v in d:
#         if v > prev_max:
#             prev_max = v
#         dd = v/prev_max
#         if dd < max_dd:
#             max_dd = dd
#     return 1-max_dd


# def plot_portfolio_dd(ds1, ds2, interval,label, **kwargs):
#     yld = []
#     rsk = []
#     for f in np.linspace(0, 1, 11):
#         pf = portfolio(f, ds1, ds2)
#         rt = pf.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         yld.append(rt.median())
#         rsk.append(pf.rolling(interval).apply(max_drawdown, raw=True).max()*100)
#     ln = ax.plot(rsk, yld, label=label, marker=".", **kwargs)[0]
#     add_arrow(ln)

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_portfolio_dd(snp, bonds, 25, 'акции+облигации')
# plot_portfolio_dd(snp, gold, 25, 'акции+золото')
# plot_portfolio_dd(bonds, gold, 25, 'облигации+золото')

# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('риск, % капитала')

# plt.legend()
# fig.tight_layout()
# plt.savefig('portfolios_ry_dd.svg')
# plt.clf()


# def plot_portfolio_tg(ds1, ds2, interval,label=None, **kwargs):
#     yld = []
#     rsk = []
#     for f in np.linspace(0, 1, 11):
#         pf = portfolio(f, ds1, ds2)
#         rt = pf.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         y = rt.median()/100+1
#         yld.append(rt.median())
#         rsk.append(pf.rolling(interval).apply(lambda d: (d[0]*y**len(d)-d[-1])/(d[0]*y**len(d))*100, raw=True).max())
#     ln = ax.plot(rsk, yld, label=label, marker=".", **kwargs)[0]
#     add_arrow(ln)
#     return ln

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# ln1 = plot_portfolio_tg(snp, bonds, 25, 'акции+облигации')
# ln2 = plot_portfolio_tg(snp, gold, 25, 'акции+золото')
# ln3 = plot_portfolio_tg(bonds, gold, 25, 'облигации+золото')

# plot_portfolio_tg(snp, bonds, 5, linestyle="dashed", color=ln1.get_color())
# plot_portfolio_tg(snp, gold, 5, linestyle="dashed", color=ln2.get_color())
# plot_portfolio_tg(bonds, gold, 5, linestyle="dashed", color=ln3.get_color())


# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('риск, % недополученного капитала')

# plt.legend()
# fig.tight_layout()
# plt.savefig('portfolios_ry_tg_max.svg')
# plt.clf()



# def plot_portfolio_tg(ds1, ds2, interval,label=None, **kwargs):
#     yld = []
#     rsk_max = []
#     rsk_avg = []
#     for f in np.linspace(0, 1, 11):
#         pf = portfolio(f, ds1, ds2)
#         rt = pf.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True)
#         rt = (rt[~ pd.isnull(rt)]-1)*100
#         y = rt.median()/100+1
#         yld.append(rt.median())
#         r = pf.rolling(interval).apply(lambda d: (d[0]*1.05**len(d)-d[-1])/(d[0]*1.05**len(d))*100, raw=True)
#         rsk_max.append(r.max())
#         #rsk_avg.append(r[r>0].sum()/len(r[r>0]))

#     ln = ax.plot(rsk_max, yld, label=label, marker=".", **kwargs)[0]
#     add_arrow(ln)
#     # ln = ax.plot(rsk_avg, yld, label=label, marker=".", linestyle="dashed", color=ln.get_color(), **kwargs)[0]
#     # add_arrow(ln)
#     return ln

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# ln1 = plot_portfolio_tg(snp, bonds, 25, 'акции+облигации')
# ln2 = plot_portfolio_tg(snp, gold, 25, 'акции+золото')


# ax = plt.gca()
# ax.set_ylabel('реальная доходность, %г')
# ax.set_xlabel('риск, % недополученного капитала до 5%г')

# plt.legend()
# fig.tight_layout()
# plt.savefig('portfolios_ry_5p.svg')
# plt.clf()


# def plot_risk(ds, label):
#     interval = [2,5,10,15,20,25,30]
#     rsk = []
#     for i in interval:
#         r = ds.rolling(i).apply(lambda d: (d[0]*1.05**len(d)-d[-1])/(d[0]*1.05**len(d))*100, raw=True)
#         rsk.append(r.max())
#     ln = plt.plot(interval, rsk, label=label)[0]

# fig, ax = plt.subplots(figsize=full_size)
# setup_ax(ax, False)

# plot_risk(snp, 'акции')
# plot_risk(portfolio(0.6, snp, bonds), "60/40")
# plot_risk(bonds, 'облигации')

# ax = plt.gca()
# ax.set_ylabel('риск, % недополученного капитала до 5%г')
# ax.set_xlabel('горизонт инвестирования, лет')

# plt.legend()

# fig.tight_layout()
# plt.savefig('risk_prag_t.svg')
# plt.clf()


interval = 2
snp_roll = snp.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True).dropna()
bonds_roll = bonds.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True).dropna()
gold_roll = gold.rolling(interval).apply(lambda d: (d[-1]/d[0])**(1.0/len(d)), raw=True).dropna()

def calc_eff(returns, cov):
    icov = la.inv(cov)
    ones = np.ones(len(returns))
    a = np.matmul(np.matmul(ones, icov), ones)
    b = np.matmul(np.matmul(ones, icov), returns)
    c = np.matmul(np.matmul(returns, icov), returns)
    d = a*c - b**2
    return lambda r: (a*r**2 - 2*b*r + c)/d

returns = np.array([snp_roll.median(), bonds_roll.median(), gold_roll.median()])-1
cov = np.cov([np.array(snp_roll), np.array(bonds_roll), np.array(gold_roll)])
std = np.diag(cov)
eff = calc_eff(returns, cov)


x = np.linspace(np.min(returns), np.max(returns), 50)

def utility(x):
    return (1+x-0.6/100)**100 * x


rsk = eff(x)
fig, ax = plt.subplots(figsize=full_size)
setup_ax(ax, False)

ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('риск')

plt.plot(rsk*100, x*100, label='линейный риск')
plt.plot(utility(rsk)*100, x*100, label='нелинейный риск')
plt.scatter(std*100, returns*100)
plt.scatter(utility(std)*100, returns*100)

plt.legend()
fig.tight_layout()
plt.savefig('_eff.png')
plt.clf()


def utility(x):
    return np.array([v-10*abs(v-0.02)**2 if v < 0.02 else (((0.03-v)*20+1)**(1/2) if v > 0.03 else 1)*v for v in x])


fig, ax = plt.subplots(figsize=full_size)
setup_ax(ax, False)

ax = plt.gca()
ax.set_ylabel('доходность, %г')
ax.set_xlabel('риск')

plt.plot(rsk*100, x*100, label='линейная доходность')
plt.plot(rsk*100, utility(x)*100, label='нелинейная доходность')
plt.scatter(std*100, returns*100)
plt.scatter(std*100, utility(returns)*100)

plt.legend()
fig.tight_layout()
plt.savefig('_eff_y.png')
plt.clf()
