#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy shared utilities for all ISM thesis projects.

Provides figure saving (saveFig), pickle serialization (savePkl/loadPkl),
directory creation (mkdir_p), and miscellaneous helpers (clumpyRandom,
fit_exp_linear, kl_divergence, delaydist, state2nap).

New code should prefer utils/serialization.py for pathlib-based save/load.
"""

from pathlib import Path

import numpy as np 

from errno import EEXIST
import os 

import pickle
from typing import Optional



def clumpyRandom(size, choices, seedprobability, numiter=1):
    pattern = np.random.choice(choices, (size, size), p=seedprobability)
    for _ in range(numiter):
        for _ in range(size ** 2):
            x = np.random.choice(range(size))
            y = np.random.choice(range(size))
            adjacent = pattern[max(x-1, 0):x+2, max(y-1, 0):y+2]
            pattern[x, y] = np.random.choice(adjacent.flatten())
    return pattern



def saveFig(fig, savename, savepath=None, filetype='pdf'):
    mkdir_p(savepath)
    fig.savefig(f'{savepath}/{savename}.{filetype}', format=filetype)


def savePkl(obj, savename: str, savepath: Optional[str] = None):
    dest = Path(savepath) if savepath else Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)
    filename = dest / f'{savename}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def loadPkl(savename: str, savepath: Optional[str] = None):
    dest = Path(savepath) if savepath else Path.cwd()
    filename = dest / f'{savename}.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line.
    If the path exists, no error."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise




def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K


from scipy.special import rel_entr
def kl_divergence(p, q):
    return np.sum(rel_entr(p, q))



def delaydist(signal,numdelays=10, maxdist=15, firstdelay=1, sqdist=False,
             dist = 'cityblock'):
    delay_dist = np.zeros((maxdist,numdelays+1-firstdelay))
    delay_kl = np.zeros((numdelays+1-firstdelay,))
    
    #For the unconditioned distirbution
    r,c = np.triu_indices(np.size(signal,0),1)
    distances_null = np.sum(np.abs(signal[r,:] - signal[c,:]), axis=1)
    nulldist,_ = np.histogram(distances_null,np.arange(-0.5,maxdist+0.5))
    nulldist = nulldist/np.sum(nulldist)
    
    for delay in range(firstdelay, numdelays + 1):
        lastindex = None
        if delay>0:
            lastindex = 0-delay
        if dist == 'cityblock':
            distances = np.sum(np.abs(signal[:lastindex,:] - signal[delay:,:]), axis=1)
        if dist == 'euclidian':
            distances = np.sqrt(np.sum(np.square(signal[:lastindex,:] - signal[delay:,:]), axis=1))
        delay_dist[:,delay-firstdelay],_ = np.histogram(distances,np.arange(-0.5,maxdist+0.5))
        delay_dist[:,delay-firstdelay] = delay_dist[:,delay-firstdelay]/np.sum(delay_dist[:,delay-firstdelay])
        if sqdist:
            delay_kl[delay-firstdelay] = np.mean(distances**2.0)
        else:
            delay_kl[delay-firstdelay] = kl_divergence(delay_dist[:,delay-firstdelay],nulldist)
    return delay_dist, delay_kl


import pynapple as nap
def state2nap(state):
    data = np.vstack((state['agent_pos'][:-1,0],state['agent_pos'][:-1,1])).T
    state_nap = nap.TsdFrame(t=np.arange(np.size(data,0)),
                             d=data,
                             time_units='s',
                             columns=('x','y'))
    

        
    return state_nap



