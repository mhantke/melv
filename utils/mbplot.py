#!/usr/bin/python
# -*- coding: latin-1 -*-
import numpy as np
import matplotlib.pyplot as pypl
import mbtools as t

def plot_proj(v, d, proj_ax, ideal_out=True, filename=None):
    """
    Plot prjojection images of given vertices (v) and DOT (d) coordinates along given projection axis [\"x\","\y\",\"z\"].
    """
    a = [0, 1, 2]
    labels = ["x", "y", "z"]
    a.remove(proj_ax)
    labels.pop(proj_ax)
    fig = pypl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    dist0 = v[:, proj_ax]
    isort = dist0.argsort()
    lw = 0.
    f = 0.2
    ori = v[:, proj_ax].mean()
    off = (v[:, proj_ax].max() - v[:, proj_ax].min())
    colors = np.array(["blue", "purple"] + 18*["green"])
    for vi, ci in zip(v[isort], colors[isort]):
        ax.scatter(vi[a[0]], vi[a[1]], c=ci, s=f*(vi[proj_ax]-ori+off), linewidths=lw, alpha=0.5)
    ax.scatter(d[a[0]], d[a[1]], c="red",  s=f*(d[proj_ax]-ori+off), linewidths=lw, alpha=0.5)
    if ideal_out:
        ico_scaled = t.match_scaling(v, d)
        dist0 = ico_scaled[:, proj_ax]
        isort = dist0.argsort()
        for vi in ico_scaled[isort]:
            ax.scatter(vi[a[0]], vi[a[1]], c=(1, 1, 1, 0), s=f*(vi[proj_ax]-ori+off), linewidths=1.)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_xlabel(labels[0] + u" [Å]")
    ax.set_ylabel(labels[1] + u" [Å]")
    if filename is not None:
        fig.savefig(filename)
        pypl.close(fig)

def _get_params(v, d):
    h = d[2]
    b = np.sqrt(d[0]**2 + d[1]**2)
    alpha = np.arctan2(-d[0], d[1])
    r_u = _get_scaling_factor(v, d)
    return h, b, alpha, r_u

def plot_params(V, D, outdir=None):
    """
    Visualise parameters of many models (list of vertices (V) and list of DOTs (D)).
    """
    N = len(V)
    h = np.zeros(N)
    b = np.zeros(N)
    alpha = np.zeros(N)
    r_u = np.zeros(N)
    for i,(v,d) in enumerate(zip(V, D)):
        h[i], b[i], alpha[i], r_u[i] = _get_params(v, d)            
    d = np.zeros(shape=(N, 3))
    d[:, 0] = b*np.sin(-alpha)
    d[:, 1] = b*np.cos(alpha)
    d[:, 2] = h
    for proj_ax in [0, 1, 2]:
        a = [0, 1, 2]
        labels = ["x", "y", "z"]
        a.remove(proj_ax)
        labels.pop(proj_ax)
        fig = pypl.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        lw = 1.
        f = 0.2
        ori = 0
        off = r_u.mean()
        for di in d:
            ax.scatter(di[a[0]], di[a[1]], c="red",  s=f*(di[proj_ax]-ori+off), linewidths=lw, alpha=0.25)
        ico_scaled = t.ico*r_u.mean()
        dist0 = ico_scaled[:, proj_ax]
        isort = dist0.argsort()
        for vi in ico_scaled[isort]:
            ax.scatter(vi[a[0]], vi[a[1]], c=(1, 1, 1, 0), s=f*(vi[proj_ax]-ori+off), linewidths=1.)
        ax.set_xlim(-1500, 1500)
        ax.set_ylim(-1500, 1500)
        ax.set_xlabel(labels[0] + " [A]")
        ax.set_ylabel(labels[1] + " [A]")
        if outdir is not None:
            fig.savefig("%s/params_%s.png" % (outdir, ["x", "y", "z"][proj_ax]))
            pypl.close(fig)
    tags = ["h", "b", "alpha", "r_u"]
    labels = ["h [A]", "b [A]", "alpha [deg]","r_u [A]"]
    ranges = [None, None, None, (1160, 1170)]
    nbins = [10, 10, 10, 10, 100]
    for tag, label, data, r, n in zip(tags, labels, [h, b, abs(alpha/2./np.pi*360), r_u], ranges, nbins): 
        fig = pypl.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(data, bins=n, range=r)
        ax.set_xlabel(label)
        ax.set_ylabel("number")
        if tag == "r_u":
            #ax.set_xlim((0, 2000))
            ax.set_xlim(r)
            #ax.text(r_u.mean()+100, 20, "r_u = (%i +/- %i) A" % (int(r_u.mean().round()), int(r_u.std().round())), ha="left")
        if outdir is not None:
            fig.savefig("%s/%s.png" % (outdir, tag))
            pypl.close(fig)
        #alpha2 = abs(abs(alpha) - 2/10.*np.pi)
    fig = pypl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(b, h)
    ax.set_xlabel("b [A]")
    ax.set_ylabel("h [A]")
    if outdir is not None:
        fig.savefig("%s/b_h.png" % (outdir))
        pypl.close(fig)
