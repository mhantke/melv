#!/usr/bin/python
# -*- coding: latin-1 -*-
import xml.etree.ElementTree as ET
import condor
import condor.utils.linalg as linalg
import condor.utils.rotation as rotation
import scipy.optimize
import numpy as np

def load_points(filename):
    """
    Read manually picked points from cmm file. Returns coordinates of vertices and the DOT.
    """
    tree = ET.parse(filename)
    v = []
    d = None
    for element in tree.getiterator():
        if "g" in element.keys():
            g = int(element.get("g"))
            vector = np.array([float(element.get("x")), float(element.get("y")), float(element.get("z"))], dtype="float")
            if g == 1:
                v.append(vector)
            else:
                d = vector
    v = np.array(v, dtype="float")
    return v, d

def rotate(v, euler1, euler2, euler3):
    """
    Extrinsic rotations z-x-z of single vector or a list of vectors v.
    """
    R = rotation.euler_to_rotmx([euler1, euler2, euler3], "zxz")
    if v.ndim == 1:
        v_rot = R.dot(v)
    elif v.ndim == 2:
        v_rot = np.asarray([R.dot(vi) for vi in v])
    else:
        print "ERROR: rotate accepts only inputs of max. 2 dimensions."
    return v_rot

def dist(v0, v1):
    """
    Calculate minimum pair distances of two lists of vectors.
    """
    N = v0.shape[0]
    k = range(N)
    dist = []
    for i in range(N):
        d = np.sqrt(((v0[i, :] * np.ones(shape=(len(k), 3)) - v1[k, :])**2).sum(axis=1))
        amin = d.argmin()
        k.pop(amin)
        dist.append(d[amin])
    return np.array(dist, dtype="float")

def rough_align(v, d):
    """
    Roughly align vertices and dot positions.
    """
    # Sort vertices by distance from dot (smallest distance first)
    dist = np.sqrt(((v - d * np.ones(shape=v.shape))**2).sum(axis=1))
    v = v[dist.argsort(), :]
    dist = dist[dist.argsort()]
    # 5-fold axis unit vector
    c = v.mean(axis=0) - v[0, :]
    c = c / linalg.length(c)
    # Put origin in v[0]
    d = d - v[0, :]
    v = v - (v[0, :] * np.ones(shape=v.shape))
    # rotation
    # x
    tx = np.arctan2(c[1], c[2])
    for i, vi in enumerate(v):
        v[i, :] = rotation.rot_x(vi, tx)[:]
    c = rotation.rot_x(c, tx)
    d = rotation.rot_x(d, tx)
    # y
    ty = -np.arctan2(c[0], c[2])
    for i, vi in enumerate(v):
        v[i, :] = rotation.rot_y(vi, ty)[:]
    c = rotation.rot_y(c, ty)
    d = rotation.rot_y(d, ty)
    # z
    k = v[1]-(v[1]*c)*c
    tz = np.arctan2(k[0], k[1])
    for i, vi in enumerate(v):
        v[i, :] = rotation.rot_z(vi, tz)[:]
    c = rotation.rot_z(c, tz)
    d = rotation.rot_z(d, tz)
    # Sanity check
    assert not ( (abs(c[0] - 0) > 1E-15) or (abs(c[1] - 0) > 1E-15) or (abs(c[2] - 1) > 1E-15) )
    # Put origin in middle
    orig = v.mean(axis=0)
    v -= orig
    d -= orig
    c -= orig
    return v, d, c

def mean_to_origin(v, d):
    """
    Move mean coordinate (center of mass) to origin.
    """
    O = v.mean(axis=0)
    v = v - O
    d = d - O
    return v, d

# Generate ideal icosahedron model
ico = condor.utils.bodies.get_icosahedron_vertices()
# Normalise to circumference ratidus r_u = 1
ico /= np.sqrt(ico[:, 0]**2+ico[:, 1]**2+ico[:, 2]**2).mean()
# Align by rotation around x axis
_phi = (1+np.sqrt(5))/2.0
ico = np.asarray([rotation.rot_x(icoi, np.arctan(1/_phi)) for icoi in ico])

def match_rotation(v, d):
    """
    Refine alignment by least-square fit.
    """
    v, d = mean_to_origin(v, d)
    err = lambda p: dist(rotate(v, p[0], p[1], p[2]), ico)
    [euler1, euler2, euler3] = scipy.optimize.leastsq(err,  [0., 0., 0.])[0]
    v = rotate(v, euler1, euler2, euler3)
    d = rotate(d, euler1, euler2, euler3)
    return v, d

def _get_scaling_factor(v, d):
    """
    Determine overall spatial scaling for best fit by least-squares approach.
    """
    v, d = mean_to_origin(v, d)
    err = lambda p: dist(v, p*ico)
    c = scipy.optimize.leastsq(err, 1/np.sqrt((v**2).sum(1)).mean())[0]
    return c
    
def match_scaling(v, d):
    """
    Match overall spatial scaling constant.
    """
    c = _get_scaling_factor(v, d)
    return ico*c
        

def hist3d(points, N, v_min=None, v_max=None, norm_nm3=False):
    if v_min is None or v_max is None:
        v_max = max([abs(points.min()),  abs(points.max())])
        v_min = -v_max
    d = (v_max-v_min)/float(N-1)
    Z, Y, X = np.meshgrid(np.linspace(v_min, v_max, N), 
                          np.linspace(v_min, v_max, N), 
                          np.linspace(v_min, v_max, N), 
                          indexing="ij")
    H = np.zeros((N, N, N))
    for point in points:
        xi = int(np.round((point[0]-v_min) / d))
        yi = int(np.round((point[1]-v_min) / d))
        zi = int(np.round((point[2]-v_min) / d))
        #print xi, yi, zi
        H[zi, yi, xi] += 1
    if norm_nm3:
        Lnm = float(v_max-v_min)/10.
        dV = (Lnm/N)**3
        H = H/len(points)/dV
    return H

