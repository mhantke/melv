#!/usr/bin/env python
import os, h5py
import numpy as np

import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator

import mayavi
from mayavi import mlab

import condor
import condor.utils.rotation as rot

import utils.mbrender as mbr

# Initialise output directory
outdir = "./rendered_image"
# Initialise output directory
if os.path.exists(outdir):
    files = [f for f in os.listdir(outdir) if f.endswith(".png")]
    for f in files:
        os.system("rm %s/%s" % (outdir, f))
else:
    os.mkdir(outdir)

cache_filename = "./data/render_image_cache.h5"
cache = {}
if os.path.exists(cache_filename):
    print "Reading data from cache (%s)." % cache_filename
    with h5py.File(cache_filename, "r") as f:
        for k, v in f.items():
            cache[k] = np.asarray(v)
    empty_cache = False
else:
    print "No cache available."
    empty_cache = True

if empty_cache:
    print "Loading LDB histogram."
    cache["H_D_sm"] = mbr.load_ldb_hist()
H_D_sm = cache["H_D_sm"]

if empty_cache:
    print "Loading MelV densities."
    M = mbr.load_melv_densities()
    z_angle = 2*np.pi/10.+np.pi/8.-0.015*2*np.pi
    rotation = rot.Rotation(formalism="quaternion", values=rot.quat(z_angle,1,0,0))
    cache["M_rot"] = mbr.rotate_map(M, rotation)
M_rot = cache["M_rot"]

if empty_cache:
    print "Isolating substructures membrane and capsid."
    tmp = mbr.isolate_membrane_and_capsid(M_rot)
    M_cap, M_cap1, cache["M_cap2"], M_mem, M_mem1, cache["M_mem2"], R1_sm, R2_sm = tmp
M_cap2 = cache["M_cap2"]
M_mem2 = cache["M_mem2"]

if empty_cache:
    print "Writing data to cache (%s)." % cache_filename
    with h5py.File(cache_filename, "w") as f:
        for k, v in cache.items():
            f[k] = v

print "Rendering data."

size = 1000
mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(size, size))

mbr.draw_scene(capsid=M_cap2, membrane=M_mem2, ldb_hist=H_D_sm)

mlab.gcf().scene.z_minus_view()
mlab.gcf().scene.camera.view_angle = 10
mlab.move(forward=-3)

filename = "%s/melv.png" % outdir
if filename is not None:
    mlab.savefig(filename)
    os.system("convert %s -transparent white %s" % (filename,filename))
    mlab.close()
else:
    mlab.show()

print "Done."
