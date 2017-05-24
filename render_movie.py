#!/usr/bin/env python
import os, h5py
import numpy as np

import condor.utils.rotation as rot
import utils.mbrender as mbr

import scipy.ndimage

import mayavi
from mayavi import mlab

from scipy.interpolate import RegularGridInterpolator

import matplotlib

# Number of frames distribution
#N_360 = 3
#N_cut = N_360
#N_angles = N_360
N_360 = 200
N_cut = N_360
N_angles = N_360

# Initialise output directory
outdir = "./rendered_movie"
# Initialise output directory
if os.path.exists(outdir):
    files = [f for f in os.listdir(outdir) if f.endswith(".png")]
    for f in files:
        os.system("rm %s/%s" % (outdir, f))
else:
    os.mkdir(outdir)


cache_filename = "./data/render_movie_cache.h5"
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
    M = mbr.load_melv_densities(blur=2.0)
    z_angle = 2*np.pi/10.+np.pi/8.-0.015*2*np.pi
    rotation = rot.Rotation(formalism="quaternion", values=rot.quat(z_angle,1,0,0))
    cache["M_rot"] = mbr.rotate_map(M, rotation)
M_rot = cache["M_rot"]
N = M_rot.shape[0]

# Cuts
cuths_cap = np.linspace(1.0, 0.55, N_cut)
cuths_mem = np.linspace(1.0, 0.6, N_cut)

size = 1000
mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(size, size))   

if empty_cache:
    print "Isolating substructures membrane and capsid."
    tmp = mbr.isolate_membrane_and_capsid(M_rot)
    cache["M_cap"], M_cap1, M_cap2, M_mem, M_mem1, M_mem2, cache["R1_sm"], cache["R2_sm"] = tmp
    (x, y, z), (X, Y, Z), cache["R"] = mbr.generate_grid(N)
M_cap = cache["M_cap"]
R = cache["R"]
R1_sm = cache["R1_sm"]
R2_sm = cache["R2_sm"]

k = 0

if empty_cache:
    cache["membrane_1"] = []
    cache["capsid_1"] = []
for i, (cuth_cap, cuth_mem) in enumerate(zip(cuths_cap, cuths_mem)):
    print "Opening capsid (%i/%i)" % (i, len(cuths_cap))
    if empty_cache:
        cache["membrane_1"].append(mbr.make_membrane_map(M_rot, R, cuth_mem, R1_sm, R2_sm, M_cap))
        cache["capsid_1"].append(mbr.make_capsid_map(M_rot, R, cuth_cap, R1_sm, R2_sm))
    membrane = cache["membrane_1"][i]
    capsid = cache["capsid_1"][i]
    mlab.clf()
    mbr.draw_scene(capsid, membrane)
    (azimuth, elevation, distance, focalpoint) = mlab.view()
    mlab.view(azimuth=azimuth+360./N_cut,
              elevation=elevation,
              distance=distance,
              focalpoint=[0., 0., 0.],
              roll=None,
              reset_roll=False,
              figure=None)
    mlab.savefig("%s/melv_%04i.png" % (outdir,k))
    k +=1

k0 = k
(azimuth0, elevation0, distance0, focalpoint0) = mlab.view()
mlab.view(azimuth=azimuth0,
          elevation=elevation0,
          distance=distance0,
          focalpoint=focalpoint0,
          roll=None,
          reset_roll=False,
          figure=None)
(azimuth, elevation, distance, focalpoint) = (azimuth0, elevation0, distance0, focalpoint0)

k = k0
for i in range(N_angles):
    print "Rotate without LDB (%i/%i)" % (i, N_angles)
    mlab.view(azimuth=azimuth+360./N_angles,
              elevation=elevation,
              distance=distance,
              focalpoint=[0., 0., 0.],
              roll=None,
              reset_roll=False,
              figure=None)
    (azimuth, elevation, distance, focalpoint) = mlab.view()
    mlab.savefig("%s/_1melv_%04i.png" % (outdir,k))
    k +=1

k = k0
mlab.clf()
mbr.draw_scene(capsid, membrane, ldb_hist=H_D_sm)
for i in range(N_angles):
    print "Rotate with LDB (%i/%i)" % (i, N_angles)
    mlab.view(azimuth=azimuth+360./N_angles,
              elevation=elevation,
              distance=distance,
              focalpoint=focalpoint0,
              roll=None,
              reset_roll=False,
              figure=None)
    (azimuth, elevation, distance, focalpoint) = mlab.view()
    mlab.savefig("%s/_2melv_%04i.png" % (outdir,k))
    k +=1

k = k0
f1s = np.linspace(1, 0, N_angles)
f2s = 1-f1s
for i,f1,f2 in zip(range(N_angles), f1s, f2s):
    print "Overlaying with transparency (%i/%i)" % (i, N_angles)
    cmd = "convert %s/_1melv_%04i.png %s/_2melv_%04i.png -poly '%f,1 %f,1' %s/melv_%04i.png" % (outdir,k,outdir,k,f1,f2,outdir,k)
    os.system(cmd)
    k +=1
    
# Continue rotation with LDB density iso surfaces
for i in range(2*N_angles):
    print "Rotating more (%i/%i)" % (i, 2*N_angles)
    mlab.view(azimuth=azimuth+360./N_angles,
              elevation=elevation,
              distance=distance,
              focalpoint=[0., 0., 0.],
              roll=None,
              reset_roll=True,
              figure=None)
    (azimuth, elevation, distance, focalpoint) = mlab.view()
    mlab.savefig("%s/melv_%04i.png" % (outdir,k))
    k +=1

# Close capsid again
if empty_cache:
    cache["membrane_2"] = []
    cache["capsid_2"] = []
for i, (cuth_cap, cuth_mem) in enumerate(zip(cuths_cap[::-1], cuths_mem[::-1])):
    print "Closing capsid (%i/%i)" % (i, len(cuths_cap))
    if empty_cache:
        cache["membrane_2"].append(mbr.make_membrane_map(M_rot, R, cuth_mem, R1_sm, R2_sm, M_cap))
        cache["capsid_2"].append(mbr.make_capsid_map(M_rot, R, cuth_cap, R1_sm, R2_sm))
    membrane = cache["membrane_2"][i]
    capsid = cache["capsid_2"][i]
    mlab.clf()
    mbr.draw_scene(capsid, membrane, ldb_hist=H_D_sm)
    mlab.view(azimuth=azimuth+360./N_cut,
              elevation=elevation,
              distance=distance,
              focalpoint=[0., 0., 0.],
              roll=None,
              reset_roll=False,
              figure=None)
    (azimuth, elevation, distance, focalpoint) = mlab.view()
    mlab.savefig("%s/melv_%04i.png" % (outdir,k))
    k +=1

if empty_cache:
    print "Writing data to cache (%s)." % cache_filename
    with h5py.File(cache_filename, "w") as f:
        for k, v in cache.items():
            f[k] = v
    
mlab.close()

outfile = "%s/melv_ldb.mp4" % outdir
print "Converting PNGs to MP4. (%s)" % outfile
os.system("ffmpeg -n -i \"%s/melv_%%04d.png\" %s" % (outdir,outfile))

print "Done."
