import h5py
import numpy as np

import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator

import matplotlib.cm as cm

import mayavi
from mayavi import mlab

import mbtools as mbt

def load_melv_densities(downsample=2, blur=2.5):
    """
    Load cryo EM reconstruction model.
    """
    # Read from file
    with h5py.File("./data/MelV_EM.h5","r") as f:
        M = np.asarray(f["data"], dtype="float64")
    # Downsample
    if downsample is not None:
        M = M[::downsample,::downsample,::downsample] 
    # Blur
    if blur is not None:
        M = scipy.ndimage.gaussian_filter(M, blur)
    return M

def load_melv_densities_2x2x2(blur=2.5):
    """
    Load cryo EM reconstruction model.
    """
    # Read from file
    with h5py.File("./data/MelV_EM_2x2x2.h5","r") as f:
        M = np.asarray(f["data"], dtype="float64")
    # Blur
    M = scipy.ndimage.gaussian_filter(M, blur)
    return M

def generate_grid(N):
    """
    Generate grid for map of edge length with N nodes.
    """
    x = np.arange(N) - N/2.
    y = np.arange(N) - N/2.
    z = np.arange(N) - N/2.
    X,Y,Z = np.meshgrid(x, y, z, indexing="ij")
    R =  np.sqrt(X**2+Y**2+Z**2)
    return (x, y, z), (X, Y, Z), R

def rotate_map(M, rotation):
    """
    Rotate map with given condor rotation.
    """
    N = M.shape[0]
    (x, y, z), (X, Y, Z), R = generate_grid(N)
    RGI = RegularGridInterpolator((x,y,z), M, bounds_error=False)
    points = np.dstack([X.ravel(),Y.ravel(),Z.ravel()])[0]
    points_rot = rotation.rotate_vectors(points)
    M_rot = RGI(points_rot).reshape((N,N,N))
    return M_rot

def load_ldb_hist(blur=1.5):
    # Reading coordinates
    with h5py.File("./results/params.h5","r") as f:
        D = np.array(f["D"])
    # Computing 3D histogram of LDB coordinates
    H_D = mbt.hist3d(D, 100,-1500,1500,norm_nm3=True)
    H_D_sm = scipy.ndimage.gaussian_filter(H_D, blur)
    return H_D_sm

def make_capsid_map(M, R, cuth, R1_sm, R2_sm, blur=1.5, r1=40, r2=100):
    M_cap = M.copy()
    M_cap[R>r2] = 0
    M_cap *= R1_sm
    M_cap *= R2_sm
    M_cap = scipy.ndimage.gaussian_filter(M_cap, blur)
    if cuth is not None:
        if cuth > 0:
            M_cap[:,:,int(cuth*M.shape[2]):] = 0
        else:
            M_cap[:,:,:int(cuth*M.shape[2])] = 0
    return M_cap

def make_membrane_map(M, R, cuth, R1_sm, R2_sm, M_cap, blur=0.8, r1=40, r2=100):
    M_mem = M.copy()
    M_mem[R>r2] = 0
    M_mem[R<r1] = 0
    M_mem *= M_cap < 10
    M_mem *= R2_sm
    M_mem = scipy.ndimage.gaussian_filter(M_mem, blur)
    if cuth is not None:
        if cuth > 0:
            M_mem[:,:,int(cuth*M.shape[2]):] = 0
        else:
            M_mem[:,:,:int(cuth*M.shape[2])] = 0
    return M_mem

def isolate_membrane_and_capsid(M, memloc=72., caploc=90., cutv=0.5, cuth_cap=0.5, cuth_mem=0.425, r1=40, r2=100, blur_cap=1.5, blur_mem=0.8):
    """
    Isolate substructures membrane and capsid from EM map.
    """
    N = M.shape[0]
    (x, y, z), (X, Y, Z), R = generate_grid(N)

    R1 = np.asarray(R > memloc, dtype='float')
    R1_sm = scipy.ndimage.gaussian_filter(R1, 1.)
    R1_sm /= R1_sm.max()
    R2 = np.asarray(R < caploc, dtype='float')
    R2_sm = scipy.ndimage.gaussian_filter(R2, 1.5)
    R2_sm /= R2_sm.max()

    M_cap = make_capsid_map(M, R, None, R1_sm, R2_sm)
    M_cap1 = make_capsid_map(M, R, cuth_cap, R1_sm, R2_sm)
    M_cap2 = make_capsid_map(M, R, cuth_cap-1, R1_sm, R2_sm)

    M_mem = make_membrane_map(M, R, None, R1_sm, R2_sm, M_cap)
    M_mem1 = make_membrane_map(M, R, cuth_mem, R1_sm, R2_sm, M_cap)
    M_mem2 = make_membrane_map(M, R, cuth_mem-1, R1_sm, R2_sm, M_cap)
    
    return M_cap, M_cap1, M_cap2, M_mem, M_mem1, M_mem2, R1_sm, R2_sm


def draw_scene(capsid, membrane, ldb_hist=None, clev_cap=40):    
    cmap = cm.jet
    llev_ldb = np.linspace(30E-7, 220E-7, 3)
    clev_ldb = [cmap(tmp) for tmp in np.linspace(0, 1, len(llev_ldb))]
    olev_ldb = [0.6]*len(llev_ldb)
                    
    if ldb_hist is not None:
        Nh = ldb_hist.shape[0]
        xh = np.linspace(-0.5,0.5,Nh)# * 1.1
        Xh, Yh, Zh = np.meshgrid(xh, xh, xh, indexing="ij")
        src1 = mlab.pipeline.scalar_field(Xh,Yh,Zh,ldb_hist)
        for l,c,o in zip(llev_ldb,clev_ldb,olev_ldb):
            mlab.pipeline.iso_surface(src1, opacity=o,color=(c[0],c[1],c[2]),contours=[l])
    
    if membrane is not None or capsid is not None:
        Nm = capsid.shape[0]
        xm = np.linspace(-0.5,0.5,Nm)
        Xm, Ym, Zm = np.meshgrid(xm, xm, xm, indexing="ij")

    if membrane is not None:
        src2 = mlab.pipeline.scalar_field(Xm, Ym, Zm, membrane)
        # Gray(0.8,0.8,0.8)
        iso2 = mlab.pipeline.iso_surface(src2, color=(0.75,0.75,0.8), opacity=0.5, contours=[0.6])
    
    if capsid is not None:
        src3 = mlab.pipeline.scalar_field(Xm, Ym, Zm, capsid)
        iso3 = mlab.pipeline.iso_surface(src3, color=(0.5,0.5,0.5), opacity=1., contours=[clev_cap])


def render_models(dot_hist, capsid=None, membrane=None, filename=None, delevation=0., dazimuth=0., distance=None, view_angle=None,
                  size=350, transparent=True):
    """
    Render image from given 3D models.
    """

    do_contour = True

    cmap = cm.jet
    llev_dot = np.linspace(30E-7, 220E-7, 3)
    clev_dot = [cmap(tmp) for tmp in np.linspace(0, 1, len(llev_dot))]
    olev_dot = [0.25]*len(llev_dot)
    
    if membrane is not None:
        Nm = capsid.shape[0]
        xm = np.linspace(-0.5,0.5,Nm)
        Xm, Ym, Zm = np.meshgrid(xm, xm, xm, indexing="ij")
    

    if do_contour:
        Nh = dot_hist.shape[0]
        xh = np.linspace(-0.5,0.5,Nh)# * 1.1
        Xh, Yh, Zh = np.meshgrid(xh, xh, xh, indexing="ij")

        for l,c,o in zip(llev_dot,clev_dot,olev_dot):
            mlab.contour3d(Xh, Yh, Zh, dot_hist,opacity=o,color=(c[0],c[1],c[2]),contours=[l])#0.35])
    if membrane is not None:
        mlab.contour3d(Xm, Ym, Zm, membrane, color=(0.8,0.8,0.8), opacity=1., contours=[2])#[40])
    if capsid is not None:
        mlab.contour3d(Xm, Ym, Zm, capsid_so, color=(0.5,0.5,0.5), opacity=1., contours=[40])
    if capsid_tr is not None:
        mlab.contour3d(Xm, Ym, Zm, capsid_tr, color=(0.5,0.5,0.5), opacity=0.05, contours=[40])
