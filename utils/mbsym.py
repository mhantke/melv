def apply_ico(points):
    # Assuming that Z-axis conincides with 5-fold axis and X-axis with the 2-fold axis (X = first index, Y = second index, Z = third index)
    R = np.loadtxt("icorot.txt")[:, :3]
    R = R.reshape((R.shape[0]/3, 3, 3))
    r_m = 1.
    a = r_m / (1/4.*(1.+np.sqrt(5)))
    r1 = condor.utils.rotation.Rotation(formalism="quaternion", values=condor.utils.rotation.quat(-np.arctan(a/2.), 1., 0., 0.))
    r2 = condor.utils.rotation.Rotation(formalism="quaternion", values=condor.utils.rotation.quat(np.arctan(a/2.), 1., 0., 0.))
    points2 = []
    for point in points:
        for Ri in R:
            points2.append(r2.rotate_vector(Ri.dot(r1.rotate_vector(point))))
    points2 = np.array(points2)
    return points2

def test_apply_ico():
    p = np.array([[0, 0, 1]])
    p_ico = apply_ico(p)
    from mayavi import mlab
    mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(350, 350))
    mlab.points3d(p_ico[:, 0],  p_ico[:, 1],  p_ico[:, 2])
    mlab.points3d(p[:, 0],  p[:, 1],  p[:, 2],  color=(1, 0, 0))
    mlab.show()

def apply_5fold(points):
    points2 = []
    for point in points:
        for i in range(5):
            r = condor.utils.rotation.Rotation(formalism="quaternion", values=condor.utils.rotation.quat(i*2*np.pi/5., 0., 0., 1.))
            points2.append(r.rotate_vector(point))
    points2 = np.array(points2)
    return points2

