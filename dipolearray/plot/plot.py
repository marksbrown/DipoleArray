'''
Plotting functions for _dipolearray (requires matplotlib >= 1.1.0)
'''

from __future__ import division, print_function
from numpy import pi, meshgrid, sin, cos, array, ptp, min, max, zeros, shape
import os

# Constants
nm = 1e-9
Degrees = pi / 180


def savefig(name, fig, verbose=0, **kwargs):
    '''
    Saves figure

    kwargs:
    imagesavelocation : imagesavelocation
    ext : extensions to save as and into
    '''

    imagesavelocation = kwargs.get('imagesavelocation',
                                   os.path.join(os.path.expanduser('~'), 'Desktop', 'images'))

    if not os.path.exists(imagesavelocation):
        if verbose > 1:
            print("Creating ImageSaveLocation directory")
        os.mkdir(imagesavelocation)

    Ext = kwargs.get('ext', ['png', 'pdf', 'svg'])

    for ext in Ext:
        savetopath = os.path.join(imagesavelocation, ext)
        if not os.path.exists(savetopath):
            if verbose > 1:
                print("Creating savetopath directory")
            os.mkdir(savetopath)

        saveto = os.path.join(savetopath, name + "." + ext)

        if verbose > 0:
            print("Saving image to", saveto)
        fig.savefig(saveto)


def getpositions(N1, N2, lc):
    '''
    Returns cartesian position of an array of points defined by the lattice
    _lc_ with min/max numbers defined by the tuples _N1_ and _N2_
    '''
    u1, u2 = meshgrid(range(*N1), range(*N2))  # grid of integers

    if len(lc) == 4:
        dx, tx, dy, ty = lc
        dvary = 0
        tvary = 0
    elif len(lc) == 6:
        dx, tx, dy, ty, dvary, tvary = lc

    xpos = lambda u, d, t: u * d * cos(t)
    ypos = lambda u, d, t: u * d * sin(t)

    X = xpos(u1, dx, tx) + xpos(u2, dy, ty) + xpos(u1 ** 2, dvary, tvary)
    Y = ypos(u1, dx, tx) + ypos(u2, dy, ty) + ypos(u2 ** 2, dvary, tvary)
    Z = zeros(shape(X))

    return X, Y, Z


def PlotLattice(axis, lc, N1, N2, verbose=0, **kwargs):
    '''
    Plots given lattice on subplot using matplotlib

    axis : matplotlib subplot instance

    lc : 2D bravais lattice  : (length-1,angle-1,length-2,angle-2)
        example of square lattice [100*nm,0*Degrees,100*nm,90*Degrees]

    N1, N2 : number of dipoles in first and second lattice axis

    ---

    kwargs

    dipolelength : Length of dipole
    dipoleangle : Angle with respect to cartesian x axis (phi)
    title : Title of plot
    xlim, ylim : x, y limits
    '''

    DipoleLength = kwargs.get('dipolelength', 50)
    DipoleAngle = 90 - kwargs.get('dipoleangle', 45)
    Title = kwargs.get('title', "")
    limfactor = kwargs.get('limfactor', 1.1)
    divideby = kwargs.get('divideby', nm)

    if verbose > 0:
        print("Number of Dipoles Generated", ptp(N1) * ptp(N2))

    X, Y, Z = getpositions(N1, N2, lc)

    X /= divideby
    Y /= divideby

    xlim = (min(X) - limfactor * min(X), max(X) + limfactor * min(X))
    ylim = (min(Y) - limfactor * min(Y), max(Y) + limfactor * min(Y))

    axis.grid()
    try:
        axis.scatter3D(X, Y, Z, marker=(2, 0, DipoleAngle), s=DipoleLength)
        axis.set_zlabel("Z Position (nm)")
    except AttributeError:
        axis.scatter(X, Y, marker=(2, 0, DipoleAngle), s=DipoleLength)

    axis.set_xlabel("X Position (nm)")
    axis.set_ylabel("Y Position (nm)")
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_title(Title)
    axis.grid(True)
    return axis


def tocube(axis, anum=1):
    axis.set_xlabel("x", size=20)
    axis.set_ylabel("y", size=20)
    axis.set_zlabel("z", size=20)
    axis.set_xlim(-anum, anum)
    axis.set_ylim(-anum, anum)
    axis.set_zlim(-anum, anum)


def fetchmaxnum(x, y, z):
    '''
    Returns maximum dimension allowing us to scale sensibly
    '''
    anumx = max(x[invert(isnan(x))])
    anumy = max(y[invert(isnan(y))])
    anumz = max(z[invert(isnan(z))])

    return max([anumx, anumy, anumz])


def ContourPlot3D(axis, x, y, z, filled=False, *args, **kwargs):
    '''
    Plots contour lines of data in 3D
    '''
    if filled == False:
        ctr = axis.contour3D(x, y, z, *args, **kwargs)
    elif filled == True:
        ctr = axis.contourf3D(x, y, z, *args, **kwargs)
    else:
        print("Non bool!")
        return None

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

    NegativeMaximumDeviation = 0
    MaximumDeviation = 0
    a, b = axis.get_xlim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b
    a, b = axis.get_zlim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b
    a, b = axis.get_ylim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b

    Deviation = (abs(NegativeMaximumDeviation), MaximumDeviation)[
        abs(NegativeMaximumDeviation) < MaximumDeviation]
    Deviation *= 1.1
    axis.set_xlim(-Deviation, Deviation)
    axis.set_ylim(-Deviation, Deviation)
    axis.set_zlim(-Deviation, Deviation)
    return ctr
