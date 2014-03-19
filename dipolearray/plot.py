"""
Plotting functions for dipolearray (requires matplotlib >= 1.1.0)
"""

from __future__ import division, print_function
from numpy import pi, ptp, min, max, array
from numpy import invert, isnan
from . import dipolearray as da

def PlotLattice(axis, lc, N1, N2, **kwargs):
    """
    Plots given Bravais lattice
    
    ---args--
    axis : matplotlib subplot instance
    lc : 2D bravais lattice  : (length-1, angle-1, length-2, angle-2)
        e.g a square lattice can be [100*nm, 0*Degrees, 100*nm, 90*Degrees]
    N1, N2 : number of dipoles in first and second lattice axis

    ---kwargs---
    verbose : verbosity control
    dipolelength : Length of dipole
    dipoleangle : Angle with respect to cartesian x axis (phi)
    title : Title of plot
    limfactor : padding about dipole array to show
    divideby : unit chosen - defaults to nm
    """

    verbose = kwargs.pop('verbose', 0)
    DipoleLength = kwargs.pop('dipolelength', 50)
    DipoleAngle = 90 - kwargs.pop('dipoleangle', 0)
    Title = kwargs.pop('title', "")
    limfactor = kwargs.pop('limfactor', 1.1)
    divideby = kwargs.pop('divideby', 1e-9)

    if verbose > 0:
        print("Number of Dipoles Generated", ptp(N1) * ptp(N2))

    X, Y, Z = da.getpositions(N1, N2, lc)

    X /= divideby
    Y /= divideby

    xlim = (min(X) - limfactor * min(X), max(X) + limfactor * min(X))
    ylim = (min(Y) - limfactor * min(Y), max(Y) + limfactor * min(Y))
    
    ##If we pass a 3D object we will plot correctly implicitly
    try:
        axis.scatter3D(X, Y, Z, marker=(2, 0, DipoleAngle),
                         s=DipoleLength, **kwargs)
        axis.set_zlabel("Z Position (nm)")
    except AttributeError:
        axis.scatter(X, Y, marker=(2, 0, DipoleAngle), 
                        s=DipoleLength, **kwargs)

    axis.set_xlabel("X Position (nm)")
    axis.set_ylabel("Y Position (nm)")
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_title(Title)
    axis.grid(True)
    return axis


def tocube(axis, cubeedge=1, size=20):
    """
    Sets 3D axis to cube
    """
    axis.set_xlabel("x", size=size)
    axis.set_ylabel("y", size=size)
    axis.set_zlabel("z", size=size)
    axis.set_xlim(-cubeedge, cubeedge)
    axis.set_ylim(-cubeedge, cubeedge)
    axis.set_zlim(-cubeedge, cubeedge)


def fetchmaxnum(x, y, z):
    """
    Returns maximum dimension allowing us to scale sensibly
    """
    anumx = max(x[invert(isnan(x))])
    anumy = max(y[invert(isnan(y))])
    anumz = max(z[invert(isnan(z))])

    return max([anumx, anumy, anumz])


def Farfield3DImage(n0, k, N1, N2, lc, p, axis, **kwargs):
    """
    Generate 3D matplotlib (>= 1.30 required if you wish save the image as svg)
    image of farfield pattern. 3 faces are plotted with the respective direction
    cosines

    ---Args---
    n0 - incident direction
    k - wavenumber
    N1, N2 : number of scatterers in x, y
    p : dipole moment
    figure : MatPlotlib figure
    axis : MatPlotlib subplot instance

    --Kwargs--
    steps : number of discrete steps in the plot to err...plot
    verbose : verbosity control
    """
    
    steps = kwargs.pop('steps', 200)
    verbose = kwargs.pop('verbose', 0)
    dist = kwargs.pop('dist', 'normal')
    
    theta, phi = da.AnglesofHemisphere('all', steps)
    alldirections = da.DirectionVector(theta, phi, verbose=verbose)    
    F = da.OutgoingDirections(alldirections, n0, N1, N2, lc, k, verbose=verbose)
    
    if dist == 'analytical':
        dsdo = da.DipoleDistribution(alldirections, p=p, k=k, const=False)
    else:
        dsdo = da.DifferentialCrossSection(F, n0, alldirections, p=p,
                                        k=k, const=False, verbose=verbose)

    adir = da.DirectionVector(theta, phi, dsdo)
    maxnum = max(adir)
    x = adir[..., 0] / maxnum
    y = adir[..., 1] / maxnum
    z = adir[..., 2] / maxnum

    axis.plot_surface(x, y, z, alpha=0.2)

    tocube(axis)

    return axis

def Farfield3DDirectionCosines(n0, k, N1, N2, lc, p, axis, **kwargs):
    """
    Generate 3D image of direction cosines of farfield pattern due to
    dipole array.

    ---Args---
    n0 - incident direction
    k - wavenumber
    N1, N2 : number of scatterers in x, y
    p : dipole moment
    axis : MatPlotlib subplot instance

    --Kwargs--
    verbose : verbosity control
    """
    steps = kwargs.pop('steps', 200)
    verbose = kwargs.pop('verbose', 0)
    N = kwargs.pop("N",50)
    dist = kwargs.pop('dist', 'normal')
    
    for adir in ['x', 'y', 'z']:

        theta, phi = da.AnglesofHemisphere(adir, steps)
        alldirections = da.DirectionVector(theta, phi)

        F = da.OutgoingDirections(alldirections, n0, N1, N2, lc, k)
                
        if dist == 'analytical':
            dsdo = da.DipoleDistribution(alldirections, p=p, k=k, const=False)
        else:
            dsdo = da.DifferentialCrossSection(F, n0, alldirections, 
                                              p=p, k=k, const=False)
        dsdo /= max(dsdo)

        ux, uy = da.GetDirectionCosine(adir, steps)

        if adir == 'x':
            ctf = axis.contourf(dsdo, ux, uy, N, 
                zdir=adir, offset=-1, **kwargs)
        elif adir == 'y':
            ctf = axis.contourf(ux, dsdo, uy, N,
                zdir=adir, offset=1, **kwargs)
        elif adir == 'z':
            ctf = axis.contourf(ux, uy, dsdo, N,
                zdir=adir, offset=-1, **kwargs)

    cb = axis.figure.colorbar(ctf, ax=axis, use_gridspec=True, shrink=0.5)
    cb.set_label("Scaled Absolute \nElectric Field Squared", size=15)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])

    return axis
