"""
Plotting functions for dipolearray (requires matplotlib >= 1.1.0)
"""

from __future__ import division, print_function

from numpy import ptp, min, max, pi
from numpy import invert, isnan, meshgrid, linspace

from . import dipolearray as da

def structure_factor(axisone, axistwo, lc, N1, N2, **kwargs):
    """
    Plots the structure factor
    """

    nm = 1e-9
    Degrees = pi/180

    steptheta = kwargs.pop('steptheta', 400)
    stepphi = kwargs.pop('stepphi', 200)
    verbose = kwargs.pop('verbose', 0)
    dist = kwargs.pop('dist', 'analytical')
    k = kwargs.pop('k', (2*pi)/(420*nm)) #wavenumber of incident wave

    theta, phi = meshgrid(linspace(0,180*Degrees, steptheta), linspace(0,360*Degrees, stepphi))

    n0 = da.radial_direction_vector(theta, phi)
    n1, theta, phi = da.spherical_unit_vectors(0*Degrees, 0*Degrees)

    F = da.structure_factor(n0, n1, N1, N2, lc, k, dist=dist)

    theta, phi = linspace(0,180*Degrees, steptheta), linspace(0,360*Degrees, stepphi)


    ctf = axisone.contourf(theta/Degrees, phi/Degrees, F)
    cb = axisone.figure.colorbar(ctf, ax=axisone)
    cb.set_label("Structure Factor")
    axisone.set_xlabel("phi (Degrees)")
    axisone.set_ylabel("theta (Degrees)")

    axistwo.plot(theta/Degrees, F[stepphi/2,...])
    axistwo.set_xlabel("theta (Degrees)")
    axistwo.set_ylabel("Structure Factor")
    axistwo.set_xlim(0,180)

    return axisone, axistwo

def lattice(axis, lc, N1, N2, **kwargs):
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

    X, Y, Z = da.periodic_lattice_positions(N1, N2, lc)

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

def farfield_3D(n0, k, N1, N2, lc, p, axis, **kwargs):
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

    steptheta = kwargs.pop('steptheta', 400)
    stepphi = kwargs.pop('stepphi', 200)
    verbose = kwargs.pop('verbose', 0)
    dist = kwargs.pop('dist', 'normal')
    const = kwargs.pop('const', False)

    theta, phi, dsdo = da.differential_cross_section_volume(n0, p, k, N1, N2, lc, 'all', steptheta=steptheta,
                                                          stepphi=stepphi, const=const, dist=dist, verbose=verbose)

    dsdo[isnan(dsdo)] = 0

    adir = da.radial_direction_vector(theta, phi, dsdo)
    maxnum = max(adir)
    x = adir[..., 0] / maxnum
    y = adir[..., 1] / maxnum
    z = adir[..., 2] / maxnum

    axis.plot_surface(x, y, z, alpha=0.2)

    _to_cube(axis)

    return axis


def farfield_directioncosines3D(n0, k, N1, N2, lc, p, axis, **kwargs):
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
    steps = kwargs.pop('steps', 100)  #square array must be square!
    verbose = kwargs.pop('verbose', 0)
    N = kwargs.pop("N", 50)
    dist = kwargs.pop('dist', 'normal')
    const = kwargs.pop('const', False)

    for adir in ['x', 'y', 'z']:
        theta, phi, dsdo = da.differential_cross_section_volume(n0, p, k, N1, N2, lc, adir, steptheta=steps,
                                                              stepphi=steps, const=const, dist=dist, verbose=verbose)

        dsdo[isnan(dsdo)] = 0 #sets NaN to zero
        dsdo /= max(dsdo)

        ux, uy = da.direction_cosine(adir, steptheta=steps, stepphi=steps)

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


def _to_cube(axis, cubeedge=1, size=20):
    """
    Sets 3D axis to cube
    """
    axis.set_xlabel("x", size=size)
    axis.set_ylabel("y", size=size)
    axis.set_zlabel("z", size=size)
    axis.set_xlim(-cubeedge, cubeedge)
    axis.set_ylim(-cubeedge, cubeedge)
    axis.set_zlim(-cubeedge, cubeedge)


def _fetch_max_dimension(x, y, z):
    """
    Returns maximum dimension allowing us to scale sensibly
    """
    anumx = max(x[invert(isnan(x))])
    anumy = max(y[invert(isnan(y))])
    anumz = max(z[invert(isnan(z))])

    return max([anumx, anumy, anumz])
