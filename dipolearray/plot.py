"""
Plotting functions for dipolearray (requires matplotlib >= 1.1.0)
"""

from __future__ import division, print_function

from numpy import ptp, min, max, pi, arccos, arctan2
from numpy import invert, isnan, linspace

from . import dipolearray as da


def structure_factor(axis, light, metasurface, adir='z', **kwargs):
    """
    Plots the structure factor for the metasurface given
    axis : matplotlib subplot instance
    light : light class object instance
    metasurface : metasurface object instance
    adir : direction of interest
    """

    Degrees = pi/180

    N = kwargs.pop('N', 100)
    verbose = kwargs.pop('verbose', 0)
    dist = kwargs.pop('dist', 'analytical')

    F = metasurface.structure_factor(adir, light, dist=dist, verbose=verbose)
    theta, phi = da.angles_of_hemisphere(adir, light.steptheta, light.stepphi)

    ctf = axis.contourf(phi, theta/Degrees,  F.T, N, label=dist, **kwargs)
    cb = axis.figure.colorbar(ctf, ax=axis, use_gridspec=True, format="%2.1e")
    cb.set_label("Structure Factor")

    return F

def lattice(axis, metasurface, **kwargs):
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

    X, Y, Z = metasurface.periodic_lattice_positions()

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

    return X, Y, Z

def farfield_polar_2D(axes, light, farfield_pattern, **kwargs):
    """
    Plots the far field pattern onto a polar contour plot

    axes : 3 matplotlib subplot instances in list with polar=True active
    light : light object for system
    farfield_pattern : function of form func(adir) to return far field pattern in _adir_ direction
    """

    Degrees = pi/180
    N = kwargs.pop("N", 100)

    for j, adir in enumerate(['x', 'y', 'z']):
        dsdo = farfield_pattern(adir)

        dsdo[isnan(dsdo)] = 0
        dsdo /= max(dsdo)

        outgoing_directions = light.outgoing_vectors(adir)


        if adir == 'x':
            theta = arccos(outgoing_directions[...,0]).T
            phi = arctan2(outgoing_directions[...,2], outgoing_directions[...,1]).T
        if adir == 'y':
            theta = arccos(outgoing_directions[...,1]).T
            phi = arctan2(outgoing_directions[...,2], outgoing_directions[...,0]).T
        if adir == 'z':
            theta = arccos(outgoing_directions[...,2]).T
            phi = arctan2(outgoing_directions[...,1], outgoing_directions[...,0]).T

        plotkwargs = {'N': N, 'zdir' : adir, 'levels' : linspace(0, 1, N)}

        plotkwargs = dict(plotkwargs.items()+kwargs.items())

        axes[j].set_title(adir)

        if adir == 'x':
            ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plotkwargs)
        elif adir == 'y':
            ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plotkwargs)
        elif adir == 'z':
            ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plotkwargs)

        cb = axes[j].figure.colorbar(ctf, ax=axes[j], use_gridspec=True, shrink=0.5)
        cb.set_label("Scaled Absolute \nElectric Field Squared", size=15)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1])


def farfield_direction_cosines_z(ax, light, farfield_pattern, **kwargs):
    """
    Generate single hemisphere in z only
    """

    N = kwargs.pop("N", 100)
    dsdo = farfield_pattern('z')

    dsdo[isnan(dsdo)] = 0
    dsdo /= max(dsdo)

    ux, uy = light.direction_cosine('z')

    plotkwargs = {'N': N, 'zdir' : 'z', 'levels' : linspace(0, 1, N)}
    plotkwargs = dict(plotkwargs.items()+kwargs.items())

    ctf = ax.contourf(ux, uy, dsdo, **plotkwargs)
    cb = ax.figure.colorbar(ctf, ax=ax, use_gridspec=True, shrink=0.5)
    cb.set_label("Scaled Absolute \nElectric Field Squared", size=15)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])


def farfield_direction_cosines_2D(axes, light, farfield_pattern, **kwargs):
    """
    Generate 3 polar contour plots of farfield pattern using
    direction cosines due to dipole array.

    ---Args---
    axes : 3 matplotlib subplot instances in a list
    light : light class object
    farfield_pattern : function to generate farfield pattern

    --Kwargs--
    verbose : verbosity control
    """
    N = kwargs.pop("N", 100)

    for j, adir in enumerate(['x', 'y', 'z']):
        dsdo = farfield_pattern(adir)

        dsdo[isnan(dsdo)] = 0
        dsdo /= max(dsdo)

        ux, uy = light.direction_cosine(adir)

        plotkwargs = {'N': N, 'zdir' : adir, 'levels' : linspace(0, 1, N)}

        plotkwargs = dict(plotkwargs.items()+kwargs.items())

        axes[j].set_title(adir)
        ctf = axes[j].contourf(ux, uy, dsdo,  **plotkwargs)

        cb = axes[j].figure.colorbar(ctf, ax=axes[j], use_gridspec=True, shrink=0.5)
        cb.set_label("Scaled Absolute \nElectric Field Squared", size=15)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1])


def farfield_direction_cosines_3D(axis, light, farfield_pattern, **kwargs):
    """
    Generate 3 polar contour plots of farfield pattern using
    direction cosines due to dipole array.

    ---Args---
    axes : 3 matplotlib subplot instances in a list
    light : light class object
    farfield_pattern : function to generate farfield pattern

    --Kwargs--
    verbose : verbosity control
    """
    N = kwargs.pop("N", 100)

    for adir in ['x', 'y', 'z']:
        dsdo = farfield_pattern(adir)

        dsdo[isnan(dsdo)] = 0
        dsdo /= max(dsdo)

        ux, uy = light.direction_cosine(adir)

        plotkwargs = {'N': N, 'zdir' : adir, 'offset' : -1*(-1)**(adir=='y'), 'levels' : linspace(0, 1, N)}

        plotkwargs = dict(plotkwargs.items()+kwargs.items())

        if adir == 'x':
            axis.contourf(dsdo, ux, uy, **plotkwargs)
        elif adir == 'y':
            axis.contourf(ux, dsdo, uy,  **plotkwargs)
        elif adir == 'z':
            ctf = axis.contourf(ux, uy, dsdo,  **plotkwargs)
            cb = axis.figure.colorbar(ctf, ax=axis, use_gridspec=True, shrink=0.5)
            cb.set_label("Scaled Absolute \nElectric Field Squared", size=15)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1])

def farfield_surface_3D(axis, light, farfield_pattern, **kwargs):
    """
    Farfield pattern plotted in 3D
    """

    dsdo = farfield_pattern('all')
    dsdo[isnan(dsdo)] = 0

    direction_vectors = light.outgoing_vectors('all', amplitudes=dsdo)

    maxnum = max(direction_vectors)
    x = direction_vectors[..., 0] / maxnum
    y = direction_vectors[..., 1] / maxnum
    z = direction_vectors[..., 2] / maxnum

    axis.plot_surface(x, y, z, **kwargs)

    _to_cube(axis)


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
