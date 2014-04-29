"""
Plotting functions for dipolearray (requires matplotlib >= 1.1.0)
"""

from __future__ import division, print_function

from numpy import ptp, min, max, pi, arccos, arctan2
from numpy import invert, isnan, linspace, log10
from collections import Iterable

from . import dipolearray as da


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
        print("Number of Dipoles Generated", ptp(metasurface.x_scatterers) * ptp(metasurface.y_scatterers))

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



def plot_2d(plot_type, axes, light, farfield_pattern, directions='xyz', **kwargs):
    """
    Plots the _far field pattern_ (or any function of form func(adir)) onto plots for the directions provided

    plot_type : 'polar', 'dc2d', 'dc3d' (polar, direction cosine 2d, direction cosine 3d)
    axes : 3 matplotlib subplot instances in list with polar=True active
    light : light object for system
    farfield_pattern : function of form func(adir) to return far field pattern in _adir_ direction
    directions : directions to plot
    """

    Degrees = pi/180
    N = kwargs.pop("N", 100)
    log_scale = kwargs.pop("log_scale", False)
    normalised = kwargs.pop("normalised", True)
    verbose = kwargs.pop("verbose", 0)


    if not isinstance(axes, Iterable):
        axes = [axes]

    for j, adir in enumerate(directions):

        if verbose > 0:
                print("On run {} plotting in {} direction for {} plot".format(j, adir, plot_type))

        dsdo = farfield_pattern(adir)

        dsdo[isnan(dsdo)] = 0

        if log_scale:
            dsdo = log10(dsdo)

        if normalised:
            default_plot_kwargs = {'levels': linspace(0, 1, N)}
            dsdo /= max(dsdo)
        else:
            default_plot_kwargs = {'levels': linspace(0, max(dsdo), N)}


        if plot_type == 'polar':

            assert len(axes) == len(directions), "Ambiguous number of axes provided!"
            outgoing_directions = light.outgoing_vectors[adir]

            if adir == 'x':
                theta = arccos(outgoing_directions[...,0]).T
                phi = arctan2(outgoing_directions[...,2], outgoing_directions[...,1]).T
            if adir == 'y':
                theta = arccos(outgoing_directions[...,1]).T
                phi = arctan2(outgoing_directions[...,2], outgoing_directions[...,0]).T
            if adir == 'z':
                theta = arccos(outgoing_directions[...,2]).T
                phi = arctan2(outgoing_directions[...,1], outgoing_directions[...,0]).T

            plot_kwargs = dict(default_plot_kwargs.items()+kwargs.items())

            axes[j].set_title(adir)

            if adir == 'x':
                ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plot_kwargs)
            elif adir == 'y':
                ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plot_kwargs)
            elif adir == 'z':
                ctf = axes[j].contourf(phi, theta/Degrees, dsdo, **plot_kwargs)

            _add_colorbar(axes[j], ctf, normalised)

        elif plot_type == 'dc_2d':

            assert len(axes) == len(directions), "Ambiguous number of axes provided!"
            ux, uy = light.direction_cosine(adir)


            plot_kwargs = dict(default_plot_kwargs.items()+kwargs.items())

            axes[j].set_title(adir)
            ctf = axes[j].contourf(ux, uy, dsdo,  **plot_kwargs)

            _add_colorbar(axes[j], ctf, normalised)

        elif plot_type == 'dc_3d':
            ux, uy = light.direction_cosine(adir)

            extra_kwargs = {'zdir': adir, 'offset' : -1*(-1)**(adir=='y')}
            plot_kwargs = dict(default_plot_kwargs.items()+extra_kwargs.items()+kwargs.items())

            if adir == 'x':
                axes[0].contourf(dsdo, ux, uy, **plot_kwargs)
            elif adir == 'y':
                axes[0].contourf(ux, dsdo, uy,  **plot_kwargs)
            elif adir == 'z':
                ctf = axes[0].contourf(ux, uy, dsdo,  **plot_kwargs)
                _add_colorbar(axes[0], ctf, normalised)          # add for z plot only
        else:
            raise KeyError, "Plot type is unknown!"


def _add_colorbar(axis, ctf, normalised):
    """
    Adds matplotlib colorbar to passes axis instance
    """

    cb = axis.figure.colorbar(ctf, ax=axis, use_gridspec=True, shrink=0.5)

    if normalised:
        cb.set_label("Normalised Absolute \nElectric Field Squared", size=15)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    else:
        cb.set_label("Absolute \nElectric Field Squared", size=15)


def surface_3D(axis, light, farfield_pattern, **kwargs):
    """
    Farfield pattern plotted in 3D
    """

    dsdo = farfield_pattern('all')
    dsdo[isnan(dsdo)] = 0

    direction_vectors = light.calculate_outgoing_vectors('all', amplitudes=dsdo)

    maxnum = max(direction_vectors)
    x = direction_vectors[..., 0] / maxnum
    y = direction_vectors[..., 1] / maxnum
    z = direction_vectors[..., 2] / maxnum

    axis.plot_surface(x, y, z, **kwargs)

    _to_cube(axis)


def _to_cube(axis, cube_edge=1, size=20):
    """
    Sets 3D axis to cube
    """
    axis.set_xlabel("x", size=size)
    axis.set_ylabel("y", size=size)
    axis.set_zlabel("z", size=size)
    axis.set_xlim(-cube_edge, cube_edge)
    axis.set_ylim(-cube_edge, cube_edge)
    axis.set_zlim(-cube_edge, cube_edge)


def _fetch_max_dimension(x, y, z):
    """
    Returns maximum dimension allowing us to scale sensibly
    """
    anumx = max(x[invert(isnan(x))])
    anumy = max(y[invert(isnan(y))])
    anumz = max(z[invert(isnan(z))])

    return max([anumx, anumy, anumz])
