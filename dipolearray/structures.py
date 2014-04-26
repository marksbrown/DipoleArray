"""
The differential scattering cross section

Author : Mark S. Brown
Started : 23rd April 2014

In this file we define four functions :

1) Single incident (arbitrary) polarisation to calculate the differential scattering cross section
2) Single dipole to calculate the power per unit solid angle
3) Incoherent sum of unpolarised light incident upon metasurface to calculate the differential scattering cross section
4) Incoherent sum of two orthogonal dipoles to calculate the power per unit solid angle

These will be functions of the form _func(adir, *args, **kwargs)_

This function will be represented by additional functions in the _plot.py_ file
"""

from __future__ import division
from numpy import cross, conj, ones, sqrt, linspace, arccos, arctan, arctan2, subtract, where, eye, argmax, ndarray
from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum, zeros, ptp, log, product, tile, newaxis
from . import dipolearray as da


def electric_dipole(light, p):
    """
    Power per unit solid angle for an electric dipole (eqn 9.22 Jackson)
    """

    def farfield_pattern(adir):
        """
        Far field pattern for a single electric dipoles
        """

        mu0 = 4 * pi * 1e-7  # Permeability of free space
        eps0 = 8.85418782E-12  # permittivity of free space
        Zzero = sqrt(mu0 / eps0)  # Impedance of free space
        c = 3E8  # speed of light
        constant_term = (c ** 2 * Zzero) / (32 * pi ** 2) * light.k ** 4

        n1 = light.outgoing_vectors(adir)

        a = cross(n1, p)
        a = cross(a, n1)

        mag = lambda mat: sum(mat * conj(mat), axis=-1)

        return real(constant_term * mag(a)).T

    return farfield_pattern


def electric_dipole_sum(light, P):
    """
    returns sum of arbitrary number of dipoles
    """

    individual_functions = [electric_dipole(light, p) for p in P]

    def farfield_pattern(adir):
        """
        Far field pattern for a sum of electric dipoles
        """
        return sum([func(adir) for func in individual_functions], axis=0)

    return farfield_pattern

def differential_scattering_cross_section(light, metasurface, **kwargs):
    """
    Differential scattering cross section for induced dipole moment due to unpolarised light

    adir : x, y, z, all
    n0 : incident direction
    p : dipole moment
    k : wavenumber
    metasurface : object defining the metasurface
    """
    dist = kwargs.pop('dist', 'analytical')  #analytical is quicker!
    verbose = kwargs.pop('verbose', 0)

    def farfield_pattern(adir):

        eps0 = 8.85418782E-12  # permittivity of free space
        constant_term = (light.k ** 2 / (4 * pi * eps0)) ** 2

        n0 = light.incoming_vector
        n1 = light.outgoing_vectors(adir)

        epsilon_1, epsilon_2 = light.orthogonal_incident_polarisations(adir)

        induced_dipole_one = metasurface.induced_dipole_moment(epsilon_1)
        induced_dipole_two = metasurface.induced_dipole_moment(epsilon_2)

        F = metasurface.structure_factor(adir, light, dist=dist, verbose=verbose)

        return (sum(constant_term * (induced_dipole_one ** 2 + dot(n1, n0)[..., newaxis] ** 2 *induced_dipole_two ** 2), axis=-1) *F).T

    return farfield_pattern

def differential_scattering_cross_section_polarised(light, metasurface, out=2, **kwargs):
    """
    Differential scattering cross section for induced dipole moment due to polarised light

    adir : x, y, z, all
    n0 : incident direction
    p : dipole moment
    k : wavenumber
    metasurface : object defining the metasurface
    """
    dist = kwargs.pop('dist', 'analytical')  #analytical is quicker!
    verbose = kwargs.pop('verbose', 0)

    def farfield_pattern(adir):

        eps0 = 8.85418782E-12  # permittivity of free space
        constant_term = (light.k ** 2 / (4 * pi * eps0)) ** 2

        induced_dipole = metasurface.induced_dipole_moment(light.incident_polarisation) #single induced dipole

        epsilon_1, epsilon_2 = light.orthogonal_polarisations(adir)

        return (dot(epsilon_1, induced_dipole)+dot(epsilon_2, induced_dipole))**2

        F = metasurface.structure_factor(adir, light, dist=dist, verbose=verbose)

        if out==0:
            return (sum(constant_term * (dot(epsilon_1,induced_dipole)**2)**2) * F).T
        elif out==1:
            return (sum(constant_term * (dot(epsilon_2,induced_dipole)**2)**2) * F).T
        if out ==2:
            return (sum(constant_term * (dot(epsilon_1,induced_dipole)**2+dot(epsilon_1,induced_dipole)**2)) * F).T

    return farfield_pattern


# def differential_scattering_cross_section_polarised(adir, light, incident_polarisation, metasurface, **kwargs):
#     """
#     Differential scattering cross section for induced dipole moment
#
#     adir : x, y, z, all
#     n0 : incident direction
#     p : dipole moment
#     k : wavenumber
#     metasurface : object defining the metasurface
#     """
#     dist = kwargs.pop('dist', 'analytical')  #analytical is quicker!
#     verbose = kwargs.pop('verbose', 0)
#
#     eps0 = 8.85418782E-12  # permittivity of free space
#     constant_term = (light.k ** 2 / (4 * pi * eps0)) ** 2
#
#     n1 = light.outgoing_vectors(adir)
#     epsilon_1, epsilon_2 = light.orthogonal_incident_polarisations()
#
#     p = metasurface.induced_dipole_moment(incident_polarisation)
#     F = metasurface.structure_factor(adir, light, dist=dist, verbose=verbose)
#
#     return (constant_term * (dot(epsilon_1, p) ** 2 + dot(epsilon_2, p) ** 2) * F).T