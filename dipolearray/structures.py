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
from numpy import cross, conj, ones, sqrt, linspace, arccos, arctan, arctan2, subtract, where, eye, argmax, ndarray, log10, max
from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum, zeros, ptp, log, product, tile, newaxis, mean, std
from . import dipolearray as da

def structure_factor_plane_wave(light, metasurface, **kwargs):
    """
    Structure factor as form func(adir)
    """

    dist = kwargs.pop('dist', 'analytical')  #analytical is quicker!
    verbose = kwargs.pop('verbose', 0)

    def structure_factor(adir):
        """
        Calculate the structure factor in the direction adir
        """

        return metasurface.structure_factor(adir, light, dist, verbose).T

    return structure_factor

def electric_dipole(light, p, verbose=0):
    """
    Power per unit solid angle for an electric dipole (eqn 9.22 Jackson)
    """

    def farfield_pattern(adir):
        """
        Far field pattern for a single electric dipoles
        """
        if verbose > 0:
            print("direction : {}".format(adir))

        mu0 = 4 * pi * 1e-7  # Permeability of free space
        eps0 = 8.85418782E-12  # permittivity of free space
        Zzero = sqrt(mu0 / eps0)  # Impedance of free space
        c = 3E8  # speed of light
        constant_term = (c ** 2 * Zzero) / (32 * pi ** 2) * light.k ** 4

        n1 = light.outgoing_vectors[adir]

        a = cross(n1, p)
        a = cross(a, n1)

        mag = lambda mat: sum(mat * conj(mat), axis=-1)

        return real(constant_term * mag(a))

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


def induced_dipole_moments(light, metasurface, **kwargs):
    """
    Calculates the induced dipole moments for unpolarised incident light
    """

    verbose = kwargs.pop('verbose', 0)

    def farfield_pattern(adir):

        if verbose > 0:
            print("direction : {}".format(adir))

        epsilon_1, epsilon_2 = light.incident_polarisation[adir]

        if verbose > 0:
            print("shape of first is {}\nshape of second is {}".format(shape(epsilon_1), shape(epsilon_2)))

        induced_dipole_one = metasurface.induced_dipole_moment(epsilon_1)
        induced_dipole_two = metasurface.induced_dipole_moment(epsilon_2)

        return induced_dipole_one, induced_dipole_two

    return farfield_pattern

def total_dipole_power(light, metasurface, **kwargs):
    """
    Calculates the induced dipole moments for unpolarised incident light
    """

    verbose = kwargs.pop('verbose', 0)

    def farfield_pattern(adir):

        if verbose > 0:
            print("direction : {}".format(adir))

        epsilon_1, epsilon_2 = light.incident_polarisation[adir]

        if verbose > 0:
            print("shape of first is {}\nshape of second is {}".format(shape(epsilon_1), shape(epsilon_2)))

        total_dipole_one = metasurface.total_dipole_power(epsilon_1, light.k)
        total_dipole_two = metasurface.total_dipole_power(epsilon_2, light.k)

        return total_dipole_one, total_dipole_two

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

        if verbose > 0:
            print("direction : {}".format(adir))
        eps0 = 8.85418782E-12  # permittivity of free space
        constant_term = (light.k ** 2 / (4 * pi * eps0)) ** 2

        n0 = light.incoming_vector
        n1 = light.outgoing_vectors[adir]

        epsilon_1, epsilon_2 = light.incident_polarisation[adir]

        induced_dipole_one = metasurface.induced_dipole_moment(epsilon_1)
        induced_dipole_two = metasurface.induced_dipole_moment(epsilon_2)

        F = metasurface.structure_factor(adir, light, dist=dist, verbose=verbose)

        return (sum(constant_term * (induced_dipole_one * conj(induced_dipole_one) + dot(n1, n0)[..., newaxis] ** 2 *induced_dipole_two * conj(induced_dipole_two)), axis=-1) *F)

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

        n1 = light.outgoing_vectors[adir]
        epsilon_1, epsilon_2 = light.orthogonal_polarisations[adir]

        #return (dot(epsilon_1, induced_dipole)+dot(epsilon_2, induced_dipole))**2

        F = metasurface.structure_factor(adir, light, n1, dist=dist, verbose=verbose)

        if out==0:
            return (sum(constant_term * (dot(epsilon_1,induced_dipole)**2)**2) * F)
        elif out==1:
            return (sum(constant_term * (dot(epsilon_2,induced_dipole)**2)**2) * F)
        if out ==2:
            return (sum(constant_term * (dot(epsilon_1,induced_dipole)**2+dot(epsilon_1,induced_dipole)**2)) * F)

    return farfield_pattern