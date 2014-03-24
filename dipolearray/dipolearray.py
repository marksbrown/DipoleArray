"""
Electric Dipole Array with Structure Factor
See : Chapters (initial sections of) 9, 10 of Electrodynamics, Jackson

Author : Mark S. Brown
Started : 9th July 2013

Description : Periodic 2D structures currently described as a finite array
of dipoles with a known Bravais lattice. The code calculates the structure
factor and the resulting differential cross section.
"""
from __future__ import division, print_function
from collections import Iterable

from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum, zeros
from numpy import cross, conj, dstack, ones, sqrt, linspace, newaxis, arccos, arctan2

def incident_phase_addition(n0, R, k, verbose=0, **kwargs):
    '''
    Returns phase addition term due to incident wave
    '''
    p0 = kwargs.get('intersection', array([0, 0, 0]))  # planes cross at origin

    return k * (dot(R, n0) + dot(p0, n0))


def periodic_lattice_positions(N1, N2, lc):
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

def spherical_unit_vectors(theta=0, phi=0):
    """
    Spherical unit vectors as cartesian unit vectors
    """
    r = array((sin(theta) * cos(phi),
               sin(theta) * sin(phi),
               ones(shape(phi)) * cos(theta))).T
    th = array((cos(theta) * cos(phi),
                cos(theta) * sin(phi),
                -1 * ones(shape(phi)) * sin(theta))).T
    ph = array((-1 * ones(shape(theta)) * sin(phi),
                ones(shape(theta)) * cos(phi),
                zeros(shape(theta)) * zeros(shape(phi)))).T

    return r, th, ph


def radial_direction_vector(theta=0, phi=0, amplitude=1, verbose=0):
    """
    returns r unit vector in cartesian coordinates
    """

    return array((amplitude * sin(theta) * cos(phi),
                  amplitude * sin(theta) * sin(phi),
                  amplitude * ones(shape(phi)) * cos(theta))).T


def angles_of_hemisphere(adir, steptheta=400, stepphi=400):
    """
    theta and phi ranges for hemispheres in 'x', 'y', 'z' direction
    ---
    adir : a chosen direction ('x' or 'y' or 'z')
    steps :
    """

    Degrees = pi / 180

    if adir == 'x':
        Theta = linspace(0, 180 * Degrees, steptheta)
        Phi = linspace(-90 * Degrees, 90 * Degrees, stepphi)
    elif adir == 'y':
        Theta = linspace(0, 180 * Degrees, steptheta)
        Phi = linspace(0, 180 * Degrees, stepphi)
    elif adir == 'z':
        Theta = linspace(0, 90 * Degrees, steptheta)
        Phi = linspace(0, 360 * Degrees, stepphi)
    elif adir == 'all':
        Theta = linspace(0, 180 * Degrees, steptheta)
        Phi = linspace(0, 360 * Degrees, stepphi)
    else:
        print("None Selected")
        return

    return meshgrid(Theta, Phi)


def direction_cosine(adir, steptheta=400, stepphi=400):
    """
    Direction cosines for cartesian unit vector 'x', 'y' or 'z' defined
    for theta and phi using _AnglesOfHemisphere_
    """
    theta, phi = angles_of_hemisphere(adir, steptheta=steptheta, stepphi=stepphi)

    if adir == "x":
        return sin(phi) * sin(theta), cos(theta)  # y,z
    elif adir == "y":
        return cos(phi) * sin(theta), cos(theta)  # x,z
    elif adir == "z":
        return cos(phi) * sin(theta), sin(phi) * sin(theta)  # x,y
    elif adir == "all":
        return cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)
    else:
        print("None Selected")
        return


def differential_crosssection_single(F, n1, p, k, const=True, split=False, verbose=0):
    """
    Differential cross section

    n0 : incident direction
    n1 : outgoing direction
    p : electric dipole moment vector
    k : wavenumber
    const : include constants
    """
    mag = lambda mat: mat * conj(mat)

    if const:
        eps0 = 8.85418782E-12  # permittivity of free space
        constterm = (k ** 2 / (4 * pi * eps0)) ** 2
    else:
        constterm = 1.0

    theta = arccos(n1[..., 2])
    phi = arctan2(n1[..., 1], n1[..., 0])

    r, theta, phi = spherical_unit_vectors(theta, phi)

    if split:
        return constterm * dot(theta, p) ** 2, constterm * dot(phi, p) ** 2
    else:
        return constterm * dot(theta, p) ** 2 + constterm * dot(phi, p) ** 2


def differential_crossection_volume(n0, p, k, N1, N2, lc, adir, **kwargs):
    """
    Calculate the differential scattering cross section

    ---args--
    n0 - incident direction
    p - induced dipole moment
    N1, N2 - numbers of scatterers in X, Y
    lc - lattice
    adir - direction of interest
    """

    steptheta = kwargs.pop('steptheta', 400)
    stepphi = kwargs.pop('stepphi', 200)
    verbose = kwargs.pop('verbose', 0)
    dist = kwargs.pop('dist', 'normal')
    const = kwargs.get('const', False)

    theta, phi = angles_of_hemisphere(adir, steptheta=steptheta, stepphi=stepphi)
    alldirections = radial_direction_vector(theta, phi, verbose=verbose)

    if dist == 'analytical':
        dsdo = electric_dipole_dpdo(alldirections, p=p, k=k, const=const).T
    else:
        F = structure_factor_volume(alldirections, n0, N1, N2, lc, k, verbose=verbose)
        dsdo = differential_crosssection_single(F, alldirections, p=p,
                                                k=k, const=const, verbose=verbose)

    return theta, phi, dsdo


def structure_factor_volume(alldirections, n0, N1, N2, lc, k, verbose=0):
    """
    Calculates the structure factor over each incident direction
    """
    R = dstack(periodic_lattice_positions(N1, N2, lc))  # positions of each dipole
    divideby = (N1[1] - N1[0]) * (N2[1] - N2[0])
    #divdeby = ptp(N1)/ptp(N2)

    if divideby == 1:
        return ones(shape(alldirections)[:-1])

    if verbose > 0:
        print("shape of all directions", shape(alldirections))

    return array([[structure_factor_single(n0, n1, R, k, verbose=verbose) / divideby
                   for n1 in row] for row in alldirections])


def structure_factor_single(n0, n1, R, k, verbose=0):
    """
    The Structure Factor (eqn 10.19 Jackson)

    n0 : incident direction
    n1 : outgoing direction
    R : position of each dipole
    k : incident wavelength
    """

    q = k * (n0[..., newaxis] - n1)

    incidentphaseterm = incident_phase_addition(n0, R, k, verbose=verbose)

    amp = sum(exp(1j * dot(R, q) + 1j * incidentphaseterm))

    a1, b1 = amp.real, amp.imag

    return (a1 ** 2 + b1 ** 2)


def electric_dipole_dpdo(n1, p, k, const=False, verbose=0):
    """
    Power per unit solid angle for an electric dipole (eqn 9.22 Jackson)
    """

    if const:
        mu0 = 4 * pi * 1e-7  # Permeability of free space
        eps0 = 8.85418782E-12  # permittivity of free space
        Zzero = sqrt(mu0 / eps0)  # Impedance of free space
        c = 3E8  # speed of light
        constterm = (c ** 2 * Zzero) / (32 * pi ** 2) * k ** 4
    else:
        constterm = 1.0

    a = cross(n1, p)
    a = cross(a, n1)

    mag = lambda mat: sum(mat * conj(mat), axis=2)

    return real(constterm * mag(a))


def polarisability_sphere(episilon, epsilon_media, radius):
    '''
    Equation (10.5) from Jackson
    '''
    eps0 = 8.85418782E-12
    return 4 * pi * ((episilon - epsilon_media) / (episilon + 2 * epsilon_media)) * radius ** 3


def induced_dipole_moment(epsilon_media, alpha, E0, Flag=False, **kwargs):
    """
    Calculate the induced dipole moment due to an incident plane wave
    assumes no time or position variation unless specified
    """
    c = 3E8
    nm = 1e-9

    wavevector = kwargs.get('wavevector', [0, 0, (2 * pi) / (420 * nm)])
    position = kwargs.get('position', 0)
    omega = kwargs.get('frequency', (2 * pi * c) / (420 * nm))
    time = kwargs.get('time', 0)

    if Flag:
        return epsilon_media * alpha * E0 * exp(1j * dot(wavevector, position)) * exp(-1j * omega * time)
    else:
        return epsilon_media * alpha * E0
