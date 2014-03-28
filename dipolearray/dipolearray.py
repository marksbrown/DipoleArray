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

from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum, zeros, ptp
from numpy import cross, conj, ones, sqrt, linspace, arccos, arctan2, subtract


def incident_phase_addition(n0, R, k, intersection=array([0, 0, 0]), verbose=0):
    """
    Returns phase addition term due to incident wave
    n0: incident direction
    R: scatter position
    k: wavenumber
    intersection: intersection position (defaults to origin)
    verbose: verbosity control
    """
    return k * (dot(R, n0) + dot(intersection, n0))

def periodic_lattice_positions(N1, N2, lc):
    """
    Returns cartesian position of an array of points defined by the lattice
    _lc_ with min/max numbers defined by the tuples _N1_ and _N2_
    """
    u1, u2 = meshgrid(range(*N1), range(*N2))  # grid of integers

    dx, tx, dy, ty = lc

    xpos = lambda u, d, t: u * d * cos(t)
    ypos = lambda u, d, t: u * d * sin(t)

    X = xpos(u1, dx, tx) + xpos(u2, dy, ty)
    Y = ypos(u1, dx, tx) + ypos(u2, dy, ty)
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
    steptheta : number of steps in theta
    stepphi : number of steps in phi
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


def differential_cross_section_single(F, n0, n1, k, const=True, split=False, verbose=0):
    """
    Differential cross section for unpolarised incident light

    F : structure factor
    n0 : incident direction
    n1 : outgoing direction
    k : wavenumber
    const : include constants
    """
    mag = lambda mat: mat * conj(mat)

    if const:
        eps0 = 8.85418782E-12  # permittivity of free space
        constterm = (k ** 2 / (4 * pi * eps0)) ** 2
    else:
        constterm = 1.0

    if split:
        return F * constterm * ones(shape(n1)), F * constterm * dot(n1, n0) ** 2
    else:
        return (F * constterm * (1 + dot(n1, n0) ** 2)).T


def differential_cross_section_volume(n0, k, N1, N2, lc, adir, **kwargs):
    """
    Calculate the differential scattering cross section

    ---args--
    n0 - incident direction
    k - wavenumber
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
    n1 = radial_direction_vector(theta, phi, verbose=verbose)

    if dist == 'analytical':
        dsdo = combined_dipole_dpdo(n0, n1, k, const, verbose)

    elif dist == 'normal':
        F = structure_factor(n0, n1, N1, N2, lc, k, verbose=verbose)
        dsdo = differential_cross_section_single(F, n0, n1, k=k, const=const, verbose=verbose)

    else:
        raise KeyError, "{0} is unknown".format(dist)

    return theta, phi, dsdo


def structure_factor_analytical(n0, n1, nx, ny, lc, k, verbose=0):
    """
    Calculates the Structure Factor

    --args--
    n0 : incident direction
    n1 : outgoing directions
    k : Wavenumber
    nx : Number of points in first axis
    ny : Number of points in second axis
    lc : Lattice definition
    verbose : verbosity control
    """

    d1, t1, d2, t2 = lc

    q = k*subtract(n0, n1)

    f = lambda length, angle : length*(q[...,0]*cos(angle)+q[...,1]*sin(angle))
    F = lambda N, f : sin(N*f/2)**2/sin(f/2)**2

    N1 = ptp(nx)
    N2 = ptp(ny)

    Fx = 1 / N1 * F(N1, f(d1, t1))
    Fy = 1 / N1 * F(N2, f(d2, t2))

    return (Fx * Fy)


def structure_factor_sum(n0, n1, nx, ny, lc, k, verbose=0):
    """
    Calculates the Structure Factor as sum

    --args--
    n0 : incident direction
    n1 : outgoing directions
    k : Wavenumber
    nx : Number of points in first axis
    ny : Number of points in second axis
    lc : Lattice definition
    verbose : verbosity control
    """

    d1, t1, d2, t2 = lc

    q = k*subtract(n0, n1)

    f = lambda length, angle : length*(q[...,0]*cos(angle)+q[...,1]*sin(angle))

    F = lambda i, f: exp(1j*i*f)

    N1 = ptp(nx)
    N2 = ptp(ny)

    Fx = sum([F(n, f(d1, t1)) for n in range(*nx)], axis=0)
    Fx *= 1 / ptp(nx) * conj(Fx)

    Fy = sum([F(n, f(d2, t2)) for n in range(*ny)], axis=0)
    Fy *= 1 / ptp(ny) * conj(Fy)

    return (real(Fx) * real(Fy))


#incident_phase_term = incident_phase_addition(n0, R, k, verbose=verbose)


def structure_factor(n0, n1, nx, ny, lc, k, dist='analytical', verbose=0):
    """
    Calculate the structure factor : F(q)
    """
    if dist == "analytical":
        return structure_factor_analytical(n0, n1, nx, ny, lc, k, verbose)
    elif dist == "sum":
        return structure_factor_sum(n0, n1, nx, ny, lc, k, verbose)


def combined_dipole_dpdo(n0, n1, k, const=False, verbose=0):
    """
    Combine two orthogonal electric dipoles to the incident direction to predict
    the behaviour of unpolarised light
    """

    theta = arccos(n0[..., 2])
    phi = arctan2(n0[..., 1], n0[..., 0])

    n0, eperp, epara = spherical_unit_vectors(theta, phi)

    return (electric_dipole_dpdo(n1, eperp, k, const, verbose) + electric_dipole_dpdo(n1, epara, k, const, verbose)).T


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

    mag = lambda mat: sum(mat * conj(mat), axis=-1)

    return real(constterm * mag(a))


def polarisability_sphere(epsilon, epsilon_media, radius):
    """
    Equation (10.5) from Jackson
    """
    eps0 = 8.85418782E-12
    return 4 * pi * ((epsilon - epsilon_media) / (epsilon + 2 * epsilon_media)) * radius ** 3


def induced_dipole_moment(epsilon_media, alpha, E0, Flag=False, **kwargs):
    """
    Calculate the induced dipole moment due to an incident plane wave
    assumes no time or position variation unless specified
    """
    c = 3E8
    nm = 1e-9

    wavenumber = kwargs.get('wavenumber', [0, 0, (2 * pi) / (420 * nm)])
    position = kwargs.get('position', 0)
    omega = kwargs.get('frequency', (2 * pi * c) / (420 * nm))
    time = kwargs.get('time', 0)

    if Flag:
        return epsilon_media * alpha * E0 * exp(1j * dot(wavenumber, position)) * exp(-1j * omega * time)
    else:
        return epsilon_media * alpha * E0
