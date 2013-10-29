'''
Electric Dipole Array with Structure Factor
See : Chapters (initial sections of) 9, 10 of Electrodynamics, Jackson

Author : Mark S. Brown
Started : 9th July 2013

Description : Periodic 2D structures currently described as a finite array
of dipoles with a known Bravais lattice. The code calculates the structure
factor and the resulting differential cross section.
'''
from __future__ import division, print_function
from numpy import meshgrid, cos, sin, pi, exp, real, imag, array, dot, shape, sum
from numpy import random, cross, conj, dstack, issubdtype, ones, sqrt, linspace, ptp
import os

# Constants
nm = 1e-9
Degrees = pi / 180
eps0 = 8.85418782E-12  # permittivity of free space
mu0 = 4 * pi * 1e-7  # Permeability of free space
Zzero = sqrt(mu0 / eps0)  # Impedance of free space

def DirectionVector(theta = 0*Degrees, phi=0*Degrees, amplitude=1, verbose=0):
    '''
    Spherical coordinates (r,theta,phi) --> cartesian coordinates (x,y,z)
    '''
    if issubdtype(type(theta), float) and issubdtype(type(phi), float):
        return array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
    
    x = amplitude * sin(theta) * cos(phi)
    y = amplitude * sin(theta) * sin(phi)
    if issubdtype(type(theta), float):
        z = amplitude * cos(theta) * ones(shape(x))
    else:
        z = amplitude * cos(theta)
    if verbose > 0:
        print("x", x)
        print("y", y)
        print("z", z)

    return dstack([x, y, z])


def ElasticScatteringVector(n0, n, verbose=0):
    '''
    Returns Elastic Scattering Vector x,y components
    nx = (n_0-n).x
    ny = (n_0-n).y
    '''
    diff = n0 - n
    return diff[..., 0], diff[..., 1]


def SpacingCoefficient(lc, nx, ny, verbose=0):
    '''
    Returns dx,dy for a given 2D lattice
    '''
    if len(lc) == 4:
        d1, t1, d2, t2 = lc
    elif len(lc) == 6:
        d1, v1, t1, d2, v2, t2 = lc
    else:
        print("Incorrect length of lattice definition")
        return None

    dx = d1 * (nx * cos(t1) + ny * sin(t1))
    dy = d2 * (nx * cos(t2) * ny * sin(t2))
    if verbose > 0:
        print("Spacing Coefficients are", dx / nm, "nm", dy / nm, "nm")
    return dx, dy

def AnglesofHemisphere(adir, steps=400):
    '''
    theta and phi ranges for hemispheres in 'x','y' or 'z' direction
    ---
    adir : a chosen direction ('x' or 'y' or 'z')
    steps : 
    '''
    if adir == 'x':
        Theta = linspace(0, 180*Degrees, steps)
        Phi = linspace(-90*Degrees, 90*Degrees, steps)
    elif adir == 'y':
        Theta = linspace(0, 180*Degrees, steps)
        Phi = linspace(0, 180*Degrees, steps)
    elif adir == 'z':
        Theta = linspace(0, 90*Degrees, steps)
        Phi = linspace(0, 360*Degrees, steps)
    elif adir == 'all':
        Theta = linspace(0,180*Degrees,steps)
        Phi = linspace(0,360*Degrees,steps)
    else:
        print("None Selected")
        return

    return meshgrid(Theta, Phi)

def GetDirectionCosine(adir, steps=400):
    '''
    Direction cosines for cartesian unit vector 'x', 'y' or 'z' defined
    for theta and phi using _AnglesOfHemisphere_
    '''
    theta,phi = AnglesofHemisphere(adir, steps)

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


def StructureFactor(k, N1, N2, dx, dy, lc, mode='1bf', verbose=0):
    """
    Structure Factor

    Variables
    k : Wavenumber
    N1 : Number of points in first axis
    N2 : Number of points in second axis
    dx : x-spacing coefficient
    dy : y-spacing coefficient
    lc : Lattice definition
    mode : mode of calculation (see below)

    Modes of operation
    1bf : 1st order brute force
    2bf : 2nd order brute force
    pa : Periodic Analytical

    """

    if len(lc) == 4:
        d1, t1, d2, t2 = lc
    elif len(lc) == 6:
        d1, v1, t1, d2, v2, t2 = lc
    else:
        print("Incorrect length of lattice definition")
        return None

    if mode == '1bf':  # 1st order (Periodic) Brute Force Method
        amp = sum(exp(1j * k * U1 * dx) for U1 in range(1, N1 + 1))
        a1, b1 = amp.real, amp.imag
        amp = sum(exp(1j * k * U2 * dy) for U2 in range(1, N2 + 1))
        a2, b2 = amp.real, amp.imag

        F = (a1 ** 2 + b1 ** 2) * (a2 ** 2 + b2 ** 2)
        F /= (N1 * N2)
        if verbose > 0:
            print(
                "The Brute-Force Structure factor is",
                F,
                ":",
                a1,
                b1,
                a2,
                b2)
        return F

    elif mode == '2bf':  # 2nd order term Brute Force Method
        if len(lc) != 6:
            print("Varying lattice constants not defined, setting to zero")
            v1 = 0
            v2 = 0

        amp = sum(exp(1j * k * U1 * dx * (1 + U1 * v1 / d1))
                  for U1 in range(1, N1 + 1))
        a1, b1 = amp.real, amp.imag
        amp = sum(exp(1j * k * U2 * dy * (1 + U2 * v2 / d2))
                  for U2 in range(1, N2 + 1))
        a2, b2 = amp.real, amp.imag

        F = (a1 ** 2 + b1 ** 2) * (a2 ** 2 + b2 ** 2)
        F /= (N1 * N2)
        if verbose > 0:
            print(
                "The Brute-Force Structure factor is",
                F,
                ":",
                a1,
                b1,
                a2,
                b2)
        return F

    elif mode == 'pa':  # periodic analytical method
        Fx = 1 / N1 * sin(N1 * k * dx / 2) ** 2 / sin(k * dx / 2) ** 2
        Fy = 1 / N2 * sin(N2 * k * dy / 2) ** 2 / sin(k * dy / 2) ** 2
        if verbose > 0:
            print("The Analytical Structure factor is", Fx * Fy, ":", Fx, Fy)
        return Fx * Fy
    else:
        print("Unknown method")
        return None


def DifferentialCrossSection(
        F, n0, n1, p, k, const=True, split=False, verbose=0):
    '''
    Differential cross section

    n0 : incident direction
    n1 : outgoing direction
    p : electric dipole moment vector
    k : wavevector
    const : include constants
    '''
    mag = lambda mat: mat * conj(mat)

    if const:
        constterm = (k ** 2 / (4 * pi * eps0)) ** 2
    else:
        constterm = 1.0

    eperp = cross(n0, n1)
    epara = cross(eperp, n0)

    firstterm = dot(eperp, p)
    secondterm = dot(epara, p)

    if split:
        return (
            real(constterm * mag(firstterm) * F), real(
                constterm * mag(secondterm) * F)
        )
    else:
        return (
            (real(constterm * mag(firstterm) * F)
             + real(constterm * mag(secondterm) * F))
        )


def DipoleDistribution(n1, p, k, const=False, verbose=0):
    '''
    Power per unit solid angle for an electric dipole (eqn 9.22 Jackson)
    '''

    if const:
        constterm = (c ** 2 * Zzero) / (32 * pi ** 2) * k ** 4
    else:
        constterm = 1.0

    a = cross(n1, p)
    a = cross(a, n1)

    mag = lambda mat: sum(mat * conj(mat), axis=2)

    return real(constterm * mag(a))

    # return constterm*(1-dot(n1,p)**2)


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
