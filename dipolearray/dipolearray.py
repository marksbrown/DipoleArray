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
from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum
from numpy import cross, conj, dstack, issubdtype, ones, sqrt, linspace
from structuredefinition import *

# Constants
nm = 1e-9
Degrees = pi / 180
eps0 = 8.85418782E-12  # permittivity of free space
mu0 = 4 * pi * 1e-7  # Permeability of free space
Zzero = sqrt(mu0 / eps0)  # Impedance of free space
c = 3E8  # speed of light


def DirectionVector(
        theta=0 * Degrees, phi=0 * Degrees, amplitude=1, verbose=0):
    '''
    Spherical coordinates (r,theta,phi) --> cartesian coordinates (x,y,z)
    '''
    if issubdtype(type(theta), float) and issubdtype(type(phi), float):
        return (
            array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
        )

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


def AnglesofHemisphere(adir, steps=400):
    '''
    theta and phi ranges for hemispheres in 'x','y' or 'z' direction
    ---
    adir : a chosen direction ('x' or 'y' or 'z')
    steps :
    '''
    if adir == 'x':
        Theta = linspace(0, 180 * Degrees, steps)
        Phi = linspace(-90 * Degrees, 90 * Degrees, steps)
    elif adir == 'y':
        Theta = linspace(0, 180 * Degrees, steps)
        Phi = linspace(0, 180 * Degrees, steps)
    elif adir == 'z':
        Theta = linspace(0, 90 * Degrees, steps)
        Phi = linspace(0, 360 * Degrees, steps)
    elif adir == 'all':
        Theta = linspace(0, 180 * Degrees, steps)
        Phi = linspace(0, 360 * Degrees, steps)
    else:
        print("None Selected")
        return

    return meshgrid(Theta, Phi)


def GetDirectionCosine(adir, steps=400):
    '''
    Direction cosines for cartesian unit vector 'x', 'y' or 'z' defined
    for theta and phi using _AnglesOfHemisphere_
    '''
    theta, phi = AnglesofHemisphere(adir, steps)

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


def OutgoingDirections(alldirections, n0, N1, N2, lc, k, verbose=0):
    '''
    Calculates the structure factor over each incident direction
    '''
    R = dstack(getpositions(N1, N2, lc))  # positions of each dipole
    divideby = (N1[1]-N1[0])*(N2[1]-N2[0])
    
    return array([[structurefactor(n0, n1, R, k, verbose=verbose)/divideby for n1 in row] 
                         for row in alldirections])


def structurefactor(n0, n1, R, k, verbose=0):
    '''
    The Structure Factor (eqn 10.19 Jackson)

    n0 : incident direction
    n1 : outgoing direction
    R : position of each dipole
    k : incident wavelength
    '''
    
    q = k*(n0-n1)
    
    incidentphaseterm = IncidentPhaseArray(n0, R, k, verbose=verbose)

    amp = sum(exp(1j * dot(R, q)+1j*incidentphaseterm))

    a1, b1 = amp.real, amp.imag

    return (a1 ** 2 + b1 ** 2)


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
