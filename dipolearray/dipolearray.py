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

from numpy import cross, conj, ones, sqrt, linspace, arccos, arctan, arctan2, subtract, where, eye, argmax, ndarray, log10
from numpy import meshgrid, cos, sin, pi, exp, real, array, dot, shape, sum, zeros, ptp, log, product, tile, newaxis, random
from numpy import vstack, einsum, reshape
from collections import Iterable

class light(object):
    """
    encapsulates properties relating to incoming and outgoing plane waves
    """
    def __init__(self, k, incident_vector=array([0, 0, 1]),
                 directions='x y z all',
                 steptheta=400, stepphi=200,
                 incident_polarisation=None,
                 method='ordered'):

        self.k = k
        self.steptheta = steptheta
        self.stepphi = stepphi
        self.incoming_vector = incident_vector

        self.outgoing_vectors = {adir: self.calculate_outgoing_vectors(adir, method=method)
                                 for adir in directions.split()}

        self.theta = {adir: arccos(self.outgoing_vectors[adir][..., 2]) for adir in directions.split()}
        self.phi = {adir: arctan2(self.outgoing_vectors[adir][..., 1], self.outgoing_vectors[adir][..., 0]) for adir in directions.split()}

        normalize_phi = lambda phi : where(phi < 0, phi+2*pi, phi)
        self.phi = {adir : normalize_phi(self.phi[adir]) for adir in self.phi}

        if isinstance(incident_polarisation, Iterable):
            self.incident_polarisation = incident_polarisation
        else:
            self.incident_polarisation = {adir: self.orthogonal_incident_polarisations(adir)
                                          for adir in directions.split()}

        self.outgoing_polarisation = {adir : self.orthogonal_polarisations(adir) for adir in directions.split()}

    def calculate_outgoing_vectors(self, adir, amplitudes=1, method='ordered'):
        """
        Calculates 2D grid of outgoing vectors
        """

        if method == 'ordered':
            theta, phi = angles_of_hemisphere(adir, self.steptheta, self.stepphi)
            return radial_direction_vector(theta, phi, amplitudes)
        elif method == 'random':
            return reshape(amplitudes*random_points_on_a_sphere(self.steptheta*self.stepphi, adir),
                (self.steptheta, self.stepphi, 3))

    def orthogonal_incident_polarisations(self, adir):
        """
        Orthogonal incident polarisations defined by scattering plane

        returns (perpendicular, parallel) polarisations
        """

        n0 = self.incoming_vector
        n1 = self.outgoing_vectors[adir]

        return cross(n0, n1), n1 - dot(n1, n0)[..., newaxis]*n0

    def orthogonal_polarisations(self, adir):
        """
        Orthogonal incident polarisations defined by scattering plane

        returns (perpendicular, parallel) polarisations
        """
        n0 = self.incoming_vector
        n1 = self.outgoing_vectors[adir]

        return cross(n0, n1), n1*dot(n1, n0[..., newaxis]) - n0

    def direction_cosine(self, adir):
        """
        Direction cosines for cartesian unit vector 'x', 'y' or 'z'
        """

        if adir == 'x':
            i, j = 1, 2
        elif adir == 'y':
            i, j = 0, 2
        elif adir== 'z':
            i, j = 0, 1
        else:
            raise KeyError, "direction not used or defined"

        return self.outgoing_vectors[adir][..., i], self.outgoing_vectors[adir][..., j]

class metasurface(object):
    """
    Metasurface defined by collection of scatterers
    """
    def __init__(self, x_scatterers, y_scatterers, lattice, alpha=1, epsilon_media=1):

        self.x_scatterers = x_scatterers
        self.y_scatterers = y_scatterers
        self.lattice = lattice
        self.epsilon_media = epsilon_media

        if shape(alpha)==(3,3):
            self.alpha = alpha
        else:
            self.alpha = eye(3)*alpha #assumes number is passed, #TODO requires test to prevent stupidity here


    def structure_factor(self, adir, light, dist='analytical', verbose=0):
        """
        Calculates the Structure Factor : F(q)

        --args--
        n0 : incident direction
        n1 : outgoing directions
        k : Wavenumber
        x_scatterers : Number of points in first axis
        y_scatterers : Number of points in second axis
        lc : Lattice definition
        verbose : verbosity control
        """

        d1, t1, d2, t2 = self.lattice

        n0 = light.incoming_vector
        n1 = light.outgoing_vectors[adir]

        q = light.k*subtract(n0, n1)

        exponent_factor = lambda length, angle: length*(q[...,0]*cos(angle)+q[...,1]*sin(angle))

        N1 = ptp(self.x_scatterers)
        N2 = ptp(self.y_scatterers)

        if verbose > 0:
            print("{0}x{1}".format(N1,N2))

        if dist == "analytical":
            #returns the analytical expression


            fx = exponent_factor(d1, t1)
            fy = exponent_factor(d2, t2)

            F = sin(N1*fx/2)**2/sin(fx/2)**2*sin(N2*fy/2)**2/sin(fy/2)**2
            F[(fx==0) |(fy==0)] = N1**2+N2**2

            return F

        elif dist == "sum":

            structure_factor_1d = lambda i, f: where(f!=0, exp(1j*i*f), 1)

            Fx = sum([structure_factor_1d(n, exponent_factor(d1, t1)) for n in range(*self.x_scatterers)], axis=0)
            Fx *= conj(Fx)

            Fy = sum([structure_factor_1d(n, exponent_factor(d2, t2)) for n in range(*self.y_scatterers)], axis=0)
            Fy *= conj(Fy)

            return real(Fx) * real(Fy)


    def induced_dipole_moment(self, E0):
        """
        Calculate the (time averaged) induced dipole moment due to an incident plane wave
        """
        try:
            return self.epsilon_media*dot(self.alpha, E0)
        except ValueError:
            return einsum('ij,...klj->...kli', self.alpha, E0)


    def total_dipole_power(self, E0, k):
        """
        Calculates the total power emitted from a given induced dipole moment
        """


        p = self.induced_dipole_moment(E0)
        p = einsum('...l->...', p*conj(p))

        mu0 = 4 * pi * 1e-7  # Permeability of free space
        eps0 = 8.85418782E-12  # permittivity of free space
        Zzero = sqrt(mu0 / eps0)  # Impedance of free space
        c = 3E8  # speed of light

        constant_term = ((c**2 * Zzero * k ** 4) / (12 * pi))

        return constant_term * p

    def periodic_lattice_positions(self):
        """
        Returns cartesian position of an array of points defined by the lattice
        _lc_ with min/max numbers defined by the tuples _N1_ and _N2_
        """
        u1, u2 = meshgrid(range(*self.x_scatterers), range(*self.y_scatterers))  # grid of integers

        dx, tx, dy, ty = self.lattice

        xpos = lambda u, d, t: u * d * cos(t)
        ypos = lambda u, d, t: u * d * sin(t)

        X = xpos(u1, dx, tx) + xpos(u2, dy, ty)
        Y = ypos(u1, dx, tx) + ypos(u2, dy, ty)
        Z = zeros(shape(X))

        return X, Y, Z


def incident_phase_addition(n0, R, k, intersection=array([0, 0, 0])):
    """
    Returns phase addition term due to incident wave
    n0: incident direction
    R: scatter position
    k: wavenumber
    intersection: intersection position (defaults to origin)
    verbose: verbosity control
    """
    return k * (dot(R, n0) + dot(intersection, n0))


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


def radial_direction_vector(theta=0, phi=0, amplitude=1):
    """
    returns r unit vector in cartesian coordinates
    """

    return array((amplitude * sin(theta) * cos(phi),
                  amplitude * sin(theta) * sin(phi),
                  amplitude * ones(shape(phi)) * cos(theta))).T


def angles_of_hemisphere(adir, steptheta=400, stepphi=400, meshed=True):
    """
    theta and phi ranges for hemispheres in 'x', 'y', 'z' direction
    ---
    adir : a chosen direction ('x' or 'y' or 'z')
    steptheta : number of steps in theta
    stepphi : number of steps in phi
    meshed : return angles as 2D arrays?
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

    if meshed:
        return meshgrid(Theta, Phi)
    else:
        return Theta, Phi


def total_scattering_cross_section_analytical(epsilon, a, k, type='metallic'):
    """
    Calculates the total scattering cross section for electric dipole scattering
    by a small dielectric sphere
    """

    if type == 'dielectric':
        term = (epsilon - 1) / (epsilon + 2)
        return (8 * pi) / 3 * k ** 4 * a ** 6 * term * conj(term)
    elif type == 'metallic':
        return (10 * pi * k ** 4 * a ** 6) / 3


def polarisability_sphere(epsilon, epsilon_media, a):
    """
    Polarisability tensor of a sphere
    """
    eps0 = 8.85418782E-12  # permittivity of free space
    return eps0 * 4*pi*a**3*(epsilon-epsilon_media)/(epsilon+2*epsilon_media)


def polarisability_spheroid(epsilon, epsilon_media, a, b, c, with_properties=False, verbose=0):
    """
    Polarisability tensor of spheroid aligned in z

    a, b : principal semiaxes

    a==b : Oblate
    b==c : prolate
    a==b==c : sphere
    a!=!b!=c : ellipsoid (Not Implemented)

    verbose : verbosity control
    """

    if isinstance(a, float) and isinstance(b, float) and isinstance(c, float): #single value catch
        a = array([a])
        b = array([b])
        c = array([c])


    arr_shapes = [product(shape(a)), product(shape(b)), product(shape(c))]

    if argmax(arr_shapes) == 0:
        geometry_factor = 1/3 * ones(shape(a))
        eccentricity = zeros(shape(a))
        alpha = tile(eye(3, dtype=complex), shape(a)).T
    if argmax(arr_shapes) == 1:
        geometry_factor = 1/3 * ones(shape(b))
        eccentricity = zeros(shape(b))
        alpha = tile(eye(3, dtype=complex), shape(b)).T
    if argmax(arr_shapes) == 2:
        geometry_factor = 1/3 * ones(shape(c))
        eccentricity = zeros(shape(c))
        alpha = tile(eye(3, dtype=complex), shape(c)).T

    if verbose > 0:
        print("alpha is of shape {}".format(shape(alpha)))

    prolate_condition = (b == c) & (a != b) & (a != c)
    oblate_condition = (a == b) & (a != c) & (b != c)
    sphere_condition = (a == b) & (a == c) & (b == c)
    ellipsoid_condition = (a != b) & (a != c) & (b != c)

    if verbose > 0:
        print("There are {} polarisabilities to find".format(product(shape(eccentricity))))
        print("There are {} prolates".format(sum(prolate_condition)))
        print("There are {} oblates".format(sum(oblate_condition)))
        print("There are {} spheres".format(sum(sphere_condition)))

    try:
        if any(ellipsoid_condition):
            print("Ellipsoid's are not implemented!")
    except TypeError:
        if ellipsoid_condition:
            print("Ellipsoid's are not implemented!")


    if isinstance(a,Iterable) and isinstance(b,Iterable):
        eccentricity[prolate_condition] = 1 - (b[prolate_condition]/a[prolate_condition])**2
    elif isinstance(a,Iterable) and not isinstance(b, Iterable):
        eccentricity[prolate_condition] = 1 - (b/a[prolate_condition])**2
    elif not isinstance(a,Iterable) and isinstance(b, Iterable):
        eccentricity[prolate_condition] = 1 - (b[prolate_condition]/a)**2
    else:
        eccentricity[prolate_condition] = 1 - (b/a)**2

    if verbose > 0:
        print(eccentricity[prolate_condition])

    if isinstance(a,Iterable) and isinstance(c, Iterable):
        eccentricity[oblate_condition] = 1 - (c[oblate_condition]/a[oblate_condition])**2
    elif isinstance(a,Iterable) and not isinstance(c, Iterable):
        eccentricity[oblate_condition] = 1 - (c/a[oblate_condition])**2
    elif not isinstance(a,Iterable) and isinstance(c, Iterable):
        eccentricity[oblate_condition] = 1 - (c[oblate_condition]/a)**2
    else:
        eccentricity[oblate_condition] = 1 - (c/a)**2

    g_eccentricity = sqrt((1-eccentricity[oblate_condition]**2)/eccentricity[oblate_condition]**2)
    geometry_factor[oblate_condition] = g_eccentricity / (2*eccentricity[oblate_condition]**2) * (pi/2 - arctan(g_eccentricity))-g_eccentricity**2/2

    geometry_factor[prolate_condition] = (1 - eccentricity[prolate_condition]**2) / eccentricity[prolate_condition]**2*(
        -1+1/(2*eccentricity[prolate_condition])*log((1+eccentricity[prolate_condition])/(1-eccentricity[prolate_condition])))

    alpha_lambda = lambda L : 4*pi*a*b*c*(epsilon - epsilon_media)/(3*epsilon_media+3*L*(epsilon-epsilon_media))

    condition = oblate_condition | prolate_condition #oblate or prolate

    assert not any(oblate_condition & prolate_condition), "Can't be oblate and prolate at the same time!"

    ##populates 3x3 diagonal positions

    if any(condition):
        alpha[::3, 0][condition] = alpha_lambda(geometry_factor[condition])
        alpha[1::3, 1][condition] = alpha_lambda(geometry_factor[condition])
        alpha[2::3, 2][condition] = alpha_lambda(1-2*geometry_factor[condition])

    if any(sphere_condition):
        sphere_alpha = polarisability_sphere(epsilon, epsilon_media, a[sphere_condition])
        alpha[::3, 0][sphere_condition] = sphere_alpha
        alpha[1::3, 1][sphere_condition] = sphere_alpha
        alpha[2::3, 2][sphere_condition] = sphere_alpha


    eps0 = 8.85418782E-12  # permittivity of free space
    if with_properties:
        return geometry_factor, eccentricity, eps0 * alpha
    else:
        return eps0 * alpha


def random_points_on_a_sphere(N, adir):
    """
    Generate N uniforms points on a hemisphere/sphere
    adir : outcome
    'x' : +ve x hemisphere
    'y' : +ve y hemisphere
    'z' : +ve z hemisphere
    'all' : full sphere
    """
    Values = []

    while len(Values) < N:
        x1 = 2 * random.random() - 1
        x2 = 2 * random.random() - 1

        if (x1 ** 2 + x2 ** 2 > 1):
            continue

        if adir == 'x' and x1 >=0:
            Values.append((x1, x2))

        if adir == 'y' and x2 >=0:
            Values.append((x1, x2))

        if adir == 'z' and (x1 ** 2 + x2 ** 2 < 0.5):
            Values.append((x1, x2))

        if adir == 'all':
            Values.append((x1, x2))

    x1, x2 = zip(*Values)
    x1 = array(x1)
    x2 = array(x2)
    x = 2 * x1 * sqrt(1 - x1 ** 2 - x2 ** 2)
    y = 2 * x2 * sqrt(1 - x1 ** 2 - x2 ** 2)
    z = 1 - 2 * (x1 ** 2 + x2 ** 2)

    return array((x,y,z)).T


def uniform_angle_segment(N, theta=(0, pi/6), phi=None, verbose=0):
    """
    Returns N directions from sphere point picking within theta and phi ranges only
    """

    if phi is None:
        return hemisphere_segment(N, *theta)
    else:
        return None

def hemisphere_segment(N, mintheta=0, maxtheta=pi/6, verbose=0):
    """
    Returns N directions from sphere point picking (Marsaglia 1972)
    """

    while True:
        ListOfDirections = random_points_on_a_sphere(N, 'all')

        mincondition = ListOfDirections[..., 2] < cos(mintheta)
        maxcondition = ListOfDirections[..., 2] > cos(maxtheta)

        try:  # matching directions aren't wasted when we generate more
            AllowedDirections = vstack((AllowedDirections,
                                        ListOfDirections[mincondition & maxcondition]))
        except UnboundLocalError:
            AllowedDirections = ListOfDirections[mincondition & maxcondition]

        if verbose > 0:
            print("Additional matching is", sum(mincondition & maxcondition))
            print("Total is now", len(AllowedDirections))
            print("Shape of AllowDirections is", shape(AllowedDirections))

        if len(AllowedDirections) < N:
            continue
        else:
            return AllowedDirections[:N]
