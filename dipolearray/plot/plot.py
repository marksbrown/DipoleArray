'''
Plotting functions for _dipolearray (requires matplotlib >= 1.1.0)
'''

from __future__ import division, print_function
from numpy import pi, meshgrid, sin, cos, array

# Constants
nm = 1e-9
Degrees = pi / 180

def __PositionsOfPoints(lc, u1, u2):
    '''
    Returns the (x,y) coordinates of points defined
    by the lattice parameters : d1 (distance) ,d2 (distance),
                                v1 (delta d1), v2 (delta d2) (optional)
                                t1 (angle with x), t2 (angle with x)
    '''
    if len(lc) == 4:  # non-varying structure
        d1, t1, d2, t2 = lc
        x = u1 * d1 * cos(t1) + u2 * d2 * cos(t2)
        y = u1 * d1 * sin(t1) + u2 * d2 * sin(t2)
    elif len(lc) == 6:  # spatially varying structure
        d1, v1, t1, d2, v2, t2 = lc
        x = u1 * d1 * cos(t1) * (1 + u1*v1/d1) + \
            u2 * d2 * cos(t2) * (1 + u2*v2/d2)
        y = u1 * d1 * sin(t1) * (1 + u1*v1/d1) + \
            u2 * d2 * sin(t2) * (1 + u2 * v2 / d2)
    else:
        print("Incorrect length of lattice definition")
        return None

    return x, y

def PlotLattice(axis, lc, N1, N2, verbose=0, **kwargs):
    '''
    Plots given lattice on subplot using matplotlib

    axis : matplotlib subplot instance

    lc : 2D bravais lattice  : (length-1,angle-1,length-2,angle-2)
        example of square lattice [100*nm,0*Degrees,100*nm,90*Degrees]

    N1, N2 : number of dipoles in first and second lattice axis

    ---

    kwargs 

    dipolelength : Length of dipole
    dipoleangle : Angle with respect to cartesian x axis (phi)
    title : Title of plot    
    xlim, ylim : x, y limits
    '''

    DipoleLength=kwargs.get('dipolelength',10*nm)
    DipoleAngle=kwargs.get('dipoleangle',0*Degrees)
    Title=kwargs.get('title',"") 
    xlim=kwargs.get('xlim',(0, 500))
    ylim=kwargs.get('ylim',(0, 500)) 

    u1, u2 = meshgrid(range(*N1), range(*N2))  # grid of integers

    if verbose > 0:
        print("Number of Dipoles Generated",ptp(N1)*ptp(N2))
    
    if verbose > 1:
        print(u1, u2)
    X, Y = __PositionsOfPoints(lc, u1, u2)

    for px, py in zip(X, Y):
        x = array(
            [(px - DipoleLength) * cos(DipoleAngle) + py * sin(DipoleAngle),
             (px + DipoleLength) * cos(DipoleAngle) + py * sin(DipoleAngle)])
        y = array(
            [-1 * (
                px - DipoleLength) * sin(DipoleAngle) + py * cos(DipoleAngle),
             -1 * (px + DipoleLength) * sin(DipoleAngle) + py * cos(DipoleAngle)])

        axis.plot(x / nm, y / nm, 'k-')

    axis.set_xlabel("X Position (nm)")
    axis.set_ylabel("Y Position (nm)")
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_title(Title)
    axis.grid()
    return axis


def ContourPlot3D(axis, x, y, z, filled=False, *args, **kwargs):
    '''
    Plots contour lines of data in 3D
    '''
    if filled == False:
        ctr = axis.contour3D(x, y, z, *args, **kwargs)
    elif filled == True:
        ctr = axis.contourf3D(x, y, z, *args, **kwargs)
    else:
        print("Non bool!")
        return None

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

    NegativeMaximumDeviation = 0
    MaximumDeviation = 0
    a, b = axis.get_xlim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b
    a, b = axis.get_zlim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b
    a, b = axis.get_ylim()
    if a < NegativeMaximumDeviation:
        NegativeMaximumDeviation = a
    if b > MaximumDeviation:
        MaximumDeviation = b

    Deviation = (abs(NegativeMaximumDeviation), MaximumDeviation)[
        abs(NegativeMaximumDeviation) < MaximumDeviation]
    Deviation *= 1.1
    axis.set_xlim(-Deviation, Deviation)
    axis.set_ylim(-Deviation, Deviation)
    axis.set_zlim(-Deviation, Deviation)
    return ctr
