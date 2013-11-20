'''
Plotting functions for _dipolearray (requires matplotlib >= 1.1.0)
'''

from __future__ import division, print_function
from numpy import pi, ptp, min, max
from numpy import invert, isnan
import os
from . import structuredefinition as _sd
from . import dipolearray as da
from matplotlib.pyplot import cm

# Constants
nm = 1e-9
Degrees = pi / 180


def savefig(name, fig, verbose=0, **kwargs):
    '''
    Saves figure

    kwargs:
    imagesavelocation : imagesavelocation
    ext : extensions to save as and into
    '''

    imagesavelocation = kwargs.get('imagesavelocation',
                                   os.path.join(os.path.expanduser('~'), 'Desktop', 'images'))

    if not os.path.exists(imagesavelocation):
        if verbose > 1:
            print("Creating ImageSaveLocation directory")
        os.mkdir(imagesavelocation)

    Ext = kwargs.get('ext', ['png', 'pdf', 'svg'])

    for ext in Ext:
        savetopath = os.path.join(imagesavelocation, ext)
        if not os.path.exists(savetopath):
            if verbose > 1:
                print("Creating savetopath directory")
            os.mkdir(savetopath)

        saveto = os.path.join(savetopath, name + "." + ext)

        if verbose > 0:
            print("Saving image to", saveto)
        fig.savefig(saveto)


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

    DipoleLength = kwargs.get('dipolelength', 50)
    DipoleAngle = 90 - kwargs.get('dipoleangle', 45)
    Title = kwargs.get('title', "")
    limfactor = kwargs.get('limfactor', 1.1)
    divideby = kwargs.get('divideby', nm)

    if verbose > 0:
        print("Number of Dipoles Generated", ptp(N1) * ptp(N2))

    X, Y, Z = _sd.getpositions(N1, N2, lc)

    X /= divideby
    Y /= divideby

    xlim = (min(X) - limfactor * min(X), max(X) + limfactor * min(X))
    ylim = (min(Y) - limfactor * min(Y), max(Y) + limfactor * min(Y))

    axis.grid()
    try:
        axis.scatter3D(X, Y, Z, marker=(2, 0, DipoleAngle), s=DipoleLength)
        axis.set_zlabel("Z Position (nm)")
    except AttributeError:
        axis.scatter(X, Y, marker=(2, 0, DipoleAngle), s=DipoleLength)

    axis.set_xlabel("X Position (nm)")
    axis.set_ylabel("Y Position (nm)")
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_title(Title)
    axis.grid(True)
    return axis


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

def GenerateFarFieldImage(n0, k, N1, N2, lc, p, steps, afig, axis, verbose=0):
    '''
    Generate 3D matplotlib (>= 1.30 required if you wish save the image as svg)
    image of farfield pattern. 3 faces are plotted with the respective direction
    cosines
    '''
    theta,phi = da.AnglesofHemisphere('all',steps)
    alldirections = da.DirectionVector(theta,phi)
        
    F = da.OutgoingDirections(alldirections, n0, N1, N2, lc, k)
        
    dsdo = da.DifferentialCrossSection(F,n0,alldirections,p=p,k=k,const=False)
    
    adir = da.DirectionVector(theta,phi,dsdo)
    maxnum = max(adir)
    x = adir[...,0]/maxnum
    y = adir[...,1]/maxnum
    z = adir[...,2]/maxnum
    
    axis.plot_surface(x,y,z,alpha=0.2)
    
    tocube(axis)
    
    for adir in ['x','y','z']:
        
        theta,phi = da.AnglesofHemisphere(adir,steps)
        alldirections = da.DirectionVector(theta,phi)
        
        dsdo = da.DifferentialCrossSection(F,n0,alldirections,p=p,k=k,const=False)
        dsdo /= max(dsdo)
        
        ux,uy = da.GetDirectionCosine(adir,steps)
        
        if adir == 'x':
            ctf = axis.contourf(dsdo,ux,uy,50,zdir=adir,cmap=cm.hot,offset=-1)
        elif adir == 'y':
            ctf = axis.contourf(ux,dsdo,uy,50,zdir=adir,cmap=cm.hot,offset=1) 
        elif adir == 'z':
            ctf = axis.contourf(ux,uy,dsdo,50,zdir=adir,cmap=cm.hot,offset=-1)
        
    
    cb = afig.colorbar(ctf,ax=axis,use_gridspec=True,shrink=0.5)
    cb.set_label("Scaled Absolute Electric Field Squared",size=15)
    cb.set_ticks([0,0.25,0.5,0.75,1])
    
    afig.tight_layout()
    
    return
