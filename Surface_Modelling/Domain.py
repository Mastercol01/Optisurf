#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions and classes related to specification and
construction of a domain over which an explicit test surface can be defined.

"""

#%%                 IMPORTATION OF LIBRARIES
import numpy as np
from itertools import product

#%%                 DEFINITION OF FUNCTIONS
          

def extract_rect_params(x_vals, y_vals):
    
    """
    This function allows to compute some relevant parameters from x_vals
    and y_vals array.
    
    Parameters
    ----------
    x_vals : 1D numpy.array of floats
        One dimensional numpy array of floats, with the discretised x-axis
        values. Must have equal-spacing dsicretization and be monotonic-increasing.
        
    y_vals : 1D numpy.array of floats
        One dimensional numpy array of floats, with the discretised y-axis
        values. Must have equal-spacing dsicretization and be monotonic-increasing.
        
    Returns
    -------
    res : list of floats
        List of floats holding the following parameters in order:
        x_dim, y_dim, dx, dy
        
    Raises
    ------
    1) Exception : "Domain must have equal number of nodes in x and y."
    
    
    """
    
    if(len(x_vals)!=len(y_vals)):
        raise Exception("Domain must have equal number of nodes in x and y.")
    
    # Coordinate dimensions.    
    x_dim = x_vals[-1] - x_vals[0]
    y_dim = y_vals[-1] - y_vals[0]
        
    # x-axis and y-axis spacing.
    dx = x_dim/(len(x_vals) - 1)
    dy = y_dim/(len(y_vals) - 1)
        
    return [x_dim, y_dim, dx, dy]







def rect_c2i(x, y, x_dim, y_dim, dx, dy):
    
    """
    Method converts from coordinate to index, when having the same coordinate
    system as in 'RectDomain'.
        
    Returns
    -------
    x : float
        Value of x.
        
    y : float
        Value of y.
        
    x_dim : float 
        Actual dimensions of the x-axis (i.e, the total range of x.)
        
    y_dim : float 
        Actual dimensions of the y-axis (i.e, the total rage of y.)
        
    dx : float
        Spacing of x-axis.
        
    dy : float
        Spacing of y-axis.
        
        
    
    Parameters
    ----------
    i: int
        Row index corresponding to value x,y.   
    
    j: int
        Column index corresponding to value x,y.  
        
    """
    j = np.around((x + x_dim/2 )/dx)
    i = np.around((y + y_dim/2 )/dy)
    return i,j 






def rect_c2v(x, y, tolx, toly, vertices):
    
    """
    Function converts from coordinates to vertices.
    
    Parameters
    ----------
    
    x : float
        Value of x.
        
    y : float
        Value of y.
        
    tolx : float
        Tolerance allowed in the value of x, for determining its vertex.
        
    toly : float
        Tolerance allowed in the value of y, for determining its vertex.
        
    vertices : 2D numpy.array of floats.
        Array of mesh vertices.
        
    Returns
    -------
    vertex_num : list of ints
        Number of the vertices computed by the function which the coordinates
        x,y could belong to.
    
    """
    
    x_min, x_max= x-tolx, x+tolx
    y_min, y_max= y-toly, y+toly
    
    vertex0 = set(np.where( np.logical_and( vertices[:,0]>x_min , vertices[:,0]<x_max) )[0])
    vertex1 = set(np.where( np.logical_and( vertices[:,1]>y_min , vertices[:,1]<y_max) )[0])
    
    vertex_num = list(vertex0.intersection(vertex1))
    
    return vertex_num


def make_XYpows(X, Y, max_powX=3, max_powY=3):
    
    """
    Compute dictionary containing all combiantions of the values of (X**i)*(Y**j), 
    where X,Y are meshgrid x-axis and y-axis of a given domain and i, j range 
    0 to max_powX and 0 to max_powY, respectively.
    
    Parameters
    ----------
    X : float, numpy.array of floats or 2D numpy.array of floats
        x value, array of x values or Mesgrid x-axis values.
    
    Y : float, numpy.array of floats or 2D numpy.array of floats
        y value, array of y values or Mesgrid y-axis values.
    
    max_powX : int
        Maximum power of x to compute.
    
    max_powY : int
       Maximum power of x to compute.
    
    
    Returns
    -------
    XYpows : dict of 2D numpy.array of floats
        Dict with the following key-value pair format.
        
        Keys : Values
        -------------
        (i, j) : 2D numpy.array of floats
            Term equal to (X**i)*(Y**j).
    
    """
    
    try:
        XYpows = {(0, 0) : np.ones(X.shape), (1, 0): X, (0, 1): Y}
    except AttributeError:
        XYpows = {(0, 0) : 1, (1, 0): X, (0, 1): Y}
        
        
    for i in range(2, max_powX + 1):
        XYpows[(i, 0)] = XYpows[(i-1, 0)]*X
        
    permutations = list(product(range(max_powX + 1), range(max_powY + 1)))
    permutations = [i for i in permutations if i not in  XYpows.keys()]
    
    for i, j in permutations:
        XYpows[(i, j)] = XYpows[(i, j-1)]*Y
        
    return XYpows







#%%                 DEFINITION OF CLASSES

class RectDomain:
    
    """
    This class holds all relevant info regarding the rectangular domain over
    which test surfaces will be generated.
    
    Parameters
    ----------
    dx : float
        x-axis spacing. Must be non-ngeative.
        
    dy : float
        y-axis spacing. Must be non-ngeative.
        
    dims : 3-tuple of floats
        3-tuple specifying the maximum dimensions of the domain in x, y and z
        respectively. Must be non-ngeative.
        
    max_powX : int
        Maximum power of X coordinates used for the definition of the 
        'self.XYpows' attribute. Must be non-ngeative.
        
    max_powY : int
        Maximum power of Y coordinates used for the definition of the 
        'self.XYpows' attribute. . Must be non-ngeative.
    
        
    
    Raises
    ------
    1) Exception: "Domain must have equal number of nodes in x and y."
    
    
    
    """
    
    def __init__(self, dx, dy, dims=(1,1,1), max_powX=10, max_powY=10):
        
        # We store the 'intended' x-axis and y-axis spacing.
        self.dx_intended = dx
        self.dy_intended = dy
        
        # We store the coordinate dimensions.
        self.xyz_dims = dims
        self.x_dim = dims[0]
        self.y_dim = dims[1]
        self.z_dim = dims[2]
        
        # We compute the number of nodes required to discretise the x-axis
        # and y-axis, according to the dimensions and spacing given.
        self.nj = round(self.x_dim/self.dx_intended + 1)
        self.ni = round(self.y_dim/self.dy_intended + 1)
        
        # The code requires a square matrix of nodes.
        if self.ni != self.nj:
            raise Exception("Domain must have equal number of nodes in x and y.")
            
        # Compute the meshgrid of idxs.
        self.i = np.arange(self.ni)
        self.j = np.arange(self.nj)
        self.J, self.I = np.meshgrid(self.j, self.i)

        # We compute node dimesnions. I.e, we compute the shape of an 
        # matrix of nodes.
        self.shape = (self.ni, self.nj)
        
        # We compute the number of vertices.
        self.num_nodes = self.ni*self.nj 
        
        # We compute the x-axis and y-axis arrays.
        self.x = np.linspace(-self.x_dim/2, self.x_dim/2, self.nj)
        self.y = np.linspace(-self.y_dim/2, self.y_dim/2, self.ni)
        
        # We compute the actual x-axis and y-axis spacing.
        self.dx = self.x_dim/(self.nj - 1)
        self.dy = self.y_dim/(self.ni - 1)
        
        # NOTE: it is necessary to recompute the x-axis and y-axis spacing, 
        # since i_dim and j_dim can't be fractional.
        
        # We compute the meshgrid of values for the domain as well as its powers.
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.XYpows = make_XYpows(self.X, self.Y, max_powX, max_powY)

        
        
        def i2c(self, i, j): 
            
            """
            Method converts from index to coordinate:
            
            Parameters
            ----------
            i: int
                Row index.    
            
            j: int
                Column index.
                
            Returns
            -------
            x : float
                Value of x at i,j.
                
            y : float
                Value of y at i,j.
            
            """
            
            x = self.X[i,j]
            y = self.Y[i,j]
            
            return x,y
        
        def c2i(self, x, y):
            
            """
            Method converts from coordinate to index:
                
            Returns
            -------
            x : float
                Value of x.
                
            y : float
                Value of y.
            
            Parameters
            ----------
            i: int
                Row index corresponding to value x,y.   
            
            j: int
                Column index corresponding to value x,y.  
                
            """
            
            j = np.around( (x + self.x_dim/2 )/self.dx )
            i = np.around( (y + self.y_dim/2 )/self.dy )
            return i,j 
        
        
#%%            EXAMPLES

if __name__ == '__main__':
    
    # We initialize a RectDomain Object
    xy_plane = RectDomain(dx = 0.017, dy = 0.017, dims = (1,1,1))
    
    # Let us have a look at some relevant and useful attributes
    x,   y                      = xy_plane.x,  xy_plane.y
    X,   Y                      = xy_plane.X,  xy_plane.Y
    dx, dy                      = xy_plane.dx, xy_plane.dy
    dx_intended, dy_intended    = xy_plane.dx_intended, xy_plane.dy_intended
    i, j                        = xy_plane.i, xy_plane.j
    I, J                        = xy_plane.I, xy_plane.J
    XYpows                      = xy_plane.XYpows
    
    

    
    



        
    
                   




