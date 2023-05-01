#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions related to handling polynomial surfaces
in coefficient matrix representation.

--- COEFFICIENT MATRIX REPRESENTATION ---

A matrix of shape (3,3) can represent a polynomial surface up to order (2,2),
where the first number is the maximum power in x, while the second number is the 
maximum power in y. An example would be:

poly = [[C00, C01, C02],
        [C10, C11, C12],
        [C20, C21, C22]]

Here, Cij, represents the polynomial's coefficient corresponding to the term
(X**i)*(Y**j), where X,Y is the x-axis and y-axis of the domain over which
the surface is defined. As such, written out explictely. The above poly,
 would encode the following polynomial surface:
    
poly = C00 + C10*(X) + C01*(Y) + C11*(X)*(Y)    + 
       C20*(X**2) + C02*(Y**2) + C21*(X**2)*(Y) +
       C12*(X)*(Y**2) + C22*(X**2)*(Y**2)


"""
#%%              IMPORTATION OF LIBRARIES

import numpy as np
import sympy as sp
from math import factorial
from scipy.optimize import curve_fit
import Surface_Modelling.Domain as dom
from numpy.polynomial.polynomial import polyder


#%%              DEFINITION OF FUNCTIONS FOR NUMERICAL OPERATIONS
    
def polydiff(poly, order):
    
    """
    Computes the derivative of order (n,m) of the polynomial (coefficient 
    matrix representation) inputted.
    
    Parameters
    ----------
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        The i,j element of poly holds the coefficient of the X**i * Y**j
        term. It should be, preferably, a square matrix.
        
    order : tuple of ints
        Order of the differentiation to be performed. First element of tuple
        indicates the order of differentiation in x, while the second element
        inidcates the order of differentiation in y.
        
    Returns
    -------
    poly_D : 2D numpy.array of floats
        Differentiated polynomial, in coefficient matrix representation.
        
    Notes
    -----
    1) Axis 0 is associated with x-axis and axis 1 is associated with the y-axis.
    
    """
    
    poly_D = np.zeros(poly.shape)
    
    diff = polyder(polyder(poly, m=order[0], axis=0), m=order[1], axis=1)
    
    poly_D[0:diff.shape[0], 0:diff.shape[1]] = diff
    
    return poly_D



def polymul(poly1, poly2):
    
    """
    Computes the multiplication of the 2 inputted polynomials of orders
    (n1, m1) and (n2, m2), respectively, in coefficient matrix representation.
    
    Parameters
    ----------
    poly1 : 2D numpy.array of floats
        First polynomial, in coefficient matrix representation, to multiply.
        It is of order (n1, m1). The i,j element of poly holds the coefficient
        of the X**i * Y**j term. It should be, preferably, a square matrix.
        
    poly2 : 2D numpy.array of floats
        Second polynomial, in coefficient matrix representation, to multiply.
        It is of order (n2, m2). The i,j element of poly holds the coefficient
        of the X**i * Y**j term. It should be, preferably, a square matrix.
        
    Returns
    -------
   poly3_global : 2D numpy.array of floats
        polynomial of order (n1*n2, m1*m2), in coefficient matrix representation,
        resulting from the multiplication of poly1 and poly2.
        
    Notes
    -----
    1) Axis 0 is associated with x-axis and axis 1 is associated with the y-axis.
    
    """
    
    n1, m1 = poly1.shape
    n2, m2 = poly2.shape
    n3, m3 = n1*n2, m1*m2
    
    poly3_local  = np.zeros((n3, m3))
    poly3_global = np.zeros((n3, m3))
    
    for i in range(n1):
        for j in range(m1):
            
            poly3_local[i:n2+i, j:m2+j] = poly1[i,j]*poly2  
            poly3_global += poly3_local
            poly3_local *= 0
            
    return poly3_global
            
            
    
    
def polyeval(poly, XYpows):
    
    """
    This function evaluates a polynomial, in coefficient matrix representation,
    over a domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    poly_eval : 2D numpy.array of floats
        Two dimensional numpy array of floats holding the values of the 
        inputted polynomial, evaluated over the domain.
    
    """
    
    poly_eval = 0
    for i in range(poly.shape[0]):
        for j in range(poly.shape[1]):
            
            if poly[i,j]==0:
                continue
            else:
                poly_eval += poly[i,j]*XYpows[i,j]
                
                
    if isinstance(poly_eval, int):
       poly_eval = np.zeros(XYpows[0,0].shape) 
            
    return poly_eval    





def poly_unit_normals(poly, XYpows):
    
    """
    This function computes the unit normals a polynomial, in coefficient matrix 
    representation, evaluated over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    unit_normals = 3D numpy.array of floats
        Unit normals of  the polynomial evaluated at each point specified in 
        the domain. unit_normals[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of the unit normal vector (correspondingly)
        evaluated over the whole domain.
        
    Notes
    -----
    1) The vector field of surface unit normals is also known as the Gauss Map.
    see https://youtu.be/e-erMrqBd1w fro more info.
        
        
    """
    
    # For a surface with parametrization r(x,y) = [x, y, f(x,y)],
    # the normal vector can be computed as [-∂f/∂x, -∂f/∂y, 1]. 
    # Normalizing said vector gives us the unit vector.
    
    # We retrieve the matrix dimensions of the xy_plane.
    try:
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError:
            i_dim, j_dim = (1, 1)
    
    # We compute the x and y derivatives, analytically.
    neg_poly_Dx = -polydiff(poly, order=(1,0))
    neg_poly_Dy = -polydiff(poly, order=(0,1))

    # We evaluate the derivatives and compute an array of unit normals.
    unit_nvecs = np.ones((i_dim, j_dim, 3))
    unit_nvecs[:,:,0] = polyeval(neg_poly_Dx, XYpows)
    unit_nvecs[:,:,1] = polyeval(neg_poly_Dy, XYpows)
    unit_nvecs /=\
    np.linalg.norm(unit_nvecs, axis=-1).reshape((i_dim, j_dim, 1))
    
    return unit_nvecs



def poly_orthonormal_frames(poly, XYpows):
    
    """
    This function computes a local orthonormal reference frame for a parametric
    surface of the form rvec = [x, y, f(x,y)], where f(x,y) is a polynomial
    in coefficient matrix representation, and returns the values of each
    base vector over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    n = 3D numpy.array of floats
        Unit normals of  the polynomial evaluated at each point specified in 
        the domain. unit_normals[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of the unit normal vector (correspondingly)
        evaluated over the whole domain.
        
    e1 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e2, evaluated at each point of the
        specified in the domain. e1[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain.
        
    e2 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e1, evaluated at each point of the
        specified in the domain. e2[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain.
        
    Notes
    -----
    1) Together n, e1 and e2 form a local/moving orthonormal frame of
    reference.
        
    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)
        
        
    # We compute all the derivatives of the polynomial analytically.
    poly_Dx = polydiff(poly, order=(1,0))
    poly_Dy = polydiff(poly, order=(0,1))
    
    # Then we evaluate said derivatives over the domain in question.
    Zx = polyeval(poly_Dx, XYpows)
    Zy = polyeval(poly_Dy, XYpows)
    
    if i_dim == j_dim == 1:
        Zx = np.array(Zx).reshape(i_dim, j_dim)
        Zy = np.array(Zy).reshape(i_dim, j_dim)
    
    # We then compute ∂rvec/∂x and normalize it.
    e1  = np.stack([np.ones((i_dim, j_dim)), np.zeros((i_dim, j_dim)), Zx]).T
    e1 /= np.linalg.norm(e1, axis=-1).reshape(i_dim, j_dim, 1)

    # We then compute ∂rvec/∂y and normalize it.    
    e2 = np.stack([np.zeros((i_dim, j_dim)), np.ones((i_dim, j_dim)), Zy]).T
    e2 /= np.linalg.norm(e2, axis=-1).reshape(i_dim, j_dim, 1) 
        
    # We compute the surface's unit normals at each point.
    n = poly_unit_normals(poly, XYpows)

    return n, e1, e2




def poly_orthonormal_frames2(poly, XYpows):
    
    """
    This function computes a local orthonormal reference frame for a parametric
    surface of the form r_vec = [x, y, f(x,y)], where f(x,y) is a polynomial
    in coefficient matrix representation, and returns the values of each
    base vector over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    n = 3D numpy.array of floats
        Unit normals of  the polynomial evaluated at each point specified in 
        the domain. unit_normals[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of the unit normal vector (correspondingly)
        evaluated over the whole domain.
        
    e1 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e2, evaluated at each point of the
        specified in the domain. e1[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain.
        
    e2 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e1, evaluated at each point of the
        specified in the domain. e2[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain.
        
    Notes
    -----
    1) Together n, e1 and e2 form a local/moving orthonormal frame of
    reference.
        
    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)

    
    # We compute the surface's unit normals at each point.
    n = poly_unit_normals(poly, XYpows)
    
    # We compute an orthonormal vector to the local unit surface normal at
    # each point. This is done as follows: A vector 'e1' is orthogonal to the
    # unit normal 'n' if it satisfies nx*e1x + ny*e1y + nz*e1z == 0. Where nx,
    # ny, nz are the unit normal's components and e1x, e1y, e1z are the 
    # the components of 'e1'. Then, choosing e1x = 1 and e1y = 0, we find that
    # e1z = -nx/nz. and, with this, e1 = (1, 0, -nx/nz); which is clearly 
    # orthogonal to 'n'. Having done this we simply divide e1 by its magnitude 
    # in toder to normalize it. The end result is an orthonormal vector to unit 
    # normal.
    
    e1 = np.zeros((i_dim, j_dim, 3))
    e1[:,:,0] = 1
    e1[:,:,2] = - n[:,:,0]/n[:,:,2]
    e1 /= np.linalg.norm(e1, axis=2).reshape(i_dim, j_dim, 1)
    
    # In order to compute a vector that is orthonormal to 'n' and to 'e1' we
    # just simply calcualte the cross product of n with e1.
    
    e2 = np.cross(n, e1)
        
    return n, e1, e2






def poly_curvature(poly, XYpows):
    
    """
    This function computes the mean and gaussian curvature of a polynomial,
    in coefficient matrix representation, evaluated over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    H : 2D numpy.array of floats
        Mean curvature of the polynomial, evaluated over the domain.
        
    K : 2D numpy.array of floats
        Gaussian curvature of the polynomial, evaluated over the domain.
        
    """
    
    # We compute all the derivatives of the polynomial analytically.
    poly_Dx = polydiff(poly, order=(1,0))
    poly_Dy = polydiff(poly, order=(0,1))
    poly_Dx2 = polydiff(poly, order=(2,0))
    poly_Dy2 = polydiff(poly, order=(0,2))
    poly_DxDy = polydiff(poly, order=(1,1))
    
    # Then we evaluate said derivatives over the domain in question.
    Zx = polyeval(poly_Dx, XYpows)
    Zy = polyeval(poly_Dy, XYpows)
    Zxx = polyeval(poly_Dx2, XYpows)
    Zyy = polyeval(poly_Dy2, XYpows)
    Zxy = polyeval(poly_DxDy, XYpows)
    
    # We compute the Mean (H) and Gaussian curvatures (K) using the 
    # 'Monge patch' formula. See: https://mathworld.wolfram.com/MongePatch.html
    # This formula is just a special case of the general formula, applicable
    # when r_vec = [x, y, f(x,y)]
    
    K = (Zxx*Zyy - Zxy**2)/(1 + Zx**2 + Zy**2)**2
    
    num_H = (1 + Zy**2)*Zxx - 2*Zx*Zy*Zxy + (1 + Zx**2)*Zyy
    dem_H = 2*(1 + Zx**2 + Zy**2)**1.5
    H = num_H/dem_H
    
    return H, K 


def poly_df(poly, XYpows):
    
    """
    This function computes the partial derivatives of a parametric surface of
    the form r_vec = [x, y, f(x,y)], where f(x,y) is a polynomial
    in coefficient matrix representation, and returns the values of the said
    partial derivatives evaluated over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    rvec_Dx = 3D numpy.array of floats
        Derivative with respect to x of the position vector parametrizing the
        surface (analytically it has the form [1,0,df/dx]), evaluated at each
        point of the specified in the domain.
        
    rvec_Dx = 3D numpy.array of floats
        Derivative with respect to x of the position vector parametrizing the
        surface (analytically it has the form [0,1,df/dy]), evaluated at each
        point of the specified in the domain.

    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)
    
    # We compute all the derivatives of the polynomial analytically.
    poly_Dx = polydiff(poly, order=(1,0))
    poly_Dy = polydiff(poly, order=(0,1))
    
    # We compute the partial derivatives of the position vector parametrizing
    # the surface for each point of the specified domain.
    rvec_Dx = np.zeros((i_dim, j_dim, 3))
    rvec_Dx[:,:,0] = 1
    rvec_Dx[:,:,2] = polyeval(poly_Dx, XYpows)
    
    rvec_Dy = np.zeros((i_dim, j_dim, 3))
    rvec_Dy[:,:,1] = 1
    rvec_Dy[:,:,2] = polyeval(poly_Dy, XYpows)
    
    return rvec_Dx, rvec_Dy
    

def poly_dn(poly, XYpows):
    
    """
    This function computes the partial derivatives of the unit normals vector 
    field of a parametric surface of the form r_vec = [x, y, f(x,y)], where 
    f(x,y) is a polynomial in coefficient matrix representation. It then returns 
    the values of the said partial derivatives evaluated over the specified 
    domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    unit_nvecs_Dx = 3D numpy.array of floats
        Derivative with respect to x of the surface's unit normals vector field,
        evaluated at each point of the specified in the domain.
        
    unit_nvecs_Dy = 3D numpy.array of floats
        Derivative with respect to y of the surface's unit normals vector field,
        evaluated at each point of the specified in the domain.

    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)
    
    # We compute all the derivatives of the polynomial analytically.
    poly_Dx = polydiff(poly, order=(1,0))
    poly_Dy = polydiff(poly, order=(0,1))
    poly_Dx2 = polydiff(poly, order=(2,0))
    poly_Dy2 = polydiff(poly, order=(0,2))
    poly_DxDy = polydiff(poly, order=(1,1))

    # Then we evaluate said derivatives over the domain in question.
    Zx = polyeval(poly_Dx, XYpows)
    Zy = polyeval(poly_Dy, XYpows)
    Zxx = polyeval(poly_Dx2, XYpows)
    Zyy = polyeval(poly_Dy2, XYpows)
    Zxy = polyeval(poly_DxDy, XYpows)
    
    
    # We compute the surface-normals vector field as well as its magnitude.
    # Analytically it is given by: n_vec = [-∂f/∂x, -∂f/∂y, 1].
    nvecs = np.zeros((i_dim, j_dim, 3))
    nvecs[:,:,0] = -Zx
    nvecs[:,:,1] = -Zy
    nvecs[:,:,2] = 1
        
    nvecs_norm = np.linalg.norm(nvecs, axis=2).reshape((i_dim, j_dim, 1))
    
    unit_nvecs = nvecs/nvecs_norm
    
    # We compute the partial derivatives of the surface-normals vector field.
    # Analythically these are given by: ∂n_vec/∂x = [-d^2f/∂x^2, -∂^2f/∂xdy, 0]
    # ∂n_vec/∂y = [-∂^2f/∂x∂y, -∂^2f/∂y^2, 0].
    
    nvecs_Dx = np.zeros((i_dim, j_dim, 3))
    nvecs_Dx[:,:,0] = -Zxx
    nvecs_Dx[:,:,1] = -Zxy
    
    nvecs_Dy = np.zeros((i_dim, j_dim, 3))
    nvecs_Dy[:,:,0] = -Zxy
    nvecs_Dy[:,:,1] = -Zyy
    
    # We compute the partial derivatives of the surface's unit normals vector
    # field. Analythically these are given by: 
        
    # ∂unit_n_vec/∂x = 
    # (|n_vec|*∂n_vec/∂x - dot(n_vec, ∂n_vec/∂x)*unit_n_vec)/(|n_vec|^2) 
    unit_nvecs_Dx = nvecs_norm*nvecs_Dx 
    unit_nvecs_Dx -= (nvecs*nvecs_Dx).sum(axis=2).reshape((i_dim, j_dim, 1))*unit_nvecs
    unit_nvecs_Dx /= nvecs_norm**2
    
    
    # ∂unit_n_vec/∂y = 
    # (|n_vec|*∂n_vec/∂y - dot(n_vec, ∂n_vec/∂y)*unit_n_vec)/(|n_vec|^2)
    unit_nvecs_Dy = nvecs_norm*nvecs_Dy 
    unit_nvecs_Dy -= (nvecs*nvecs_Dy).sum(axis=2).reshape((i_dim, j_dim, 1))*unit_nvecs
    unit_nvecs_Dy /= nvecs_norm**2
    

    return unit_nvecs_Dx, unit_nvecs_Dy



def poly_shape_operator(poly, XYpows):
    
    """
    This function computes the shape operator of a parametric surface of the
    form r_vec = [x, y, f(x,y)], where f(x,y) is a polynomial in coefficient 
    matrix representation. It then returns the values of the said shape operator
    evaluated over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    shape_operator = 3D numpy.array of floats
        Shape operator of f(x,y), evaluated at each point of the specified in 
        the domain. Note that:
            
            1) shape_operator[:,:,0] gives the S11 component of the shape
               operator evaluated at each point of the specified the domain.
               
            2) shape_operator[:,:,1] gives the S21 component of the shape
               operator evaluated at each point of the specified the domain.
              
            3) shape_operator[:,:,2] gives the S12 component of the shape
               operator evaluated at each point of the specified the domain.
               
            4) shape_operator[:,:,3] gives the S22 component of the shape
               operator evaluated at each point of the specified the domain.
        

    Notes
    -----
    1) Let J(h) be the jacobian matrix of function h, where h is of the type
       h : R^n → R^m. That is, h takes an n-th dimensional vector as input and
       produces an m-th dimesnional vector as output. Then J(h) is a matrix
       of size m x n, where i-th, j-th element of J(h) is given by: ∂hi/∂xj,
       i.e, the partial derivative of the i-th component of h with
       respect to j-th variable.
       

    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)
    
    # Given the Jacobian matrix of rvec and of unit_nvec, the following
    # relationship must hold for all points:
        
    # Jmat(rvec) @ S = Jmat(unit_nvec)
    
    # Where Jmat(rvec) is the Jacobian matrix of rvec, S is the shape
    # operator matrix, Jmat(unit_nvec) is the Jacobian matrix of the
    # surface's unit normal vector field and @ denotes matrix multiplication.
    # This system can be written more explicitely as:
        
    # [[a11, a12],    [[S11, S12],       [[b11, b12],
    #  [a21, a22],  @  [S21, S22]]  ==    [b21, b22],
    #  [a31, a32]]                        [b31, b32]]
    
    # We see that Jmat(rvec) and Jmat(unit_nvec) are 3x2 matrices while
    # S is a 2x2 matrix. Now, using the definition of the Jacobian, for a 
    # surface with parametrization rvec = [x, y, f(x,y)], the above expression
    # simplifies to:
        
    # [[1,     0],    [[S11, S12],       [[b11, b12],
    #  [0,     1],  @  [S21, S22]]  ==    [b21, b22],
    #  [a31, a32]]                        [b31, b32]]
    
    # Then, carrying out the multiplication we find the following equality:
        
    # [[S11,                             S12],        [[b11, b12],
    #  [S21,                             S22],  ==     [b21, b22],
    #  [a31*S11 + a32*s21, a31*s12 + a32*s22]]         [b31, b32]]
    
    # From which it is easy to see that:
        
    #   S == [[S11, S12], ==  [[b11, b12], 
    #         [S21, S22]]      [b21, b22]] 
     
    
    # We need not to worry about equalities a31*S11 + a32*s21 == b31 and 
    # a31*s12 + a32*s22 == b32, since it can be shown that these are 
    # automatically satisfied with the values for S chosen above.
    
    # We compute the partial derivatives of the surface's unit normal vector
    # field. These derivatives make up the components of Jmat(unit_nvec).
    unit_nvecs_Dx, unit_nvecs_Dy = poly_dn(poly, XYpows)
    
    # We create the shape operator array and store the value of the shape 
    # operator for each point in the specified domain. In order to do this,
    # we flatten the shape operator matrix into the following form:
    # [S11, S21, S12, S22] and stroe it in a third dimesnion.
    
    shape_operator = np.zeros((i_dim, j_dim, 4))
    shape_operator[:,:,0:2] = unit_nvecs_Dx[:,:,:-1]
    shape_operator[:,:,2:4] = unit_nvecs_Dy[:,:,:-1]

    return shape_operator


def poly_principal_values(poly, XYpows):
    
    """
    This function computes the principal curvatures and principal directions of 
    a parametric surface of the form r_vec = [x, y, f(x,y)], where f(x,y) is a 
    polynomial in coefficient matrix representation. It then returns the values
    of the said curvatures/directions evaluated over the specified domain.
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    min_kappa = 2D numpy.array of floats
        Minimum principal curvature of f(x,y), evaluated at each point of the 
        specified the domain.
        
    max_kappa = 2D numpy.array of floats
        Maximum principal curvature of f(x,y), evaluated at each point of the 
        specified the domain.
        
    min_dir = 3D numpy.array of floats
        Minimum principal direction of f(x,y), evaluated at each point of the 
        specified the domain. min_dir[:,:,v], for v = 0,1,2 gives the x,y,z 
        component (respectively) of the tangent vector that describes the
        minimum principal direction.
        
    max_dir = 3D numpy.array of floats
        Maximum principal direction of f(x,y), evaluated at each point of the 
        specified the domain. min_dir[:,:,v], for v = 0,1,2 gives the  x,y,z 
        component (respectively) of the tangent vector that describes the
        maximum principal direction.
        
    Notes
    -----
    1) For more info on the maths behind this procedure, check out: 
       https://youtu.be/e-erMrqBd1w

    """
    
    # We retrieve the matrix dimensions of the xy_plane.
    try: 
        i_dim, j_dim = XYpows[(0,0)].shape
    except AttributeError: 
        i_dim, j_dim = (1, 1)
    
    # We compute the partial derivatives of the surface. With these we can 
    # construct its Jacobian and, as such, foward the 2D principal direction
    # vectors onto 3D. 
    rvec_Dx, rvec_Dy = poly_df(poly, XYpows)   
    
    # We compute the surface's shape operator.
    S = poly_shape_operator(poly, XYpows)

    # We compute the shape operator matrix for each point and store it in a
    # list. In this way we may passed it to numpy.linalg.eig and compute all
    # eigenvalues and eigenvectors at once.
    S_locals = [ np.array([[S[i,j,0], S[i,j,2]], [S[i,j,1], S[i,j,3]]]) 
                 for i in range(i_dim) for j in range(j_dim) ]
    
    # We compute the eigenvalues and eigenvectors of each local shape operator
    # matrix, these define the principal curvatures and directions.
    eigenvals, eigenvecs = np.linalg.eig(S_locals)
    eigenvals[abs(eigenvals) < 1e-12] = 0
    
    max_kappa = np.zeros((i_dim, j_dim))
    max_dir = np.zeros((i_dim, j_dim, 3))
    min_kappa = np.zeros((i_dim, j_dim))
    min_dir = np.zeros((i_dim, j_dim, 3))
    
    # We classify the curvatures, checking whether it is the maximum 
    # or minimum.
    c=0
    for i in range(i_dim):
        for j in range(j_dim):
            
            kappa0, kappa1 = eigenvals[c,:]
            
            # We foward the principal directions from 2D onto 3D using the 
            # Jacobian matrix of the surface. We then normalize the vectors.
            Jdf = np.vstack((rvec_Dx[i,j,:], rvec_Dy[i,j,:])).T
            dirs = Jdf @ eigenvecs[c,:,:]
            dirs /= np.linalg.norm(dirs, axis=0)
            
            if kappa0 >= kappa1:
                
               max_kappa[i,j] = kappa0  
               min_kappa[i,j] = kappa1
               max_dir[i,j,:] = dirs[:,0]
               min_dir[i,j,:] = dirs[:,1]
               
            else:
               max_kappa[i,j] = kappa1  
               min_kappa[i,j] = kappa0
               max_dir[i,j,:] = dirs[:,1]
               min_dir[i,j,:] = dirs[:,0]

            c+=1        
    
    return min_kappa, max_kappa, min_dir, max_dir
    

def poly_principal_orthonormal_frames(poly, XYpows):
    
    """
    This function computes a local orthonormal reference frame for a parametric
    surface of the form r_vec = [x, y, f(x,y)], where f(x,y) is a polynomial
    in coefficient matrix representation, and returns the values of each
    base vector over the specified domain. The particularity with this function 
    is that it uses the principal direction vectors of the surface for 
    constructing the reference frame.
    
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    Returns
    -------
    
    n = 3D numpy.array of floats
        Unit normals of  the polynomial evaluated at each point specified in 
        the domain. unit_normals[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of the unit normal vector (correspondingly)
        evaluated over the whole domain.
        
    e1 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e2, evaluated at each point of the
        specified in the domain. e1[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain. It is equal to the minimum principal direction of the 
        surface.
        
    e2 = 3D numpy.array of floats
        Vectors that are orthonormal to n and to e1, evaluated at each point of the
        specified in the domain. e2[:,:,k] (where k = 0,1,2) gives the value of 
        the x, y and z components of e1 (correspondingly) evaluated over the
        whole domain. It is equal to the maximum principal direction of the 
        surface.
        
    Notes
    -----
    1) Interestingly, the principall curvature directions of a surface are not 
       only orthogonal to each other but also to the normal of the surface. As
       such, by normalizing these 3, one can construct a very natural moving/
       local orthonomral reference frame.
        
    """
    
    # We compute the surface's unit normals at each point.
    n = poly_unit_normals(poly, XYpows)
    
    _, _, e1, e2 = poly_principal_values(poly, XYpows)

    return n, e1, e2


def poly_plane_projection(poly, x0, y0, config = "default" ):
    
    """
    This function computes an orthonormal reference frame/local coordinate 
    system for the point on the surface r0 = [x0, y0, f(x0, y0)]. It
    then computes the normal projection of f(x,y) onto the plane defined by said
    local coordinate system. The proected function is called g(s,t) and it
    is calculated over the rectangular domain [config["s2lim"][0], 
    config["s2lim"][1]] X [config["t2lim"][0], config["t2lim"][1]]. g(s,t) can 
    basically be thought of as f(x,y), but in terms of the local refrence 
    frame's coordinates s and t.
    
    
    Parameters
    ----------
    
    poly : 2D numpy.array of floats
        Polynomial, in coefficient matrix representation, to differentiate.
        Must be a square matrix.
        
    x0 : float
        x-coordinate of the origin of the reference frame, in the
        global coordinate system.
        
    y0 : float
        y-coordinate of the origin of the reference frame, in the
        global coordinate system.
        
    config : "default" or dict
        If "default", default system values are used. If dict, it must contain
        some or all of the following key-value pairs (for those whose value is 
        not specified here, the default value will be used.):
            
            Keys : Values
            -------------
            "xlim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the x-coordinate
                in the global coordinate system. Default is (-0.5, 0.5).
                
            "ylim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the y-coordinate
                in the global coordinate system. Defualt is (-0.5, 0.5).
                
            "zlim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the z-coordinate
                in the global coordinate system. Defualt is (0, 1).
                
            "s1lim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the s-coordinate
                for exploration, in the local coordinate system. Defualt is (-0.5, 0.5).
                
            "t1lim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the t-coordinate
                for exploration, in the local coordinate system. Defualt is (-0.5, 0.5).
                
            "s2lim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the s-coordinate
                in the local coordinate system. Default is (-0.065, 0.065).
                
            "t2lim" : 2-tuple of float
                Allowed lower and upper bounds (in that order) of the t-coordinate
                in the local coordinate system. Default is (-0.065, 0.065).
                
            "m" : int
                Number of samples to generate for s and t cooridnates. 
                A greater value of m indicates that the mesh is more refined. 
                Defualt is 2000.

                
        
    Returns
    -------
    
    new_S2 = numpy.array of float
        Sample values of the s-coordinate in the local reference frame. Its
        corresponding basis vector is e1.
        
    new_T2 = numpy.array of float
        Sample values of the t-coordinate in the local reference frame. Its
        corresponding basis vector is e2.
        
    new_lmbda = numpy.array of float
        Sample values of g(s,t) in the local reference frame. Its corresponding
        basis vector is -n. This is basically 
        
    Notes
    -----
    1) Since the surface varies, that means that the normal projection of f(x,y)
       onto the local plane will also vary. As such, in general, the values
       of new_S2 and new_T2  won't be evenly spaced.
        
    """
    
    
#    global r0, n, e1, e2, S1, T1, rp1, unit_rp1, r0_rp1, rs, rp2, S2, T2
#    global lmbda, r0_rp2, x_logic, y_logic, z_logic, s_logic, t_logic, logic

    # Default configurations dictionary.
    config_ = {
    "xlim"  : (-0.5, 0.5),
    "ylim"  : (-0.5, 0.5),
    "zlim"  : (0, 1),
    "s1lim" : (-0.5, 0.5),
    "t1lim" : (-0.5, 0.5),
    "s2lim" : (-0.065, 0.065),
    "t2lim" : (-0.065, 0.065),
    "m"     : 2000}
    
    if isinstance(config, dict):
        for key, val in config.items():
            config_[key] = val 
            
    m = config_["m"]
    max_powX = poly.shape[0] - 1
    max_powY = poly.shape[1] - 1
    
    # LIST OF SYMBOLS, DESCRIPTIONS AND DEFINITIONS
    
    # r0  : Position vector from the origin of the global coordinate system to
    #       the chosen point on the surface.
    # n   : Outward-facing unit normal vector of the surface at r0.
    # e1  : Unitary vector orthogonal to n and to e2.
    # e2  : Unitary vector orthogonal to n and to e1.
    
    # x, y, z : Coordinates of the global coordinate system.
    # i, j, k : Orthonormal basis for the local coordinate system.
    # i and j span the global plane.
    # i, j and k span the global space.
    
    # s,  t,  lmbda : Coordinates of the local coordinate system.
    # e1, e2, -n    : Orthonormal basis for the local coordinate system.
    # e1 and e2 span the local plane.
    # e1, e2 and -n span the local space.
    
    # rp1 : Position vector from the origin of the local coordinate system
    #       to a point on the local plane  that is to be projected,
    #       in the direction -k, onto the polynomial surface.
     
    # r0_rp1 : Position vector from the origin of the global coordinate system
    #          to a point on the local plane  that is to be projected,
    #          in the direction -k, onto the polynomial surface.      
    
    # rp2 : Position vector from the origin of the local coordinate system
    #       to a point on the local plane that is to be projected,
    #       in the direction -n, onto the polynomial surface.
     
    # r0_rp2 : Position vector from the origin of the global coordinate system
    #          to a point on the local plane  that is to be projected,
    #          in the direction -n, onto the polynomial surface.      
    
    # unit_rp1 : Normalized version of rp1.
    
    # rs : Position vector from the origin of the global coordinate system to
    #      a given on the surface. It is equal to rp1 after it is projected,
    #      in the direction -k, onto the surface.
    

    
    
    # We compute the position vector r0. r0 will constitute the origin of the 
    # local plane.
    XYpows_0 = dom.make_XYpows(x0, y0, max_powX, max_powY)
    r0 = np.array([x0, y0, polyeval(poly, XYpows_0)])
    
    # We compute the local reference frame/local coordinate system.
    n, e1, e2 = poly_orthonormal_frames(poly, XYpows_0)
    n, e1, e2 = n.flatten(), e1.flatten(), e2.flatten()
    
    # We mesh the local plane using limits for s and t greater than those
    # required by the logical conditions below. This is done because when 
    # computing rp2 from rp1, it is probable that no all vectors will map in
    # such a way as to yield the desired range of s and t in the conditions 
    # below. The solution for this is to compute more values for rp1, further
    # away from the local origin than is allowed by the conditions, compute rp2
    # from said values and just filter out those which do not comply.
    s1 = np.linspace(config_["s1lim"][0], config_["s1lim"][1], m)
    t1 = np.linspace(config_["t1lim"][0], config_["t1lim"][1], m)
    S1, T1 = np.meshgrid(s1, t1)
    
    # We compute the rp1 vectors for the meshed local plane, using i,j,k.
    rp1 = np.zeros((m, m, 3))
    rp1[:,:,0] = S1*e1[0] + T1*e2[0]
    rp1[:,:,1] = S1*e1[1] + T1*e2[1]
    rp1[:,:,2] = S1*e1[2] + T1*e2[2]

    # We compute the unitary version of rp1.
    unit_rp1 = rp1/np.linalg.norm(rp1, axis=2).reshape(m,m,1)
    
    # We compute r0_rp1.
    r0_rp1 = r0 + rp1        
    
    # We compute rs.
    rs = r0_rp1.copy()
    XYpows_1 = dom.make_XYpows(r0_rp1[:,:,0], r0_rp1[:,:,1], max_powX, max_powY)
    rs[:,:,2] = polyeval(poly, XYpows_1)
    
    # We compute rp2 = dot((rs - r0), unit_rp1)*unit_rp1
    rp2 = ((rs-r0)*unit_rp1).sum(axis=2).reshape(m, m, 1)*unit_rp1

    # Compute the components of rp2 in local coordinates.
    S2 = e1[0]*rp2[:,:,0] + e1[1]*rp2[:,:,1] + e1[2]*rp2[:,:,2]
    T2 = e2[0]*rp2[:,:,0] + e2[1]*rp2[:,:,1] + e2[2]*rp2[:,:,2]
    
    # We compute lmbda = dot(rp1 + r0 - rs, n)
    lmbda = (r0_rp1 - rs)
    lmbda = n[0]*lmbda[:,:,0] + n[1]*lmbda[:,:,1] + n[2]*lmbda[:,:,2]

    # We compute r0_rp2.
    r0_rp2 = rp2 + r0

    # We identify the points on the local plane, using global coordinates,
    # that lie within the specified domain of x.
    x_logic = np.logical_and(r0_rp2[:,:,0] >= config_["xlim"][0],
                             r0_rp2[:,:,0] <= config_["xlim"][1] )
    
    # We identify the points on the local plane, using global coordinates,
    # that lie within the specified domain of y.
    y_logic = np.logical_and(r0_rp2[:,:,1] >= config_["ylim"][0],
                             r0_rp2[:,:,1] <= config_["ylim"][1] )
    
    # We identify the points on the local plane, using global coordinates,
    # that lie within the specified domain of z.
    z_logic = np.logical_and(r0_rp2[:,:,2] >= config_["zlim"][0],
                             r0_rp2[:,:,2] <= config_["zlim"][1] )
    
    # We identify the points on the local plane, using local coordinates,
    # that lie within the specified domain of s.
    s_logic = np.logical_and(S2 >= config_["s2lim"][0],
                             S2 <= config_["s2lim"][1] )
    
    # We identify the points on the local plane, using local coordinates,
    # that lie within the specified domain of t.
    t_logic = np.logical_and(T2 >= config_["t2lim"][0],
                             T2 <= config_["t2lim"][1] )
    
    # A valid point on the plane (as per the needs of our analysis)
    # satisfies all 5 conditions stated above.
    logic = np.logical_and(np.logical_and(x_logic, y_logic), z_logic)
    logic = np.logical_and(np.logical_and(s_logic, t_logic), logic)
    
    # We select only the valid points.
    new_S2 = S2[logic]
    new_T2 = T2[logic]
    new_lmbda = lmbda[logic]

    
    return new_S2, new_T2, new_lmbda





def fitting_func1(data, C00, C01, 
                        C10, C11):
    
    """
    Helper function for 'poly_fit'.
    It helps evaluate a polynomial of order (1,1) forthe inputted data,
    in a way in which it can be utilized by the 'poly_fit' function.
    """
    
    x = data[0]
    y = data[1]
    
    local_poly = np.array([C00, C01,
                           C10, C11]).reshape(2, 2)
    
    xy_pows = dom.make_XYpows(X = x, Y = y, 
                              max_powX = local_poly.shape[0] - 1,
                              max_powY = local_poly.shape[1] - 1)

    return polyeval(local_poly, XYpows = xy_pows)



def fitting_func2(data, C00, C01, C02, 
                        C10, C11, C12, 
                        C20, C21, C22):
    
    """
    Helper function for 'poly_fit'.
    It helps evaluate a polynomial of order (2,2) forthe inputted data,
    in a way in which it can be utilized by the 'poly_fit' function.
    """
    
    x = data[0]
    y = data[1]
    
    local_poly = np.array([C00, C01, C02,
                           C10, C11, C12,
                           C20, C21, C22]).reshape(3,3)
    
    xy_pows = dom.make_XYpows(X = x, Y = y, 
                              max_powX = local_poly.shape[0] - 1,
                              max_powY = local_poly.shape[1] - 1)

    return polyeval(local_poly, XYpows = xy_pows)



def fitting_func3(data, C00, C01, C02, C03, 
                        C10, C11, C12, C13,
                        C20, C21, C22, C23, 
                        C30, C31, C32, C33):
    
    """
    Helper function for 'poly_fit'.
    It helps evaluate a polynomial of order (3,3) forthe inputted data,
    in a way in which it can be utilized by the 'poly_fit' function.
    """
    
    x = data[0]
    y = data[1]
    
    local_poly = np.array([C00, C01, C02, C03,
                           C10, C11, C12, C13,
                           C20, C21, C22, C23,
                           C30, C31, C32, C33]).reshape(4,4)
    
    xy_pows = dom.make_XYpows(X = x, Y = y, 
                              max_powX = local_poly.shape[0] - 1,
                              max_powY = local_poly.shape[1] - 1)

    return polyeval(local_poly, XYpows = xy_pows)



def fitting_func4(data, C00, C01, C02, C03, C04,
                        C10, C11, C12, C13, C14, 
                        C20, C21, C22, C23, C24, 
                        C30, C31, C32, C33, C34, 
                        C40, C41, C42, C43, C44):
    
    """
    Helper function for 'poly_fit'.
    It helps evaluate a polynomial of order (4,4) forthe inputted data,
    in a way in which it can be utilized by the 'poly_fit' function.
    """
    
    x = data[0]
    y = data[1]
    
    local_poly = np.array([C00, C01, C02, C03, C04,
                           C10, C11, C12, C13, C14, 
                           C20, C21, C22, C23, C24, 
                           C30, C31, C32, C33, C34,
                           C40, C41, C42, C43, C44]).reshape(5,5)
    
    xy_pows = dom.make_XYpows(X = x, Y = y, 
                             max_powX = local_poly.shape[0] - 1,
                             max_powY = local_poly.shape[1] - 1)

    return polyeval(local_poly, XYpows = xy_pows)


def fitting_func5(data, C00, C01, C02, C03, C04, C05,
                        C10, C11, C12, C13, C14, C15, 
                        C20, C21, C22, C23, C24, C25,
                        C30, C31, C32, C33, C34, C35,
                        C40, C41, C42, C43, C44, C45,
                        C50, C51, C52, C53, C54, C55):
    """
    Helper function for 'poly_fit'.
    It helps evaluate a polynomial of order (5,5) forthe inputted data,
    in a way in which it can be utilized by the 'poly_fit' function.
    """
    
    x = data[0]
    y = data[1]
    
    local_poly = np.array([C00, C01, C02, C03, C04, C05,
                           C10, C11, C12, C13, C14, C15, 
                           C20, C21, C22, C23, C24, C25,
                           C30, C31, C32, C33, C34, C35,
                           C40, C41, C42, C43, C44, C45,
                           C50, C51, C52, C53, C54, C55]).reshape(6,6)
    
    xy_pows = dom.make_XYpows(X = x, Y = y, 
                              max_powX = local_poly.shape[0] - 1,
                              max_powY = local_poly.shape[1] - 1)

    return polyeval(local_poly, XYpows = xy_pows)




def poly_fit(X, Y, Z, max_pow = 2):
    
    """
    Fit polynomial surface to data.
    
    Parameters
    ----------
    X : 2D numpy.array of floats
      Mesgrid x-axis values.
      
    Y : 2D numpy.array of floats
      Mesgrid y-axis values.
    
    Z : 2D numpy.array of floats
      Data to fit. It is defined over the domain specified
      by 'X' and 'Y'.
      
     max_pow : int
         Maximum order of the polynomial to fit.
      
    Returns
    -------
    fitted_coeffs : 2D numpy.array of floats
        Fitted polynomial in matrix coefficient form.
    
    """
    
    if max_pow == 1:
        fitted_coeffs, covariance =\
        curve_fit(fitting_func1, [X.flatten(), Y.flatten()], Z.flatten()) 

    elif max_pow == 2:
        fitted_coeffs, covariance =\
        curve_fit(fitting_func2, [X.flatten(), Y.flatten()], Z.flatten()) 
        
    elif max_pow == 3:
        fitted_coeffs, covariance =\
        curve_fit(fitting_func3, [X.flatten(), Y.flatten()], Z.flatten()) 
        
    elif max_pow == 4:
        fitted_coeffs, covariance =\
        curve_fit(fitting_func4, [X.flatten(), Y.flatten()], Z.flatten()) 
        
    elif max_pow == 5:
        fitted_coeffs, covariance =\
        curve_fit(fitting_func5, [X.flatten(), Y.flatten()], Z.flatten()) 

    
    return fitted_coeffs.reshape((max_pow + 1, max_pow + 1))
    


#%%          DEFINITION OF FUNCTIONS FOR SYMBOLIC OPERATIONS



def create_symbolic_matrix(shape, symbol="C"):
    
    """
    This function creates a 2D numpy.array of symbolic constants of 
    specified shape.
    
    Parameters
    ----------
    
    shape : 2-tuple of int
        Shape of the array to bre created. Eg.: shape = (2,2) creates a 2x2
        matrix, while shape = (3,4) creates a 3x4 matrix.
        
    symbol : str
        Symbol by which the elements of the new matrix will be represented.
        Default is 'C', which means that a 2x2 matrix would, for instance,
        have elements 'C00', 'C01', 'C10' and 'C11'. Were the symbol = 'A',
        for example, the elements of the same matrix would now be represented
        as: 'A00', 'A01', 'A10' and 'A11'.
        
    Returns
    -------
    
    symbolic_mat : 2D numpy.array of sympy.Symbol objects
        Matrix of symbolic constants.
        
    """
    
    symbolic_mat = np.zeros(shape).astype(object)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            symbolic_mat[i,j] = sp.Symbol(f"{symbol}{i}{j}")
            
    return symbolic_mat




def sym_zeros(shape):
    """
    Thsi function is analogous to numpy.zeros() function. The difference is
    this function produces a zero-matrix for symbolic operations.
    
    Parameters
    ----------
    
    shape : 2-tuple of int
        Shape of the array to bre created. Eg.: shape = (2,2) creates a 2x2
        matrix, while shape = (3,4) creates a 3x4 matrix.

    Returns
    -------
    
    symbolic_mat : 2D numpy.array of sympy.Zero objects
        Matrix of symbolic zeros.
        
    """
    symbolic_mat = 0*create_symbolic_matrix(shape, symbol="C")
    return symbolic_mat



def sym_polydiff(sym_poly, order):
    
    """
    Computes the derivative of order (n,m) of the polynomial (symbolic 
    coefficient matrix representation) inputted, for symbolic operations.
    
    Parameters
    ----------
    sym_poly : 2D numpy.array of sympy.Symbol objects
        Polynomial, in symbolic coefficient matrix representation,
        to differentiate. The i,j element of poly holds the coefficient of 
        the X**i * Y**j term. It should be, preferably, a square matrix.
        
    order : tuple of ints
        Order of the differentiation to be performed. First element of tuple
        indicates the order of differentiation in x, while the second element
        inidcates the order of differentiation in y.
        
    Returns
    -------
    sym_poly_diffd : 2D numpy.array of sympy.Symbol objects
        Differentiated polynomial, in symbolic coefficient matrix
        representation.
        
    Notes
    -----
    1) Axis 0 is associated with x-axis and y-axis is associated with axis 1.
    
    """
    
    ndx, ndy = order
    sym_poly_diffd = sym_zeros(sym_poly.shape)
    
    for i in range(ndx, sym_poly.shape[0]):
        for j in range(ndy, sym_poly.shape[1]):
            
            new_coef = sym_poly[i,j]
            new_coef *= int(factorial(i)/factorial(i - ndx))
            new_coef *= int(factorial(j)/factorial(j - ndy))
            sym_poly_diffd[i - ndx, j - ndy] = new_coef
    
    return sym_poly_diffd




def sym_polymul(sym_poly1, sym_poly2):
    
    """
    Computes the multiplication of the 2 inputted polynomials of orders
    (n1, m1) and (n2, m2), respectively, in symbolic coefficient matrix 
    representation.
    
    Parameters
    ----------
    poly1 : 2D numpy.array of sympy.Symbol objects
        First polynomial, in symbolic coefficient matrix representation,
        to multiply. It is of order (n1, m1). The i,j element of poly holds
        the coefficient of the X**i * Y**j term. It should be, preferably, 
        a square matrix.
        
    poly2 : 2D numpy.array of sympy.Symbol objects
        Second polynomial, in symbolic coefficient matrix representation,
        to multiply. It is of order (n1, m1). The i,j element of poly holds
        the coefficient of the X**i * Y**j term. It should be, preferably, 
        a square matrix.
        
    Returns
    -------
   poly3_global : 2D numpy.array of sympy.Symbol objects
        polynomial of order (n1*n2, m1*m2), in symbolic coefficient matrix
        representation, resulting from the multiplication of poly1 and poly2.
        
    Notes
    -----
    1) Axis 0 is associated with x-axis and y-axis is associated with axis 1.
    
    """
    
    n1, m1 = sym_poly1.shape
    n2, m2 = sym_poly2.shape
    n3, m3 = n1*n2, m1*m2
    
    poly3_local = sym_zeros((n3, m3))
    poly3_global = sym_zeros((n3, m3))
    
    for i in range(n1):
        for j in range(m1):
            
            poly3_local[i:n2+i, j:m2+j] = sym_poly1[i,j]*sym_poly2  
            poly3_global += poly3_local
            poly3_local *= 0
            
    return poly3_global


    


#%%                       EXAMPLES: NUMERIC

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import Surface_Modelling.Domain as dom
    from scipy.interpolate import griddata
    # We create the rectangular over which a surface will be defined. 
    xy_plane = dom.RectDomain(dx = 0.01, dy = 0.01, dims = (1,1,1))
    
    # Let us extract the XYpows attribute. This is attribute is they to
    # evaluating polynomial surface relatively fast.
    XYpows = xy_plane.XYpows
    

    # Initialize polynomial surface in matrix coeff representation.
    poly = np.array([ [0, 1, 1],
                      [1, 0, 5],
                      [1, 5, 2] ])
    
    poly = np.array([ [0,  2, -1],
                      [1,  4,  0],
                      [-4, 0,  0] ])*0.05
    
    
    # Evaluate the polynomial over the entire defined domain.
    Z = polyeval(poly, XYpows)
    
    # Plot said surface.
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xy_plane.X, xy_plane.Y, Z)
    ax.set_title("Original Poly")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    # Evaluate the polynomial at a single point not defined in the domained.
    Zpt = polyeval(poly, XYpows = dom.make_XYpows(np.pi/5, -np.pi/10))
    print(Zpt)
    
    # We differentiate the polynomial in coef-matrix form.
    diffd_poly = polydiff(poly, order=(1,1))
    
    # Evaluate the differentiated polynomial over the entire defined domain.
    Zdiffd = polyeval(diffd_poly, XYpows)
    
    # Plot said surface.
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xy_plane.X, xy_plane.Y, Zdiffd)
    ax.set_title("Differentiated Poly")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    # We compute the curvatures of the original polynomial.
    H, K = poly_curvature(poly, XYpows)
    kappa_max =  H + np.sqrt(H**2 - K)
    kappa_min =  H - np.sqrt(H**2 - K)
    Pmaxmin = np.maximum(abs(kappa_max), abs(kappa_min))
    
    
    # Plot the gaussian curvature of the original surface.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(xy_plane.X, xy_plane.Y, K)
    plt.title("Original Poly: Gaussian curvature")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()
    
    # Plot of the absolute value of mean curvature of the original surface.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(xy_plane.X, xy_plane.Y, H)
    plt.title("Original Poly: Mean curvature")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()
    
    # Plot of the absolute value of mean curvature of the original surface.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(xy_plane.X, xy_plane.Y, Pmaxmin)
    plt.title("max(|kappa_max|, |kappa_min|)")
    plt.suptitle("Maximum Absolute Principal curvature of Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.grid()
    plt.show()
    
    
#%% 
    
    
    # We define 2 new polynomials  in order to test polymul
    poly_1 = np.array([ [1, 0, 3],
                        [0, 0, 0],
                        [2, 0, 0] ])
    
    poly_2 = np.array([ [0, 2, 0],
                        [1, 0, 0],
                        [0, 0, 0] ])
    
    
    poly_3 = polymul(poly_1, poly_2)
    
    print(poly_1)
    print(poly_2)
    print(poly_3)
    
    
    # Compute unit vectors of surface over the whole specified domain.
    unit_vecs =  poly_unit_normals(poly, XYpows)
    
    # Again, if we want to compute the just one unit vector at one point, we can.
    unit_vec = poly_unit_normals(poly, XYpows = dom.make_XYpows(np.pi/5, -np.pi/10))
    print(unit_vec)
    
    # Compute an orthonormal frame over the whole specified domain.
    n, e1, e2 = poly_orthonormal_frames(poly, XYpows)
    
    # Compute shape operator over the whole specified domain.
    shape_operators = poly_shape_operator(poly, XYpows)
    
    # Compute, minimum and maximum principal curvatures and directions of
    # over the whole specified domain.
    min_kappa, max_kappa, min_dir, max_dir = poly_principal_values(poly, XYpows)
    
    # Compute an orthonormal frame over the whole specified domain, using the
    # surface normals and the principal directions.
    n_, e1_, e2_ = poly_principal_orthonormal_frames(poly, XYpows)
    
    
    # Compute plane projection at point x0, y0.
    config  = {
    "xlim"  : (-0.5, 0.5),
    "ylim"  : (-0.5, 0.5),
    "zlim"  : (0, 1),
    "s1lim" : (-0.5, 0.5),
    "t1lim" : (-0.5, 0.5),
    "s2lim" : (-0.065, 0.065),
    "t2lim" : (-0.065, 0.065),
    "m"     : 2000}
    
    
    new_S2, new_T2, new_lmbda =\
    poly_plane_projection(poly, x0 = 0, y0 = 0, config = config )
    
    
    # Let us define a new function and fit the closest polynomial approximation:
    new_Z = np.sin(3*xy_plane.Y)*np.cos(3*xy_plane.X)    
    
    # Plot said surface. 
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xy_plane.X, xy_plane.Y, new_Z)
    plt.title("New Poly")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
        
    # Fit surface polynomial
    fitted_coeffs = poly_fit(xy_plane.X, xy_plane.Y, new_Z, max_pow = 4)
    fitted_Z = polyeval(fitted_coeffs, XYpows)
    
    # Plot fitted surface. 
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xy_plane.X, xy_plane.Y, fitted_Z)
    plt.title(f"Fitted Poly: Polynomial order {(fitted_coeffs.shape[0]-1, fitted_coeffs.shape[1]-1)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    
    
#%%                       EXAMPLES: SYMBOLIC
                    
        
    # We initialize a symbolic polynomial in coef-matrix form.
    sym_poly = create_symbolic_matrix(shape=(3,3), symbol="C")
    print(sym_poly)
    print("-------------")
    
    # We differentiate the symbolic polynomial in coef-matrix form.
    diffd_sym_poly = sym_polydiff(sym_poly, order=(1,1))
    print(diffd_sym_poly)
    print("-------------")
    
    # We define 2 new polynomials  in order to test sym_polymul
    poly_1 = create_symbolic_matrix(shape=(2,2), symbol="C")
    poly_1[1,1] *=0
    
    poly_2 = create_symbolic_matrix(shape=(2,2), symbol="A")


    poly_3 = sym_polymul(poly_1, poly_1)
    poly_4 = sym_polymul(poly_1, poly_2)
    
    print(poly_1)
    print("-------------")
    print(poly_2)
    print("-------------")
    print(poly_3)
    print("-------------")
    print(poly_4)
            
    
    
#%%

    poly = np.array([[ 0.62401339, -0.93556008,  -0.43358944],
                     [ 0.05675702,  0.07461949,   0.],
                     [-0.00321045,  0.,           0.]])
    
    Z = polyeval(poly, xy_plane.XYpows)
    H, K = poly_curvature(poly, xy_plane.XYpows)
    
    # Plot fitted surface. 
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xy_plane.X, xy_plane.Y, Z)
    plt.title(f"Fitted Poly: Polynomial order {(fitted_coeffs.shape[0]-1, fitted_coeffs.shape[1]-1)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    # Plot of the absolute value of mean curvature of the original surface.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(xy_plane.X, xy_plane.Y, 2*abs(H))
    plt.title("max(|kappa_max|, |kappa_min|)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()


    
    config = {"m" : 4000}
    new_S2, newT2, new_lmbda = poly_plane_projection(poly   = poly, 
                                                     x0     =  0.4,
                                                     y0     = -0.4,
                                                     config = config )
    new_lmbda -= new_lmbda.min()
    
    
    
    
    
    # We compute how f(x,y) looks from the perspective of the solar cell being 
    # situated at tangent point normal to the mold surface.
    new_S2, new_T2, new_lmbda = poly_plane_projection(poly=poly, x0=0.4, y0=-0.4)   
    new_lmbda -= new_lmbda.min()
    
    n = 100
    S, T = np.meshgrid(np.linspace(-0.0625, 0.0625, n),
                       np.linspace(-0.0625, 0.0625, n))
    

    g_st = griddata(points = np.stack([new_S2, new_T2]).T, 
                    values = new_lmbda, 
                    xi = np.stack([S.flatten(), T.flatten()]).T,
                    method = 'cubic')
    
    g_st_2D = g_st.reshape((n, n))
    
    


    # We plot the local surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
    ax.plot_surface(S, T, g_st_2D)
    ax.set_zlim(0, 0.006)
    ax.set_xlabel("S [m]")
    ax.set_ylabel("T [m]")
    ax.set_zlabel("λ [m]")
    ax.set_title("Local surface")
    ax.view_init(30, 60)
    plt.show()

    # We also plot a countour plot for conviniency.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(S, T, g_st_2D)
    plt.colorbar()
    plt.xlabel("S [m]")
    plt.ylabel("T [m]")
    plt.title("Local surface")
    plt.show()    
    


    # We attempt to compute the local surface's polynomial representation.
    STpows           = dom.make_XYpows(S, T, 5, 5)
    g_st_2D_hat_poly = poly_fit(S, T, g_st_2D, max_pow=3)
    g_st_2D_hat      = polyeval(g_st_2D_hat_poly, STpows)        


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
    ax.plot_surface(S, T, g_st_2D_hat)
    ax.set_zlim(0, 0.005)
    ax.set_xlabel("S [m]")
    ax.set_ylabel("T [m]")
    ax.set_zlabel("λ [m]")
    ax.set_title("Local surface hat")
    ax.view_init(30, 60)
    plt.show()

    # We also plot a countour plot for conviniency.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(S, T, g_st_2D_hat)
    plt.colorbar()
    plt.xlabel("S [m]")
    plt.ylabel("T [m]")
    plt.ylabel("T [m]")
    plt.title("Local surface hat")
    plt.show() 

    # We also plot the absolute error.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(S, T, abs(g_st_2D - g_st_2D_hat))
    plt.colorbar()
    plt.xlabel("S [m]")
    plt.ylabel("T [m]")
    plt.ylabel("T [m]")
    plt.title("Local surface hat: Abs Error")
    plt.show() 


    H, K = poly_curvature(g_st_2D_hat_poly, STpows)

    # We also plot a countour plot for conviniency.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(S, T, K)
    plt.colorbar()
    plt.xlabel("S [m]")
    plt.ylabel("T [m]")
    plt.title("Local surface hat: Gaussian Curvature")
    plt.show() 

    # We also plot a countour plot for conviniency.
    fig = plt.figure(figsize=(12,8))
    plt.contourf(S, T, H)
    plt.colorbar()
    plt.xlabel("S [m]")
    plt.ylabel("T [m]")
    plt.title("Local surface hat: Mean Curvature")
    plt.show() 
    
    
    
 










    