#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions related to computing an initial population
of polynomial surfaces for later optimization.

"""
#%%                  IMPORTATION OF LIBRARIES

import numpy as np
import Surface_Modelling.Mesh as mg
import Surface_Modelling.taylor_funcs as tayf

#%%                 IMPORTATION OF LIBRARIES


def create_init_pop_3x3(RectDomain_obj, pop_size, distribution="normal", scale=3, max_Zrange=1, Kconds=(10**-4, 0.01), Pconds=(1/3, 0.05)):
    
    """
    This function computes an initial population of surfaces represented by 
    3x3 Taylor polynomials, with zero gaussian curvature.
    
    Parameters
    ----------
    RectDomain_obj : DomainDef.RectDomain object
        Object specifying the domain of optimization.
        
    pop_size : int
        Number of surfaces to initialize over the domain.
        
    distribution : str
        Distribution from which the surface's taylor coefficients are to be
        sampled. It can either be "normal" or "uniform".
        
    scale : float
        When 'distribution' is "normal", it specifies its standard deviation.
        When 'distribution' is "uniform" it specifies the upper and lower 
        limits likes so: [-scale/2, scale/2]. As such, in both cases it 
        should be a positive number.
        
    max_Zrange : float
        Maximum desired range of the taylor polynomial.
        
    Kconds : 2-tuple of float
        Taylor polynomial gaussian curvature conditions. The function rejects
        any polynomial surface where a 'Kconds[1]' fraction of (equally-spaced) 
        sampled points from said polynomial surface have a value of gaussian 
        curvature greater than 'Kconds[0]'.
        
    Pconds : 2-tuple of float
        Taylor polynomial principal curvature conditions. The function rejects
        any polynomial surface where a 'Pconds[1]' fraction of (equally-spaced) 
        sampled points from said polynomial surface have a value of absolute
        principal curvature (any of the two) greater than 'Pconds[0]'.
        
    Returns
    -------
    init_population : dict of Mesh objects.
        Initial population of surfaces. Each key contains a Mesh object of
        the produced surface. These Mesh objects, have also an extra attribute
        called 'self.poly', which holds the coefficient matrix representation
        of the surface.
        
    
        
    Notes
    -----
    1) Both "normal" and "uniform" have a mean eauql to zero.
    
    """
    
    i = 0
    population = {}
    while len(population) < pop_size:
        

        if distribution == "normal":
            C10 =  np.random.normal(loc = 0, scale = scale)
            C01 =  np.random.normal(loc = 0, scale = scale)
            C20 =  np.random.normal(loc = 0, scale = scale)
            C02 =  np.random.normal(loc = 0, scale = scale)

                
        elif distribution == "uniform":
            C10 =  np.random.uniform(low = -scale/2, high = scale/2)
            C01 =  np.random.uniform(low = -scale/2, high = scale/2)
            C20 =  np.random.uniform(low = -scale/2, high = scale/2)
            C02 =  np.random.uniform(low = -scale/2, high = scale/2)
                            
                
        # if len(population) <= pop_size//2:
        #     C20 = abs(C20)
        #     C02 = abs(C02)
            
        # elif len(population) > pop_size//2:
        #     C20 = -abs(C20)
        #     C02 = -abs(C02)

        C02 = C20
        
        
        C11 = 2*np.sqrt(C20*C02)
        
    
        poly = np.array([ [0,    C01,    C02],
                          [C10,  C11,    0  ],
                          [C20,    0,    0, ]])
            
        
        # We normalize the surface when its range is greater than the maximum
        # range allowed.
        Z = tayf.polyeval(poly, RectDomain_obj.XYpows)
        
        Zmin = Z.min()
        Zrange = Z.max() - Zmin
        poly[0,0] -= Z.min()
        
        if Zrange > max_Zrange:
            poly *= np.random.uniform(low=0.1, high=max_Zrange)/Zrange

        
        Z = tayf.polyeval(poly = poly, XYpows = RectDomain_obj.XYpows)
        H, K = tayf.poly_curvature(poly = poly, XYpows = RectDomain_obj.XYpows)
        
        Pmin = H - np.sqrt(H**2 - K)
        Pmax = H + np.sqrt(H**2 - K)
        Pmaxmin = np.maximum(abs(Pmin), abs(Pmax))
        
        if (Pmaxmin > Pconds[0]).sum()/Pmaxmin.size > Pconds[0]:
            continue
        
        if (abs(K) > Kconds[0]).sum()/K.size > Kconds[1]:
            continue
        
        
        
        surf_obj = mg.Mesh(X=RectDomain_obj.X, Y=RectDomain_obj.Y, Z=Z)
        surf_obj.poly = poly
        
        population[i] = surf_obj
        
        i += 1
        
    return population

#%%        EXAMPLES

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import Surface_Modelling.Domain as dom
    
    
    # Initialize RectDomain object.
    xy_plane = dom.RectDomain(dx = 0.05, dy = 0.05, dims = (1,1,1))
    
    # Create inital population.
    inital_pop =\
    create_init_pop_3x3(RectDomain_obj  = xy_plane,
                        pop_size        = 30,
                        distribution    = "normal",
                        scale           = 3, 
                        max_Zrange      = 1,
                        Kconds          = (10**-4, 0.01),
                        Pconds          = (1/3, 0.05))

    # Plot each of the generated surfaces and the curvatures.
    for key, Mesh_obj in inital_pop.items(): 
        
        config = {"title" : f"Mesh No. {key}"}
        Mesh_obj.visualize(config = config)
        
        H, K = tayf.poly_curvature(Mesh_obj.poly, xy_plane.XYpows)
        
        Pmin = H - np.sqrt(H**2 - K)
        Pmax = H + np.sqrt(H**2 - K)
        Pmaxmin = np.maximum(abs(Pmin), abs(Pmax))
        
        # Plot the gaussian curvature of the original surface.
        fig = plt.figure(figsize=(12,8))
        plt.contourf(xy_plane.X, xy_plane.Y, K)
        plt.title(f"Mesh No. {key}: abs(Gaussian) curvature")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()
        plt.show()
        
        # Plot of the absolute value of mean curvature of the original surface.
        fig = plt.figure(figsize=(12,8))
        plt.contourf(xy_plane.X, xy_plane.Y, Pmaxmin)
        plt.title(f"Mesh No. {key}: max(abs(Minimum Principal Curvature), abs(Maximum Principal Curvature))")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()
        plt.show()
    
    
    


