#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions and classes related to the optimization
of surfaces represented as taylor polynomials. Currently supported is the 
optimization of developable taylor polynomials of order (2,2).
"""

#%%                     IMPORTATION OF LIBRARIES
import os
import math
import warnings
import itertools
import time as tm
import numpy as np
import matplotlib.pyplot as plt
import Surface_Modelling.Mesh as mg
import Surface_Modelling.Domain as dom
import Surface_Modelling.taylor_funcs as tayf
import Ambience_Modelling.auxiliary_funcs as aux

#%%                 DEFINITION OF FUNCTIONS


def num_top_surfs(pop_size):
    """
    Compute the number of top surfaces required to at least generate a 
    population of surfaces as big as pop_size via surface pairing.

    Parameters
    ----------
    pop_size : int
        Number of surfaces in the population.

    Returns
    -------
    num_top_surfs : int
        Minimum number of top-surfaces required to replenish the surface
        population.

    """
    
    num_top_surfs = 0
    while math.comb(num_top_surfs, 2) < pop_size:
        num_top_surfs += 1
    return num_top_surfs

    

def evolve(population, rank_list, XYpows, mutprob=0.15, mutweight=0.15, max_Zrange=1):
    
    """
    Compute new population of surfaces via an established algorithim that
    simulates sexual reproduction in animals.
    

    Parameters
    -------
    population : dict of mg.Mesh objects.
        Population of surfaces to be reproduced.
        
    rank_list : list of tuples
        List of tuples containing the key for each surface in population dict
        as well as its computed fitness score. List must be sorted from best 
        fitness score to worst fitness score.
        
    XYpows : dict of 2D numpy.array of floats
        Dictionary storing the products of the meshgird values of X and Y
        elevated to the i-th and j-th power respectively. In other words,
        the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
        where X,Y are the meshgrid values of the x and y coordinates over
        the domain in question.
        
    mutprob : float
        Probabilty that a given child surface will mutate. Must be a number
        between 0 and 1.
        
    mutweight : float
        Standard deviation of the normal distribution (with mean equal to 0),
        from which the coefficients of the mutation polynomial are sampled.
        
    max_Zrange : float
        Maximum range allowed for a child surface.
        

    Population
    ----------
    new_population : dict of mg.Mesh objects.
        Offspring population of parent surfaces.

    """
    
    
    # We compute the mating pairs for the reproduction of surfaces.
    pop_size = len(population)
    num_top_surfs_ = num_top_surfs(pop_size)
    surf_mating_pairs = list(itertools.combinations(range(num_top_surfs_), 2))
    surf_mating_pairs = surf_mating_pairs[0:pop_size-1]
    
    # We initialize the new population dict. The best surface of the previous
    # generations gets a direct pass to the next generation.
    new_population = { 0:population[rank_list[0][0]] }
    
    # We retrieve and compute some very important Domain parameters (these 
    # do not change surface to surface).
    X = population[0].X
    Y = population[0].Y

    
    
    # We retrieve the order of the polynomials being processed and compute
    # the condition for which the components of the polynomial must be zero
    # in order for it to have zero gaussian curvature.
    (n, n) = population[0].poly.shape

    c = 1
    for dad_surf_rank, mom_surf_rank in surf_mating_pairs:
        
        # We get the keys of the surfaces (in population dict) whose ranks
        # correspond to dad_surf_rank and mom_surf_rank, respectively.
        dad_surf_idx = rank_list[dad_surf_rank][0]
        mom_surf_idx = rank_list[mom_surf_rank][0]
        
        # We get the polynomial coefficients of both parent surfaces.
        dad_poly = population[dad_surf_idx].poly
        mom_poly = population[mom_surf_idx].poly
        
        #------------ MATING AND MUTATION -------------------------
        
        # Parent surfaces interchange roughly half of their genes on average,
        # to conceive a child surface.
        parent_polys = (dad_poly, mom_poly)
        idx = round(np.random.uniform())
        
        if (n,n) == (3,3):
            C10 = parent_polys[idx][1,0]
            C01 = parent_polys[1-idx][0,1]
            C20 = parent_polys[idx][2,0]
            C02 = parent_polys[1-idx][0,2]
            
            # There's a mutprob % chance that a child surface will suffer a mutation 
            # after being conceived.
            if np.random.uniform() < mutprob:
                C10 += np.random.normal(loc = 0, scale = mutweight)
                C01 += np.random.normal(loc = 0, scale = mutweight)
                C20 += np.random.normal(loc = 0, scale = mutweight)
                C02 += np.random.normal(loc = 0, scale = mutweight)
                
            if abs(C10) < 10**-6: C10 = 0
            if abs(C01) < 10**-6: C01 = 0
            if abs(C20) < 10**-6: C20 = 0
            if abs(C02) < 10**-6: C02 = 0
            
            sgn_C20 = np.sign(C20)
            sgn_C02 = np.sign(C02)
            if C20!=0 and C02!=0 and sgn_C20!=sgn_C02:
                rand_sgn = (sgn_C20, sgn_C02)
                C20 = rand_sgn[idx]*abs(C20)
                C02 = rand_sgn[idx]*abs(C02)
                
                
                
            C11 = 2*np.sqrt(C20*C02)
            child_poly = np.array([ [0,    C01,    C02],
                                    [C10,  C11,    0  ],
                                    [C20,    0,    0, ]])
                        

        #------------- NORMALIZATION ------------------------
        
        # We evaluate the child polynomial.
        child_Z = tayf.polyeval(child_poly, XYpows)
        child_Zrange = child_Z.max() - child_Z.min()
        
        if child_Zrange > max_Zrange:
            dad_Z = tayf.polyeval(dad_poly, XYpows)
            mom_Z = tayf.polyeval(mom_poly, XYpows)
            dad_Zrange = dad_Z.max() - dad_Z.min()
            mom_Zrange = mom_Z.max() - mom_Z.min()
            
            child_poly[0,0] -= child_Z.min()
            child_poly *= 0.5*(mom_Zrange + dad_Zrange)/child_Zrange
        else:
            child_poly[0,0] -= child_Z.min()

            
        #----------- NEW POPULATION-------------------------
            
        new_child_Z = tayf.polyeval(child_poly, XYpows) 
        
        surf_obj = mg.Mesh(X, Y, new_child_Z)
        
        surf_obj.poly = child_poly
            
        new_population[c] = surf_obj
        
        c+=1
            
        
    return new_population    

    


def fitness_function1(Mesh_obj, params):
    
    """
    Compute fitness score for mg.Mesh object.
    This particular fitness function works as follows:
        
        1) Sample values (via the analythical expression) of abs(K) and 
           max(abs(Pmax), abs(Pmin)) for each face of the meshed surface. 
          
        2) Compute the fraction of sampled points that satisfy the curvature
           conditions specified by the user, at each face.
           
        3) Multiply the previously computed fraction by the energy absorbed 
           by each face. 
           
        4) Sum the adjusted absorbed energy of all faces in order to get the
           adjusted total aborbed energy for the whole surface. This value
           becomes our new benchmark for judging the suitability of the surface.
           I.e, its fitness score.
    
    Where 'K' is Gaussian curvature, 'Pmax' is Maximum Principal Curvature
    and 'Pmin' is Minimum Principal Curvature.
          
          
    Parameters
    ----------
    Mesh_obj : mg.Mesh object
        Instance of class 'Surface_Modelling.Mesh.Mesh'. This fucntion requires 
        that the Mesh object have the following 2 attributes defined:
            
        1) 'Mesh_obj.poly' : numpy.array of floats with shape (3,3)
               3x3 matrix corresponding to the surface's representation in
               coefficient matrix form.
    
        2) 'Mesh_obj.absorbed_incident_energy' : numpy.array of floats with shape (Mesh_obj.num_faces,)
                Absorbed incident energy by each face in watt-hours.
                
    params : dict
        Dictionary of extra-parameters. It must have the following key-value
        pairs:
            
        Keys : Values
        -------------
        "U" : numpy.array of floats with shape (1,1,n)
            Array containing the barycentric u-coordinates at which we should
            evaluate the curvature values for each face. Array must satisfy:
            U[0,0,i] ≥ 0 and U[0,0,i] + V[0,0,i] ≤ 1, for all i.
           
        "V" : numpy.array of floats with shape (1,1,n)
            Array containing the barycentric v-coordinates at which we should
            evaluate the curvature values for each face. Array must satisfy:
            V[0,0,i] ≥ 0 and U[0,0,i] + V[0,0,i] ≤ 1, for all i.
        
        "W" : numpy.array of floats with shape (1,1,n)
            Array containing the barycentric w-coordinates at which we should
            evaluate the curvature values for each face. Array must satisfy:
            W[0,0,i] = 1 - U[0,0,i] - V[0,0,i], for all i.
            
        "abs(K) max" : float
            Maximum allowed value of abs(K) for any sampled point. Above
            this value, energy penalization occurs.
            
        "Pmaxmin max" : float
            Maximum allowed value of max(abs(Pmax), abs(Pmin)) for any sampled 
            point. Above this value, energy penalization occurs.
            
        
    Returns
    -------
    penalty_adjusted_total_absorbed_incident_energy : float
        Surface fitness score. More specifically, the total absorbed energy
        by the surface,. after pensalization has been factored in.
            

    """
    # Get sample points in local barycentric coordiantes.
    U, V, W = params["U"], params["V"], params["W"]
    
    # Get sample points in global coordinates for each face
    # (only the x,y values though).
    eval_pts_for_each_face =\
    U*Mesh_obj.A[:,:2].reshape(Mesh_obj.num_faces, 2, 1) +\
    V*Mesh_obj.B[:,:2].reshape(Mesh_obj.num_faces, 2, 1) +\
    W*Mesh_obj.C[:,:2].reshape(Mesh_obj.num_faces, 2, 1)
                             
    XYpows = dom.make_XYpows(X = eval_pts_for_each_face[:,0,:],
                             Y = eval_pts_for_each_face[:,1,:],
                             max_powX = 2, max_powY = 2)
    
    # Compute curvatures at sample points for each face.
    H, K = tayf.poly_curvature(Mesh_obj.poly, XYpows)
    
    Pmin = H - np.sqrt(H**2 - K)
    Pmax = H + np.sqrt(H**2 - K)
    Pmaxmin = np.maximum(abs(Pmin), abs(Pmax))
    
    # Test wether sampled curvature values satisfy the user-specified conditions.
    logic = np.logical_and(abs(K)  <= params["abs(K) max"], 
                           Pmaxmin <= params["Pmaxmin max"])
    
    
    # Compute fraction of compliant-sample points for each face.
    energy_penalty_per_face = logic.sum(axis=1)/logic.shape[1] 
    
    # Use the aforementioned fraction as a penalty weight for each face.
    # Then sum the adjusted absorbed energies to get the total adjusted
    # absrobed energy by the surface.
    penalty_adjusted_total_absorbed_incident_energy =\
    (Mesh_obj.absorbed_incident_energy*energy_penalty_per_face).sum()

    # We use the  penalty-adjusted total absorbed incident energy
    # as the fitness score for our Surface.
    return penalty_adjusted_total_absorbed_incident_energy
    


#%%                  DEFINTIION OF OPTIMIZER CLASS


class Optimizer:
    
    """
    Class for storing attributes and methods related to the optimization
    of curved PV surfaces.
    

    Parameters
    ----------
    Sky_obj : Sky object
        Instance of class 'Ambience_Modelling.Sky.Sky'.
        Must have attribute 'time_integrated_spectral_irradiance_res'.
        For this, utilize the Sky objetc's
        'compute_time_integrated_spectral_irradiances_for_a_date_interval'
        method or set the attribute manually for the Sky object. 

    absorbance_function : callable
        A function that takes in a numpy.array of floats with shape (n,2).
        We call this input numpy array 'eval_pts'. The first column of this
        array contains the angles of incidence, while the second column
        contains the wavelengths for which the fraction of absorbed incident 
        energy is to be calculated.
        
    radiation_mode : str
        Type of radiation to use for computing the absorbed incident energy
        for each meshed surface of the population. Supported are: "direct",
        "diffuse", "global". Default is "global".
        
    self_shading_config : dict or None
        This parameter specifies all parameters related to the computation of
        self-shading for meshed surfaces. If None, the default configuration
        is used. If dict, it should contain at least one of the following
        key-value pairs:
            
        Keys : Values
        -------------
        rad : float
            Cylinder radius for considering faces. From each ray-facing face center
            imagine an axis passing through that face center having the same
            direction as the ray. We then compute the closest distance from each of the 
            face centers of the surface to each of face centers, in general.
            If this distance is supirior to 'rad', the face element is not
            considered for ray-tracing. If it inferior, it is considered.
            Default is 0.1 .
            
        u : float
            Barycentric u-coordinate of the position used within the each face
            element to determine the origin vector of each ray. Default is 1/3.
        
        v : float
            Barycentric v-coordinate of the position used within the each face
            element to determine the origin vector of each ray. Default is 1/3.
        
        lmbda : float 
            Distance up to which to extend each of the rays. It should be greater
            than the dimensions of the surface. Default is 3.0 .
        
    Returns
    -------
    None.

    """
    
    def __init__(self, Sky_obj, absorbance_function, radiation_mode = "global", self_shading_config = None):
        """
        Constructor function.
        """
        
        # The 'self.generations' dict contains all the populations 
        # across all iterations/generations, as well as their rankings.
        self.generations = {}
        self.absorbance_function = absorbance_function
        self.num_divisions = Sky_obj.num_divisions
        
        # Save time_integrated_spectral_irradiance_res attribute. 
        self.time_integrated_spectral_irradiance_res =\
        Sky_obj.time_integrated_spectral_irradiance_res
        
        # Set direction vectors and set
        # time-integrated spectral irradiance magnitudes.
        self.set_radiation_mode(radiation_mode)
        
        # Save wavelengths array.
        self.wavelengths =\
        Sky_obj.time_integrated_spectral_irradiance_res["wavelengths"]
        


        # Initialize default configuration for self-shading computation.
        self.self_shading_config_ = {"rad"   : 0.1, 
                                     "u"     : 1/3,
                                     "v"     : 1/3,
                                     "lmbda" : 3.0}
        
        # User's configuration overwrites default configuration.
        if isinstance(self_shading_config, dict):
            for key, val in self_shading_config.items():
                self.self_shading_config_[key] = val
                
                
        
        # --- DEFAULT FITNESS FUNCTION ---
        # We initialize a default fitness function. If no other fitness function
        # is passed to the optimizer, the surfaces at each generation get ranked 
        # by the total amount of incident energy they abosorbed.
        self.fitness_function =\
        lambda Mesh_obj : Mesh_obj.total_absorbed_incident_energy
                
                
                
                
                
                
    def set_radiation_mode(self, radiation_mode) :
        
        """
        Set radiation mode for computing the absorbed incident energy.
        
        radiation_mode : str
            Type of radiation to use for computing the absorbed incident energy
            for each meshed surface of the population. Supported are: "direct",
            "diffuse", "global". Default is "global".
            
        Returns
        -------
        None
        
        Produces
        --------
        self.radiation_mode : str
            Selected radiation mode.
            
        self.time_integrated_spectral_irradiance_magnitudes : numpy.array of floats with shape (self.num_divisions, 122)
            Array containing the Magnitude of the 'radiation_mode' Spectral Irradiance vector 
            for each of the sky patches. Each row has units of Wh/m^2/nm.
            
        self.direction_vectors : numpy.array of floats with shape (self.num_divisions, 3)
            Unit ray-direction vectors. 'self.direction_vectors[i,:]' encodes the unit vector, in 
            cartesian coordinates, of the i-th ray/ direction that is to be 
            considered for ray tracing.
            
        """
        
        self.radiation_mode = radiation_mode
        
        key_magnitudes = f"magnitude_{radiation_mode}"
        self.time_integrated_spectral_irradiance_magnitudes =\
        self.time_integrated_spectral_irradiance_res[key_magnitudes]
        
        key_dvecs = f"spectrally_averaged_unit_{radiation_mode}" 
        self.direction_vectors =\
        self.time_integrated_spectral_irradiance_res[key_dvecs]
        
        return None
    



        
    def set_fitness_function(self, fitness_function, fitness_function_params):
        
        """
        Define the function to be used for ranking the surfaces at each 
        generation. 
        
        Parameters
        ----------
        fitness_function : callable
            Function for computing the fitness score of a Mesh object.
            It must take in the following 2 arguments:
                
            Parameters
            ----------
            Mesh_obj : Mesh object
                'fitness_function' has to take in an instance of the class
                'Surface_Modelling.Mesh.Mesh' as its first argument.
                
            config : dict
                As its second argument, 'fitness_function' must take in
                a dictionary containing all other necessary parameters required
                by the function in order to compute the fitness score. 
                
            Finally the function has to return one single output:
                
            Returns
            -------
            fitness_score : float
                Fitness score computed for the Mesh object.
                
        fitness_function_params : dict
            Dictionary of extra paramters required by 'fitness_function'
            in order to calculate a Mesh object's fitness score.
            
        Returns
        -------
        None
        
        Produces
        --------
        
        self.fitness_function_params : dict
            Dictionary of extra paramters required by 'fitness_function'.
            These are passed automatically to 'self.fitness_function',
            so it only need have one argument. 
        
        self.fitness_function : callable
            Fitness function to be used for the optimization process.
            It takes in only one parameter, a Mesh object. and computes
            its fitness score. All extra paramters reuqired for such a 
            process are passed autormatically.
            
            
       Notes
       ------
       1) self.fitness_function_params remains constant during the whole
          optimization process.
          
       2) If this method is never called, the program defaults to using
          'lambda Mesh_obj : Mesh_obj.total_absorbed_incident_energy'
          as the fitness function for the optimization procedure. However,
          if this method is indeed called, said function gets overwritten by
          the new inputted one.
          
       3) Make sure that all attributes of 'Mesh_obj' required for the calculation
          of the firness score, already are computed.
            
        """
        
        
        
        self.fitness_function_params = fitness_function_params
        
        self.fitness_function =\
        lambda Mesh_obj : fitness_function(Mesh_obj, self.fitness_function_params)
        
        return None        
    
    
    

        
    def set_population_for_generation(self, population, num_gen = 0):
        
        """
        Set 'self.generations[num_gen]["population"]' attribute.
        
        Parameters
        ----------
        population: dict of mg.Mesh objects
            Dict of 'Surface_Modelling.Mesh.Mesh' objects, representing
            the population of surfaces that are to be optimized. Its keys
            are ints, going from zero up until the number of surfaces - 1. 
            The keys act as identification for the meshed surfaces. Finally,
            these Mesh objects, however, must have a 'Mesh_obj.poly' attribute;
            where 'Mesh_obj.poly' is a 3x3 matrix corresponding to the 
            surface's representation in coefficient matrix form.
            
        num_gen : int
            Number of the generation for which the population is to be set.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.generations[num_gen]["population"] : dict of Mesh objects
            num-gen-th entry in the self.generations dict for "population".
        
        
        """
        
        self.generations[num_gen] = {"population" : population}
        return None
    
    


    
    def set_generations(self, generations):
        
        """
        Load 'generations' dict and overwrite 'self.generations' attribute.
        Use this method to load the results of a previous optimization run.
        With them, you can keep optimizing from where you last left off.
        
        Parameters
        ----------
        generations : dict of dicts
            Dictionary contaning the surface populations and ranking
            lists of each generation. Its keys are ints, whose value indicates
            the generation number (must start at 0 and be monotonic-increasing).
            Its values are dictionaries, each following this structure:
                
                Keys : Values
                -------------
                "population" : dict of Mesh objects
                    Dict of 'Surface_Modelling.Mesh.Mesh' objects, representing
                    the population of surfaces that are to be optimized. Its keys
                    are ints, going from zero up until the number of surfaces - 1. 
                    The keys act as identification for the meshed surfaces. Finally,
                    these Mesh objects, however, must have a 'Mesh_obj.poly' attribute;
                    where 'Mesh_obj.poly' is a 3x3 matrix corresponding to the 
                    surface's representation in coefficient matrix form.
                    
                    
                    
                "ranking_list" : list of 2-tuples
                    List of tuples containing the key of each surface (first tuple-
                    element) in 'generations[num_gen]["population"]', next to its 
                    computed fitness score (second tuple-element). This list must sorted 
                    from best fitness score to worst fitness score. That is, the
                    first element of this list contains the key, fitness score
                    of the best-performing surface, while the last element of this list
                    contains the key, fitness score of the worst-performing one.
                    
        
        """
        
        self.generations = generations
        
        return None
    
    
    
    


    def clear_generations(self):
        """
        Errase all key-value pairs in 'self.generations'
        
        Returns
        -------
        None.
        
        Produces
        --------
        self.generations : dict
             Empty dictionary of generations.

        """

        self.generations.clear()
        return None
        
    


    def compute_absorbed_energy_for_generation(self, num_gen):
        
        """
        Compute absorbed incident energy by all meshed surfaces
        stored in generation 'num_gen'.
        
        Parameters
        ----------
        num_gen : int
            Number of generation
            
        Returns
        -------
        None
        
        Notes
        -----
        1) Requires self.generations[num_gen]["population"] to exist.
        
        2) The absorbed incident energy is stored within each of the meshed
            surfaces. What we do here is to call a method of each Mesh obj.
            The results are stored as attributes in each mesh object.
        
        """
        
        # Loop over all meshed surfaces in 'num_gen'.
        for key in self.generations[num_gen]["population"].keys():
            
            # Compute self-shading/Ray-tracing for each surface.
            self.generations[num_gen]["population"][key].\
            compute_directions_logic(
            dvecs = self.direction_vectors,
            rad   = self.self_shading_config_["rad"],
            u     = self.self_shading_config_["u"], 
            v     = self.self_shading_config_["v"], 
            lmbda = self.self_shading_config_["lmbda"]
            )
            
            # Compute absorbed incident energy by each surface.
            self.generations[num_gen]["population"][key].\
            compute_absorbed_incident_energy(
            absorbance_function = self.absorbance_function,
            dvecs               = self.direction_vectors,
            time_integrated_spectral_irradiance_magnitudes =\
            self.time_integrated_spectral_irradiance_magnitudes, 
            wavelengths         = self.wavelengths
            )
                
        return None
    
    
    
    
    
    
    def compute_fitness_score_for_generation(self, num_gen):
        
        """
        Compute fitness score of all Mesh objects inside the specified
        generation.
        
        Parameters
        ----------
        num_gen : int
            Number of generation
            
        Returns
        -------
        None
        
        Produces
        --------
        self.generations[num_gen]["population"][key].fitness_score : float
            The fitness score of each Mesh object is calculated and gets
            stored inside each Mesh object as the 'Mesh_obj.fitness_score'
            attribute.
            
        
        Notes
        -----
        1) Requires self.generations[num_gen]["population"] to exist.
    
        2) This function requires most probably requires that the absorbed
           incident energy of each Mesh object in 
           self.generations[num_gen]["population"] already be computed and 
           stored inside each of the Mesh object.
           
        """
        
        for key, Mesh_obj in self.generations[num_gen]["population"].items():
            
            fitness_score = self.fitness_function(Mesh_obj)
            self.generations[num_gen]["population"][key].fitness_score = fitness_score
        
        return None
        
    
    
    def compute_rank_list_for_generation(self, num_gen):
        
        """
        Compute rank-list for of specified generation.
        
        Parameters
        ----------
        num_gen : int
            Number of generation
            
        Returns
        -------
        None
        
        Produces
        --------
        self.generations[num_gen]["rank_list"] : list of 2-tuples
            List of tuples containing the key of each surface (first tuple-
            element) in self.generations[num_gen]["population"], next to its 
            computed fitness score (second tuple-element). The list is sorted 
            from best fitness score to worst fitness score. That is, the
            first element of this list contains the key, fitness score
            of the best-performing surface, while the last element of this list
            contains the key, fitness score of the worst-performing one.
            
        
        Notes
        -----
        1) Requires self.generations[num_gen]["population"] to exist.
        
        2) This function requires that the fitness-score of each Mesh object in
           self.generations[num_gen]["population"] already be computed and 
           stored inside each of the Mesh objects, as 'Mesh_obj.fitness_score'
           attribute.
        
        """
        
        
        rank_list = []
        for key, Mesh_obj in self.generations[num_gen]["population"].items():
            rank_list.append((key, Mesh_obj.fitness_score))
            
        rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True)
        self.generations[num_gen]["rank_list"] = rank_list
        
        
        return None
    
    
    def compute_summary_data_for_generation(self, num_gen):
        
        """
        Compute dict of 'self.generations[num_gen]["summary data"]'
        dict. This dictiornay shall store relevent data pertaining to each
        Mesh object in 'self.generations[num_gen]["population"]'. In this way,
        all relevant data for analysis, plotting and even recreation of the 
        population, will be availble across all generations, without the
        necessity of storing the entirety of Mesh objects. This is relevant, as 
        we'll most probably have to delete most of the populations
        as they just occupy too much space in memory.
        
        Parameters
        ----------
        num_gen : int
            Number of generation
            
        Returns
        -------
        None
        
        Produces
        --------
        self.generations[num_gen]["summary data"] : dict of dicts
            Dict containing the most relevant data of each Mesh object stored at 
            self.generations["population"]. It's keys are of type int, and are 
            equal to the number id/key of the Mesh object whose info they store.
            In other words self.generations[num_gen]["summary data"][surface_key] 
            contains the summary data of the Mesh object stored at
            self.generations[num_gen]["population"][surface_key].
            
            self.generations[num_gen]["summary data"][surface_key] may contain
            one or more of the following values:
                
            Values
            -------
            "area" : float
                Total area of the surface.
                
            "poly" : numpy.array of floats with shape (3,3)
                3x3 matrix corresponding to the surface's representation in
                coefficient matrix form.
                
            "fitness score" : float
                Fitness score obtained by the meshed surface, when it was passed
                to the 'self.fitness_function' method.
                
            "absorbed energy" float
                Total absorbed incident energy by the surface. It has units
                of watt-hours.
                
            "ranking" : int
                Ranking of the surface when compared to its peers in the 
                same generation.
                
        Notes
        -----
        1) Requires self.generations[num_gen]["population"] to exist.

        """
        
        
        self.generations[num_gen]["summary data"] = {}
        for key, Mesh_obj in self.generations[num_gen]["population"].items():
            
            
            self.generations[num_gen]["summary data"][key] = {}
            
            try:
                self.generations[num_gen]["summary data"][key]["area"] =\
                Mesh_obj.total_area
            except AttributeError:
                pass
            
            try:
                self.generations[num_gen]["summary data"][key]["poly"] =\
                Mesh_obj.poly
            except AttributeError:
                pass
            
            try:
                self.generations[num_gen]["summary data"][key]["fitness score"] =\
                Mesh_obj.fitness_score
            except AttributeError:
                pass
            
            try:
                self.generations[num_gen]["summary data"][key]["absorbed energy"] =\
                Mesh_obj.total_absorbed_incident_energy
            except AttributeError:
                pass
            
        
        if "rank_list" in self.generations[num_gen].keys():
            for ranking, (surface_key, _) in enumerate(self.generations[num_gen]["rank_list"]):
                self.generations[num_gen]["summary data"][surface_key]["ranking"] = ranking
                
            
        return None    
    
    
    
    
    def compute_population_for_next_generation(self, num_gen):
        
        """
        Compute the next generation of surfaces. That is, compute a new 
        population of surfaces via an established algorithim that
        simulates sexual reproduction in animals.
        
        Parameters
        ----------
        num_gen : int
            Number of generation.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.generations[num_gen + 1]["population"] : dict of Mesh objects
             New population of Mesh objects. 
            
            
        Notes
        -----
        1) Requires the existance of self.generations[num_gen]["population"]
           self.generations[num_gen]["rank_list"]. 
    
        2) This function requires the existance of the attribute 
           'self.algorithm_params'.
           
        3) This fucntion requires that each Mesh object in the population 
           have the 'Mesh_obj.poly' attributes. Where 'Mesh_obj.poly' is
           a 3x3 matrix corresponding to the surface's representation in
           coefficient matrix form.
            
        """
        
        self.generations[num_gen + 1] = {"population" :\
        evolve(population = self.generations[num_gen]["population"],
               rank_list  = self.generations[num_gen]["rank_list"],
               XYpows     = self.algorithm_params["XYpows"],
               mutprob    = self.algorithm_params["mutprob"],
               mutweight  = self.algorithm_params["mutweight"],
               max_Zrange = self.algorithm_params["max_Zrange"]
               )}
        
    
        return None
    
    
    def delete_population_for_generation(self, num_gen):
        
        """
        Delete population of Meshed surfaces stored in 'self.generations'
        attribute. This function may be necessary use periodically, as storing
        large populations of meshed surfaces becomes increasingly costly due to
        the amount of space taken up in memory by even a single "population"
        of Meshed surfaces.
        
        Parameters
        ----------
        num_gen : int
            Number of generation.
            
            
        Returns
        -------
        None
        
        """
        
        
        try:
            del self.generations[num_gen]["population"]
        except KeyError:
            pass
        
        return None
    
    

    
    def optimize(self, algorithm_params, num_iterations, config = None):
        
        """
        Optimize a population of surfaces, iteratively, by employing an 
        evolutive algorithm that simulates sexual reproduction in animals.
        
        Parameters
        ----------
        
        algorithm_params : dict
            Dict containing the extra-parameters required by the evolutive
            algorithm for optimizing the population. It has the following
            key-value pairs:
                
            Keys : Values
            -------------
            "XYpows" : dict of 2D numpy.array of floats
                Dictionary storing the products of the meshgird values of X and Y
                elevated to the i-th and j-th power respectively. In other words,
                the key of the dictionary: (i,j) stores the value: (X**i)*(Y**j),
                where X,Y are the meshgrid values of the x and y coordinates over
                the domain in question.
                
            "mutprob" : float
                Probabilty that a given child surface will mutate. Must be a number
                between 0 and 1.
                
            "mutweight" : float
                Standard deviation of the normal distribution (with mean equal to 0),
                from which the coefficients of the mutation polynomial are sampled.
                
            "max_Zrange" : float
                Maximum range allowed for a child surface.
                
                
        
        num_iterations : int
            How many new iterations to compute for optimizing the 
            population surface. Must be greater than 0.
            
            
        config : dict or None
            Configuration/settings for 'self.optimize' method. If None,
            the default configuration/settings are used. If dict, it must
            contain at least one of the forllowing key-value pairs:
                
            Keys : Values
            --------------
            "verbose" : bool
                If True, print to console the status of the optimization
                process, every time a new iteration is finished. If False,
                print nothing. Default is True.   
                
            "periodic_save" : bool
                If true, the 'self.generations' attribute is saved to a
                during the optimization runtime. If False, it is not.
                Default is False.
            
            "periodic_save_num" : int
                Number of iterations in-between saves. Must be grater than zero.
                This only applies if "periodic_save" is equal to True. Default
                is 10.
                
            "periodic_save_path" : int
                Path of the pickle file to which 'self.generations' is to be saved.
                Default is: os.path.join(os.getcwd(), "generations.pkl"). 
            
            "periodic_del" : bool
                If true, the 'self.generations[num_gen]["population"]' attribute 
                is periodically as to avoid running out of memory. If False,
                it does nothing. Default is True.
                
            "periodic_del_num" : int
                Delete all 'self.generations[num_gen]["population"]' attributes
                that comply with num_gen ≤ current_num_gen  - config["periodic_del_num"].
                This only applies if "periodic_del" is True. Default is 1.
                

        Returns
        -------
        None
        
        Produces
        --------
        self.algorithm_params : dict
            Dict containing the extra-parameters required by the evolutive
            algorithm for optimizing the population.
            
        self.generations : dict of dicts
            Updated self.generations attribute. A new population is created
            and ranked.
            
        
        """
        
        # --- DEFINE DEAFULT CONFIGURATION OF OPTIMIZER --- 
        
        config_ = {"verbose"            : True,
                   "periodic_save"      : False,
                   "periodic_save_num"  : 5,
                   "periodic_save_path" : os.path.join(os.getcwd(), "generations.pkl") ,
                   "periodic_del"       : True,
                   "periodic_del_num"   : 1}
        
        
        self.algorithm_params = algorithm_params
        
        
        # --- OVERWRITE DEAFULT CONFIGURATION OF OPTIMIZER WITH USER'S --- 
        if isinstance(config, dict):
            config_.update(config)
        

      # --- DELETE ALL POPULATIONS THAT COMPLY WITH: num_gen ≤ current_num_gen  - config["periodic_del_num"]---   
        if config_["periodic_del"]:  
            for num_gen in self.generations.keys():
                self.delete_population_for_generation(num_gen - config_["periodic_del_num"])
                
                
                
        # --- COMPUTE GENERATION NUMBER FROM WHCIH TO START OPTIMIZATION --- 
        init_num_gen = - 1
        for num_gen in self.generations.keys():
            if "population" in self.generations[num_gen].keys():
                init_num_gen = max(init_num_gen, num_gen)
                
        
        # --- ERROR HANDLING --- 
        if init_num_gen < 0:
            raise Exception("ERROR: No surface population to optimize was found.")
        
        
        # --- WARN USER OF POSSIBLE MEMORY ERROR --- 
        if not config_["periodic_del"]:
            msg = "WARNING: 'periodic_del' was set to False. The program is"
            msg = f"{msg} at risk of running out of memory during optimization."
            warnings(msg)
        
        elif config_["periodic_del_num"] > 4:
            msg = "WARNING: 'periodic_del_num' is greater than 4. The program is"
            msg = f"{msg} at risk of running out of memory during optimization."
            warnings(msg)
            
        # --- INITALIZE TIMER FOR DISPLAY --- 
        if config_["verbose"]:
            t0 = tm.time()
            
            
        
                      # --- OPTIMIZATION OF SURFACES --- 
        for num_gen in range(init_num_gen, init_num_gen + num_iterations + 1):
            
            # Delete previous generation's population.
            if config_["periodic_del"]:
                self.delete_population_for_generation(num_gen - config_["periodic_del_num"])
            
            # Optimize
            self.compute_absorbed_energy_for_generation(num_gen)
            self.compute_fitness_score_for_generation(num_gen)
            self.compute_rank_list_for_generation(num_gen)
            self.compute_summary_data_for_generation(num_gen)
            self.compute_population_for_next_generation(num_gen)
            
            
            # Print status of optimization.
            if config_["verbose"]:
                print("--- OPTIMIZATION STATUS ---")
                print(f"Iteration Finished: {num_gen}")
                print(f"Time Ellapsed [min]: {(tm.time()-t0)/60}")
                best_key, best_fitness_score = self.generations[num_gen]['rank_list'][0]
                print(f"Best Surface: {best_key}")
                print(f"Best Fitness Score: {best_fitness_score}")
                print(f"Absorbed Energy [Wh]: {self.generations[num_gen]['population'][best_key].total_absorbed_incident_energy}")
                print(f"Area [m^2]: {self.generations[num_gen]['population'][best_key].total_area}")
                
                
                
            # Save 'self.generations' dict.
            if config_["periodic_save"]:
                if num_gen % config_["periodic_save_num"] == 0:
                    aux.save_obj_with_pickle(self.generations,
                                             config_["periodic_save_path"])
                    

        return None
    
    
 
    
    
    
    def export_generations_as_pickle(self, path):
        """
        Save 'self.generations' attribute as a pickle file.

        Parameters
        ----------
        path : path-str
            Path to which to save 'self.generations' attribute.

        Returns
        -------
        None

        """
        
        aux.save_obj_with_pickle(class_obj = self.generations,
                                 path = path)
        
        return None
    
    
    def set_generations_from_path_with_pickle(self, path):
        """
        Load 'generations' dict from a pickle file and set it as the
        'self.generations' attribute.

        Parameters
        ----------
        path : path-str
            Path of pickle obj from which to load 'generations' dict.

        Returns
        -------
        None

        """
        self.set_generations(aux.load_obj_with_pickle(path))
        
        return None
        
    
    
    
    def plot_evolution_of_variable(self, var = "fitness score", mode = "top", config = None):
        
        """
        Plot the evolution of the chosen surface variable across generations.
        
        Parameters
        ----------
        var : str
            Variable to plot. Supported are: "fitness score", "area" and 
            absorbed energy.
        
        mode : str
            Type of plot. Supported are: "top", "bottom" and "comprehensive".
            If equal to "top", the value of the variable belonging to the top
            surface (i.e, that with the maximum fitness score) is plotted at
            each generation. If equal to "bottom", , the value of the variable
            belonging to the bottom surface (i.e, that with the minimum fitness
            score) is plotted at  each generation. If equal to "comprehensive",
            the average, 25th, 50th and 75th percentiles of the variable, with
            respect to all surfaces, is plotted at each generation.
            
        config : dict or None
            Plot's settings. If None, the default plot settings/configuration
            are/is used to the draw the plot. If dict, it must contain at least
            one of the following key-value pairs:
                
            Keys : Values
            -------------
            "title" : str
                Title of plot.
                
            "figsize" : 2-tuple of int
                Figure size. Default is (12, 8).
                
            "xlabel" : str
                Label of x-axis. Default is "Number of generations".
                
            "ylabel" : str
                Label of y-axis. 
                
            "xlims" : 2-tuple of float
                Lower and upper bounds for the x-axis.
                
            "ylims" : 2-tuple of float
                Lower and upper bounds for the y-axis.

        """
        
        #           ------ ERROR HANDLING ----
        
        if var not in ["fitness score", "area", "absorbed energy"]:
            raise Exception(f"Inputted variable: '{var}' is not valid.")
            
        if mode not in ["top", "bottom", "comprehensive"]:
            raise Exception(f"Inputted mode: '{mode}' is not valid.")
            
            
            
            
            
        
        #     ------ RETRIEVE VARIABLE TO BE PLOTTED -----
        
        # Retrieve rank lists for each generation that has one.
        res = {}
        num_gens   = [num_gen for num_gen, val in self.generations.items() if "rank_list" in val.keys()]
        rank_lists = [self.generations[num_gen]["rank_list"] for num_gen in num_gens]
        
        
        
        if mode in ["top", "bottom"]:
            res[mode] = []
            for num_gen in num_gens:
                if mode == "top"   : key = rank_lists[num_gen][0][0]
                if mode == "bottom": key = rank_lists[num_gen][-1][0]
                val = self.generations[num_gen]["summary data"][key][var]
                res[mode].append(val)
                
            if   var == "fitness score":
                config_ = {"title"  : "Fitness score of top surface",
                           "ylabel" : "Fitness Score"}
            
            elif var == "area":
                config_ = {"title"  : "Area of top surface", 
                           "ylabel" : "Area [m^2]"}
            
            elif var == "absorbed energy":
                config_ = {"title"  : "Absorbed incident energy by top surface", 
                           "ylabel" : "Absorbed energy [Wh]"}
                
                
                

        elif mode == "comprehensive":
            res = {"p25":[], "p50":[], "p75":[], "avg":[]}
            
            for num_gen in num_gens:
                vals = [self.generations[num_gen]["summary data"][key][var]
                        for key in self.generations[num_gen]["summary data"].keys()]
                
                vals = np.array(vals)
                res["p25"].append(np.percentile(vals, q=0.25))
                res["p50"].append(np.percentile(vals, q=0.50))
                res["p75"].append(np.percentile(vals, q=0.75))
                res["avg"].append(vals.mean())
            

            if var == "fitness score":
                config_ = {"title"  : "Fitness score of surfaces: average, 25th, 50th and 75th percentile",
                           "ylabel" : "Fitness Score"}
                
            elif var == "area":
                config_ = {"title"  : "Area of Surfaces: average, 25th, 50th and 75th percentile",
                           "ylabel" : "Area [m^2]"}
                
            elif var == "absorbed energy":
                config_ = {"title"  : "Absorbed incident energy by Surfaces: average, 25th, 50th and 75th percentile",
                           "ylabel" : "Absorbed energy [Wh]"}
                
                
                
            
        #     ------ PLOT VARIABLE -----
        
        # Set default configuration.
        config_.update({"figsize" : (12,8),
                        "xlabel" : "Number of generation",
                        "xlims"  : (min(num_gens), max(num_gens)),
                        "ylims"  : (None, None)})
                
        
                
        # Overwritte default configuration with user's configuration if inputted.
        if isinstance(config, dict):
            config_.update(config)

        # Plot the fitness scores.
        _ = plt.figure(figsize = config_["figsize"])      
        for key, val in res.items():
            
            if   key == "p25": linestyle = "-."
            elif key == "p75": linestyle = "--"
            elif key == "avg": linestyle = ":"
            else:              linestyle = "-"

            plt.plot(num_gens, val, color="black", linestyle=linestyle, label=key)
            
        plt.xlim(config_["xlims"])
        plt.ylim(config_["ylims"])
        plt.title(config_["title"])
        plt.xlabel(config_["xlabel"])
        plt.ylabel(config_["ylabel"])  
        
        if key not in ["top", "bottom"]:
            plt.legend()
            
        plt.grid()
        plt.show()

        return None
    
    
    

    
    
    
#%%          EXAMPLES

if __name__ == "__main__":
    from Ambience_Modelling.Sky import Sky
    from scipy.interpolate import RegularGridInterpolator
    import Surface_Modelling.taylor_initpop_gen as tay_init
    
    
    # --- LOAD SKY OBJ AND ABSORBANCE FUNCTION ---
    
    # Load absorbance array of Silicon.
    absorbance_arr_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\silicon_absorbance.npy" 
    absorbance_arr = np.load(absorbance_arr_path)
    
    # We interpolate the absorbance array in order to generate the absorbance function.
    
    absorbance_function = RegularGridInterpolator(points = (absorbance_arr[:,0,2],
                                                            absorbance_arr[0,:,1]), 
                                                  values =  absorbance_arr[:,:,0])
    
    # Test absorbance function for 45° at 500 nm.
    print(absorbance_function([45, 500]))
    
    
    # Load Sky_obj.
    Sky_obj_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Sky_obj_Medellin_2022.pkl"  
    Sky_obj = aux.load_obj_with_pickle(Sky_obj_path)


    # --- INITIALIZE INITIAL POPULATION OF MESHED SURFACES---
    xy_plane = dom.RectDomain(dx = 0.05, dy= 0.05, dims = (1,1,1))
    init_pop = tay_init.create_init_pop_3x3(RectDomain_obj = xy_plane, 
                                            pop_size       = 30,
                                            distribution   = "normal",
                                            scale          = 3,
                                            max_Zrange     = 1,
                                            Kconds         = (10**-4, 0.01),
                                            Pconds         = (1/6, 0.05))
    
#%%
    config = {"figsize":(10,10),
              "axis_view" : (30,-5)} 
    init_pop[10].visualize(config=config)



#%%
    
    
            
    for key, Mesh_obj in init_pop.items():
        config = {"title" : f"Surface id {key}"}
        Mesh_obj.visualize(config=config)
        
        
    # --- INITIALIZE OPTIMIZER ---
    Optimizer_obj = Optimizer(Sky_obj = Sky_obj, 
                              absorbance_function = absorbance_function,
                              self_shading_config = None)
    
    
    
    
    
    # Check out some of the relevan attrubutes.
    direction_vectors = Optimizer_obj.direction_vectors
    
    time_integrated_spectral_irradiance_res =\
    Optimizer_obj.time_integrated_spectral_irradiance_res
    
    time_integrated_spectral_irradiance_magnitudes =\
    Optimizer_obj.time_integrated_spectral_irradiance_magnitudes
    
    
    # --- LOAD INITIAL POPULATION ---
    Optimizer_obj.set_population_for_generation(init_pop)
    
    # Errase un-used memory-consuming variables.
    del Sky_obj
    del init_pop
    
            
    # --- DEFINE FITNESS FUNCTION ---
    num_samples_per_axis = 6
    u = v = np.linspace(0, 1, num_samples_per_axis)
    U, V  = np.meshgrid(u, v)
    
    filer_logic = U + V <= 1
    U, V = U[filer_logic], V[filer_logic]
    U, V = U.reshape(1,1,len(U)), V.reshape(1,1,len(V))
    W = 1 - U - V
    
    fitness_function_params = {"U" : U, "V" : V, "W" : W,
                               "abs(K) max"  : 10**-4,
                               "Pmaxmin max" : 1/6}
    
    Optimizer_obj.set_fitness_function(fitness_function1,
                                       fitness_function_params)
    
    
        
    # --- TEST OPTIMIZER METHOD ---
    
    algorithm_params  = {"XYpows" : xy_plane.XYpows,
                         "mutprob"    : 0.2,
                         "mutweight"  : 0.2,
                         "max_Zrange" : 1}
    
#%%

    generations_save_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Optimization\generations.pkl"
    config = {"periodic_save"      : True,
              "periodic_save_num"  : 3,
              "periodic_save_path" : generations_save_path}
    
    Optimizer_obj.optimize(algorithm_params = algorithm_params,
                           num_iterations    = 100,
                           config           = config)
    
    
    # --- RE-PLOT ---
    Optimizer_obj.plot_evolution_of_variable(var = "fitness score",   mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "area",            mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "absorbed energy", mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "fitness score",   mode = "comprehensive")
    Optimizer_obj.plot_evolution_of_variable(var = "area",            mode = "comprehensive")
    Optimizer_obj.plot_evolution_of_variable(var = "absorbed energy", mode = "comprehensive")
    
#%%     --- LOAD SAVED GENERATIONS DICT AND KEEP OPTIMIZING ---

    
    generations_save_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Optimization\generations_buya_1m.pkl"

    Optimizer_obj.set_generations_from_path_with_pickle(generations_save_path)
    
    config = {"periodic_save"      : True,
              "periodic_save_num"  : 5,
              "periodic_save_path" : generations_save_path}
    
    
    Optimizer_obj.optimize(algorithm_params = algorithm_params,
                           num_iterations    = 56,
                           config           = config)
    
    # --- RE-PLOT ---
    Optimizer_obj.plot_evolution_of_variable(var = "fitness score",   mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "area",            mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "absorbed energy", mode = "top")
    Optimizer_obj.plot_evolution_of_variable(var = "fitness score",   mode = "comprehensive")
    Optimizer_obj.plot_evolution_of_variable(var = "area",            mode = "comprehensive")
    Optimizer_obj.plot_evolution_of_variable(var = "absorbed energy", mode = "comprehensive")

#%%
    
    for surf_key, Mesh_obj in Optimizer_obj.generations[99]["population"].items():
         
         area = Optimizer_obj.generations[99]["summary data"][surf_key]["area"]
         ranking = Optimizer_obj.generations[99]["summary data"][surf_key]["ranking"]
         fitness_score = Optimizer_obj.generations[99]["summary data"][surf_key]["fitness score"]
         absorbed_energy = Optimizer_obj.generations[99]["summary data"][surf_key]["absorbed energy"]

         
         config = {"title" : f"Surface: {surf_key}. Ranking: {ranking}. Fitness Score: {round(fitness_score,2)}. Area: {round(area,3)} [m^2]. Absorbed Energy {round(absorbed_energy/1000,3)} [kWh]"}
         Mesh_obj.visualize(config = config)
        
        
        
        
        
        
        
    
 #%%   
    
tayf.poly_curvature(poly, XYpows)


#%%

fig = plt.figure(figsize=(10,8))
plt.contourf(xy_plane.X, xy_plane.Y, a)
plt.colorbar()
plt.suptitle("Maximum absolute principal curvature of top surface")
plt.title("max(|κ_max|, |κ_min|)")
plt.xlabel("X[m] (↑ == N, ↓ == S)")
plt.ylabel("Y[m] (↑ == E, ↓ == W)")
plt.show()


fig = plt.figure(figsize=(10,8))
plt.contourf(xy_plane.X, xy_plane.Y, abs(K))
plt.colorbar()
plt.suptitle("Absolute Gaussian Curvature of top top surface")
plt.title("Abs(K)")
plt.xlabel("X[m] (↑ == N, ↓ == S)")
plt.ylabel("Y[m] (↑ == E, ↓ == W)")
plt.show()








