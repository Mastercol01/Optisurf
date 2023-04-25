#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
modelling of radiation coming from the sky.
"""


#%%                   IMPORTATION OF LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from Ambience_Modelling.auxiliary_funcs import load_obj_with_pickle
from Ambience_Modelling.spectral_radiance_model import compute_spectral_radiance


#%%                     DEFINITION OF CLASSES


class Sky:
    
    def __init__(self, Site_obj, num_divisions = 400, sunrise_sunset_apel = -0.25):
        
        """
        Class for spectrally and spatially modelling the radiation coming
        from the sky.
        
        Parameters
        ----------
        Site_obj : Site object
            Instance of class 'Ambience_Modelling.Site.Site', whose 
            self.site_data, self.sun_data, self.ground_albedo, 
            self.single_scattering_albedo and  self.aerosol_asymmetry_factor 
            have already been computed/filled with data.

                    
        num_divisions : int
            Number of patches into which the Sky Vault will be discretised.
        
        sunrise_sunset_apel : float
            Sun's apparent elevation at which we consittutes the sunrise and 
            sunset times.
            
        """
        
        
        self.Site_obj = Site_obj
        self.discretise(num_divisions)
        self.sunrise_sunset_apel = sunrise_sunset_apel
        
        
        # Compute vectorized versions of sky-patch localization
        # functions using coordinates.
        self.sky_points_to_zones_patches =\
        np.vectorize(self.sky_point_to_zone_patch)
        
        self.disk_points_to_zones_patches =\
        np.vectorize(self.disk_point_to_zone_patch)
        
        
    def discretise(self, num_divisions):
        
        """
        Discretise the Sky Vault into non-congruent square-like patches of
        similar area, according to the procedure proposed in the paper "A 
        general rule for disk and hemisphere partition into equal-area cells".
        This discretisation is required for the later computation of 
        equivalent-irradiance vectors.
        
        Parameters
        ----------
        num_divisions : int
            Number of patches into which the Sky Vault will be discretised.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.zone_data : dict of dicts
            Dictionary containing all relevant info regarding the Sky Vault's
            discretization zones, stored in another dictionary. The keys of
            'self.zone_data' are ints, corresponding the the zone number whose
            information is stored there. There are as many keys of 'self.zone_data'
            as there are zones. The component dictionaries (i.e, those stored at 
            'self.zone_data[zone_num]', for all zone numbers) have the following 
            Key-Value pairs:
                
                Keys : Values
                -------------
                "num_patches" : int
                    Number of sky patches contained inside the sky zone.
                    
                "inf_zen" : float
                    Inferior zenith angle bound delimiting the sky zone,
                    in degrees.
                    
                "sup_zen" : float
                    Superior zenith angle bound delimiting the sky zone,
                    in degrees.
                    
                "inf_rad" : float
                    Inferior radius bound delimiting the sky zone's plane 
                    projection [adm].
                    
                "sup_rad" : float
                    Superior radius bound delimiting the sky zone's plane 
                    projection [adm].
                    
                "azimuths" : numpy.array of floats
                    Array containing the azimuth angle intervals delimiting
                    each sky patch inside the zone, in degrees.
                    
                "patch_area" : float
                    Solid angle/area, taken up by each sky patch inside the
                    sky zone, in steradians.
                    
                "zone_area" : float
                    Total solid angle/area of the whole sky zone, in steradians.
                    
                
        self.patch_data : dict of dicts
            Dictionary containing all relevant info regarding the Sky Vault's
            discretization patches, stored in another dictionary. The keys of
            'self.patch_data' are 2-tuples of ints, corresponding the patch, as
            identified by its (zone number, local patch number), whose
            information is stored there. There are as many keys of 'self.patch_data'
            as there are patches. The component dictionaries (i.e, those stored at
            'self.patch_data[(zone_num, local_patch_num)]', for all zone and 
            local patch numbers) have the following Key-Value pairs:
                
                Keys : Values
                -------------
                "inf_zen" : float
                    Inferior zenith angle bound delimiting the sky patch,
                    in degrees.
                    
                "sup_zen" : float
                    Superior zenith angle bound delimiting the sky patch,
                    in degrees.
                    
                "inf_az" : float
                    Inferior azimuth angle bound delimiting the sky patch,
                    in degrees.
                    
                "sup_az" : float
                    Superior azimuth angle bound delimiting the sky patch,
                    in degrees.
                    
                "patch_area" : float
                    Solid angle/area, taken up by the sky patch, in steradians.
                    
                "unit_vector" : np.array of floats with shape (3,)
                    Unit solid angle vector of the center of the sky patch.
                    It is basically a unit vector with tail at the origin and 
                    which points to the center position of the sky patch.
                    "unit_vector"[i], with i = 0,1,2; gives the unit vector's
                    x,y,z component respecitvely.
                    
        """
        
        # ----- SKY VAULT DISCRETIZATION --------
        # Compute radius and zenith defining each sky zone, as well as the number
        # of elements present within each concentric circle/ring.

        rad = [1]
        zenith = [np.pi/2]
        num_elem_in_rad = [num_divisions]

        i = 0
        while num_elem_in_rad[-1]>1:

            zenith.append\
            (zenith[i] - np.sqrt(2)*np.sin(zenith[i]/2)*np.sqrt(np.pi/num_elem_in_rad[i]))

            rad.append\
            (np.sqrt(2)*np.sin(zenith[i+1]/2))

            num_elem_in_rad.append\
            (round(num_elem_in_rad[i]*(rad[i+1]/rad[i])**2))

            i += 1
        
        
        # If the algorithm tells us that there are zero elements within the
        # innermost ring, we have to adjust the data of the last element 
        # to reflect that. 
        if num_elem_in_rad[-1] == 0:
            rad[-1] = 0
            zenith[-1] = 0
            i -= 1
            
        # If the algorithm tells us that there is 1 element within the
        # innermost ring, we have to add the missing data.
        elif num_elem_in_rad[-1] == 1:
            rad += [0]
            zenith += [0]
            num_elem_in_rad += [0]
            
            
        # We sort from min to max (i.e, reverse teh lists, in this case).
        rad = np.array(rad[::-1])
        zenith = np.array(zenith[::-1])
        num_elem_in_rad = np.array(num_elem_in_rad[::-1])
        
        
        # We initialize relevant attributes.
        self.zone_data  = {}
        self.patch_data = {}
        self.zone_max_key = i
        self.num_divisions = num_divisions
        
        
       # ------- CREATION OF DATABASE FOR ZONE AND PATCH INFO --------
       
        # We store all relevant information in 2 custom databases. one
        # for zone infor and another for patch info. 
        for zone_num in range(self.zone_max_key + 1):
            
            num_patches =\
            num_elem_in_rad[zone_num + 1] - num_elem_in_rad[zone_num]
            
            self.zone_data[zone_num] =\
            { "num_patches" : num_patches,
              "inf_zen"     : np.rad2deg(zenith[zone_num]),
              "sup_zen"     : np.rad2deg(zenith[zone_num + 1]),
              "inf_rad"     : rad[zone_num],
              "sup_rad"     : rad[zone_num + 1],
              "azimuths"    : np.rad2deg(np.linspace(0, 2*np.pi, num_patches + 1)),
              "patch_area"  : (2*np.pi/num_patches)*(np.cos(zenith[zone_num])-np.cos(zenith[zone_num + 1])),
              "zone_area"   : 2*np.pi*(np.cos(zenith[zone_num])-np.cos(zenith[zone_num + 1]))
              
            }
            
            # Compute mean zenith angle for each zone in radians.
            mid_theta  = zenith[zone_num]
            mid_theta += zenith[zone_num + 1]
            mid_theta  = mid_theta/2
            
            for local_patch_num in range(num_patches):
                
                # Compute mean azimuth angle for each sky patch in radians.
                mid_az  = self.zone_data[zone_num]["azimuths"][local_patch_num]
                mid_az += self.zone_data[zone_num]["azimuths"][local_patch_num + 1]
                mid_az  = np.deg2rad(mid_az/2)
                
                
                self.patch_data[(zone_num, local_patch_num)] =\
                { "inf_zen"     : np.rad2deg(zenith[zone_num]),
                  "sup_zen"     : np.rad2deg(zenith[zone_num + 1]),
                  "inf_az"      : self.zone_data[zone_num]["azimuths"][local_patch_num],
                  "sup_az"      : self.zone_data[zone_num]["azimuths"][local_patch_num + 1],
                  "patch_area"  : self.zone_data[zone_num]["patch_area"],
                  "unit_vector" : np.array([np.cos(mid_az)*np.sin(mid_theta),
                                            np.sin(mid_az)*np.sin(mid_theta),
                                            np.cos(mid_theta)])
                }
            
        
        return None   
    
    
    
    
    # ------ METHODS FOR ZONE-PATCH LOCALIZATION FROM COORDINATES: SPHERE -------

    def sky_point_to_zone_patch(self, zen, az):
        
        """
        Bin sky point into the correct sky patch. That is, given a sky point
        represented by a tuple of (zenith, azimuth) values, return the sky patch, 
        represented by a tuple of (zone_num, local_patch_num), to which said
        sky point belongs.
        
        Parameters
        ----------
        zen : float
            Zenith of sky point in degrees. Must be between 0 and 90.
        
        az : float
            Azimuth of sky point in degrees. Must be between 0 and 360.
            
        Returns
        -------
        zone_num : int or str
            Sky zone (int) to which the sky point belongs, or "not found" if
            search failed.
        
        local_patch_num : int or str
            Sky patch (int) (identified by its local patch number in reference 
            to the sky zone) to which the sky point belongs, or "not found" if 
            search failed.
        
        
        """
        
        zone_num = self.zenith_to_zone(zen)
        local_patch_num = self.azimuth_to_patch(zone_num, az)
        return zone_num, local_patch_num



    def zenith_to_zone(self, zen, start=None, end=None):
        
        """
        Bin zenith value into the correct sky zone via binary search.
        
        Parameters
        ----------
        zen : float
            Zenith value in degrees. Must be between 0 and 90.
        
        start : int or None
            Lower search bound for zone. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for zone. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        zone_num : int or str
            Sky zone (int) to which the zenith coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): start = 0
        if(end is None): end = self.zone_max_key

        if(start > end):
            return "Not found"

        zone_num = int((start + end)/2)
        
        inf_zen = self.zone_data[zone_num]["inf_zen"]
        sup_zen = self.zone_data[zone_num]["sup_zen"]

        if(zen <= sup_zen):

            if(zen >= inf_zen):
                return zone_num

            else:
                return self.zenith_to_zone(zen, start, zone_num-1)

        else:
            return self.zenith_to_zone(zen, zone_num+1, end)
        
        
        
        
    def azimuth_to_patch(self, zone_num, az, start=None, end=None):
        
        """
        Bin azimuth value into the correct sky patch via binary search.
        
        Parameters
        ----------
        zone_num : int
            Sky zone to which the azimuth value "belongs".
        
        az : float
            Azimuth value in degrees. Must be between 0 and 360.
        
        start : int or None
            Lower search bound for patch. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for patch. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        local_patch_num : int
            Sky patch (int) to which the zenith coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): 
            start = 0
        
        if(end is None):
            end = self.zone_data[zone_num]["num_patches"] - 1
            

        if(start > end):
            return "Not found"

        local_patch_num = int((start + end)/2)
        inf_az = self.patch_data[(zone_num, local_patch_num)]["inf_az"]
        sup_az = self.patch_data[(zone_num, local_patch_num)]["sup_az"]
        
        if(az <= sup_az):

            if(az >= inf_az):
                return local_patch_num

            else:
                return self.azimuth_to_patch(zone_num, az, start, local_patch_num-1)

        else:
            return self.azimuth_to_patch(zone_num, az, local_patch_num+1, end)
        
        
        
    # ------ METHODS FOR ZONE-PATH LOCALIZATION FROM COORDINATES: DISK -------       
        
    def disk_point_to_zone_patch(self, rad, az):
        
        """
        Bin disk point into the correct sky patch. That is, given a disk point
        represented by a tuple of (radius, azimuth) values, return the sky patch, 
        represented by a tuple of (zone_num, local_patch_num), to which said
        disk point belongs.
        
        Parameters
        ----------
        rad : float
            Radius of disk point [adm]. Must be between 0 and 1.
        
        az : float
            Azimuth of disk point in degrees. Must be between 0 and 360.
            
        Returns
        -------
        zone_num : int
            Sky zone to which the disk point belongs.
        
        local_patch_num : int
            Sky patch (int) (identified by its local patch number in reference
            to the sky zone) to which the disk point belongs, or "not found"
            if search failed.
        
        
        """
        
        zone_num = self.rad_to_zone(rad)
        local_patch_num = self.azimuth_to_patch(zone_num, az)
        return zone_num, local_patch_num
        
        
    def rad_to_zone(self, rad, start=None, end=None):
        
        """
        Bin radius value into the correct sky zone via binary search.
        
        Parameters
        ----------
        rad : float
            radius value [adm]. Must be between 0 and 1.
        
        start : int or None
            Lower search bound for zone. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for zone. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        zone_num : int
            Sky zone (int) to which the radius coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): start = 0
        if(end is None): end = self.zone_max_key
        

        if(start > end):
            return "Not found"

        zone_num = int((start + end)/2)
        
        inf_rad = self.zone_data[zone_num]["inf_rad"]
        sup_rad = self.zone_data[zone_num]["sup_rad"]

        if(rad <= sup_rad):

            if(rad >= inf_rad):
                return zone_num

            else:
                return self.rad_to_zone(rad, start, zone_num-1)

        else:
            return self.rad_to_zone(rad, zone_num+1, end)
        
    
    # ------ METHODS FOR VISUALIZATION OF DISCRETISED SKY VAULT: DISK -------   
    
    def plot_disk_patches(self, figsize=(12,12)):
        
        """
        Visualize discretized Sky Vault in 2D.
        
        Paramters
        ---------
        figsize : 2-tuple of int
            Size of figure.
    
        """

        _, ax = plt.subplots(figsize=figsize, 
        subplot_kw={'projection': 'polar'})
        kwargs = {"edgecolor":"k",  "facecolor":"white"}

        for zone_num, zone_dict in self.zone_data.items():
            
            r0 = zone_dict["inf_rad"]
            r1 = zone_dict["sup_rad"]

            for i in range(zone_dict["num_patches"]):

                theta0 = np.deg2rad(zone_dict["azimuths"][i])
                theta1 = np.deg2rad(zone_dict["azimuths"][i+1])

                ax.bar(x = 0.5*(theta0 + theta1),
                       height = r1 - r0, 
                       width = theta1 - theta0, 
                       bottom = r0, 
                       **kwargs)

                ax.set_rlim(0, 1)
                ax.set_title("Discretised Sky Vault : 2D Visualization | N = 0°, E = 90°, S = 180°, W = 270°")

        plt.show()

        return None
    
    # ------ METHODS FOR VISUALIZATION OF DISCRETISED SKY VAULT: SPHERE -------
    
    
    def _compute_sphere_patch_lines(self, phis, thetas, n):
        """
        Private helper function. Computes the borders of a sky patch for 
        plotting.

        Parameters
        ----------
        phis : array-like
            Array-like of length 2 containing the lower and upper limits of the 
            range of phi (zenith angle) values for the patch, in radians.
            
        thetas : array-like
            Array-like of length 2 containing the lower and upper limits of the
            range of theta values for the patch, in radians.
            
        n : int
            Number of points to use for drawing the border of the patch.
            

        Returns
        -------
        lines : dict
            The function returns a dictionary lines with four keys, each 
            representing one of the borders of the patch. The values for each 
            key are dictionaries containing the x, y, and z coordinates of the
            points defining that border. 

        """

        lines = {}
        phi_arr   = np.linspace(phis[0], phis[1], n)
        theta_arr = np.linspace(thetas[0], thetas[1], n)

        for i in [0, 1]:

            lines[i] = {"x" : np.cos(phis[i])*np.sin(theta_arr), 
                        "y" : np.sin(phis[i])*np.sin(theta_arr), 
                        "z" : np.cos(theta_arr)}

            lines[i+2] = {"x" : np.cos(phi_arr)*np.sin(thetas[i]), 
                          "y" : np.sin(phi_arr)*np.sin(thetas[i]), 
                          "z" : np.cos(thetas[i])}

        return lines
    
    
    def plot_sphere_patches(self, figsize=(12,12), axis_view=(25, 30)):
        
        """
        Visualize discretized Sky Vault in 3D.
        
        Paramters
        ---------
        figsize : 2-tuple of int
            Size of figure.
            
        axis_view = 2-tuple of int
            Plot's elevation, azimuth in degrees.
    
        """
            
        n = 10        
        _ = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")
 
        el, az = axis_view
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(el, az)

        for zone_num, zone_dict in self.zone_data.items(): 
            
            thetas = np.deg2rad([zone_dict["inf_zen"], zone_dict["sup_zen"]])

            for i in range(zone_dict["num_patches"]):

                phis = np.deg2rad(zone_dict["azimuths"][i:i+2])

                lines = self._compute_sphere_patch_lines(phis, thetas, n)

                for line in lines.values():
                    ax.plot(line["x"], line["y"], line["z"], color="k")
                    
        ax.set_title("Discretised Sky Vault : 3D Visualization")     
        ax.set_xlabel("X (↑ == N, ↓ == S)")
        ax.set_ylabel("Y (↑ == E, ↓ == W)")
        plt.show()

        return None
    

    
    

        
        
    def _compute_simulation_times(self, nt, year, month, day, t_initial = None, t_final = None):
        
        """
        Private helper function. Compute simulation times required by other 
        methods of this class. Computes the times of simulation based on the 
        sunrise and sunset times computed via the 'self.sunrise_sunset_apel' 
        parameter, in case that t_inital or t_final are not specified.
        
        Parameters
        ----------
        nt : int
            Number of time samples to use when computing the simulation times.
            
        year : int
            Year for which the simulation times are to be computed. Must be
            present in 'self.Site_obj'.
            
        month : int
            Month for which the simulation times are to be computed. Must be
            present in 'self.Site_obj'.
            
        day : int
            Day for which the simulation times are to be computed. Must be
            present in 'self.Site_obj'.
            
        t_inital : float or None
            If float: 
                It is the initial fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[0] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[0] is used as
                the first simulation hour.
                
                Else: The sunrise time is computed and used as the first simulation hour.
                
    
        t_final : float or None
            If float: 
                It is the final fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[-1] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[-1] is used as 
                the last simulation hour.
                
                Else: The sunset time is computed and used as the last simulation hour.
            
            
             
        Returns
        -------
        sim_times : pandas.Series of pandas.Timestamp obj
            Simulation times.
        
        """
        
        
        # ------ RETIEVAL OF NECESSARY VARIABLES ------
        
        Timestamp_index = self.Site_obj.time_data[(year, month, day)]
        sun_apel        = self.Site_obj.sun_data[(year, month, day)]["apel"]
        hms_float       = self.Site_obj.site_data[(year, month, day)]["hms_float"]
        
        
        # ------ COMPUTATION OF INITIAL SIMULATION TIME ------
        
        if t_initial is None:
            if sun_apel.iloc[0] >= self.sunrise_sunset_apel:
                t_init  = hms_float.iloc[0]
                ts_init = Timestamp_index.loc[0]
            else:
                c = 0
                while sun_apel.iloc[c] < self.sunrise_sunset_apel: c += 1
                apel0, apel1 = sun_apel.iloc[c - 1],  sun_apel.iloc[c]
                t0,    t1    = hms_float.iloc[c - 1], hms_float.iloc[c]
                t_init = (t1-t0)/(apel1-apel0)*(self.sunrise_sunset_apel - apel0) + t0
                ts_init = Timestamp_index.loc[c] - pd.Timedelta(t1 - t_init, unit = "h")
        
        else:
            t_init = t_initial
            hour   = int(t_init)
            minute = int((t_init % 1)*60)
            second = ((t_init % 1)*60 % 1)*60
            ts_init = pd.Timestamp(f"{year}-{month}-{day} {hour}:{minute}:{second}",
                      tz = Timestamp_index.loc[0].tz)
        
        


        # ------ COMPUTATION OF FINAL SIMULATION TIME ------
            
        if t_final is None:
            if sun_apel.iloc[-1] >= self.sunrise_sunset_apel:
                t_fin = hms_float.iloc[-1]
            else:
                c = -1
                while sun_apel.iloc[c] < self.sunrise_sunset_apel: c -= 1
                apel0, apel1 = sun_apel.iloc[c], sun_apel.iloc[c + 1]
                t0,    t1 = hms_float.iloc[c],   hms_float.iloc[c + 1]
                t_fin = (t1-t0)/(apel1-apel0)*(self.sunrise_sunset_apel - apel0) + t0
        else:
            t_fin = t_final
            
        
        # ------ COMPUTATION OF FINAL SIMULATION TIMES ------
            
        dt = (t_fin - t_init)/(nt - 1)
        new_hms_float = np.linspace(t_init, t_fin, nt)
        
        new_Timestamp_index = [ts_init]
        timedelta = pd.Timedelta(dt, unit="h")
        for i in range(nt-1):
            new_Timestamp_index.append(new_Timestamp_index[i] + timedelta)
            
            
        sim_times = pd.Series(data = new_hms_float, index = new_Timestamp_index)
        
        
        return sim_times
    



    def compute_spectral_radiance_for_a_date(self, year, month, day, nel = 46, naz = 181, nt = 241, kind = "linear", mean_surface_tilt = 0, num_iterations = 500, t_initial = None, t_final = None):
        
        """
        Compute spectral radiance across time for a complete day on a specific
        date, using the data stored in the 'self.Site_obj' attribute.
        
        Parameters
        ----------
        year : int
            Year for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj'.
            
        month : int
            Month for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj'.
            
        day : int
            Day for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj'.
            
        nel : int
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.

        nt : int
            Number of time samples used for computing the sky radiance across 
            time. Default is 241.
            
        kind : str
            Interpolation method for 'self.Site_obj' variables. Supported are 
            those specified in scipy's documentation for 'interp1d' function.
            
        mean_surface_tilt : float or numpy.array of floats with shape (nt,)
            Mean panel tilt from horizontal [degrees] across time. Default is 0.
        
        num_iterations : int
            Number of iterations to use when filling NaN data. Default is 500.
            
        t_inital : float or None
            If float: 
                It is the initial fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[0] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[0] is used as
                the first simulation hour.
                
                Else: The sunrise time is computed and used as the first simulation hour.
                
    
        t_final : float or None
            If float: 
                It is the final fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[-1] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[-1] is used as 
                the last simulation hour.
                
                Else: The sunset time is computed and used as the last simulation hour.
            
        Returns
        -------
        None

        Produces
        -------
        self.spectral_radiance_res : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : numpy.array of floats with shape (nel,naz)
                    Azimuth array of meshgrid of Azimuth, Elevation values. It contains
                    the azimuth (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'Az' varies
                    along axis 1. Values are between 0 and 360.
                
                "El" : numpy.array of floats with shape (nel,naz)
                    Elevation array of meshgrid of Azimuth, Elevation values. It contains
                    the elevation (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'El' varies
                    along axis 0. Values are between 0 and 90.
                
                "Siv" : numpy.array of floats with shape (nt,)   
                    Igawa's 'Sky Index' parameter across time.
                
                "Kc" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Clear Sky Index' parameter across time.
                    
                "Cle" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Cloudless Index' parameter across time.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "Timestamp_index" : pandas.Series of pandas.Timestamp objects.
                    Series of Timestamp values detailing the times at which each of the
                    samples of the time-dependent variables were taken. We denote its 
                    length as nt.
                    
                "direct" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Direct component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "dffuse" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Diffuse component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
        
                                        
        Notes
        -----
        1) Initial time and final time of simulation are taken to be 
           self.Site_obj.time_data[(year, month, day)][0] and 
           self.Site_obj.time_data[(year, month, day)][-1] (respectively), 
           unless those times correspond to a time before the sun has come
           out/after the sun has set. If this is the case, the sunrise and 
           sunset times are estimated and those used.
           
         2) Angular resolution in the Elevation coordinate is equal to 90/(nel - 1).
    
         3) Angular resolution in the Elevation coordinate is equal to 360/(naz - 1).
    
         4) The time resolution used varies with the length of each day, but
            is constant for one single day. The exact time resolution used for
            the computation can extracted from "Timestamp_index".
    
         5) "mean_surface_tilt" variable really only affects the computation of
            the spectral distribution of diffuse radiance. It has no effect on 
            the actual value. 

        """


        # ------ DISCRETIZE SKY VAULT ------#
        dAz, dEl = 360/(naz-1), 90/(nel-1)    
        Az, El = np.meshgrid(np.linspace(0, 360, naz), np.linspace(0, 90, nel)) 
        
        # ------ COMPUTE SIMULATION TIMES FOR THE DATE ------
        sim_times = self._compute_simulation_times(nt, year, month, day, t_initial, t_final)
        new_Timestamp_index, new_hms_float = sim_times.index, np.array(sim_times)


        # ------ COMPUTE RELATIVE AIRMASS ------#
        interpd_rel_airmass =\
        self.Site_obj.time_interpolate_variable(
        col           = "rel_airmass", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind)
        
        # Replace possible NaN values with the maximum value avilable.
        interpd_rel_airmass[pd.isnull(interpd_rel_airmass)] = 38
        
        
        # ------- COMPUTE SPECTRAL RADIANCE --------
        res =\
        compute_spectral_radiance(
        Az = Az, 
        
        El = El,
        
        dAz = dAz,
        
        dEl = dEl,
        
        Timestamp_index = new_Timestamp_index,  
        
        sun_apel =\
        self.Site_obj.time_interpolate_variable(
        col           = "apel", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        sun_az =\
        self.Site_obj.time_interpolate_variable(
        col           = "az", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        Gh =\
        self.Site_obj.time_interpolate_variable(
        col           = "G(h)", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        extra_Gbn =\
        self.Site_obj.time_interpolate_variable(
        col           = "extra_Gb(n)", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        Gbn =\
        self.Site_obj.time_interpolate_variable(
        col           = "Gb(n)", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind), 
        
        Gdh =\
        self.Site_obj.time_interpolate_variable(
        col           = "Gd(h)", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        SP =\
        self.Site_obj.time_interpolate_variable(
        col           = "SP", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        rel_airmass =\
        interpd_rel_airmass,
        
        H2O =\
        self.Site_obj.time_interpolate_variable(
        col           = "H2O", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        O3 =\
        self.Site_obj.time_interpolate_variable(
        col           = "O3", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        AOD_500nm =\
        self.Site_obj.time_interpolate_variable(
        col           = "AOD_500nm", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        alpha_500nm =\
        self.Site_obj.time_interpolate_variable(
        col           = "alpha_500nm", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        spectrally_averaged_aaf =\
        self.Site_obj.time_interpolate_variable(
        col           = "spectrally_averaged_aaf", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        single_scattering_albedo =\
        self.Site_obj.time_interpolate_variable(
        col           = "single_scattering_albedo", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        ground_albedo =\
        self.Site_obj.time_interpolate_variable(
        col           = "ground_albedo", 
        year          = year, 
        month         = month, 
        day           = day, 
        new_hms_float = new_hms_float,
        kind          = kind),
        
        mean_surface_tilt =\
        mean_surface_tilt,
        
        num_iterations =\
        num_iterations)
            
        res["Az"] = Az
        res["El"] = El
        res["Timestamp_index"] = new_Timestamp_index
        
        self.spectral_radiance_res = res
        
        return None
    
    
    
    def compute_radiance_for_a_date(self):
        """
        Compute the radiance for a given date by integrating the (already
        computed) spectral radiance over the wavelength axis.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        -------
        self.radiance_res : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : numpy.array of floats with shape (nel,naz)
                    Azimuth array of meshgrid of Azimuth, Elevation values. It contains
                    the azimuth (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'Az' varies
                    along axis 1. Values are between 0 and 360.
                
                "El" : numpy.array of floats with shape (nel,naz)
                    Elevation array of meshgrid of Azimuth, Elevation values. It contains
                    the elevation (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'El' varies
                    along axis 0. Values are between 0 and 90.
                
                "Siv" : numpy.array of floats with shape (nt,)   
                    Igawa's 'Sky Index' parameter across time.
                
                "Kc" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Clear Sky Index' parameter across time.
                    
                "Cle" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Cloudless Index' parameter across time.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "Timestamp_index" : pandas.Series of pandas.Timestamp objects.
                    Series of Timestamp values detailing the times at which each of the
                    samples of the time-dependent variables were taken. We denote its 
                    length as nt.
                    
                "direct" : numpy.array of floats with shape (nel,naz,nt)
                    Direct component of radiance across time. It has units
                    of W/m^2/sr.
                    
                "dffuse" : numpy.array of floats with shape (nel,naz,nt)
                    Diffuse component of radiance across time. It has units
                    of W/m^2/sr.
                    
        Notes
        -----
        1) This method requires the attribute 'self.spectral_radiance' to
           already be defined.  For this, please check the 
           'self.compute_spectral_radiance_for_a_date' method.
        """
        
        # Get number of sample points across space and time.
        nel, naz = self.spectral_radiance_res["Az"].shape
        nt = len(self.spectral_radiance_res["Timestamp_index"])
        
        # Initialize arrays for storing the radiance across time for all
        # points in the sky.
        new_direct  = np.zeros((nel, naz, nt))
        new_diffuse = np.zeros((nel, naz, nt))
        
        # --- COMPUTE RADIANCE BY INTEGRATING SPECTRAL RADIANCE ACROSS THE WAVELENGTHS AXIS ---
        for t in range(nt):
            
            new_direct[:,:,t] =\
            simpson(
            y    = self.spectral_radiance_res["direct"][t], 
            x    = self.spectral_radiance_res["wavelengths"],
            axis = 2
            )
            
            new_diffuse[:,:,t] =\
            simpson(
            y    = self.spectral_radiance_res["diffuse"][t], 
            x    = self.spectral_radiance_res["wavelengths"],
            axis = 2
            )
            
        # Copy all other related variables that are not 'diffuse', 'direct', 'wavelengths'.
        self.radiance_res =\
        {key : val for key, val in self.spectral_radiance_res.items() 
         if key not in ["direct","diffuse", "wavelengths"]}

        # Include the direct and diffuse radiances.
        self.radiance_res["direct"]  = new_direct 
        self.radiance_res["diffuse"] = new_diffuse
        
            
        return None
        
                
            
    def compute_time_integrated_spectral_radiance_for_a_date_interval(self, years, months, days, nel = 46, naz = 181, nt = 241, kind = "linear", mean_surface_tilt = 0, num_iterations = 500, t_initial = None, t_final = None):
        
        """
        Compute time integral of spectral radiance for a specified interval of
        dates, using the data stored in the 'self.Site_obj' attribute.
        
        Parameters
        ----------
        years : list of years or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the years that are to be used for the computation. 
            The first element of 'years' would be the lower bound, while the 
            second element would be the upper bound. If None, the lower and
            upper bounds for the 'years' variable are automatically selected by 
            the program so that all avialable years are included.
            
        months : list of months or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the months that are to be used for the computation. 
            The first element of 'months' would be the lower bound, while the 
            second element would be the upper bound. If None, the lower and 
            upper bounds for the 'months'  variable are automatically selected 
            by the program so that all avialable months are included.
            
        days : list of days or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the days that are to be used for the computation. 
            The first element of 'days' would be the lower bound, while the 
            second element would be the upper bound. If None, the lower and 
            upper bounds for the 'days' variable are automatically selected by 
            the program so that all avialable days are included.
                
        nel : int
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.

        nt : int
            Number of time samples used for computing the sky radiance across 
            time. Default is 241.
            
        kind : str
            Interpolation method. Supported are those specified in scipy's 
            documentation for 'interp1d' function.
            
        mean_surface_tilt : float or numpy.array of floats with shape (nt,)
            Mean panel tilt from horizontal [degrees] across time. Default is 0.
        
        num_iterations : int
            Number of iterations to use when filling NaN data. Default is 500.
            
        t_inital : float or None
            If float: 
                It is the initial fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[0] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[0] is used as
                the first simulation hour.
                
                Else: The sunrise time is computed and used as the first simulation hour.
                
    
        t_final : float or None
            If float: 
                It is the final fraction-hour of simulation. It should
                be a number that lies between self.Site_obj.time_data[(year, month, day)].iloc[0]
                and self.Site_obj.time_data[(year, month, day)].iloc[-1], while
                satisfying t_inital < t_final.
            
            If None (default):
                If self.Site_obj.sun_data[(year, month, day)]["apel"].iloc[-1] ≥ self.sunrise_sunset_apel:
                The hour encoded by self.Site_obj.time_data[(year, month, day)].loc[-1] is used as 
                the last simulation hour.
                
                Else: The sunset time is computed and used as the last simulation hour.

        Returns
        -------

        Produces
        -------
        self.time_integrated_spectral_radiance_res : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                
                "Az" : numpy.array of floats with shape (nel,naz)
                    Azimuth array of meshgrid of Azimuth, Elevation values. It contains
                    the azimuth (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'Az' varies
                    along axis 1. Values are between 0 and 360.
                
                "El" : numpy.array of floats with shape (nel,naz)
                    Elevation array of meshgrid of Azimuth, Elevation values. It contains
                    the elevation (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'El' varies
                    along axis 0. Values are between 0 and 90.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "direct" : numpy.array of floats with shape (nel,naz,122)
                    Time integral of direct component of spectral radiance across time.
                    It has units of Wh/m^2/sr/nm.
                    
                "dffuse" : numpy.array of floats with shape (nel,naz,122)
                    Time integral of diffuse component of spectral radiance across time.
                    It has units of Wh/m^2/sr/nm.
        
                                        
        Notes
        -----
        1) Initial time and final time of simulation are taken to be 
           self.Site_obj.time_data[(year, month, day)][0] and 
           self.Site_obj.time_data[(year, month, day)][-1] (respectively), 
           unless those times correspond to a time before the sun has come
           out/after the sun has set. If this is the case, the sunrise and 
           sunset times are estimated and those used.
           
         2) Angular resolution in the Elevation coordinate is equal to 90/(nel - 1).
    
         3) Angular resolution in the Elevation coordinate is equal to 360/(naz - 1).
    
         4) The time resolution used varies with the length of each day, but
            is constant for one single day. The exact time resolutions used for
            the computation can extracted from "self.time_integrated_sim_times".
    
         5) "mean_surface_tilt" variable really only affects the computation of
            the spectral distribution of diffuse radiance. It has no effect on 
            the actual value. 

        """
        
        # Process the data years, months, days data in order to define an
        # date interval.
        if not isinstance(years,  list):  years  = [years, years]
        if not isinstance(months, list):  months = [months, months]
        if not isinstance(days,   list):  days   = [days, days] 
            
        if years[0]  is None : years[0]  = - np.inf
        if years[1]  is None : years[1]  =   np.inf
        if months[0] is None : months[0] = - np.inf
        if months[1] is None : months[1] =   np.inf
        if days[0]   is None : days[0]   = - np.inf
        if days[1]   is None : days[1]   =   np.inf
        
        years  = [min(years),  max(years)]
        months = [min(months), max(months)]
        days   = [min(days),   max(days)]
        
        
        # ------ DISCRETIZE SKY VAULT ------#
        dAz, dEl = 360/(naz-1), 90/(nel-1)    
        Az, El = np.meshgrid(np.linspace(0, 360, naz), np.linspace(0, 90, nel)) 
        
        
        # Initialize results dict.
        self.time_integrated_spectral_radiance_res =\
        {"direct"  : np.zeros((nel, naz, 122)),
         "diffuse" : np.zeros((nel, naz, 122))}
        
        
        # --- COMPUTE SPECTRAL RADIANCE ACROSS TIME FOR ALL DATES INSIDE THE SPECIFIED RANGE ---
        self.time_integrated_sim_times = {}

        for (year, month, day) in self.Site_obj.time_data.keys():
            
            # We plot the data which is inside the interval previously 
            # specified.
            if year  < years[0]  or year > years[1]:   continue
            if month < months[0] or month > months[1]: continue
            if day   < days[0]   or day > days[1]:     continue
        
            print(year, month, day)
        
            # ------ COMPUTE SIMULATION TIMES FOR A GIVEN DATE ------
            sim_times = sim_times = self._compute_simulation_times(nt, year, month, day, t_initial, t_final)
            new_Timestamp_index, new_hms_float = sim_times.index, np.array(sim_times)
            
            # Store simulated times of all dates.
            self.time_integrated_sim_times[(year, month, day)]  =  sim_times

            
            # ------ COMPUTE RELATIVE AIRMASS ------#
            interpd_rel_airmass =\
            self.Site_obj.time_interpolate_variable(
            col           = "rel_airmass", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind)
            
            # replace possible NaN values with the maximum value avilable.
            interpd_rel_airmass[pd.isnull(interpd_rel_airmass)] = 38
            
            
            # ------- COMPUTE SPECTRAL RADIANCE --------
            res =\
            compute_spectral_radiance(
            Az = Az, 
            
            El = El,
            
            dAz = dAz,
            
            dEl = dEl,
            
            Timestamp_index = new_Timestamp_index,  
            
            sun_apel =\
            self.Site_obj.time_interpolate_variable(
            col           = "apel", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            sun_az =\
            self.Site_obj.time_interpolate_variable(
            col           = "az", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            Gh =\
            self.Site_obj.time_interpolate_variable(
            col           = "G(h)", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            extra_Gbn =\
            self.Site_obj.time_interpolate_variable(
            col           = "extra_Gb(n)", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            Gbn =\
            self.Site_obj.time_interpolate_variable(
            col           = "Gb(n)", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind), 
            
            Gdh =\
            self.Site_obj.time_interpolate_variable(
            col           = "Gd(h)", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            SP =\
            self.Site_obj.time_interpolate_variable(
            col           = "SP", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            rel_airmass =\
            interpd_rel_airmass,
            
            H2O =\
            self.Site_obj.time_interpolate_variable(
            col           = "H2O", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            O3 =\
            self.Site_obj.time_interpolate_variable(
            col           = "O3", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            AOD_500nm =\
            self.Site_obj.time_interpolate_variable(
            col           = "AOD_500nm", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            alpha_500nm =\
            self.Site_obj.time_interpolate_variable(
            col           = "alpha_500nm", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            spectrally_averaged_aaf =\
            self.Site_obj.time_interpolate_variable(
            col           = "spectrally_averaged_aaf", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            single_scattering_albedo =\
            self.Site_obj.time_interpolate_variable(
            col           = "single_scattering_albedo", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            ground_albedo =\
            self.Site_obj.time_interpolate_variable(
            col           = "ground_albedo", 
            year          = year, 
            month         = month, 
            day           = day, 
            new_hms_float = new_hms_float,
            kind          = kind),
            
            mean_surface_tilt =\
            mean_surface_tilt,
            
            num_iterations =\
            num_iterations)
                
            # --- INTEGRATE SPECTRAL RADIANCE ACROSS TIME FOR THE CURRENT DATE ---
            # For this, we make use of the trapezoid rule.
            
            direct, diffuse = res["direct"], res["diffuse"]
            dt = (new_hms_float[-1] - new_hms_float[0])/(nt - 1)
            
            direct[0],   direct[-1] =  direct[0]/2,  direct[-1]/2
            diffuse[0], diffuse[-1] = diffuse[0]/2, diffuse[-1]/2
            
            direct, diffuse = sum(direct)*dt, sum(diffuse)*dt
            
            
            # --- ACCUMULATE EACH DATE'S TIME INTEGRAL TO GET THE TIME INTEGRAL OF THE FULL INTERVAL ----
            self.time_integrated_spectral_radiance_res["direct"]  += direct
            self.time_integrated_spectral_radiance_res["diffuse"] += diffuse
            
        # Add some parameters to the results dict.
        self.time_integrated_spectral_radiance_res["Az"] = Az
        self.time_integrated_spectral_radiance_res["El"] = El
        self.time_integrated_spectral_radiance_res["wavelengths"] = res["wavelengths"]
        
        return None
    
    
    
    def compute_time_integrated_radiance_for_a_date_interval(self):
        
        """
        Compute the time-integrated radiance for a given date interval by 
        integrating the (already computed) time-integrated spectral radiance
        over the wavelength axis.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        -------
        self.time_integrated_radiance_res : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : numpy.array of floats with shape (nel,naz)
                    Azimuth array of meshgrid of Azimuth, Elevation values. It contains
                    the azimuth (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'Az' varies
                    along axis 1. Values are between 0 and 360.
                
                "El" : numpy.array of floats with shape (nel,naz)
                    Elevation array of meshgrid of Azimuth, Elevation values. It contains
                    the elevation (in degrees) of each sky element to be considered in 
                    the calculation of spectral sky radiance. The values of 'El' varies
                    along axis 0. Values are between 0 and 90.
                    
                "direct" : numpy.array of floats with shape (nel,naz)
                    Spectral integral of direct component of time-integrated 
                    spectral radiance. It has units of Wh/m^2/sr.
                    
                "dffuse" : numpy.array of floats with shape (nel,naz)
                    Spectral integral of diffuse component of time-integrated 
                    spectral radiance. It has units of Wh/m^2/sr.
                    
        Notes
        -----
        1) This method requires the attribute 'self.time_integrated_spectral_radiance_res'
           to already be defined. For this, please check the 
           'compute_time_integral_of_spectral_radiance_for_a_date_interval' method.
        """
        
        # Get number of sample points across space.
        nel, naz = self.time_integrated_spectral_radiance_res["Az"].shape
        
        # --- COMPUTE TIME-INTEGRATED RADIANCE BY INTEGRATING TIME-INTEGRATED SPECTRAL RADIANCE ACROSS THE WAVELENGTHS AXIS ---
        new_direct =\
        simpson(
        y    = self.time_integrated_spectral_radiance_res["direct"], 
        x    = self.time_integrated_spectral_radiance_res["wavelengths"],
        axis = 2
        )
        
        new_diffuse =\
        simpson(
        y    = self.time_integrated_spectral_radiance_res["diffuse"], 
        x    = self.time_integrated_spectral_radiance_res["wavelengths"],
        axis = 2
        )
        
            
        # Copy all other related variables that are not 'diffuse', 'direct' or wavelengths.
        self.time_integrated_radiance_res =\
        {key : val for key, val in self.time_integrated_spectral_radiance_res.items() 
         if key not in ["direct","diffuse", 'wavelengths']}

        # Include the direct and diffuse radiances.
        self.time_integrated_radiance_res["direct"]  = new_direct 
        self.time_integrated_radiance_res["diffuse"] = new_diffuse
        
        return None
    
    
    
    
    
    def compute_time_integrated_spectral_irradiances_for_a_date_interval(self, int_nzen = 20, int_naz = 30):
        
        """
        Compute the time-integrated spectral irradiance over each Sky patch (for
        a given date interval) by integrating the (already computed) 
        time-integrated spectral radiance with respect to the solid angle, over
        each sky patch of the discretised Sky Vault.
        
        Parameters
        ----------
        int_nzen : int
            Number of samples for dicretizing each sky patch, with regards to
            the zenith coordinate, in order to compute the diffuse spectral
            irradiance via integration. Default is 20.
            
        int_naz : int
            Number of samples for dicretizing each sky patch, with regards to
            the zenith coordinate, in order to compute the diffuse spectral
            irradiance via integration. Default is 30.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.patch_data[(zone_num, local_patch_num)]["time-integrated spectral irradiance"] : dict of dicts
            Each sky patch recieves a new key in its database called 
            "time-integrated spectral irradiance". This is a dict with keys:
            "direct", "diffuse", "global", each which holds another dictionary
            containing the relevant information about the direct, diffuse
            and global spectral irradiance (respectively) related to that 
            particular sky patch. Each of these dicts contains the following
            key-value pairs:
                
                Keys : Values
                -------------
                "vector" : np.array of floats with shape (3,122)
                    Direct/Diffuse/Global (depending on the case) Spectral
                    Irradiance vector. "vector"[0,:], vector"[1,:] and
                    vector"[2,:], hold the x, y and z components of the 
                    spectral irradiance vector, respectively, for all
                    wavelengths in key "wavelnegths". Each component
                    has units of Wh/m^2/nm.
                    
                "magnitude" : np.array of floats with shape (122,)
                    Magnitude of the Direct/Diffuse/Global (depending on the
                    case) Spectral Irradiance vector. It has units of Wh/m^2/nm.
                    
                "spectrally_averaged_unit_vector" : np.array of floats with shape (3,)
                    Average position of irradiance within a sky patch. That is,
                    it is the unit vector version of the spectrally averaged
                    Direct/Diffuse/Global (depending on the case) Spectral
                    Irradiance vector. In the case, however, that said Spectral
                    Irradiance vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.
                    
        self.patch_data[(zone_num, local_patch_num)]["time-integrated spectral irradiance"]["wavelengths"] : np.array of floats with shape (122,)
                    Array of wavelengths over which the spectral irradiances 
                    are defined.
      
                    
       self.time_integrated_spectral_irradiance_res : dict of numpy.arrays
                    Dict containing the same info as above, but in another format
                    that is handier for other things. It has the following
                    key-value pairs:
                        
                    Keys : Values
                    -------------
                    "direct" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Array containing the Direct Spectral Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "magnitude_direct" : numpy.array of floats with shape (self.num_divisions, 122)
                        Array containing the Magnitude of the Direct Spectral Irradiance vector 
                        for each of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "spectrally_averaged_unit_direct" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containig the average position of irradiance within 
                        the sky patch, for each sky patch. That is,
                        it is the unit vector of the spectrally averaged
                        Direct Spectral Irradiance vector, for all
                        sky patches. In the case, however, that said Spectral
                        Irradiance vector is zero for a given sky patch, we 
                        default to using the unit vector pointing to the center 
                        of the sky patch, for the row corresponding to said sky patch.
                        Each row is adimensional.
                        
                    "diffuse" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Array containing the Diffuse Spectral Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "magnitude_diffuse" : numpy.array of floats with shape (self.num_divisions, 122)
                        Array containing the Magnitude of the Diffuse Spectral Irradiance vector 
                        for each of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "spectrally_averaged_unit_diffuse" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containig the average position of irradiance within 
                        the sky patch, for each sky patch. That is,
                        it is the unit vector of the spectrally averaged
                        Diffuse Spectral Irradiance vector, for all
                        sky patches. In the case, however, that said Spectral
                        Irradiance vector is zero for a given sky patch, we 
                        default to using the unit vector pointing to the center 
                        of the sky patch, for the row corresponding to said sky patch.
                        Each row is adimensional.
                        
                    "global" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Array containing the Global Spectral Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "magnitude_global" : numpy.array of floats with shape (self.num_divisions, 122)
                        Array containing the Magnitude of the Global Spectral Irradiance vector 
                        for each of the sky patches. Each row has units of Wh/m^2/nm.
                        
                    "spectrally_averaged_unit_global" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containig the average position of irradiance within 
                        the sky patch, for each sky patch. That is,
                        it is the unit vector of the spectrally averaged
                        Global Spectral Irradiance vector, for all
                        sky patches. In the case, however, that said Spectral
                        Irradiance vector is zero for a given sky patch, we 
                        default to using the unit vector pointing to the center 
                        of the sky patch, for the row corresponding to said sky patch.
                        Each row is adimensional.
                        
                    "wavelengths" : np.array of floats with shape (122,)
                        Array of wavelengths over which the spectral irradiances 
                        are defined.
                        
                        
        Notes
        -----
        1) This method requires that time-integrated spectral radiance already 
           be calculated. Check the method 
           "self.compute_time_integrated_spectral_radiance_for_a_date_interval"
           for more info.
                    
        """

        
        # --- INITIALIZE NEW ATTRIBUTE FOR STORING THE RESULTS IN ARRAY FORM ---
        self.time_integrated_spectral_irradiance_res =\
        {"direct"            : np.zeros((self.num_divisions, 3, 122)),
         "diffuse"           : np.zeros((self.num_divisions, 3, 122)), 
         "global"            : np.zeros((self.num_divisions, 3, 122)), 
         
         "magnitude_direct"  : np.zeros((self.num_divisions,    122)),
         "magnitude_diffuse" : np.zeros((self.num_divisions,    122)),
         "magnitude_global"  : np.zeros((self.num_divisions,    122)),
         
         "spectrally_averaged_unit_direct"  : np.zeros((self.num_divisions, 3)),
         "spectrally_averaged_unit_diffuse" : np.zeros((self.num_divisions, 3)),
         "spectrally_averaged_unit_global"  : np.zeros((self.num_divisions, 3))}
        
        
        
        # ---- RETREIVE/COMPUTE DOMAIN ARRAYS ----
        
        # Retrieve data related to the domain over which we shall compute the
        # spectral irradiance quantities.
    
        Az  = self.time_integrated_spectral_radiance_res["Az"] 
        Zen = 90 - self.time_integrated_spectral_radiance_res["El"] 
        wavelengths = self.time_integrated_spectral_radiance_res["wavelengths"]         
        Phi, Theta = np.deg2rad(Az), np.deg2rad(Zen)
        
        
        # --- INTERPOLATE DIFFUSE RADIANCE, OVER THE WHOLE DOMAIN, AT EACH WAVELENGTH -----
        
        # We need to do this in order for later calculate the diffuse spectral irradiance.
        # Since the diffuse spectral radiance is taken to be a continuous function rather 
        # than a delta dirac (unlike the direct spectral radiance), we must 
        # compute its irradiance quantity via actual integration. Hence why we 
        # need this function.
        
        interp_diffuse = []
        for wv in range(len(wavelengths)):
            interp_diffuse.append(
            RegularGridInterpolator(
            points = (Zen[:,0], Az[0,:]),
            values = self.time_integrated_spectral_radiance_res["diffuse"][:,:,wv]))


        # ---- COMPUTE SOLID ANGLE UNIT VECTORS FOR ALL SKY POINTS ----
        
        # When computing the direct spectral irradiance vector, we need to 
        # mutliply each direct spectral radiance value by its corresponding 
        # unit solid angle vector and sum them all toguether. Having these 
        # vectors already precomputed to just retrieve them later, saves us 
        # computational power.
        
        solid_angle_unit_vecs = np.zeros(list(Az.shape)+[3])
        solid_angle_unit_vecs[:,:,0] = np.cos(Phi)*np.sin(Theta)
        solid_angle_unit_vecs[:,:,1] = np.sin(Phi)*np.sin(Theta)
        solid_angle_unit_vecs[:,:,2] = np.cos(Theta)
        
        
        #     --- GET SKY POINTS WITHIN SKY PATCH ---
        
        # We go over each sky patch, retrive the sky patch's limits. Then, in the case
        # of the direct spectral irradiance, we also compute a logic array for
        # retrieving all sky points that lie within said sky patch.

        for c, ((zone_num, local_patch_num), patch_dict) in enumerate(self.patch_data.items()):
    
            # We retrieve the sky patch's limits.
            inf_az,  sup_az  = patch_dict["inf_az"],  patch_dict["sup_az"]
            inf_zen, sup_zen = patch_dict["inf_zen"], patch_dict["sup_zen"]
            
            # We compute a logic array for retrieving all sky points that
            # lie within the given sky patch. We use this for computing
            # the direct spectral radiance.
            if sup_az == 360:
                  logic_az = np.logical_and(inf_az <= Az, Az <= sup_az)
            else: logic_az = np.logical_and(inf_az <= Az, Az <  sup_az)
                
            if sup_zen == 90:
                  logic_zen = np.logical_and(inf_zen <= Zen, Zen <= sup_zen)
            else: logic_zen = np.logical_and(inf_zen <= Zen, Zen <  sup_zen)
                
            logic_patch = np.logical_and(logic_az, logic_zen) 
            
            
            
            # -------- (1) COMPUTE DIRECT COMPONENT QUANTITIES ------------


            #     --- (1.A) RETRIEVE LOCAL SOLID ANGLE UNIT VECTORS ---
            # Retrieve the unit vectors that indicate the position of each sky 
            # point, that lies within the current sky patch.
            
            num_pts = logic_patch.sum() 
            local_solid_angle_unit_vecs = np.zeros((num_pts, 3))            
            local_solid_angle_unit_vecs[:,0] = solid_angle_unit_vecs[:,:,0][logic_patch]
            local_solid_angle_unit_vecs[:,1] = solid_angle_unit_vecs[:,:,1][logic_patch]
            local_solid_angle_unit_vecs[:,2] = solid_angle_unit_vecs[:,:,2][logic_patch]
            
            
            #    --- (1.B) INTIALIZE LOCAL DIRECT SPECTRAL IRRADIANCE VECTOR --- 
            local_direct_time_integrated_spectral_irradiance_vec = np.zeros((3, 122))
            
            
            #    --- (1.C) RETRIEVE DIRECT SPECTRAL RADIANCES --- 
            # Retrieve direct component of spectral radiance, for all
            # sky points that lie within the current sky patch.
            local_direct_time_integrated_spectral_radiance_vals =\
            self.time_integrated_spectral_radiance_res["direct"][logic_patch] 
            

            # --- (1.D) COMPUTE LOCAL DIRECT SPECTRAL IRRADIANCE VECTOR ---
            
            # We compute the local direct spectral irradiance vector component
            # by component. A general procedure of the description is as follows:
            # We gather the different direct spectral radiance values within the
            # current sky patch. Multiply each value one by its corresponding 
            # unit solid angle vector and then sum them all toguethere, wavelength-wise.
            
            local_direct_time_integrated_spectral_irradiance_vec[0,:] =\
            (local_solid_angle_unit_vecs[:,0].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            local_direct_time_integrated_spectral_irradiance_vec[1,:] =\
            (local_solid_angle_unit_vecs[:,1].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            local_direct_time_integrated_spectral_irradiance_vec[2,:] =\
            (local_solid_angle_unit_vecs[:,2].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            
            # --- (1.E) COMPUTE SPECTRAL IRRADIANCE VECTOR MAGINTUDE FOR CURRENT SKY PATCH ---
            # We compute the magnitudes of the computed vector for each wavelength.
            
            local_direct_time_integrated_spectral_irradiance_magnitude =\
            np.linalg.norm(local_direct_time_integrated_spectral_irradiance_vec, axis = 0)
            

            # --- (1.F) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.
            
            local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = \
            simpson(
            y    = local_direct_time_integrated_spectral_irradiance_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd /=\
            np.linalg.norm(local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)):
                local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = patch_dict["unit_vector"]
                
                
            
        
            # --------- (2) COMPUTE DIFFUSE COMPONENT QUANTITIES --------------
            
            
            #    --- (2.A) INTIALIZE LOCAL DIFFUSE SPECTRAL IRRADIANCE VECTOR --- 
            local_diffuse_time_integrated_spectral_irradiance_vec = np.zeros((3,122))
            
            
            #   --- (2.B) COMPUTE LOCAL DOMAIN VARIABLES FOR INTEGRATION --- 
            # We are going to compute the diffuse spectral irradiance for each
            # sky patch, via integration. As such, we define a new domain for
            # such a purpose. This is necessary, as it is possible that the overall
            # angular resolution of the data is not enough to compute a good
            # approximation of the integral. Therefore, we use another domain with
            # linearly interpolated data in order to gain angular resolution and,
            # therefore, accuracy in the integration procedure.
            
            int_dAz          = np.deg2rad(sup_az  -  inf_az)/(int_naz - 1)
            int_dZen         = np.deg2rad(sup_zen - inf_zen)/(int_nzen - 1)
            int_dAzZen       = int_dAz*int_dZen
            
            # New integration domain.
            int_Az, int_Zen = np.meshgrid(np.linspace(inf_az,  sup_az,  int_naz),
                                          np.linspace(sup_zen, inf_zen, int_nzen))
            
            # Evaluation points for the interpolation of diffuse spectral radiance.
            eval_pts = np.stack([int_Zen.flatten(), int_Az.flatten()], axis=1)
            
            # We compute part of the integrand ahead of time in order so save
            # computational resources.
            int_Phi, int_Theta = np.deg2rad(int_Az), np.deg2rad(int_Zen)
            x_integrand_add_on_term = np.cos(int_Phi)*np.sin(int_Theta)**2
            y_integrand_add_on_term = np.sin(int_Phi)*np.sin(int_Theta)**2
            z_integrand_add_on_term = np.cos(int_Theta)*np.sin(int_Theta)
            
            
            #  --- (2.C) COMPUTE LOCAL DIFFUSE SPECTRAL IRRADIANCE VECTOR VIA INTEGRATION, WALENGTH BY WAVELENGTH --- 
            for wv in range(len(wavelengths)):
                
                # COMPUTE INTERPOLATED VALUES OF DIFFUSE SPECTRAL RADIANCE FOR INTEGRATION
                local_diffuse_vals_at_wv =\
                interp_diffuse[wv](eval_pts).reshape(int_nzen, int_naz)
                
                # COMPUTE INTEGRAND FOR THE 3 COMPONENTS OF THE DIFFUSE SPECTRAL IRRADIANCE VECTOR
                diffuse_x_integral_at_wv =\
                local_diffuse_vals_at_wv*x_integrand_add_on_term
                
                diffuse_y_integral_at_wv =\
                local_diffuse_vals_at_wv*y_integrand_add_on_term
                
                diffuse_z_integral_at_wv =\
                local_diffuse_vals_at_wv*z_integrand_add_on_term
                
                # COMPUTE INTEGRALS FOR THE 3 COMPONENTS OF THE DIFFUSE SPECTRAL IRRADIANCE VECTOR VIA THE TRAPEZOIDAL RULE.
                diffuse_x_integral_at_wv = 0.25*\
                (diffuse_x_integral_at_wv[:-1, :-1] +
                 diffuse_x_integral_at_wv[:-1,  1:] +
                 diffuse_x_integral_at_wv[1:,  :-1] + 
                 diffuse_x_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                                         
                diffuse_y_integral_at_wv = 0.25*\
                (diffuse_y_integral_at_wv[:-1, :-1] +
                 diffuse_y_integral_at_wv[:-1,  1:] +
                 diffuse_y_integral_at_wv[1:,  :-1] + 
                 diffuse_y_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                
                diffuse_z_integral_at_wv = 0.25*\
                (diffuse_z_integral_at_wv[:-1, :-1] +
                 diffuse_z_integral_at_wv[:-1,  1:] +
                 diffuse_z_integral_at_wv[1:,  :-1] + 
                 diffuse_z_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                
                # TRANSFER COMPUTE VALUES TO THE PREDEFINED-VARIABLE
                local_diffuse_time_integrated_spectral_irradiance_vec[0, wv] = diffuse_x_integral_at_wv
                local_diffuse_time_integrated_spectral_irradiance_vec[1, wv] = diffuse_y_integral_at_wv
                local_diffuse_time_integrated_spectral_irradiance_vec[2, wv] = diffuse_z_integral_at_wv
                
                
            # --- (2.D) COMPUTE  SPECTRAL IRRADIANCE VECTOR MAGINTUDE FOR CURRENT SKY PATCH ---
            # We compute the magnitudes of the computed vector for each wavelength.
            local_diffuse_time_integrated_spectral_irradiance_magnitude =\
            np.linalg.norm(local_diffuse_time_integrated_spectral_irradiance_vec, axis = 0)
            

            # --- (2.E) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.
            local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = \
            simpson(
            y    = local_diffuse_time_integrated_spectral_irradiance_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd /=\
            np.linalg.norm(local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)):
                local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = patch_dict["unit_vector"]
            
            
            
            # --------- (3) COMPUTE GLOBAL COMPONENT QUANTITIES --------------
            
            
            # --- (3.A) COMPUTE GLOBAL SPECTRAL IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # Global irradiance is just the sum of the direct and diffuse irradiances. As such, the 
            # global spectral irradiance vector is jsut the sum of the direct and diffuse 
            # spectral irradiance vectors, wavelength by wavelngth.
            
            local_global_time_integrated_spectral_irradiance_vec =\
            local_direct_time_integrated_spectral_irradiance_vec +\
            local_diffuse_time_integrated_spectral_irradiance_vec
            
            # --- (3.B) COMPUTE  SPECTRAL IRRADIANCE VECTOR MAGINTUDE FOR CURRENT SKY PATCH ---
            local_global_time_integrated_spectral_irradiance_magnitude =\
            np.linalg.norm(local_global_time_integrated_spectral_irradiance_vec, axis = 0)
            
            # --- (3.C) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.
            local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = \
            simpson(
            y    = local_global_time_integrated_spectral_irradiance_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd /=\
            np.linalg.norm(local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)):
                local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd = patch_dict["unit_vector"]
            
            
            
            # --------- (4) SAVE RESULTS --------------
            
            
            # Save results to each sky patch dict.
            self.patch_data[(zone_num, local_patch_num)]["time-integrated spectral irradiance"] =\
            {
            "direct"  : {"vector"      : local_direct_time_integrated_spectral_irradiance_vec,
                         "magnitude"   : local_direct_time_integrated_spectral_irradiance_magnitude,
                         "spectrally_averaged_unit_vector" : local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
                       },   
                
            "diffuse" : {"vector"      : local_diffuse_time_integrated_spectral_irradiance_vec,
                         "magnitude"   : local_diffuse_time_integrated_spectral_irradiance_magnitude,
                         "spectrally_averaged_unit_vector" : local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
                       }, 
                
            "global" : {"vector"      :  local_global_time_integrated_spectral_irradiance_vec,
                        "magnitude"   :  local_global_time_integrated_spectral_irradiance_magnitude,
                        "spectrally_averaged_unit_vector" : local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
                       },
            
            "wavelengths" : wavelengths
                
            }
            
            
            # Save results to independent attribute.
            self.time_integrated_spectral_irradiance_res["direct"][c,:,:] = local_direct_time_integrated_spectral_irradiance_vec
            self.time_integrated_spectral_irradiance_res["magnitude_direct"][c,:] = local_direct_time_integrated_spectral_irradiance_magnitude
            self.time_integrated_spectral_irradiance_res["spectrally_averaged_unit_direct"][c,:] = local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
            
            self.time_integrated_spectral_irradiance_res["diffuse"][c,:,:] = local_diffuse_time_integrated_spectral_irradiance_vec
            self.time_integrated_spectral_irradiance_res["magnitude_diffuse"][c,:] = local_diffuse_time_integrated_spectral_irradiance_magnitude
            self.time_integrated_spectral_irradiance_res["spectrally_averaged_unit_diffuse"][c,:] = local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
            
            self.time_integrated_spectral_irradiance_res["global"][c,:,:] = local_global_time_integrated_spectral_irradiance_vec
            self.time_integrated_spectral_irradiance_res["magnitude_global"][c,:] = local_global_time_integrated_spectral_irradiance_magnitude
            self.time_integrated_spectral_irradiance_res["spectrally_averaged_unit_global"][c,:] = local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd
            
            self.time_integrated_spectral_irradiance_res["wavelengths"] = wavelengths
            
        return None        
                
                
            
            
    def compute_time_integrated_irradiances_for_a_date_interval(self):
        
        """
        Compute the time-integrated irradiance over each Sky patch (for
        a given date interval) by integrating the (already computed) 
        time-integrated spectral irradiance over the wavelength axis.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        Produces
        --------
        self.patch_data[(zone_num, local_patch_num)]["time-integrated irradiance"] : dict of dicts
            Each sky patch recieves a new key in its database called 
            "time-integrated irradiance". This is a dict with keys:
            "direct", "diffuse", "global", each which holds another dictionary
            containing the relevant information about the direct, diffuse
            and global irradiance (respectively) related to that 
            particular sky patch. Each of these dicts contains the following
            key-value pairs:
                
                Keys : Values
                -------------
                "vector" : np.array of floats with shape (3,)
                    Direct/Diffuse/Global (depending on the case)
                    Irradiance vector. "vector"[0], vector"[1] and
                    vector"[2], hold the x, y and z components of the 
                    irradiance vector. Each component
                    has units of Wh/m^2.
                    
                "magnitude" : float
                    Magnitude of the Direct/Diffuse/Global (depending on the
                    case) Irradiance vector. It has units of Wh/m^2.
                    
                "unit_vector" : np.array of floats with shape (3,)
                    Unit vector version of "vector". Unless "vector" is the
                    zero-vector, in which case "unit_vector" is equal to
                    a unit vector pointing to the center of the current sky 
                    patch.
      
                    
       self.time_integrated_spectral_irradiance_res : dict of numpy.arrays
                    Dict containing the same info as above, but in another format
                    that is handier for other things. It has the following
                    key-value pairs:
                        
                    Keys : Values
                    -------------
                    "direct" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containing the Direct Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2.
                        
                    "magnitude_direct" : numpy.array of floats with shape (self.num_divisions,)
                        Array containing the Magnitude of the Direct Irradiance vector 
                        for each of the sky patches. Each element has units of Wh/m^2.
                        
                    "unit_direct" : numpy.array of floats with shape (self.num_divisions, 3)
                        Unit vector version of "direct", for each row. Unless 
                        the "direct"[c,:] has magnitude zero, in which case
                        "direct"[c,:]" is equal to a unit vector pointing to the
                        center of the sky patch, associated with to c-th row.  
                        
                    "diffuse" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containing the Diffuse Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2.
                        
                    "magnitude_diffuse" : numpy.array of floats with shape (self.num_divisions,)
                        Array containing the Magnitude of the Diffuse Irradiance vector 
                        for each of the sky patches. Each element has units of Wh/m^2.
                        
                    "unit_direct" : numpy.array of floats with shape (self.num_divisions, 3)
                        Unit vector version of "diffuse", for each row. Unless 
                        the "diffuse"[c,:] has magnitude zero, in which case
                        "diffuse"[c,:]" is equal to a unit vector pointing to the
                        center of the sky patch, associated with to c-th row.  
                        
                    "global" : numpy.array of floats with shape (self.num_divisions, 3)
                        Array containing the Global Irradiance vector for each
                        of the sky patches. Each row has units of Wh/m^2.
                        
                    "magnitude_global" : numpy.array of floats with shape (self.num_divisions,)
                        Array containing the Magnitude of the Global Irradiance vector 
                        for each of the sky patches. Each element has units of Wh/m^2.
                        
                    "unit_direct" : numpy.array of floats with shape (self.num_divisions, 3)
                        Unit vector version of "global", for each row. Unless 
                        the "global"[c,:] has magnitude zero, in which case
                        "global"[c,:]" is equal to a unit vector pointing to the
                         center of the sky patch, associated with to c-th row. 
                         
        Notes
        -----
        1) This method requires that time-integrated spectral irradiances already 
           be calculated. Check the method 
           "self.compute_time_integrated_spectral_irradiances_for_a_date_interval"
           for more info.
                    
        """
         # ----- INITIALIZE VARIABLES -----
        
        wavelengths = self.time_integrated_spectral_irradiance_res["wavelengths"]
        
        self.time_integrated_irradiance_res =\
        { "direct"           : np.zeros((self.num_divisions, 3)),
          "diffuse"          : np.zeros((self.num_divisions, 3)),
          "global"           : np.zeros((self.num_divisions, 3)),
          
          "magnitude_direct"  : np.zeros(self.num_divisions),
          "magnitude_diffuse" : np.zeros(self.num_divisions),
          "magnitude_global"  : np.zeros(self.num_divisions),
          
          "unit_direct"      : np.zeros((self.num_divisions, 3)),
          "unit_diffuse"     : np.zeros((self.num_divisions, 3)),
          "unit_global"      : np.zeros((self.num_divisions, 3))}
        
        
        
        # --- COMPUTE INTEGRAL OVER WAVELENGTHS OF EACH TYPE SPECTRAL IRRADIANCE FOR EACH SKY PATCH ---
        
        for c, ((zone_num, local_patch_num), patch_dict) in enumerate(self.patch_data.items()):
            
            new_direct_vector =\
            simpson(
            y    = patch_dict["time-integrated spectral irradiance"]["direct"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_direct_magnitude   = np.linalg.norm(new_direct_vector)
            new_direct_unit_vector = new_direct_vector/new_direct_magnitude
            
            
            new_diffuse_vector =\
            simpson(
            y    = patch_dict["time-integrated spectral irradiance"]["diffuse"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_diffuse_magnitude   = np.linalg.norm(new_diffuse_vector)
            new_diffuse_unit_vector = new_diffuse_vector/new_diffuse_magnitude
            
            
            new_global_vector =\
            simpson(
            y    = patch_dict["time-integrated spectral irradiance"]["global"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_global_magnitude   = np.linalg.norm(new_global_vector)
            new_global_unit_vector = new_global_vector/new_global_magnitude
            
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(new_direct_unit_vector)):
                new_direct_unit_vector = patch_dict["unit_vector"]
            
            if any(np.isnan(new_diffuse_unit_vector)):
                new_diffuse_unit_vector = patch_dict["unit_vector"]
                    
            if any(np.isnan(new_global_unit_vector)):
                new_global_unit_vector = patch_dict["unit_vector"]
            


            # --- SAVE THE RESULTS TO EACH SKY PATCH DATA ---
            self.patch_data[(zone_num, local_patch_num)]["time-integrated irradiance"] = \
            { "direct"  : {"vector"       : new_direct_vector,
                           "magnitude"    : new_direct_magnitude,
                           "unit_vector"  : new_direct_unit_vector
                           },
             
              "diffuse"  : {"vector"      : new_diffuse_vector,
                            "magnitude"   : new_diffuse_magnitude,
                            "unit_vector" : new_diffuse_unit_vector
                            },
              
              "global"  : {"vector"       : new_global_vector,
                           "magnitude"    : new_global_magnitude,
                           "unit_vector"  : new_global_unit_vector
                           }
              }
            
            
            # --- SAVE THE RESULTS TO A NEW ATTRIBUTE ---
            self.time_integrated_irradiance_res["direct"][c,:]         = new_direct_vector
            self.time_integrated_irradiance_res["magnitude_direct"][c] = new_direct_magnitude
            self.time_integrated_irradiance_res["unit_direct"][c]      = new_direct_unit_vector
            
            self.time_integrated_irradiance_res["diffuse"][c,:]         = new_diffuse_vector
            self.time_integrated_irradiance_res["magnitude_diffuse"][c] = new_diffuse_magnitude
            self.time_integrated_irradiance_res["unit_diffuse"][c]      = new_diffuse_unit_vector
            
            self.time_integrated_irradiance_res["global"][c,:]         = new_global_vector
            self.time_integrated_irradiance_res["magnitude_global"][c] = new_global_magnitude
            self.time_integrated_irradiance_res["unit_global"][c]      = new_global_unit_vector

        
        return None
    
    


    def _check_if_unit_vecs_are_within_sky_pacth_bounds(self):
        
        """
        Private helper method. This function goes over each of the defined sky
        pacthes and checks if the spectrally averaged unit vectors for the
        spectral irradiance (which are the same as those for irradiance) lie
        within the sky patch they belong to.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        ---------
        self.unit_vecs_within_sky_patch_bounds_check : dict of lists
            Dictiornary whose keys are (zone_num, local_patch_num) for 
            all zone_nums, patch_nums containing lists of three bool
            elements. if the first element is true, that means that 
            the spectrally averaged unit vector of direct spectral irradiance
            does fall within the sky patch bounds for that sky patch. If not,
            that means it does not. The other 2 elements do the same but for
            the diffuse and global spectral irradiances.
            
            
        Notes
        -----
        1) This method requires that time-integrated spectral irradiances already 
           be calculated. Check the method 
           "self.compute_time_integrated_spectral_irradiances_for_a_date_interval"
           for more info.
            
        """
    
        self.unit_vecs_within_sky_patch_bounds_check = {}
        
        
        for (zone_num, local_patch_num), patch_dict in self.patch_data.items():
            self.unit_vecs_within_sky_patch_bounds_check[(zone_num, local_patch_num)] = []
            
            for name in ["direct", "diffuse", "global"]:
                unit_vec = patch_dict["time-integrated spectral irradiance"]\
                                     [name]["spectrally_averaged_unit_vector"]
                                     
                zenith  = np.rad2deg(np.arccos(unit_vec[2]))
                azimuth = np.rad2deg(np.arctan2(unit_vec[1], unit_vec[0]))
                
                if azimuth < 0:
                    azimuth += 360
                    
                if patch_dict["sup_zen"] == 90:
                    zenith_within_bounds =\
                    patch_dict["inf_zen"] <= zenith <= patch_dict["sup_zen"]
                    
                else: 
                    zenith_within_bounds =\
                    patch_dict["inf_zen"] <= zenith < patch_dict["sup_zen"]
                    
                                                 
                if patch_dict["sup_az"] == 360:
                    azimuth_within_bounds =\
                    patch_dict["inf_az"] <= azimuth <= patch_dict["sup_az"]
                    
                else: 
                    azimuth_within_bounds =\
                    patch_dict["inf_az"] <= azimuth < patch_dict["sup_az"]                
                    
                    
                self.unit_vecs_within_sky_patch_bounds_check[(zone_num, local_patch_num)].\
                append(zenith_within_bounds and azimuth_within_bounds)
            
        return None    




    def plot_integrated_irradiances(self, config = None):
        
        """
        Plot time-integrated irradiances.
        
        Parameters
        ----------
        
        config : dict or None
            Dict of plot configuration options. If None (the default), it uses
            the default confifuration plot options. If dict, it should include
            one or more of the following key-value pairs:
                
            Keys : Values
            -------------
            "projection" : str
                Type of plot projection. Supported are: "disk" and "sphere".
                The first one plots the time-integrated irradiances in a 2D
                plot, while the second uses a 3D plot. Default is "disk".
                
            "mode" : str
                Component of time-integrated irradiance to plot. Supported 
                are: "direct", "diffuse" and "global". Default is "global".
                
            "figsize" : 2-tuple of int
                Figure size. Default is (13,13).
            
           "unit" : str
                Units with which to display the time integrated irradiances.
                Supported are: "Wh/m^2", "kWh/m^2", "kJ/m^2" and "MJ/m^2.
                In order, these mean: 'Watt-hours per meter squared',
                'kilo Watt-hours per meter squared', 'kilo Joules per meter squared',
                and 'Mega Joules per meter squared'. Default is "Wh/m^2".
                
            "n" : int
                Number of samples per axis to use for plot. A greater number 
                means a more detailed plot (i.e, greater resolution) but it is 
                resource intensive. Default is 500.
                
            "view" : 2-tuple of int
                Elevation, azimuth of plot camara in degrees. It applies
                only for "sphere" plot. Default is (45, 120).
                
        Returns
        -------
        None
        
        Produces
        --------
        None
        
        
        Notes
        -----
        1) This method requires that time-integrated irradiances already 
           be calculated. Check the method 
           "self.compute_time_integrated_irradiances_for_a_date_interval"
           for more info.
           
        """
        
        config_ =\
        {"projection":"disk", "mode":"global", "figsize":(13,13),
         "unit":"Wh/m^2", "n":500, "view":(45, 120)}
        
        # User-defined configuration overwrites default one.
        if(isinstance(config, dict)):
            for key, val in config.items():
                config_[key] = val
                
                
        # Sample points to plot in "disk".
        if config_["projection"] == "disk":
            Phi, R = np.meshgrid(np.linspace(0, 360, config_["n"]), 
                                 np.linspace(0,   1, config_["n"]))
            
            zone_nums, patch_nums =\
            self.disk_points_to_zones_patches(R.flatten(), Phi.flatten())
            
            Phi = np.deg2rad(Phi)
            
        # Sample points to plot in "sphere".   
        elif config_["projection"] == "sphere":
            Phi, Theta = np.meshgrid(np.linspace(0, 360, config_["n"]), 
                                     np.linspace(90,  0, config_["n"]))
            
            zone_nums, patch_nums =\
            self.sky_points_to_zones_patches(Theta.flatten(), Phi.flatten())
            
            Phi, Theta = np.deg2rad(Phi), np.deg2rad(Theta)
            
            X = np.cos(Phi)*np.sin(Theta)
            Y = np.sin(Phi)*np.sin(Theta)
            Z = np.cos(Theta)
            

            
        zone_nums  =  zone_nums.reshape(config_["n"],  config_["n"])
        patch_nums = patch_nums.reshape(config_["n"], config_["n"])
        
        
        # --- RETRIEVE DATA FOR EACH SAMPLE DATA ---
        
        Color = np.zeros((config_["n"], config_["n"]))
        
        for i in range(config_["n"]):
            for j in range(config_["n"]):
                zone_num, patch_num = zone_nums[i,j], patch_nums[i,j]
                
                Color[i,j] =\
                self.patch_data[(zone_num, patch_num)]\
                ["time-integrated irradiance"][config_["mode"]]["magnitude"]
                            
        
            
        # --- ACCOMODATE DATA TO SELECTED UNIT ---    
            
        if config_["unit"] == "kWh/m^2":
            Color /= 1000
            
        elif config_["unit"] == "kJ/m^2":
            Color *= 3.6 
            
        elif config_["unit"] == "MJ/m^2":
            Color *= 3.6/1000
            
        Color = Color.reshape(config_["n"], config_["n"])
        
        
        # --- COMPUTE TITLE ---  
        
        title = "Time-integrated irradiance by Sky-Patch "
        
        if config_["mode"]=="direct":
            cbar_title = f"Direct [{config_['unit']}]"
        
        elif config_["mode"]=="diffuse":
            cbar_title = f"Diffuse [{config_['unit']}]"
            
        elif config_["mode"]=="global":
            cbar_title = f"Global [{config_['unit']}]"
            
            
            
        # --- GET INITIAL AND FINAL DATE --- 
        for i, key in enumerate(self.time_integrated_sim_times.keys()):
            if i == 0: date_init = key
            continue
        date_fin = key
        title = f"{title} | From {date_init} to {date_fin}."
        
            
        # --- PLOT DISK DATA --- 
        if config_["projection"] == "disk":
            
            fig, ax = plt.subplots(figsize=config_["figsize"], 
            subplot_kw={'projection': 'polar'})
            ax.grid(False)
            ax.pcolormesh(Phi, R, Color, cmap="hot")
            ax.set_xlabel("N = 0°, E = 90°, S = 180°, W = 270°")

        

        # --- PLOT SPHERE DATA --- 
        elif config_["projection"] == "sphere":
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                   figsize=config_["figsize"])
            
            ax.view_init(config_["view"][0], config_["view"][1])
            ax.set_xlabel("X (↑ == N, ↓ == S)")
            ax.set_ylabel("Y (↑ == E, ↓ == W)")
            ax.set_title(title)
            
            if(np.max(Color)>1):
                Color_ = Color/np.max(Color)
            else:
                Color_ = Color
            
            ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
            
        m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
        m.set_array(Color)
        cbar = plt.colorbar(m)
        cbar.ax.set_title(cbar_title)
        ax.set_title(title)
        plt.show()
        
        
        return None
            
            

                

    
#%%                  EXAMPLES

if __name__ == '__main__':    
    from Ambience_Modelling.Site import Site
    from Ambience_Modelling import auxiliary_funcs as aux
    
    # ---- LOAD SITE_OBJ ----
    # As a test, we get the atmospheric data required for this module from
    # precomputed Site obj.

    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    Site_obj = load_obj_with_pickle(path = path)
    
    # --- INITIALIZE SKY OBJ ---
    Sky_obj = Sky(Site_obj, num_divisions = 400, sunrise_sunset_apel = -0.25)   
    
    # --- VISUALIZE SKY VAULT DISCRETIZATION IN 2D ---
    Sky_obj.plot_disk_patches(figsize=(12,12))
    
    # --- VISUALIZE SKY VAULT DISCRETIZATION IN 3D ---
    Sky_obj.plot_sphere_patches(figsize=(12,12), axis_view=(25, 30))
        
    # --- EXTRACT ZONE, PATCH DATA ---
    zone_data, patch_data = Sky_obj.zone_data, Sky_obj.patch_data
    
    # --- BIN SKY PTS INTO THE CORRECT SKY PATCHES -----
    sky_pts = np.array([[30, 90],
                        [45, 180],
                        [60, 270],
                        [77.25, 35.4]])
    
    zone_nums, local_patch_nums =\
    Sky_obj.sky_points_to_zones_patches(sky_pts[:,0], sky_pts[:, 1])
    
    
#%%      ---- COMPUTE SPECTRAL RADIANCE FOR A DAY ----
    
    year, month, day = 2022, 2, 1
    Sky_obj.compute_spectral_radiance_for_a_date(year              = year, 
                                                 month             = month,
                                                 day               = day, 
                                                 nel               = 46,
                                                 naz               = 181,
                                                 nt                = 150,
                                                 kind              = "linear",
                                                 mean_surface_tilt = 0,
                                                 num_iterations    = 500)
    
#%%      ---- PLOT SPECTRAL RADIANCE FOR A DAY ----

    component = "diffuse"
    for nt in range(len(Sky_obj.spectral_radiance_res["Timestamp_index"])):
        
        fig = plt.figure(figsize = (16, 12))
        ax = plt.axes(projection ="3d")
        color_map = plt.get_cmap("hot")
        plt.gca().invert_zaxis()
    
        x_vals, y_vals = [], []
        z_vals, colors = [], []
        for i in 5*np.arange(10):
            for j in 15*np.arange(13):
                for k in 10*np.arange(13):
                                    
                    y_vals.append(Sky_obj.spectral_radiance_res["El"][i,j])
                    x_vals.append(Sky_obj.spectral_radiance_res["Az"][i,j])
                    z_vals.append(Sky_obj.spectral_radiance_res["wavelengths"][k])
                    
                    colors.append(Sky_obj.spectral_radiance_res[component][nt][i,j,k])
                    
        
                    
        scatter_plot = ax.scatter3D(x_vals, y_vals, z_vals,
                                   c = colors,
                                   cmap = color_map)
     
        ax.set_title(f"{Site_obj.name}: {component} Spectral Radiance at time {Sky_obj.spectral_radiance_res['Timestamp_index'][nt]}")
        cbar = plt.colorbar(scatter_plot)
        cbar.ax.set_title('W/m^2/sr/nm')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.set_zlim(4000, 300)
        ax.set_xlabel("Azimuth [°]")
        ax.set_ylabel("Elevation [°]")
        ax.set_zlabel("Wavelength [nm]")
        plt.show()        

#%%       ---- COMPUTE RADIANCE FOR A DAY ----

    Sky_obj.compute_radiance_for_a_date()
    
#%%       ---- PLOT RADIANCE FOR A DAY ----

    Phi = np.deg2rad(Sky_obj.radiance_res["Az"])
    Theta = np.deg2rad(90 - Sky_obj.radiance_res["El"])
    
    X = np.cos(Phi)*np.sin(Theta)
    Y = np.sin(Phi)*np.sin(Theta)
    Z = np.cos(Theta)

    component = "diffuse"
    for nt in range(len(Sky_obj.radiance_res["Timestamp_index"])):
        
        Color = Sky_obj.radiance_res[component][:,:,nt]
        
        max_ = np.max(Color)
        if(max_ > 0):
            Color_ = Color/max_
        else:
            Color_ = Color
        
         
        title = f"{Site_obj.name}: {component} Radiance at time: {Sky_obj.spectral_radiance_res['Timestamp_index'][nt]}"
        
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.view_init(45, 180)
        ax.set_xlabel("X (↑ == N, ↓ == S)")
        ax.set_ylabel("Y (↑ == E, ↓ == W)")
        ax.set_title(title)
        ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
        
        m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
        m.set_array(Color)
        cbar = plt.colorbar(m)
        cbar.ax.set_title('W/m^2/sr')
        
        plt.show()
        
        
        
#%%    ---- COMPUTE TIME-INTEGRATED SPECTRAL RADIANCE FOR A DATE INTERVAL ----

    years, months, days = [2022, 2022], [1, 12], [1, 31]
    Sky_obj.compute_time_integrated_spectral_radiance_for_a_date_interval( years             = years,
                                                                           months            = months, 
                                                                           days              = days,
                                                                           nel               = 46,
                                                                           naz               = 181,
                                                                           nt                = 241,
                                                                           kind              = "linear",
                                                                           mean_surface_tilt = 0,
                                                                           num_iterations    = 500)
    
#%%    ---- PLOT TIME-INTEGRATED SPECTRAL RADIANCE FOR A DATE INTERVAL ----
    
    component = "diffuse"
    fig = plt.figure(figsize = (16, 12))
    ax = plt.axes(projection ="3d")
    color_map = plt.get_cmap("hot")
    plt.gca().invert_zaxis()

    x_vals, y_vals = [], []
    z_vals, colors = [], []
    for i in 5*np.arange(10):
        for j in 15*np.arange(13):
            for k in 10*np.arange(13):
                                
                y_vals.append(Sky_obj.time_integrated_spectral_radiance_res["El"][i,j])
                x_vals.append(Sky_obj.time_integrated_spectral_radiance_res["Az"][i,j])
                z_vals.append(Sky_obj.time_integrated_spectral_radiance_res["wavelengths"][k])
                
                colors.append(Sky_obj.time_integrated_spectral_radiance_res[component][i,j,k])
                
    
                
    scatter_plot = ax.scatter3D(x_vals, y_vals, z_vals,
                                c = colors,
                                cmap = color_map)
 
    #ax.set_title(f"{Site_obj.name}: Time-Integrated {component} Spectral Radiance from date {years[0]}-{months[0]}-{days[0]} to {years[1]}-{months[1]}-{days[1]}")
    plt.suptitle(f"{Site_obj.name}: Time-Integrated {component} Spectral Radiance from date {years[0]}-{months[0]}-{days[0]} to {years[1]}-{months[1]}-{days[1]}")
    cbar = plt.colorbar(scatter_plot)
    cbar.ax.set_title('Wh/m^2/sr/nm')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.set_zlim(4000, 300)
    ax.set_xlabel("Azimuth [°]")
    ax.set_ylabel("Elevation [°]")
    ax.set_zlabel("Wavelength [nm]")
    ax.view_init(30, 130)
    plt.show() 


#%%    ---- COMPUTE TIME-INTEGRATED RADIANCE FOR A DATE INTERVAL ----

    Sky_obj.compute_time_integrated_radiance_for_a_date_interval()
    
    
#%%    ---- PLOT TIME-INTEGRATED RADIANCE FOR A DATE INTERVAL ----

    component = "direct"
    Phi = np.deg2rad(Sky_obj.time_integrated_radiance_res["Az"])
    Theta = np.deg2rad(90 - Sky_obj.time_integrated_radiance_res["El"])
    
    X = np.cos(Phi)*np.sin(Theta)
    Y = np.sin(Phi)*np.sin(Theta)
    Z = np.cos(Theta)
    
    Color = Sky_obj.time_integrated_radiance_res[component]
    
    max_ = np.max(Color)
    if(max_ > 0):
        Color_ = Color/max_
    else:
        Color_ = Color
    
     
    title = f"{Site_obj.name}: {component} Radiance from date {years[0]}-{months[0]}-{days[0]} to {years[1]}-{months[1]}-{days[1]}"
    
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(45, 150)
    ax.set_xlabel("X (↑ == N, ↓ == S)")
    ax.set_ylabel("Y (↑ == E, ↓ == W)")
    #ax.set_title(title)
    plt.suptitle(title)
    ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
    
    m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
    m.set_array(Color)
    cbar = plt.colorbar(m)
    cbar.ax.set_title('Wh/m^2/sr')
    
    plt.show()
    
    
#%%     ---- COMPUTE TIME-INTEGRATED SPECTRAL IRRADIANCES FOR A DATE INTERVAL (FOR EACH SKY PATCH)----

    Sky_obj.compute_time_integrated_spectral_irradiances_for_a_date_interval(int_naz = 30,
                                                                             int_nzen = 20)
        

    
    
#%%     ---- PLOT TIME-INTEGRATED SPECTRAL IRRADIANCES FOR A DATE INTERVAL (FOR EACH SKY PATCH)----

    # --- EXTRACT ZONE, PATCH DATA ---
    zone_data, patch_data = Sky_obj.zone_data, Sky_obj.patch_data
        
    for key in patch_data.keys():

        direct_spectral_irrad_mag  = patch_data[key]["time-integrated spectral irradiance"]["direct"]["magnitude"]
        diffuse_spectral_irrad_mag = patch_data[key]["time-integrated spectral irradiance"]["diffuse"]["magnitude"]
        global_spectral_irrad_mag  = patch_data[key]["time-integrated spectral irradiance"]["global"]["magnitude"]
        
        wavelengths = patch_data[key]["time-integrated spectral irradiance"]["wavelengths"]
        az_lims = [round(patch_data[key]['inf_az'],2), round(patch_data[key]['sup_az'],2)]
        zen_lims = [round(patch_data[key]['inf_zen'],2), round(patch_data[key]['sup_zen'],2)]
        
        fig = plt.figure(figsize=(16,12))
        plt.plot(wavelengths, direct_spectral_irrad_mag,  label="direct")
        plt.plot(wavelengths, diffuse_spectral_irrad_mag, label="diffuse")
        plt.plot(wavelengths, global_spectral_irrad_mag,  label="global")
        plt.suptitle(f"From date {years[0]}-{months[0]}-{days[0]} to {years[1]}-{months[1]}-{days[1]}")
        plt.title(f"Time-integrated spectral irradiance of patch ({key[0]}, {key[1]}) with az_lims: {az_lims}°, zen_lims: {zen_lims}°")
        plt.ylabel("Time-integrated Spectral Irradiance [Wh/m^2/nm]")
        plt.xlabel("Wavelengths [nm]")
        plt.xlim(300, 4000)
        plt.ylim(0,40)
        plt.grid()
        plt.legend()
        plt.show()
        
#%%   ---- COMPUTE TIME-INTEGRATED IRRADIANCES FOR A DATE INTERVAL (FOR EACH SKY PATCH)----

    # --- EXTRACT ZONE, PATCH DATA ---
    zone_data, patch_data = Sky_obj.zone_data, Sky_obj.patch_data

    Sky_obj.compute_time_integrated_irradiances_for_a_date_interval()
    
#%%  ---- PLOT TIME-INTEGRATED IRRADIANCES FOR A DATE INTERVAL (FOR EACH SKY PATCH)----
    
    config =\
    {"projection":"disk", "mode":"global", "figsize":(15,15), 
     "unit":"kWh/m^2", "n":500, "view":(45, 120)}
    
    Sky_obj.plot_integrated_irradiances(config)
    
    config =\
    {"projection":"sphere", "mode":"global", "figsize":(15,15), 
     "unit":"kWh/m^2", "n":500, "view":(45, 120)}
    
    Sky_obj.plot_integrated_irradiances(config)
    
    config =\
    {"projection":"disk", "mode":"diffuse", "figsize":(15,15), 
     "unit":"kWh/m^2", "n":500, "view":(45, 120)}
    
    Sky_obj.plot_integrated_irradiances(config)
    
#%%              CHECK CONSERVATION OF ENERGY

    # Energy is in units of Watt-Hours.
    total_energy_on_the_horizontal_plane_original_approach = 0
    
    # NOTE: WE ASSUME A PANEL OF UNIT AREA. THAT'S WHY WE MULTIPLY BY NOTTHING BUT 
    # BUT THE UNITS ARE POWER AND ENERGY INSTEAD OF POWER/m^2 AND ENERGY/m^2.
    for (year, month, day), sim_time in Sky_obj.time_integrated_sim_times.items():
        
        apzen = Sky_obj.Site_obj.time_interpolate_variable(
               col           = "apzen", 
               year          = year, 
               month         = month, 
               day           = day, 
               new_hms_float = np.array(sim_time),
               kind          = "linear")
        
        Gbn = Sky_obj.Site_obj.time_interpolate_variable(
               col           = "Gb(n)", 
               year          = year, 
               month         = month, 
               day           = day, 
               new_hms_float = np.array(sim_time),
               kind          = "linear")
        
        Gdh = Sky_obj.Site_obj.time_interpolate_variable(
               col           = "Gd(h)", 
               year          = year, 
               month         = month, 
               day           = day, 
               new_hms_float = np.array(sim_time),
               kind          = "linear")
        
        cos_aoi = np.cos(np.deg2rad(apzen))
        cos_aoi[cos_aoi < 0] = 0
        
        power = Gdh + Gbn*cos_aoi 
        energy = simpson(y = power, x = np.array(sim_time))
        total_energy_on_the_horizontal_plane_original_approach += energy
        
        
    # Energy is in units of Watt-Hours.
    total_energy_on_the_horizontal_plane_new_approach = 0
    
    for (zone_num, local_patch_num), patch_dict in Sky_obj.patch_data.items():
        
        global_time_integrated_irradiance_vector =\
        patch_dict["time-integrated irradiance"]["global"]["vector"] 
        
        energy = np.dot([0,0,1], global_time_integrated_irradiance_vector)
        
        total_energy_on_the_horizontal_plane_new_approach += energy
        
    
    percent_error  = total_energy_on_the_horizontal_plane_new_approach
    percent_error -= total_energy_on_the_horizontal_plane_original_approach
    percent_error  = 100*abs(percent_error)
    percent_error /= total_energy_on_the_horizontal_plane_original_approach
    
    
    
    print(f"--- CALCULATION OF THE TOTAL ENERGY THAT FALLS ONTO A 1m^2 HORIZONTAL PANEL FROM THE SKY: FROM {years[0]}-{months[0]}-{days[0]} TO {years[1]}-{months[1]}-{days[1]} ---")   
    print(f"ORIGINAL ENERGY APPROACH [Wh]: {total_energy_on_the_horizontal_plane_original_approach}")
    print(f"NEW ENERGY APPROACH [Wh]: {total_energy_on_the_horizontal_plane_new_approach}")    
    print(f"PERCENTAGE ERROR: {percent_error}") 
    
    # We can see that both numbers are extremely close. The difference
    # in Wh may be explained by the difference in integration methods.
    # It satnds to reason that a higher spatial and time resolution would 
    # porduce closer results. But even so, the resoluts are extremely close.
    # The differences may also lie in the partial omission of the effect of the
    # horizon, when computing the diffuse energy in the new approach.
    # This is something for future work.
        
#%%       SAVE SKY OBJ

    Sky_obj_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Sky_obj_Medellin_2022.pkl"
    aux.save_obj_with_pickle(Sky_obj, path = Sky_obj_path)

    
        
        
    
    
    
    
    
    
#%%

# fig = plt.figure(figsize = (11,9))
# plt.plot(patch_zens, patch_areas)
# plt.xlabel("Mean zenith angle of sky patch [°]")
# plt.ylabel("Solid Angle of Sky Patch [sr]")
# plt.title("Solid Angle of Sky Patch vs Zenith Angle (num divisions = 400)")
# plt.xlim(min(patch_zens), max(patch_zens))
# plt.ylim(0.9*min(patch_areas), 1.05*max(patch_areas))
    
    
    
    
    
    

        
        

        