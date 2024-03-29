#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions related to the computation of the 
spatial distribution of direct irradiance, also called radiance. This is 
accomplished by modelling the direct irradiance as a point-like source of
radiance, i.e, a dirac delta. However, since the the discretization of the sky
does not have infinite resolution, I implement a procedure whereby the 
irradiance of a point, gets proportionately redistributed to the nearest 
points which do appear in the regular grid. 

"""
#%%                 IMPORTATION OF LIBRARIES

import numpy as np

#%%                      DEFINITION OF FUNCTIONS

def compute_direct_radiance(Az, El, Gbn, sun_az, sun_apel):
    
    """
    Compute direct sky radiance by modelling it as dirac delta function.
    
    Parameters
    ----------
    Az : numpy.array of floats with shape (E,A,1)
       Azimuth array of meshgrid of Azimuth, Elevation values. It contains
       the azimuth (in degrees) of each sky element to be considered in 
       the calculation of diffuse sky radiance. The values of 'Az' should vary 
       along axis 1. Values should be between 0 and 360.
    
    El : numpy.array of floats with shape (E,A,1)
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of diffuse sky radiance. The values of 'El' should vary 
       along axis 0. Values should be between 0 and 90.
       
    Gbn : numpy.array of floats with shape (1,1,T)  
       Direct normal irradiance [W/m^2] across time. Must be a
       non-negative array of numbers. 
       
    sun_apel : numpy.array of floats with shape (1,1,T) 
        Sun's apparent elevation (in degrees) across time. Values must lie 
        between 0 and 90 (inclusive).
        
    sun_az : numpy.array of floats with shape (1,1,T) 
        Suns's azimuth (in degrees) across time. Values must lie 
        between 0 and 360 (inclusive).
        
    Returns
    -------
    Lea : numpy.array of floats with shape (E,A,T) 
        Direct sky radiance distribution [W/m^2/sr] across time.

    """
    
    # Get arrays of azimuth and elevation values.
    az, el = Az[0,:,0], El[:,0,0]
    
    
    # ------- COMPUTE LIMS IDXS ---------
    # Compute  indices of the azimuth intervals within each sun azimuth lies.
    az_sup_lims_idxs = np.digitize(x = sun_az.flatten(), bins = az)
    az_sup_lims_idxs[az_sup_lims_idxs > len(az) - 1] = len(az) - 1
    
    az_inf_lims_idxs  = az_sup_lims_idxs - 1
    az_inf_lims_idxs[az_inf_lims_idxs < 0] = 0
    
    # Compute indices of the elevation intervals within each sun elevation lies.
    el_sup_lims_idxs = np.digitize(x = sun_apel.flatten(), bins = el)
    el_sup_lims_idxs[el_sup_lims_idxs > len(el) - 1] = len(el) - 1
    
    el_inf_lims_idxs  = el_sup_lims_idxs - 1
    el_inf_lims_idxs[el_inf_lims_idxs < 0] = 0
    
    # ------- COMPUTE LIMS VALUES ---------
    # Compute azimuth intervals within each sun azimuth lies.
    az_sup_lims = az[az_sup_lims_idxs]
    az_inf_lims = az[az_inf_lims_idxs]
    
    # Compute the elevation intervals within each sun elevation lies.
    el_sup_lims = el[el_sup_lims_idxs]
    el_inf_lims = el[el_inf_lims_idxs]
    
    
    # ------- COMPUTE WEIGHT VALUES ---------
    
    # Each combination of az, el lims define a corner of a square within which 
    # the sun's position lies. We compute these corners' coordinates as well as
    # their coordinate's indices.
    
    corners0 = np.stack([az_inf_lims, el_inf_lims], axis = 1)
    corners1 = np.stack([az_inf_lims, el_sup_lims], axis = 1)
    corners2 = np.stack([az_sup_lims, el_inf_lims], axis = 1)
    corners3 = np.stack([az_sup_lims, el_sup_lims], axis = 1)
    
    corners0_idxs = np.stack([az_inf_lims_idxs, el_inf_lims_idxs], axis = 1)
    corners1_idxs = np.stack([az_inf_lims_idxs, el_sup_lims_idxs], axis = 1)
    corners2_idxs = np.stack([az_sup_lims_idxs, el_inf_lims_idxs], axis = 1)
    corners3_idxs = np.stack([az_sup_lims_idxs, el_sup_lims_idxs], axis = 1)
    
    
    # After computing the corner's coordinates, we compute the sun's coordinates
    # across time.
    sun_pts = np.stack([sun_az.flatten(), sun_apel.flatten()], axis=1)
    
    # We compute the distance of the sun point to each cof its nearest corners,
    # for all times.
    distances0 = np.linalg.norm(corners0 - sun_pts, axis=1)
    distances1 = np.linalg.norm(corners1 - sun_pts, axis=1)
    distances2 = np.linalg.norm(corners2 - sun_pts, axis=1)
    distances3 = np.linalg.norm(corners3 - sun_pts, axis=1)
    
    # We compute the total distance for all times.
    total_ditances = distances0 + distances1 + distances2 + distances3
    
    # We compute the weights with which we shall compute the radiance at each 
    # point of the defined grid. The further away a corner is from the sun's 
    # position, the less amount of radiation said corner will contribute
    # in realtion to a corner that is closer.
    weights0 = total_ditances/distances0
    weights1 = total_ditances/distances1
    weights2 = total_ditances/distances2
    weights3 = total_ditances/distances3
    
    total_weigths = weights0 + weights1 + weights2 + weights3 
    
    weights0 /= total_weigths
    weights1 /= total_weigths
    weights2 /= total_weigths
    weights3 /= total_weigths
    
    weights0[np.isnan(weights0)] = 1
    weights1[np.isnan(weights1)] = 1
    weights2[np.isnan(weights2)] = 1
    weights3[np.isnan(weights3)] = 1
    
    
    # --------- COMPUTE WEIGTHED GBNS
    weighted_Gbns0 = Gbn.flatten()*weights0
    weighted_Gbns1 = Gbn.flatten()*weights1
    weighted_Gbns2 = Gbn.flatten()*weights2
    weighted_Gbns3 = Gbn.flatten()*weights3
    
    # --------- COMPUTE DIRECT RADIANCE ---------
    # We compute the direct radiance for all points at all times.
    Lea = np.zeros((Az.shape[0], Az.shape[1], sun_az.shape[2]))
    
    for nt in range(Gbn.shape[-1]):    
        Lea[corners0_idxs[nt,1], corners0_idxs[nt,0], nt] += weighted_Gbns0[nt]
        Lea[corners1_idxs[nt,1], corners1_idxs[nt,0], nt] += weighted_Gbns1[nt]
        Lea[corners2_idxs[nt,1], corners2_idxs[nt,0], nt] += weighted_Gbns2[nt]
        Lea[corners3_idxs[nt,1], corners3_idxs[nt,0], nt] += weighted_Gbns3[nt]
    
    return Lea
    
    

#%%

if __name__ == '__main__':
    
    # We import libraries
    import time as tm
    import matplotlib.pyplot as plt
    from Ambience_Modelling.Site import Site
    from Ambience_Modelling.auxiliary_funcs import load_obj_with_pickle
    
    # ---- DEFINE EVALUATION POINTS FOR SKY RADIANCE DISTRIBUTION ----
    angular_resolution = 1 #[degrees]
    num_pts_el = round(90/angular_resolution) + 1
    num_pts_az = 4*(num_pts_el - 1) + 1
    
    az = np.linspace(0, 360, num_pts_az)
    el = np.linspace(0,  90, num_pts_el)
    
    Az, El = np.meshgrid(az, el)
    
    # Reshape Az, El into the required shapes.
    Az = Az.reshape(list(Az.shape) + [1])
    El = El.reshape(list(El.shape) + [1])
    
    
    # ---- LOAD SITE_OBJ ----
    # As a test, we get the atmospheric data required for this module from
    # precomputed Site obj.
    
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    Site_obj = load_obj_with_pickle(path = path)
    
    # Scpecify year, day and month of the data.
    # year, month, day = 2022, 1, 1
    year, month, day = 2022, 1, 1
    
    Gbn  = Site_obj.site_data[(year, month, day)]["Gb(n)"]
    
    index = Gbn.index
    
    sun_apel    = Site_obj.sun_data[(year, month, day)]["apel"]
    sun_az      = Site_obj.sun_data[(year, month, day)]["az"]
    
    # Transform data from pandas.Series to numpy.arrays
    
    Gbn         = np.array(Gbn)
    sun_apel    = np.array(sun_apel)
    sun_az      = np.array(sun_az)
    
    # Reshape data into the required shapes.
    
    Gbn          = Gbn.reshape(1, 1, len(Gbn))
    sun_apel     = sun_apel.reshape(1, 1, len(sun_apel))
    sun_az       = sun_az.reshape(1, 1 ,len(sun_az))
    
    # Plot Irradiances
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, Gbn.flatten(), label = "Direct Normal Irradiance")
    plt.title(f"Irradiances for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Irradiance [W/m^2]")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
#%%       ----- COMPUTE DIFFUSE SKY RADIANCE DISTRIBUTION ----
    
    t = tm.time()
    Lea = compute_direct_radiance(Az              = Az,  
                                  El              = El,
                                  Gbn             = Gbn,
                                  sun_az          = sun_az,
                                  sun_apel        = sun_apel)
    
    dt = tm.time() - t
    
#%%    ---- PLOT CUMMULATIVE SKY RADIANCE DISTRIBUTION----

    
    Phi = np.deg2rad(Az).reshape(Az.shape[0], Az.shape[1])
    Theta = np.deg2rad(90 - El).reshape(El.shape[0], El.shape[1]) 
    
    X = np.cos(Phi)*np.sin(Theta)
    Y = np.sin(Phi)*np.sin(Theta)
    Z = np.cos(Theta)
    
    total_Lea = Lea.sum(axis=-1)
    
    Color = total_Lea
    
    max_ = np.max(total_Lea)
    if(max_ > 0):
        Color_ = Color/max_
    else:
        Color_ = Color
    
     
    title = f"{Site_obj.name}: Cummulative Direct Irradiance at time {year},{month},{day}"
    
    
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

        

        
#%%           ---- CHECKING CONSERVATION OF DIRECT IRRADIANCE -----   
      
    # Energy must be conserved. As such, the direct irradiance on a normal
    # plane should be constant. In other words, we should be able to recompute 
    # Gbn from Lea and still get the exact same values.

    recomputed_Gbn = Lea.sum(axis=(0,1))
    
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, Gbn.flatten(), label = "Original")
    plt.plot(index, recomputed_Gbn, label = "Recomputed")
    plt.title(f"Direct normal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Diffuse irradiance [W/m^2]")
    plt.legend()
    plt.grid()
    plt.show()
    


    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





