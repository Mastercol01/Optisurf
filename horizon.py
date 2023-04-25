#%%                    MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of a horizon profile. More specifically, it
contains functions for retrieving the horizon profile of a geographical 
location from PVGIS and plot it. 


                         ---- PVGIS ----
"PVGIS is a web site that gives you information about solar radiation and 
PhotoVoltaic (PV) system performance. You can use PVGIS to calculate how much
energy you can get from different kinds of PV systems at nearly any place in 
the world."

1) PVGIS online tool at:
https://re.jrc.ec.europa.eu/pvg_tools/en/
    
2) More info about the PVGIS tool in general, at: 
https://joint-research-centre.ec.europa.eu/pvgis-online-tool/getting-started-pvgis/pvgis-user-manual_en

3) More info about the PVGIS's Horizon profile, at: 
https://joint-research-centre.ec.europa.eu/pvgis-online-tool/pvgis-tools/horizon-profile_en
    
4) More info about the PVGIS's API Non-Interactive Service, at: 
https://joint-research-centre.ec.europa.eu/pvgis-online-tool/getting-started-pvgis/api-non-interactive-service_en

5) Examples on how PVGIS's Non-Interactive Service has been used in other projects, at:
https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/iotools/pvgis.html

"""

#%%               IMPORTATION OF LIBRARIES

import requests
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%%                 DEFINITION OF PATH CONSTANTS

# PVGIS Api url, for extracting the horizon profile.
URL = 'https://re.jrc.ec.europa.eu/api/printhorizon'


#%%          DEFINITION OF FUNCTIONS


def get_PVGIS_horizon(lat, lon, timeout=30):
    
    """
    Get a site's horizon profile, computed by PVGIS, using its API 
    Non-Interactive Service.
    
    Parameters
    ----------
    lat : float
        Site's latitude in sexagesimal degrees. Must be a number between -90
        and 90.
        
    lon : float
        Site's longitude in sexagesimal degrees. Must be a number between -180
        and 180.
        
    timeout : float
        Number of seconds after which the requests library will stop waiting
        for a response of the server. That is, if the requests library does not 
        receive a response in the specified number of seconds, it will raise a 
        Timeout error.
        
    Returns
    -------
    horizon_df : pandas.DataFrame object
        DataFrame with 2 columns: "az" and "H_hor". "H_hor" is the horizon's
        height for a given azimuth "az". Both are given in sexagesimal degrees.
        
    Note
    ----
    1) Horizon height is the angle between the local horizontal plane and the
    horizon. In other words, the Horizon height is equal to the horizon's 
    elevation angle.
                                                                        
    """
    
    
    # We request the horizon profile calculated by PVGIS from its API,
    # for given latitude and longitude.
    res = requests.get(URL, params={"lat":lat, "lon":lon}, timeout=timeout)
    
    
    # We check for any errors in the request.
    if not res.ok:
        try:
            err_msg = res.json()
        except Exception:
            res.raise_for_status()
        else:
            raise requests.HTTPError(err_msg['message'])
            
    # The values obtained from the request are given as a string. As such,
    # some procesing is necessary to extract the numerical values of the
    # horizon profile.
    lines = res.text.split("\n")[4:-9]
    horizon_df = pd.DataFrame(columns=["az", "H_hor"], index=range(len(lines)))
            
    for i, line in enumerate(lines):
        values = [float(value) for value in line.split("\t\t")]
        horizon_df.loc[i,"az"] = values[0]
        horizon_df.loc[i,"H_hor"] = values[1]
        
    horizon_df = horizon_df.astype(float)     
    
    # The coordinate system used by PVGIS for the horizon profile is different 
    # that the one used in this package. In particular, for PVGIS: N = ± 180°, 
    # E = -90°, S = 0°, and W = 90°. While, for us: N = 0°, E = 90°, S = 180°
    # and W = 270°. Adding 180° to PVGIS's azimuth resolves the problem.
        
    horizon_df["az"] += 180
        
        
    return horizon_df





def plot_horizon(horizon_df, config = None):
    
    """
    Plot horizon profile.
    
    Parameters
    ----------
    horizon_df : pandas.DataFrame object
        DataFrame with 2 columns: "az" and "H_hor". "H_hor" is the horizon's
        height for a given azimuth "az". Both are given in sexagesimal degrees.
        
    config : None or dict
        Configuration or settings of the plot. When equal to None (which is 
        the default) the default plot settings are used. When not equal to None,
        it must be a dict containing some or all of the following key-value 
        pairs:
            Keys-Values
            -----------
            "polar" : bool
                If equal to True, the Horizon profile is plotted using a polar 
                plot. If equal to False, it is plotted using a cartesian plot.
                "Default is True.
                
            "vanilla" : bool
                If 'polar' is True and "vanilla" is also equal to True, a simple
                horizon profile plot such as the one used by PVGIS is plotted
                (see link 2)). Such a plot does not include a colorbar or color 
                variation of the plotted profile. If, on the other hand, 'polar'
                is True and  "vanilla" is equal to false, a more complex horizon 
                profile plot is used. This plot does include a colorbar, as it
                uses color variation to codify the magnitude of the horizon
                Height. Default is "vanilla" equal to True.
                
            "title" : str
                Title of the plot. Default is 'Horizon Profile'.
                
            "cmap" : str
                Color map to use for the plot when 'polar' is True and 
                'vanilla' is False. It can be any of the colormaps 
                provided by Matplotlib. Default is "Greys".
                
            "interp_n" : int
                Number of interpolating samples to generate. Must be non-negative.
                It applies only when "polar" is True and "vanilla" is False. 
                Default is 720.
                
            "s" : int
             The marker size of the points making up the horizon color line.
             Must be non-negative. It applies only when "polar" is True and 
             "vanilla" is False. Default is 100.
             
             "facecolor" : str
                 Background color of the Horizon Height part of the plot.
                 Must be equal to str(x), where x is a float between 0 and 1.
                 0 means that the background color is black. 1 means that it 
                 is white. Any value in between represents a shade of gray.
                 Default is "0.5".
                 
            "figsize" : tuple of float
                Figure size of the plot.
                
                
    Returns
    -------
    None
                
    
    """
    
    # Default plot settings.
    config_ = {"polar"     : True,
               "vanilla"   : True,
               "title"     : "Horizon profile",
               "cmap"      : "Greys",
               "interp_n"  : 720,
               "s"         : 100,
               "facecolor" : "0.5",
               "figsize"   : (12,12)}
    
    # User settings overwrite default settings.
    if isinstance(config, dict):
        for key, val in config.items():
            config_[key] = val
    
    # POLAR PLOT
    if config_["polar"]:
        
        fig, ax =\
        plt.subplots(figsize = config_["figsize"], subplot_kw={"polar":True})

        rad = 90 - horizon_df["H_hor"]
        theta = np.deg2rad(horizon_df["az"])
        ax.patch.set_facecolor(config_["facecolor"])
        ax.plot(theta, rad, color='black', ls='-', linewidth=1)
        


        if not config_["vanilla"]:
            
            # We would like for the horizon height line to have color. Ideally,
            # the color of this line would codify the magnitude of the horizon
            # height at each point. The easiest way to do this is to construct
            # the line from a collection of individual points, each represented
            # by a marker. With sufficient marker density and size, the collection
            # of dots will resemble a line. 
            
            
            # We interpolate the values to be plotted in order to get as many 
            # samples as required so the line looks continuous.
            interp_func = interp1d(theta, horizon_df["H_hor"])
            interp_theta = np.linspace(0, 2*np.pi, 720)
            interp_H_hor = interp_func(interp_theta)
            interp_rad = 90 - interp_H_hor
            
            # We normalize the colormap to the min and max values of the 
            # Horizon height.
            norm = mpl.colors.Normalize(horizon_df["H_hor"].min(), 
                                        horizon_df["H_hor"].max())
        
            # We draw the color line as a collection of points using an
            # scatter plot. We also define the marker size. Increasing the 
            # marker size also has other interesting effects, such as taking
            # over the coloring of the horizon line background.
            im = ax.scatter(interp_theta, 
                            interp_rad, 
                            c = interp_H_hor,
                            s = config_["s"],  
                            cmap = config_["cmap"], 
                            norm = norm, 
                            linewidths = 0)
        
            # We add a color bar for reference.
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Horizon Height [°]', rotation = 270)


        # The Horizon height lies between 0 and 90 degrees.
        ax.set_rlim(0,90)
        ax.fill(theta,rad,'w')
        
        # We get rid of radious ticks.
        ax.set_yticklabels([])
        
        # We count the angles clockwise.
        ax.set_theta_direction(-1)
        plt.suptitle(config_["title"])
        
        # We put the 0 angle at the top.
        ax.set_theta_zero_location("N")
        ax.set_title("(N = 0°, E = 90°, S = 180°, W = 270°)")
        
        plt.show()
        plt.draw() 
        
        
    # CARTESSIAN PLOT   
    else:
        _ = plt.figure(figsize = config_["figsize"])
        plt.plot(horizon_df["az"], horizon_df["H_hor"], color="k")
        
        plt.grid()
        plt.xlim(0, 360)
        plt.title(config_["title"])
        plt.ylabel("Horizon Height [°]")
        plt.xlabel("Azimuth [°]  (N = 0°, E = 90°, S = 180°, W = 270°)")
        

    return None
 

#%%                       EXAMPLES

if __name__ == '__main__': 

    # We compute the horizon for a point in Medelín.
    horizon_df = get_PVGIS_horizon(lat = 6.230833, 
                                   lon = -75.590553, 
                                   timeout = 30)
    
    # We compute the vanilla polar plot.
    plot_horizon(horizon_df)
    
    # We compute 2 different custom polar plots.
    config  = {"vanilla" : False, "s":100, "cmap":"inferno", "facecolor":"0.85"}
    plot_horizon(horizon_df, config = config)
    
    config  = {"vanilla" : False, "s":5000}
    plot_horizon(horizon_df, config = config)
    
    # We compute a normal cartesian plot.
    config  = {"polar":False}
    plot_horizon(horizon_df, config = config)
    

    

            
            