#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
computation and manipulation of most of a site's geographical and
metheorological data. 
"""


#%%                       IMPORTATION OF LIBRARIES

import scipy
import warnings
import numpy as np
import pvlib as pv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from pvlib.atmosphere import gueymard94_pw
from scipy.integrate import cumulative_trapezoid
from pvlib.irradiance import get_extra_radiation
from pvlib.atmosphere import angstrom_aod_at_lambda


import Ambience_Modelling.Time as time
import Ambience_Modelling.ozone_column as oz
import Ambience_Modelling.horizon as horizon
import Ambience_Modelling.water_column as wat
import Ambience_Modelling.aod_550nm as aod550nm
import Ambience_Modelling.angstrom_exponent as angsexp
import Ambience_Modelling.single_scattering_albedo as ssa
import Ambience_Modelling.aerosol_asymmetry_factor as aaf



#%%                      DEFINITION OF CONSTANTS 

# Number of days each month posesses.
MONTH_DAYS =\
{0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

# Month number to month name dict.
MONTH_NAMES = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul",
               8:"Aug", 9:"Sept", 10:"Oct", 11:"Nov", 12:"Dic"}

_SPECTRL2_WAVELENGTHS = [# nm
    300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
    350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0,
    450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0,
    550.0, 570.0, 593.0, 610.0, 630.0, 656.0, 667.6, 690.0, 710.0, 718.0,
    724.4, 740.0, 752.5, 757.5, 762.5, 767.5, 780.0, 800.0, 816.0, 823.7,
    831.5, 840.0, 860.0, 880.0, 905.0, 915.0, 925.0, 930.0, 937.0, 948.0,
    965.0, 980.0, 993.5, 1040.0, 1070.0, 1100.0, 1120.0, 1130.0, 1145.0,
    1161.0, 1170.0, 1200.0, 1240.0, 1270.0, 1290.0, 1320.0, 1350.0, 1395.0,
    1442.5, 1462.5, 1477.0, 1497.0, 1520.0, 1539.0, 1558.0, 1578.0, 1592.0,
    1610.0, 1630.0, 1646.0, 1678.0, 1740.0, 1800.0, 1860.0, 1920.0, 1960.0,
    1985.0, 2005.0, 2035.0, 2065.0, 2100.0, 2148.0, 2198.0, 2270.0, 2360.0,
    2450.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0,
    3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0]



# ------------ COMPUTE POINTS OF INTERPOLATION --------------

# We compute the points of interpolation of the data, slightly different as
# they appear in the 'tmy_data' DataFrame. Since the values for each
# day of this DataFrame only go from hour 0 to hour 23, for each day. Were 
# we to try to compute the interpolation for a value at hour 23.5, for a given 
# day, an error would be raised. In order to avoid this, we compute what 
# the values for each variable would be at hour 24, using the fact that
# said value would just be the same as that one of hour 0 of the next
# day.

_GRIDDATA_MONTHS = [np.full(25*MONTH_DAYS[month], month) for month in range(1,13)]
_GRIDDATA_MONTHS = np.concatenate(_GRIDDATA_MONTHS, axis=0)

_GRIDDATA_DAYS = []
for month in range(1,13):
    for day in range(1, MONTH_DAYS[month]+1):
        _GRIDDATA_DAYS.append(np.full(25, day))
_GRIDDATA_DAYS = np.concatenate(_GRIDDATA_DAYS, axis=0)

_GRIDDATA_HOURS = np.arange(0, 25)
_GRIDDATA_HOURS = np.array([_GRIDDATA_HOURS]*365).flatten()

GRIDDATA_POINTS = np.stack([_GRIDDATA_MONTHS, 
                            _GRIDDATA_DAYS, 
                            _GRIDDATA_HOURS], axis=1)

GRIDDATA_NUMPY_ARANGE = np.arange(len(GRIDDATA_POINTS))



#%%                 DEFINITION OF FUNCTIONS 


def get_pvgis_tmy_data(lon, lat, tz_name, startyear, endyear):
    
    """
    Get Typical Meteorological Year (TMY) data from PVGIS.

    Parameters
    -----------
    lon : float
        Site's longitude in degrees. Must be a number between -180 and 180.
    
    lat : float
        Site's latitude in degrees. Must be a number between -90 and 90.
        
    tz_name : str
        Time zone string accepted by pandas.
        
    startyear: int or None
        First year to calculate TMY.

    endyear : int or None
        Last year to calculate TMY, must be at least 10 years from first year.
        
    Returns
    -------
    tmy_data : pandas.DataFrame obj of floats
        DataFrame contining the PVGIS TMY data for the whole year.
        Its index is a multiindex of (month, day, hour). Its columns, along
        with their descriptions, are as follows:
            
        1) "T2m": 2-m air temperature (degree Celsius)
    
        2) "RH": relative humidity (%)
        
        3) "G(h)": Global irradiance on the horizontal plane (W/m2)
        
        4) "Gb(n)": Beam/direct irradiance on a plane always normal to sun rays (W/m2)
        
        5) "Gd(h)": Diffuse irradiance on the horizontal plane (W/m2)
        
        6) "IR(h)": Surface infrared (thermal) irradiance on a horizontal plane (W/m2)
        
        7) "WS10m": 10-m total wind speed (m/s)
        
        8) "WD10m": 10-m wind direction (0 = N, 90 = E) (degree)
        
        9) "SP": Surface (air) pressure (Pa)
        
    

    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    2) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
       
    3) The PVGIS website uses 10 years of data to generate the TMY, whereas the
       API accessed by this function defaults to using all available years. 
       This means that the TMY returned by this function may not be identical
       to the one generated by the website. To replicate the website requests,
       specify the corresponding 10 year period using startyear and endyear. 
       Specifying endyear also avoids the TMY changing when new data becomes 
       available.

    """
    
    

    #          CONSTRUCTION OF TMY_DATA DATAFRAME
    
    
    # Get Typical Meteorological Year (TMY) data of the site in question,
    # from the PVGIS database. This is done through the 'pvlib' library 
    # function 'iotools.get_pvgis_tmy'.
    
    # CHECK THE LINKS BELOW FOR MORE INFO:
    # 1) https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.iotools.get_pvgis_tmy.html?msclkid=3aacaceecfc211ec8bec2645c1a03011
    # 2) https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/tmy-generator_en
    # 3) https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
    # 4) https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system/getting-started-pvgis/pvgis-data-sources-calculation-methods_en
    
    
    # T2m: 2-m air temperature (degree Celsius)
    # RH: relative humidity (%)
    # G(h): Global irradiance on the horizontal plane (W/m2)
    # Gb(n): Beam/direct irradiance on a plane always normal to sun rays (W/m2)
    # Gd(h): Diffuse irradiance on the horizontal plane (W/m2)
    # IR(h): Surface infrared (thermal) irradiance on a horizontal plane (W/m2)
    # WS10m: 10-m total wind speed (m/s)
    # WD10m: 10-m wind direction (0 = N, 90 = E) (degree)
    # SP: Surface (air) pressure (Pa)
    
    tmy_data = pv.iotools.get_pvgis_tmy(latitude = lat,
                                        longitude = lon, 
                                        startyear = startyear, 
                                        endyear = endyear, 
                                        outputformat = 'csv',
                                        map_variables = True)[0]
        
    
    # At this point 'tmy_data' is a DataFrame consisting of 8760 rows.
    # Each row has the TMY info of a a particular hour of a particular day
    # of the year (24*365 = 8760). The thing is, the index of the DataFrame
    # (which is given in hours), was given in terms of UTC=0, that is, in 
    # terms of greenwhich median time. We then have to convert it to local
    # time:
    
    new_tmy_index = tmy_data.index.tz_convert(tz_name)
    tmy_data = tmy_data.reindex(new_tmy_index)
    
    # After the time index conversion, we sort the values, in hierarchical
    # order, by Month, Day and Hour:
    
    tmy_data["Date"] = tmy_data.index
    
    tmy_data["Month"] = tmy_data["Date"].apply(lambda x:x.month)
    tmy_data["Day"] = tmy_data["Date"].apply(lambda x:x.day)
    tmy_data["Hour"] = tmy_data["Date"].apply(lambda x:x.hour)
    
    tmy_data = tmy_data.sort_values(by=["Month", "Day", "Hour"])
    tmy_data.drop(columns=["Date"], inplace=True)
    
    # We then create a Multiindex for the DataFrame. In this way we may
    # access any value by specifying the Month, Day and Hour.
    

    tmy_data = tmy_data.set_index(["Month", "Day", "Hour"])
    
    # We change the names of the columns to give them more standard names:
    new_cols_dict = {"temp_air":"T2m", "relative_humidity":"RH", "ghi":"G(h)",
                     "dni":"Gb(n)", "dhi":"Gd(h)", "wind_speed":"WS10m",
                     "wind_direction":"WD10m", "pressure":"SP"}
    
    tmy_data = tmy_data.rename(columns=new_cols_dict)
    
    
    return tmy_data





def interp_pvgis_tmy_data(tmy_data, col, eval_pts, method='linear'):
    
    """
    Interpolate PVGIS TMY data using scipy's griddata function.

    Parameters
    -----------
    tmy_data : pandas.DataFrame obj of floats
        DataFrame contining the PVGIS TMY data for the whole year.
        Its index must be a multiindex of (month, day, hour).
    
    col : str
        Name of the column of 'tmy_data', containing the data tha we wish
        to interpolate.
        
    eval_pts : 2D-numpy.array of floats.
        Points at which to interpolate data. Array of size (n x 3) where n is 
        the number of data points to evaluate. Each data point is composed
        of 3 coordinates: month, day, hour. As such, 
        
            eval_pts = np.array([[1,30,23],
                                 [6,15,8]])
        
        Would compute the interpolated values for the specified variable at
        January 30-th at 11pm and June 15-th at 8am.   
        
    method: str
        Method of interpolation. Supported methods are ‘linear’, ‘nearest’
        and ‘cubic’.
        
        
    Returns
    -------
    interp_data : numpy.array of floats
        Interpolated TMY data of the variable specified by 'col', at the 
        the specified 'eval_pts'.

    
    """
    
    # We make a copy of the eval_pts, as to not modify the original array.
    xi = eval_pts.copy()
    

    
    # -------GET VALUES FOR NEEDED FOR INTERPOLATION --------
    values = np.full(len(GRIDDATA_POINTS), np.nan)    
    values[GRIDDATA_POINTS[:,2] < 24] = tmy_data[col]
    
    values[-1] = values[0]
    NaN_idxs = GRIDDATA_NUMPY_ARANGE[np.isnan(values)]
    values[NaN_idxs] = values[NaN_idxs + 1]


    #    --------- DEALING WITH LEAP YEARS -----------
    # In case we get a leap day, we have to remap it to the 28th
    # of February, as the 29th of February is not defined in the data.
    
    is_february_29 = np.logical_and(xi[:,0]==2, xi[:,1]==29)
    xi[is_february_29, 1] = 28
    
    
    #    --------- DEALING WITH HOUR VALUES OVER 24 -----------      
    # In the case we get values above 24h, we remap them onto their equivalent
    # times. Therefore, for example: month, day, hour = 6, 25, 34 should
    # get remapped to month, day, hour = 6, 26, 14. month, day, hour =
    # 7, 31, 26.5 should get remapped to month, day, hour = 8, 1, 2.5. And 
    # month, day, hour = 12, 31, 29 should get remapped to month, day, hour = 
    # 1, 1, 5.
     
    while True:
        
        # We check for any hour values greater than 24.
        # If none, we break the loop.
        hour_bigger_than_24 = xi[:,2] > 24
        
        if not any(hour_bigger_than_24): 
            break
        
        # The hours which exceed 24 get substracted 24. Meanwhile, the days
        # coordinate of said hours gets increased by one.
        xi[hour_bigger_than_24, 2] -= 24
        xi[hour_bigger_than_24, 1] += 1
        
        # If any day of a given month exceeds the maxmum values of days
        # for that month, said day gets mapped to 1 and the month coordinate
        # of that same day gets increased by 1.
        for month in range(1, 13):
            day_of_month_bigger_than_MONTH_DAYS =\
            np.logical_and(xi[:,0]==month, xi[:,1] > MONTH_DAYS[month])
            
            xi[day_of_month_bigger_than_MONTH_DAYS, 1]  = 1
            xi[day_of_month_bigger_than_MONTH_DAYS, 0] += 1
            
            
        # If any month exceeds the value of 12, it gets mapped back to one.
        month_bigger_than_12 = xi[:,0] > 12
        xi[month_bigger_than_12, 0] = 1
        
        # Do this as long as it is necesary.
        
            
    # --------------INTEROPLATE THE DATA -------------------
    interp_data = griddata(points = GRIDDATA_POINTS,
                           values = values,
                           xi = xi, 
                           method = method)
    
    
    return interp_data


 
#%%                    DEFINITION OF CLASSSES

class Site:
    
    """
    Class for storing most of the geographical and metheorological information
    of a site, along with some useful methods.
    
    Parameters
    ----------
    time_data : dict of pandas.Series of pandas.Timestamp objs
        Dictionary containing the time information of when the simulation is to
        be performed. That is, the information relating the exact times for 
        which the metheorological and sun-position information will be
        extracted. It must be a dict with the following key-value pair format:
            
            Keys : Values
            -------------
            (year, month, day) : pandas.Series of pandas.Timestamp objs
                Each key is a 3-tuple of ints, specifying the year, month and
                day of simulation. Then, the value stored by this key, should
                be a pandas.Series of pandas.Timestamp objs specifying the 
                exact times of the day for which the simulation is to be 
                performed. This time series must be monotonic-increasing and
                have constant period (i.e, the time interval between samples
                should be constant).
                
    lon : float
        Site's longitude in degrees. Must be a number between -180 and 180.
    
    lat : float
        Site's latitude in degrees. Must be a number between -90 and 90.
        
    alt : float
        Site's elevation above sea level in meters. Must be non-negative.
        
    UTC : str
        Standard time zone we wish to consider. If positive, it should
        be given in the format: "hh:mm:ss" and if it is negative it should 
        be given as "-hh:mm:ss".
        
    name : str
        Custom name for the site being modelled. 
        
        
    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    2) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.  
       
    3) "hh:mm:ss" format means hour:minute:second format. Eg.: "-05:00:00"
       means 5 hours, 0 minutes and 0 seconds, west of the greenwhich meridian.
        
        
    """
    
    def __init__(self, time_data, lat, lon, alt, UTC, name):
        
        for val in time_data.values():
            if not isinstance(val, pd.core.series.Series):
                message = "ERROR: Non-pandas.Series value detected"
                message = f"{message}. Only pandas.Series objs, holding"
                message = f"{message} pandas.Timestamps objs are valid."
                raise Exception()
        
        
        # Save input parameters. 
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.utc = UTC
        self.name = name
        self.time_data = time_data
        self.utc_hour = int(UTC.split(":")[0])
        self.tz_name = time.utc_hour_to_tz_name(self.utc_hour)
        
        
        # Initialize main attributes.
        self.sun_data = {}
        self.site_data = {}
        self.ground_albedo = {}
        self.single_scattering_albedo = {}
        self.aerosol_asymmetry_factor = {}
        
        
        
        SUN_DATA_COLS =\
        ["index", "hms_float", "apzen", "zen", "apel", "el", "az", "i", "j",
         "k", "rel_airmass"]

        SITE_DATA_COLS =\
        ["index", "hms_float","G(h)", "Gb(n)", "Gd(h)", "T2m", "SP", "RH", 
         "O3", "H2O", "alpha_500nm", "AOD_500nm", "spectrally_averaged_aaf",
         "extra_Gb(n)"]
        
        SPECTRL2_WAVELENGTHS_COLS =\
        ["index", "hms_float"] + [f"{int(i)}nm" for i in _SPECTRL2_WAVELENGTHS]
        
        
       # Each attribute initialized above is a dictionary. Each dictionary 
       # will have the same keys, which are those of 'time_data'. However, each
       # dictionary will contain at each key a different pandas.DataFrame
       # containing different data. Now, although the DataFrames of each 
       # attribute will be different, all have the same 'hms_column' which will
       # be a column containing the hour of the day in float format.
        for key, series in self.time_data.items():
            
            # Create an empety DataFrame with pre-defined columns for each 
            # key of each attribute.
            self.sun_data[key] =\
            pd.DataFrame(index = range(len(series)), 
                         columns = SUN_DATA_COLS)
            
            self.site_data[key] =\
            pd.DataFrame(index = range(len(series)), 
                         columns = SITE_DATA_COLS)
            
            self.ground_albedo[key] =\
            pd.DataFrame(index = range(len(series)), 
                         columns = SPECTRL2_WAVELENGTHS_COLS)
            
            self.single_scattering_albedo[key] =\
            pd.DataFrame(index = range(len(series)), 
                         columns = SPECTRL2_WAVELENGTHS_COLS)
            
            self.aerosol_asymmetry_factor[key] =\
            pd.DataFrame(index = range(len(series)), 
                         columns = SPECTRL2_WAVELENGTHS_COLS)
            
            # Make the index of each DataFrame be equal to its corresponding
            # time series.
            self.sun_data[key]["index"] = series.copy()
            self.site_data[key]["index"] = series.copy()
            self.ground_albedo[key]["index"] = series.copy()
            self.single_scattering_albedo[key]["index"] = series.copy()
            self.aerosol_asymmetry_factor[key]["index"] = series.copy()
            
            self.sun_data[key].set_index("index", inplace=True)
            self.site_data[key].set_index("index", inplace=True)
            self.ground_albedo[key].set_index("index", inplace=True)
            self.single_scattering_albedo[key].set_index("index", inplace=True)
            self.aerosol_asymmetry_factor[key].set_index("index", inplace=True)
            
            
            # Compute 'hms_float' column.
            init_hms_float =\
            time.timestamp_hms_to_float(series.iloc[0], unit="h")  
            
            hms_float = series - series.iloc[0]
            
            hms_float =\
            [init_hms_float + 24*ts.days + ts.seconds/3600 for ts in hms_float]
            
            
            self.sun_data[key]["hms_float"] = hms_float.copy()
            self.site_data[key]["hms_float"] = hms_float.copy()
            self.ground_albedo[key]["hms_float"] = hms_float.copy()
            self.single_scattering_albedo[key]["hms_float"] = hms_float.copy()
            self.aerosol_asymmetry_factor[key]["hms_float"] = hms_float.copy()
            
            
        # Define an attribute which will contain all descriptions and units
        # of the site variables.
        self.variables_info = {"descriptions":{}, "units":{}}
        
        self.variables_info["descriptions"]  = {
        'T2m': 'Air temperature at 2m above ground',
        'RH': 'Relative humidity',
        'G(h)':'Global horizontal irradiance',
        'Gb(n)':'Beam (direct) normal irradiance',
        'Gd(h)':'Diffuse horizontal irradiance',
        'SP':'Surface Pressure',
        'int G(h)'  :'Time integral of global horizontal irradiance',
        'int Gb(n)':'Time integral of beam (direct) normal irradiance',
        'int Gd(h)':'Time integral of diffuse horizontal irradiance',
        'H2O' : 'Precipitable water column',
        'O3' : 'Atmospheric ozone column',
        'alpha_500nm':'Angstrom turbidity exponent at 500nm',
        'AOD_500nm':'Aerosol optical depth at 500nm',
        'spectrally_averaged_aaf': 'Aerosol Asymmetry Factor averaged over specified spectral range',
        'apzen' : 'apparent zenith angle of the Sun',
        'zen' : 'zenith angle of the Sun',
        'apel': 'apparent elevation angle of the Sun',
        'el' : 'elevation angle of the Sun',
        'az' : 'azimuth angle of the Sun',
        'rel_airmass' : 'Relative Airmass',
        'single_scattering_albedo' : 'Single Scattering Albedo',
        'aerosol_asymmetry_factor' : 'Aerosol Asymmetry Factor',
        'ground_albedo' : 'Albedo of the ground surface',
        'extra_Gb(n)' : 'Extraterrestrial Irradiance'}
        
        self.variables_info["units"]  = {
        'T2m': '[°C]',
        'RH': '[%]',
        'G(h)':'[W/m^2]',
        'Gb(n)':'[W/m^2]',
        'Gd(h)':'[W/m^2]',
        'SP':'[Pa]',
        'int G(h)'   :'[Wh/m^2]',
        'int Gb(n)' : '[Wh/m^2]',
        'int Gd(h)' : '[Wh/m^2]',
        'H2O' : '[cm]',
        'O3' : '[atm-cm]',
        'alpha_500nm':'[-]',
        'AOD_500nm':'[-]',
        'spectrally_averaged_aaf': '[-]',
        'apzen' : '[°]',
        'zen' : '[°]',
        'apel': '[°]',
        'el' : '[°]',
        'az' : '[°]',
        'rel_airmass' : '[-]',
        'single_scattering_albedo' : '[-]',
        'aerosol_asymmetry_factor' : '[-]',
        'ground_albedo' : '[-]',
        'extra_Gb(n)' : '[W/m^2]'}
            
        
    
    def get_pvgis_horizon(self, timeout = 30):
        
        """
        Get a site's horizon profile, computed by PVGIS, using its API 
        Non-Interactive Service.
        
        Parameters
        ----------            
        timeout : float
            Number of seconds after which the requests library will stop waiting
            for a response of the server. That is, if the requests library does not 
            receive a response in the specified number of seconds, it will raise a 
            Timeout error.
            
        Returns 
        -------
        None
        
        Produces
        -------
        self.horizon_df : pandas.DataFrame object
            DataFrame with 2 columns: "az" and "H_hor". "H_hor" is the
            horizon's height for a given azimuth "az". Both are given in 
            sexagesimal degrees.
            

        Note
        ----
        1) Horizon height is the angle between the local horizontal plane and
        the horizon. In other words, the Horizon height is equal to the 
        horizon's  elevation angle.
                                                                            
        """
        
        self.horizon_df = horizon.get_PVGIS_horizon(lat = self.lat, 
                                                    lon = self.lon,
                                                    timeout = timeout)
        
        return None
    
    
    def plot_horizon(self, config = None):
        
        """
        Plot horizon profile.
        
        Parameters
        ----------            
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
                    
        Notes
        -----
        1) It requires for 'self.horizon_df' to already be defined.
    
        """
        
        try:
            horizon.plot_horizon(horizon_df = self.horizon_df,
                                 config = config)
        except AttributeError:
            message = "There is no horizon to plot. Try running first the"
            message = f"{message} 'get_pvgis_horizon' method or define a"
            message = f"{message} ' custom 'horizon_df' attribute."
            raise Exception(message)
        
        return None
        
            
            
        
    def get_pvgis_tmy_data(self, startyear, endyear):
        """
        Get Typical Meteorological Year (TMY) data from PVGIS.
    
        Parameters
        -----------            
        
        startyear: int or None
            First year to calculate TMY.
    
        endyear : int or None
            Last year to calculate TMY, must be at least 10 years from first year.
            
        Returns
        -------
        None
            
            
        Produces
        ---------
        self.tmy_data : pandas.DataFrame obj of floats
            DataFrame containing the PVGIS TMY data for the whole year.
            Its index is a multiindex of (month, day, hour).
            
    
        Notes
        -----
        1) Latitude of -90° corresponds to the geographic South pole, while a 
           latitude of 90° corresponds to the geographic North Pole.
          
        2) A negative longitude correspondes to a point west of the greenwhich 
           meridian, while a positive longitude means it is east of the greenwhich 
           meridian.
           
        3) The PVGIS website uses 10 years of data to generate the TMY, whereas the
           API accessed by this function defaults to using all available years. 
           This means that the TMY returned by this function may not be identical
           to the one generated by the website. To replicate the website requests,
           specify the corresponding 10 year period using startyear and endyear. 
           Specifying endyear also avoids the TMY changing when new data becomes 
           available.
        
        
        """
        self.tmy_data = get_pvgis_tmy_data(lon = self.lon, 
                                           lat = self.lat, 
                                           tz_name = self.tz_name, 
                                           startyear = startyear,
                                           endyear = endyear)
        
        return None
    
    
    def use_pvgis_tmy_data(self, interp_method = "linear"):
        
        """
        Use the Typical Meteorological Year (TMY) data from PVGIS to partially
        fill the 'self.site_data' attribute using the 'interp_pvgis_tmy_data'
        function.
    
        Parameters
        -----------            
        interp_method: str
            Method of interpolation. Supported methods are ‘linear’, ‘nearest’
            and ‘cubic’.
            
        Returns
        -------
        None
            
            
        Produces
        ---------
        Partially filled 'self.site_data' attribute. By interpolating the data
        stored in the "self.tmy_data" attribute, this fucntion fills some
        of the data fields present in the "self.tmy_data" attribute. More ,
        it fills the: "G(h)", "Gb(n)", "Gd(h)", "T2m", "RH" and "SP" columns
        of all the DataFrames contained by the 'self.site_data' dict.
            
          
        """
        

        for (year, month, day), df in self.site_data.items():
            print((year, month, day))
            
            month_arr = np.full(len(df), month)
            days_arr  = np.full(len(df), day)
            hours_arr = np.array(df["hms_float"])
            eval_pts = np.stack([month_arr, days_arr, hours_arr], axis=1)
            
            columns_to_compute =\
            list(set(df.columns).intersection(set(self.tmy_data.columns)))
            
            for col in columns_to_compute:
                self.site_data[(year, month, day)].loc[:,col] =\
                interp_pvgis_tmy_data(tmy_data = self.tmy_data,
                                      col = col,
                                      eval_pts = eval_pts,
                                      method = interp_method)
            
        return None
    
    
    
    def compute_ozone_column_using_van_Heuklon_model(self):
                
        """
        Computes the Ozone Column values (in atm-cm) for the site, using a 
        van Heuklon's Ozone model. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "O3" column of all the DataFrames contained by the
        'self.site_data' dict.
        

        """
            
        for (year, month, day), df in self.site_data.items():     
            self.site_data[(year, month, day)]['O3'] =\
            oz.compute_van_Heuklon_ozone(lat = self.lat,
                                         lon = self.lon,
                                         timestamp = df.index)
            
            
        return None
    
    
    
    
    def compute_ozone_column_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):
        
        """
        Computes the monthly average of Ozone Column values (in atm-cm) for
        the site, using a 'TMY-like' procedure. The raw data used for 
        calculating said TMY-like ozone column values is extracted from the 
        'Ozone monthly gridded data from 1970 to present derived from satellite
        observations' database, belonging to the 'Climate Data Store' webpage.
        
        Parameters
        ----------
        path : path-str
            Path of the folder where the ozone column .nc files of raw data
            are stored. That is, the path to the local ozone column database.
            
        percentile : float or None
            If float, it is the percentile used for computing the TMY-like
            data. More specifically the TMY-like data is equal to the 
            'percentile'-th percentile value of the monthly averages across all
            years, currently existing in the local database. It must be a 
            number between 0 and 1. Default is 0.5. If None, instead of using
            the percentile value, the average value of the data points across
            all existing years is used.
            
    
        interp_method : str
            The method of interpolation to perform when computing the data
            for an specific location. Supported methods are the same as
            supported by scipy's RegularGridInterpolator, i.e, “linear”, 
            “nearest”, “slinear”, “cubic”, “quintic” and “pchip”. Default is
            "linear".   

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "O3" column of all the DataFrames contained by the
        'self.site_data' dict.
        
        """
        
        # We compute the TMY-like data using the speficied percentile.    
        if percentile is None:
            processed_ozone_data_funcs =\
            oz.process_CDS_ozone_data(path = path, percentile = 0.5,
                                      interp_method = interp_method)
            
            processed_ozone_data_funcs =\
            processed_ozone_data_funcs["avg_data_funcs"]
            
        # We compute the TMY-like data using the average.      
        else: 
            processed_ozone_data_funcs =\
            oz.process_CDS_ozone_data(path = path, percentile = percentile,
                                      interp_method = interp_method)
            
            processed_ozone_data_funcs =\
            processed_ozone_data_funcs["percentile_data_funcs"]
            
        
        
        for (year, month, day), df in self.site_data.items():     
            self.site_data[(year, month, day)]['O3'] =\
            float(processed_ozone_data_funcs[(month)]([self.lat, self.lon]))
            
            
        return None
    
    
    
    
    def compute_water_column_using_gueymard94_model(self):
        
        """
        Computes the Precipitable Water Column values (in atm-cm) for the site,
        using pvlib's implementation of the gueymard94 model. For this, we make 
        use of the of the "T2m" and "RH" values stored in each of the
        DataFrames of the 'self.site_data' dict.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "O3" column of all the DataFrames contained by the
        'self.site_data' dict.
        
        """
        
        for (year, month, day), df in self.site_data.items():
            
            T2m = np.array(df["T2m"]).astype(float)
            RH  = np.array(df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['RH']. Precipitable" 
                message = f"{message} Water Column for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            if any(pd.isnull(T2m)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['T2m']. Precipitable" 
                message = f"{message} Water Column for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            
            self.site_data[(year, month, day)]['H2O'] =\
            gueymard94_pw(temp_air = T2m, relative_humidity = RH)
            
        return None
    


    def compute_water_column_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):
        
        """
        Computes the monthly average of Precipitable Water Column values 
        (in cm) for the site, using a 'TMY-like' procedure. The raw data used 
        for calculating said TMY-like ozone column values is extracted from the 
        'Monthly total column water vapour over land and ocean from 2002 to 
        2012 derived from satellite observations' database, belonging to the 
        'Climate Data Store' webpage.
        
        Parameters
        ----------
        path : path-str
            Path of the folder where the ozone column .nc and .npy files of
            raw and filled-NaN data are stored. That is, the path to the local 
            water column database.
            
        percentile : float or None
            If float, it is the percentile used for computing the TMY-like
            data. More specifically the TMY-like data is equal to the 
            'percentile'-th percentile value of the monthly averages across all
            years, currently existing in the local database. It must be a 
            number between 0 and 1. Default is 0.5. If None, instead of using
            the percentile value, the average value of the data points across
            all existing years is used.
            
    
        interp_method : str
            The method of interpolation to perform when computing the data
            for an specific location. Supported methods are the same as
            supported by scipy's RegularGridInterpolator, i.e, “linear”, 
            “nearest”, “slinear”, “cubic”, “quintic” and “pchip”. Default is
            "linear".   

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "H2O" column of all the DataFrames contained by the
        'self.site_data' dict.
        
        """
        
        # We compute the TMY-like data using the speficied percentile.   
        if percentile is None:
            processed_water_data_funcs =\
            wat.process_CDS_water_column_data(path = path, percentile = 0.5,
                                              interp_method = interp_method)
            
            processed_water_data_funcs =\
            processed_water_data_funcs["avg_data_funcs"]
            
        # We compute the TMY-like data using the average.    
        else: 
            processed_water_data_funcs =\
            wat.process_CDS_water_column_data(path = path, percentile = percentile,
                                              interp_method = interp_method)
            
            processed_water_data_funcs =\
            processed_water_data_funcs["percentile_data_funcs"]
            
        
        
        for (year, month, day), df in self.site_data.items():     
            self.site_data[(year, month, day)]['H2O'] =\
            float(processed_water_data_funcs[(month)]([self.lat, self.lon]))
            
            
        return None
    
    
    

    
  
    
    def compute_angstrom_turbidity_exponent_500nm_using_SF_model(self, model):
        
        """
        Compute the Ansgtrom turbidity exponent at 500nm for the site using the
        Shettel and Fenn model, as detailed in the paper "SMARTS2, a simple
        model of the atmospheric radiative transfer of sunshine: algorithms 
        and performance assessment". For this, we make use of the of the "RH" 
        values stored in each of the DataFrames of the 'self.site_data' dict.
        
        Parameters
        ----------
        model : str
            Model to be used in the computation of the Angstrom exponent.
            Supported are “Rural”, “Urban” and "Maritime".
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "alpha_500nm" column of all the DataFrames contained by the
        'self.site_data' dict.
        
        """
        
        
        
        for (year, month, day), df in self.site_data.items():
            
            RH = np.array(df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['RH']. Angstrom" 
                message = f"{message} Turbidity Exponent for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            
            self.site_data[(year, month, day)]['alpha_500nm'] =\
            angsexp.compute_angstrom_exponent_using_SF(RH = RH,
                                                       wavelength = 500,
                                                       model = model)
            
            
        return None
    
    
    

    
    

    
    
    
    def compute_AOD_500nm_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):
        
        """
        Computes the monthly average of Aerosol Optical Depth at 500nm 
        (adm) for the site, using a 'TMY-like' procedure. The raw data used 
        for calculating said TMY-like ozone column values is extracted from the 
        'Aerosol properties gridded data from 1995 to present derived from 
        satellite observations' database, belonging to the 'Climate Data Store' 
        webpage.
        
        Parameters
        ----------
        path : path-str
            Path of the folder where the ozone column .nc and .npy files of
            raw and filled-NaN data are stored. That is, the path to the local 
            AOD_550nm database.
            
        percentile : float or None
            If float, it is the percentile used for computing the TMY-like
            data. More specifically the TMY-like data is equal to the 
            'percentile'-th percentile value of the monthly averages across all
            years, currently existing in the local database. It must be a 
            number between 0 and 1. Default is 0.5. If None, instead of using
            the percentile value, the average value of the data points across
            all existing years is used.
            
    
        interp_method : str
            The method of interpolation to perform when computing the data
            for an specific location. Supported methods are the same as
            supported by scipy's RegularGridInterpolator, i.e, “linear”, 
            “nearest”, “slinear”, “cubic”, “quintic” and “pchip”. Default is
            "linear".   

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "H2O" column of all the DataFrames contained by the
        'self.site_data' dict.
        
        """
        
        # We compute the TMY-like data using the speficied percentile.  
        if percentile is None:
            processed_AOD_550nm_data_funcs =\
            aod550nm.process_CDS_aod_550nm_data(path = path, percentile = 0.5,
                                                interp_method = interp_method)
            
            processed_AOD_550nm_data_funcs =\
            processed_AOD_550nm_data_funcs["avg_data_funcs"]
            
        # We compute the TMY-like data using the average.       
        else: 
            processed_AOD_550nm_data_funcs =\
            aod550nm.process_CDS_aod_550nm_data(path = path, percentile = percentile,
                                                interp_method = interp_method)
            
            processed_AOD_550nm_data_funcs =\
            processed_AOD_550nm_data_funcs["percentile_data_funcs"]
            
        
        
        for (year, month, day), df in self.site_data.items():
            
            alpha_500nm  = np.array(df["alpha_500nm"]).astype(float)
            
            if any(pd.isnull(alpha_500nm)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['alpha_500nm']. Aerosol" 
                message = f"{message} Optical Depth at 500nm for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
            
            aerosol_optical_depth_at_550nm =\
            float(processed_AOD_550nm_data_funcs[(month)]([self.lat, self.lon]))
            
            # We convert AOD_550nm to AOD_500nm using the pre-computed
            # values of Angstrom Turbidity Coefficient at 500nm (Note: 
            # according to the SF model, alpha becomes independent of
            # wavlength for wavelengths at or over 500nm).
            aerosol_optical_depth_at_500nm =\
            angstrom_aod_at_lambda(aod0 = aerosol_optical_depth_at_550nm, 
                                   lambda0 = 550, 
                                   alpha   = alpha_500nm,
                                   lambda1 = 500)
            
            
            self.site_data[(year, month, day)]['AOD_500nm'] =\
            aerosol_optical_depth_at_500nm
            
            
        return None
        
        
        
    def compute_single_scattering_albedo_using_SF_model(self, model, interp_method = "linear"):
        
        """
        Compute the Single Scattering Albedo of Aerosols for the spectral range
        300 nm - 4000 nm, for the site, using the Ansgtrom Shettel and Fenn
        model, as detailed in the paper "Models for the Aerosols of the Lower
        Atmosphere and the Effects of Humidity Variations on their Optical 
        Properties". For this, we make use of the of the "RH" values stored in
        each of the DataFrames of the 'self.site_data' dict.
        
        Parameters
        ----------
        model : str
            Model to be used in the computation of the Angstrom exponent.
            Supported are “Rural”, “Urban” and "Maritime".
            
        interp_method : str
            Method of interpolation to use on the data. Supported are "linear",
            "nearest" and "cubic". Default is "linear".
            
        Returns
        -------
        None
        
        Produces
        --------
        Filled 'self.single_scattering_albedo' attribute. More specifically, it
        fills completely all the DataFrames contained by the 
        'self.single_scattering_albedo' dict.
        
        """
        
        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        
        for (year, month, day), df in self.site_data.items():
            
            RH  = np.array(df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['alpha_500nm']. Single" 
                message = f"{message} Scattering Albedo for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            Wavelengths, RHs = np.meshgrid(wavelength, RH) 
            Wavelengths = Wavelengths.flatten()
            RHs = RHs.flatten()
            
            single_scattering_albedo =\
            ssa.compute_single_scattering_albedo_SF(RH = RHs, 
                                                    wavelength = Wavelengths, 
                                                    model = model, 
                                                    method = interp_method)
            
            single_scattering_albedo =\
            single_scattering_albedo.reshape(len(RH), len(wavelength))
            
            self.single_scattering_albedo[(year, month, day)].iloc[:,1:] =\
            single_scattering_albedo
            
            
        return None
    
    
    
    
    def compute_aerosol_asymmetry_factor_using_SF_model(self, model, interp_method = "linear"):
        
        """
        Compute the Aersol Asymmetry Factor for the spectral range
        300 nm - 4000 nm, for the site, using the Ansgtrom Shettel and Fenn
        model, as detailed in the paper "Models for the Aerosols of the Lower
        Atmosphere and the Effects of Humidity Variations on their Optical 
        Properties". For this, we make use of the of the "RH" values stored in
        each of the DataFrames of the 'self.site_data' dict.
        
        Parameters
        ----------
        model : str
            Model to be used in the computation of the Angstrom exponent.
            Supported are “Rural”, “Urban” and "Maritime".
            
        interp_method : str
            Method of interpolation to use on the data. Supported are "linear",
            "nearest" and "cubic". Default is "linear".
            
        Returns
        -------
        None
        
        Produces
        --------
        Filled 'self.aerosol assymetry factor' attribute. More specifically, it
        fills completely all the DataFrames contained by the 
        'self.single_scattering_albedo' dict.
        
        """
        
        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        
        for (year, month, day), df in self.site_data.items():
            
            RH  = np.array(df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['alpha_500nm']. Aerosol" 
                message = f"{message} Asymmetry factor for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            Wavelengths, RHs = np.meshgrid(wavelength, RH) 
            Wavelengths = Wavelengths.flatten()
            RHs = RHs.flatten()
            
            aerosol_asymmetry_factor =\
            aaf.compute_aerosol_asymmetry_factor_SF(RH = RHs, 
                                                    wavelength = Wavelengths, 
                                                    model = model, 
                                                    method = interp_method)
            
            aerosol_asymmetry_factor =\
            aerosol_asymmetry_factor.reshape(len(RH), len(wavelength))
            
            self.aerosol_asymmetry_factor[(year, month, day)].iloc[:,1:] =\
            aerosol_asymmetry_factor
            
            
        return None
    
    
    def compute_spectrally_averaged_aerosol_asymmetry_factor(self, spectral_range = (300, 4000)):
        
        """
        Compute the average of the aerosol asymmetry factor, in the interval of
        wavelengths given by 'spectral_range'. For this, we make use of the 
        'self.aerosol_asymmetry_factor' attribute. We go over all the DataFrames
        stored is the afroemntioned dict and compute the row-wise mean of the
        values for the interval of wavlengths specified. We then use the the 
        computed values to partially fill the self.site_data' attribute.
        
        Parameters
        ----------
        spectral_range : 2-tuple of float
            Tuple containing the lower and upper bounds of wavelengths
            (in nm) that make up the spectral range meant for averaging the
            aerosol asymmetry factor.
            
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, it
        fills the "spectrally_averaged_aaf" column of all the DataFrames
        contained by the 'self.site_data' dict.
        
        """
        
        
        if spectral_range[0] > spectral_range[1]:
            message = "The upper bound of 'spectral_range' must be greater"
            message = f"{message} than the lower bound."
            raise Exception(message)
        
        
        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        idx0 = (wavelength -  spectral_range[0])**2
        idx0 = list(idx0).index(idx0.min()) + 1 
        
        idx1 = (wavelength -  spectral_range[1])**2
        idx1 = list(idx1).index(idx1.min()) + 1



        for (year, month, day), df in self.aerosol_asymmetry_factor.items():
            self.site_data[(year, month, day)]["spectrally_averaged_aaf"] =\
            np.array(df.iloc[:, idx0:idx1+1].mean(axis=1)).astype(float)
            
            
            
        return None    
    
    
 
    
    
    
    
    def compute_sun_data(self):
        
        """
        Computes most of the data related to the position of the sun. More
        specifically, it computes the apparent sun zenith and elevation,
        as well as the actual zenith and elevation, it computes the sun 
        azimuth, it computes the three cartesian components of the unit
        vector that points to the position of the sun in the sky from the
        origin and it computes the relative airmass. It then stores all the 
        computed information in the 'self.sun_data' attribute. For this, we
        make use of the of the "T2m" and "SP" values stored in each of the 
        DataFrames of the 'self.site_data' dict.
        
        Returns
        -------
        None
        
        Produces
        --------
        Totally filled 'self.sun_data' attribute. More specifically, it
        fills all the columns of all the DataFrames contained by the 
        'self.sun_data' dict.
        

        """
        
        for (year, month, day), df in self.site_data.items():
            
            T2m = np.array(df["T2m"]).astype(float)
            SP = np.array(df["SP"]).astype(float)
            
            if any(pd.isnull(T2m)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['T2m']. Apparent" 
                message = f"{message} Zenith/Elevation for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
                
            if any(pd.isnull(SP)):
                message = f"NaN values detected in self.site_data[({year}"
                message = f"{message}, {month}, {day})]['SP']. Apparent" 
                message = f"{message} Zenith/Elevation for the NaN values"
                message = f"{message} will also be NaN."
                warnings.warn(message)
        
            local_sun_data =\
            pv.solarposition.get_solarposition(time        = df.index, 
                                               latitude    = self.lat, 
                                               longitude   = self.lon, 
                                               altitude    = self.alt,
                                               pressure    = SP,
                                               method      ='nrel_numpy', 
                                               temperature = T2m) 
            
            
            self.sun_data[(year, month, day)]["apzen"] =\
            local_sun_data["apparent_zenith"]
            
            self.sun_data[(year, month, day)]["zen"] =\
            local_sun_data["zenith"]
            
            self.sun_data[(year, month, day)]["apel"] =\
            local_sun_data["apparent_elevation"]
            
            self.sun_data[(year, month, day)]["el"] =\
            local_sun_data["elevation"]
            
            self.sun_data[(year, month, day)]["az"] =\
            local_sun_data["azimuth"]
            
            theta = np.deg2rad(local_sun_data["apparent_zenith"]).astype(float)
            phi   = np.deg2rad(local_sun_data["azimuth"]).astype(float)
            
            self.sun_data[(year, month, day)]["i"] =\
            np.array(np.cos(phi)*np.sin(theta))
            
            self.sun_data[(year, month, day)]["j"] =\
            np.array(np.sin(phi)*np.sin(theta))
            
            self.sun_data[(year, month, day)]["k"] =\
            np.array(np.cos(theta))
            
            apzen = np.array(local_sun_data["apparent_zenith"]).astype(float)
            
            self.sun_data[(year, month, day)]["rel_airmass"] =\
            pv.atmosphere.get_relative_airmass(apzen, model='kasten1966')

        return None
    
    
    
    def use_constant_ground_albedo(self, constant_ground_albedo=0):
        
        """
        Set 'self.ground_albedo' attribute to be equal to 
        'constant_ground_albedo' for all times and for all wavelengths.
        
        Parameteres
        -----------
        constant_ground_albedo : float
            Value of albedo of the ground surface to be used for all times and
            for all wavelengths. Must be a number between 0 and 1. Default is 0.
            
        Returns
        -------
        None
        
        Produces
        --------
        Completely filled 'self.ground_albedo' attribute.
        
        """
        
        if constant_ground_albedo < 0 or constant_ground_albedo > 1:
            raise Exception("Ground Albedo must be between 0 and 1.")
        
        for key in self.ground_albedo.keys():
            self.ground_albedo[key].iloc[:,1:] = constant_ground_albedo
        
        return None
            
            
            
    def compute_cummulative_time_integral_of_irradiances(self):
        
        """
        Computes the cummulative time integral of 'self.site_data[(year, month,
        day)][x]', where x is an element of {"G(h)", "Gb(n)", "Gd(h)"}, for all
        year, month, day. in 'self.site_data'. For this, we make use of the of 
        the "G(h)", "Gb(n)" and "Gd(h)" values stored in each of the 
        DataFrames of the 'self.site_data' dict.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, creates
        and fills 3 new columns in each of the DataFrames contained by the 
        'self.site_data' dict. More precisely, it creates the columns "int G(h)",
        "int Gb(n)" and "int Gd(h)" and fills them by computing the 
        cummulative time integral of "G(h)", "Gb(n)" and "Gd(h)", respectively.
        
        """
        
        # NOTE: Time integral of irradiances is given in Wh/m^2
        
        for (year, month, day), df in self.site_data.items():
            
            t_vals = np.array(df["hms_float"])
            
            for col in ["G(h)", "Gb(n)", "Gd(h)"]:
                y_vals = np.array(df[col])
                
                if any(pd.isnull(y_vals)):
                    message = f"NaN values detected in self.site_data[({year}"
                    message = f"{message}, {month}, {day})][{col}]. Time" 
                    message = f"{message} integral of {col} for the NaN values"
                    message = f"{message} will also be NaN."
                    warnings.warn(message)
                
                integral_of_y_vals = cumulative_trapezoid(y_vals, t_vals)
                integral_of_y_vals = np.insert(integral_of_y_vals, 0, 0)
                
                self.site_data[(year, month, day)][f"int {col}"] =\
                integral_of_y_vals
                    
            
        return None   


        
    def compute_extraterrestrial_normal_irradiance(self, method = "nrel"):
        """
        Determine extraterrestrial radiation from day of year, using pvlib's
        'get_extra_radiation' function, for all year, month, day. in 'self.site_data'.
        
        

        Parameters
        ----------
        method : str
            The method by which the extraterrestrial radiation should be
            calculated. Supported are: "pyephem", "spencer", "asce" and "nrel".
            The default is "nrel".

        Returns
        -------
        None.
        
        Produces
        --------
        Partially filled 'self.site_data' attribute. More specifically, creates
        and fills 1 new column in each of the DataFrames contained by the 
        'self.site_data' dict. More precisely, it creates the column "extra_Gb(n)",
        and fills it by computing the extraterrestrial normal irradiance.

        """
        
        for (year, month, day), df in self.site_data.items():
            
            extra_Gbn = get_extra_radiation(datetime_or_doy = df.index,
                                            method = method,
                                            epoch_year = year)
            
            self.site_data[(year, month, day)]["extra_Gb(n)"] = extra_Gbn
            
        
        return None
        
        
    
    def plot_data(self, col, years, months, days, hours, mode=2, interp_method = "linear", figsize = (16, 12)):
        
        """
        Plot variable specified by 'col', for the period of time specified, 
        and using the mode selected.
        
        Parameters
        ----------
        col : "str"
            Name of the variable to be plotted. Must be one of the keys of 
            'self.variables_info["descriptions"]'. 
            
        years : list of years or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the years to plot. The first element of 'years' 
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the 'years' 
            variable are automatically selected by the program so that all 
            avialable years are included.
            
        months : list of months or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the months to plot. The first element of 'months' 
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the 'months' 
            variable are automatically selected by the program so that all 
            avialable months are included.
            
        days : list of days or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the days to plot. The first element of 'days' 
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the 'days' 
            variable are automatically selected by the program so that all 
            avialable days are included.
            
        hours : list of hours
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the hours to plot. The first element of 'hours' 
            would be the lower bound, while the second element would be the 
            upper bound. 
            
        mode : int
            Mode of plotting. There are 3 options:
                1) mode=1 : Plot all variable curves for all times.
                2) mode=2 : Plot all variable curves for all times + 
                            Plot average and 25th, 50th, 75th percentiles.
                3) mode=3 : Plot average and 25th, 50th, 75th percentiles.
    
            Default is mode=2.
    
            
        interp_method : str
            Method to use for the interpolation of data before plotting.
            The methods supported are the same ones as those supported by 
            scipy's 'griddata' fucntion. Default is "linear".
            
        figsize : 2-tuple of int
            Figure size accepeted by Matplotlib. Default is (16, 12).
        
            

        """
        
        
         # We accomodate the data inot a form which is more suitable for calulations.
        if not isinstance(years,  list):  years  = [years, years]
        if not isinstance(months, list):  months = [months, months]
        if not isinstance(days,   list):  days   = [days, days]       
        if not isinstance(hours,  list):  hours  = [hours, hours]
        
            
        if years[0]  is None : years[0]  = - np.inf
        if years[1]  is None : years[1]  =   np.inf
        if months[0] is None : months[0] = - np.inf
        if months[1] is None : months[1] =   np.inf
        if days[0]   is None : days[0]   = - np.inf
        if days[1]   is None : days[1]   =   np.inf
        if hours[0]  is None : hours[0]  = - np.inf
        if hours[1]  is None : hours[1]  =   np.inf
        
        
        years  = [min(years),  max(years)]
        months = [min(months), max(months)]
        days   = [min(days),   max(days)]
        hours  = [min(hours),  max(hours)]
        
        
        
        SITE_DATA_COLS =\
        ['hms_float', 'G(h)', 'Gb(n)', 'Gd(h)', 'T2m', 'SP', 'RH', 'O3', 'H2O',
         'AOD_500nm', 'alpha_500nm', 'spectrally_averaged_aaf', 'int G(h)', 
         'int Gb(n)', 'int Gd(h)', 'extra_Gb(n)']
        
        SUN_DATA_COLS =\
        ['hms_float', 'apzen', 'zen', 'apel', 'el', 'az', 'i', 'j', 'k',
         'rel_airmass']
        
        
        
        # We defined the iterator, depending on which variable we want to plot.
        if col in SITE_DATA_COLS:
            iterator = self.site_data.items()
            
        elif col in SUN_DATA_COLS:
            iterator = self.sun_data.items()
            
        elif col == "single_scattering_albedo":
            iterator = self.single_scattering_albedo.items()
            
        elif col == "aerosol_asymmetry_factor":
            iterator = self.aerosol_asymmetry_factor.items()
            
        elif col == "ground_albedo":
            iterator = self.ground_albedo.items()
            
        else:
            message = f"{col} variable either doesn't exist or cannot be" 
            message = f"{message} plotted using this function."
            raise Exception(message)
            
            

        # --- PLOT SITE OR SUN DATA VARIABLES ----

        if col in SITE_DATA_COLS or col in SUN_DATA_COLS:
            
            dates = []
            data_to_plot = []
            x_eval = np.linspace(hours[0], hours[1], 1000)
            
            for (year, month, day), df in iterator:
                
                # We plot the data which is inside the interval previously 
                # specified.
                if year  < years[0]  or year > years[1]:   continue
                if month < months[0] or month > months[1]: continue
                if day   < days[0]   or day > days[1]:     continue
            
                # As all data may not line-up, we gotta make it line up by
                # evaluating it at the same specified hours.
                x_interp = np.array(df["hms_float"])
                y_interp = np.array(df[col])
                interp_func = scipy.interpolate.interp1d(x = x_interp,
                                                         y = y_interp, 
                                                         kind = interp_method)
                
                try:
                    y_eval = interp_func(x_eval)
                except ValueError as m:
                    message = f"{m} Range of hours specified is not valid for"
                    message = f"{message} the timeframe selected. Some days"
                    message = f"{message} within timeframe selected do not"
                    message = f"{message} contain all the hours specified. Try"
                    message = f"{message} using a smaller or a different hour"
                    message = f"{message} range."
                    raise Exception(message)
                    
                
                data_to_plot.append(y_eval)
                dates.append((year, month, day))
                
                
            # Compute percentiles and averages.        
            data_to_plot = np.stack(data_to_plot, axis=0)   
            p25 = np.percentile(data_to_plot, q = 25, axis=0)
            p50 = np.percentile(data_to_plot, q = 50, axis=0)
            p75 = np.percentile(data_to_plot, q = 75, axis=0) 
            avg = np.nanmean(data_to_plot, axis=0)
            
            
            _ = plt.figure(figsize=figsize)
            
            
            # We plot Variable vs. Time for all days specified.
            if mode == 0 or mode == 2:
                for i in range(data_to_plot.shape[0]):
                    plt.plot(x_eval, data_to_plot[i,:], color="gray", linestyle="-.")
                    
            if mode == 1 or mode == 2:
                plt.plot(x_eval, p25, color="black", linestyle="-.", label="p25")
                plt.plot(x_eval, p50, color="black", linestyle="-" , label="p50")
                plt.plot(x_eval, p75, color="black", linestyle="--", label="p75")
                plt.plot(x_eval, avg, color="black", linestyle=":",  label="avg")
            

            plt.grid()
            plt.legend(prop={'size': 12})
            plt.xlim(hours[0], hours[1])
            plt.xlabel("Hour [24h-format]", fontsize = 16)
            plt.ylabel(f"{col} {self.variables_info['units'][col]}", fontsize = 16)
            plt.suptitle(
            f"{self.name}: lat={self.lat}° lon={self.lon}°, alt={self.alt} m",
            fontsize = 16)
            plt.title(
            f"{col} vs Time. From inital date: {dates[0]} to final date: {dates[-1]}.", 
            fontsize = 16)
            plt.show()
            
            
        # --- PLOT SINGLE_SCATTERING_ALBEDO, AEROSOL_ASYMMETRY_FACTOR OR GROUND_ALBEDO----
        else:
            
            dates = []
            data_to_plot = []
            
            x_eval = np.linspace(hours[0], hours[1], 24)
            y_eval = np.array(_SPECTRL2_WAVELENGTHS)
            Y_eval, X_eval = np.meshgrid(y_eval, x_eval)
            eval_pts = np.stack([X_eval.flatten(), Y_eval.flatten()], axis=1)
            
            for (year, month, day), df in iterator:
                
                # We plot the data which is inside the interval previously 
                # specified.
                if year  < years[0]  or year > years[1]:   continue
                if month < months[0] or month > months[1]: continue
                if day   < days[0]   or day > days[1]:     continue
            
                # As all data may not line-up, we gotta make it line up by
                # evaluating it at the same specified hours and wavelengths.    
                x_interp = np.array(df["hms_float"])
                y_interp = y_eval.copy()
                Y_interp, X_interp = np.meshgrid(y_interp, x_interp)
                interp_pts = np.stack([X_interp.flatten(), Y_interp.flatten()], axis=1)
                interp_vals = np.array(df.iloc[:,1:]).flatten()
                
                try:
                    evaluated_vals = griddata(points = interp_pts, 
                                              values = interp_vals,
                                              xi = eval_pts)
                except ValueError as m:
                    message = f"{m} Range of hours specified is not valid for"
                    message = f"{message} the timeframe selected. Some days"
                    message = f"{message} within timeframe selected do not"
                    message = f"{message} contain all the hours specified. Try"
                    message = f"{message} using a smaller or a different hour"
                    message = f"{message} range."
                    raise Exception(message)
                
                evaluated_vals = evaluated_vals.reshape(len(x_eval),len(y_eval))
                
                data_to_plot.append(evaluated_vals)
                dates.append((year, month, day))
                
                
            # Compute percentiles and averages.
            data_to_plot = np.vstack(data_to_plot) 
            p25 = np.percentile(data_to_plot, q = 25, axis=0)
            p50 = np.percentile(data_to_plot, q = 50, axis=0)
            p75 = np.percentile(data_to_plot, q = 75, axis=0) 
            avg = np.nanmean(data_to_plot, axis=0)     
            
    
            _ = plt.figure(figsize=figsize)
            
            # We plot Variable vs. Wavelength for all dates specified.
            if mode == 0 or mode == 2:
                for i in range(data_to_plot.shape[0]):
                    plt.plot(y_eval, data_to_plot[i,:], color="gray", linestyle="-.")
                    
            if mode == 1 or mode == 2:
                plt.plot(y_eval, p25, color="black", linestyle="-.", label="p25")
                plt.plot(y_eval, p50, color="black", linestyle="-" , label="p50")
                plt.plot(y_eval, p75, color="black", linestyle="--", label="p75")
                plt.plot(y_eval, avg, color="black", linestyle=":",  label="avg")
            

            plt.grid()
            plt.legend(prop={'size': 12})
            plt.xlim(y_eval[0], y_eval[-1])
            plt.xlabel("Wavelengths [nm]", fontsize = 16)
            plt.ylabel(f"{col} {self.variables_info['units'][col]}", fontsize = 16)
            plt.suptitle(
            f"{self.name}: lat={self.lat}° lon={self.lon}°, alt={self.alt} m",
            fontsize = 16)
            plt.title(
            f"{col} vs Wavelength. From inital date: {dates[0]} to final date: {dates[-1]}.", 
            fontsize = 16)
            plt.show()
            
        
        return None
    
    
    def time_interpolate_variable(self, col, year, month, day, new_hms_float, kind = "linear"):
        
        """
        Interpolate variable specified by 'col' across time, for the period of 
        time specified by 'hms_fl' and using the mode selected by 'new_hms_float'.
        
        Parameters
        ----------
        col : "str"
            Name of the variable to be plotted. Must be one of the keys of 
            'self.variables_info["descriptions"]'.  
            
        year : int
            Year at which the variable is defined.
            
        month : int
            Month at which the variable is defined.
        
        day : int
            Day at which the variable is defined.
            
        new_hms_float : array-like of floats
            Fractional hours at which to evaluate the interpolated variable.
            The range of 'new_hms_float' must be the same as the the variable's
            original hms_float.
            
        kind : str
            Interpolation method. Supported are those specified in scipy's 
            documentation for 'interp1d' function. Default is "linear".
        
            
        Returns
        -------
        interpd_y : numpy.array of floats
            Array of interpolated values for the variable specified by "col",
            at the times specified by "year", "month", "day".
        
        """
        
        if col in self.site_data[(year, month, day)].columns:
            
            original_y = self.site_data[(year, month, day)][col]
            original_t = self.site_data[(year, month, day)]["hms_float"]
            interpd_y = interp1d(original_t, original_y, kind)(new_hms_float)
            
            
            
        elif col in self.sun_data[(year, month, day)].columns:
            
            original_y = self.sun_data[(year, month, day)][col]
            original_t = self.sun_data[(year, month, day)]["hms_float"]
            interpd_y = interp1d(original_t, original_y, kind)(new_hms_float)
            
            
            
        elif col == "single_scattering_albedo":
            
            interpd_y = np.zeros((len(new_hms_float), 122))
            original_t = self.single_scattering_albedo[(year, month, day)]["hms_float"]
            
            for i in range(122):
                original_y =\
                self.single_scattering_albedo[(year, month, day)].iloc[:,i+1]
                
                interpd_y[:,i] =\
                interp1d(original_t, original_y, kind)(new_hms_float)
                    


        elif col == "aerosol_asymmetry_factor":
            
            interpd_y = np.zeros((len(new_hms_float), 122))
            original_t = self.aerosol_asymmetry_factor[(year, month, day)]["hms_float"]
            
            for i in range(122):
                original_y =\
                self.aerosol_asymmetry_factor[(year, month, day)].iloc[:,i+1]
                
                interpd_y[:,i] =\
                interp1d(original_t, original_y, kind)(new_hms_float)
                
                
            
        elif col == "ground_albedo":
            
            interpd_y = np.zeros((len(new_hms_float), 122))
            original_t = self.ground_albedo[(year, month, day)]["hms_float"]
            
            for i in range(122):
                original_y =\
                self.ground_albedo[(year, month, day)].iloc[:,i+1]
                
                interpd_y[:,i] =\
                interp1d(original_t, original_y, kind)(new_hms_float)
                
                
            
        else:
            message = f"{col} variable either doesn't exist or cannot be" 
            message = f"{message} interpolated using this function."
            raise Exception(message)
        
        
        
        
        return interpd_y
        
    
    
    
    


    
    
    
    

        
    
    
    
    
    
#%%                             EXAMPLES           
           
if __name__ == '__main__':
    from Ambience_Modelling import auxiliary_funcs as aux
    
    # ---- COMPUTE TIME DATA ----
    time_data = time.geo_date_range(lat               = 6.230833,
                                    lon               = -75.590553, 
                                    alt               = 1475, 
                                    start_time        = "2022-01-01 00:00:00", 
                                    end_time          = "2022-12-31 23:59:59", 
                                    min_hms           = "sunrise", 
                                    max_hms           = "sunset", 
                                    time_interval     = "5-min",
                                    UTC               = "-05:00:00",
                                    skip_polar_nights = True,
                                    time_delta        = "10-min" )


    # ---- COMPUTE SITE_OBJ ----
    Site_obj = Site(time_data = time_data,
                    lat       = 6.230833,
                    lon       = -75.590553, 
                    alt       = 1475, 
                    UTC       = "-05:00:00",
                    name      = "Medellín")


    # ---- GET PVGIS HORIZON AND PLOT IT ----
    Site_obj.get_pvgis_horizon()
    Site_obj.plot_horizon()
    
    
    # ---- GET AND USE PVGIS TMY DATA ----
    Site_obj.get_pvgis_tmy_data(startyear=2005, endyear=2015)
    Site_obj.use_pvgis_tmy_data(interp_method="linear")
    

    # ---- COMPUTE OZONE COLUMN ----
    # Van Heuklen.
    Site_obj.compute_ozone_column_using_van_Heuklon_model()
    
    # Satelite data.
    local_ozone_column_database_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\OptiSurf\Fitness_Function\pvpowlib\Local Ozone Column Database"
    Site_obj.compute_ozone_column_using_satelite_data(path = local_ozone_column_database_path, 
                                                      percentile = 0.5,
                                                      interp_method = "linear")


    # ---- COMPUTE WATER COLUMN ----
    # Gueymard
    Site_obj.compute_water_column_using_gueymard94_model()
    
    # Satelite data.
    local_water_column_database_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\OptiSurf\Fitness_Function\pvpowlib\Local Water Column Database"
    Site_obj.compute_water_column_using_satelite_data(path = local_water_column_database_path, 
                                                      percentile = 0.5,
                                                      interp_method = "linear")


    # ---- COMPUTE ANGSTROM TURBIDITY EXPONENT ----
    # Shettel and Fenn model.
    Site_obj.compute_angstrom_turbidity_exponent_500nm_using_SF_model(model = "Urban")
    

    # ---- COMPUTE AEROSOL OPTICAL DEPTH AT 500 nm ----
    # Satelite data
    local_AOD_500nm_database_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\OptiSurf\Fitness_Function\pvpowlib\Local AOD_550nm Database"
    Site_obj.compute_AOD_500nm_using_satelite_data(path = local_AOD_500nm_database_path, 
                                                   percentile = 0.5,
                                                   interp_method = "linear")


    # ---- COMPUTE SINGLE SCATTERING ALBEDO ----
    # Shettel and Fenn model.
    Site_obj.compute_single_scattering_albedo_using_SF_model(model = "Urban", interp_method = "linear")
    

    # ----COMPUTE AEROSOL ASYMMETRY FACTOR ----
    # Shettel and Fenn model.
    Site_obj.compute_aerosol_asymmetry_factor_using_SF_model(model = "Urban", interp_method = "linear")


    # ---- COMPUTE SPECTRALLY AVERAGED AEROSOL ASYMMETRY FACTOR ----
    Site_obj.compute_spectrally_averaged_aerosol_asymmetry_factor(spectral_range=(300, 4000))


    # ---- COMPUTE SUN DATA ----
    Site_obj.compute_sun_data()


    # ----USE CONSTANT GROUND ALBEDO ----
    Site_obj.use_constant_ground_albedo(constant_ground_albedo=0)


    # ---- COMPUTE TIME INTEGRAL OF IRRADIANCES ----
    Site_obj.compute_cummulative_time_integral_of_irradiances()
    
    
    # ---- COMPUTE EXTRATERRESTRIAL IRRADIANCE ----
    Site_obj.compute_extraterrestrial_normal_irradiance(method = "nrel")
    
    
    # ----- TEST INTERPOLATING FUNCTION -------
    interpd_aaf =\
    Site_obj.time_interpolate_variable(col           = "aerosol_asymmetry_factor", 
                                       year          = 2022,
                                       month         = 1,
                                       day           = 1,
                                       new_hms_float = np.linspace(6,6.33,721),
                                       kind          = "linear")
    
    
    # ---- SAVE SITE_OBJ ----
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    aux.save_obj_with_pickle(class_obj = Site_obj, path = path)
    
#%%
    # ---- LOAD SITE_OBJ ----
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    Site_obj = aux.load_obj_with_pickle(path = path)
    
#%%
    # ---- PLOT DATA ----
    
    for col_name in ["Gb(n)", "int Gb(n)", "T2m", "alpha_500nm", "AOD_500nm",
                     "ground_albedo", "single_scattering_albedo", 
                     "aerosol_asymmetry_factor"]:
        
        Site_obj.plot_data(col = col_name, 
                           years  = None,
                           months = None, 
                           days   = None,
                           hours  = [6.1, 17.9],
                           mode   = 1,
                           figsize = (10,8))
    
    
#%%
    #  ---- EXTRACT RELEVANT DATA INTO VARIABLES ----
    tmy_data                 = Site_obj.tmy_data
    site_data                = Site_obj.site_data  
    single_scattering_albedo = Site_obj.single_scattering_albedo
    aerosol_asymmetry_factor = Site_obj.aerosol_asymmetry_factor
    sun_data                 = Site_obj.sun_data
    ground_albedo            = Site_obj.ground_albedo
    


