#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the atmospheric ozone column of a site.


                  ---- CLIMATE DATA STORE ----

"The Copernicus - Climate Data Store (CDS) is an online open and free service
that allows users to browse and access a wide range of climate datasets via a 
searchable catalogue. It also allows users to build their own applications,
maps and graphs."

1) CDS webpage at:
https://cds.climate.copernicus.eu/cdsapp#!/home

2) More info about its API, at:
https://cds.climate.copernicus.eu/api-how-to

3) Useful tutorial on how to use the API, at
https://youtu.be/cVtiVTSVdlo
    

NOTE: As described by the links in 2) and 3), it is necessary to have a CDS 
      account (and be currently logged in) account in order to be able to use
      the API. Furtheremore, the user's key and the API website link should be 
      stored in a place, recognisable by the system being used.  
"""

#%%                  IMPORTATION OF LIBRARIES

import os
import cdsapi
import zipfile
import warnings
import numpy as np
import netCDF4 as nc
from dateutil.parser import parse
from scipy.interpolate import RegularGridInterpolator



#%%


@np.vectorize
def compute_van_Heuklon_ozone(lat, lon, timestamp):
    """
    Returns the ozone contents in atm-cm for the given latitude/longitude and
    timestamp according to van Heuklon's Ozone model. The model is described in
    Van Heuklon, 'T. K. (1979). Estimating atmospheric ozone for solar radiation 
    models. Solar Energy, 22(1), 63-68'. This function uses numpy functions, 
    so you can pass arrays and it will return an array of results. The
    timestamp argument can be either an array/list or a single value. 
    If timestamp is a single value then this will be used for all lat/lon 
    values given. 
    
    Parameters
    ----------
    
    lat : float
        Site's latitude in degrees. Must a number between -90 and 90.
        
    lon : float
        Site's longitude in degrees. Must be a number between -180 and 180.
    
    timestamp : pandas.Timestamp object or array like of pandas.Timestamp objects
        The times for which the ozone is to be computed. It is strongly 
        recommend that the timestamp use an ISO 8601 format of yyyy-mm-dd.
        
    Returns
    -------
    result : float
        Ozone amount in atm-cm.
        
    Notes
    -----
    1) This function was directly taken from https://github.com/robintw/vanHOzone
       all credit goes to him. I copy-pasted the code rather than directly 
       downloading tthe pachage from pip as I wanted to add some very minor
       changes.
       
    2) The function also supports array like inputs for lat and long but in that
       case timestamp must be either a single element or an array matching the
       length of the lat and lon arrays.
      
    3) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    4) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.

        
    
    """
    # Deal with scalar values
    try:
        lat_count = len(lat)
    except:
        lat = [lat]
        lat_count = 1

    try:
        lon_count = len(lon)
    except:
        lon = [lon]
        lon_count = 1

    if lat_count != lon_count:
        raise ValueError("lan and lon arrays must be the same length")

    lat = np.array(lat)
    lon = np.array(lon)

    # Set the Day of Year
    try:
        # Try and do list-based things with it
        # If it works then it is a list, so check length is correct
        # and process
        count = len(timestamp)
        if count == len(lat):
            try:
                E = [t.timetuple().tm_yday for t in timestamp]
                E = np.array(E)
            except:
                d = [parse(t) for t in timestamp]
                E = [dt.timetuple().tm_yday for dt in d]
                E = np.array(E)
        else:
            raise ValueError("Timestamp must be the same length as lat and lon")
    except:
        # It isn't a list, so just do it once
        try:
            # If this works then it is a datetime obj
            E = timestamp.timetuple().tm_yday
        except:
            # If not then a string, so parse it and set it
            d = parse(timestamp)
            E = d.timetuple().tm_yday

    # Set parameters which are the same for both
    # hemispheres
    D = 0.9865
    G = 20.0
    J = 235.0

    # Set to Northern Hemisphere values by default
    A = np.zeros(np.shape(lat)) + 150.0
    B = np.zeros(np.shape(lat)) + 1.28
    C = np.zeros(np.shape(lat)) + 40.0
    F = np.zeros(np.shape(lat)) - 30.0
    H = np.zeros(np.shape(lat)) + 3.0
    I = np.zeros(np.shape(lat))

    # Gives us a boolean array indicating
    # which indices are below the equator
    # which we can then use for indexing below
    southern = lat < 0

    A[southern] = 100.0
    B[southern] = 1.5
    C[southern] = 30.0
    F[southern] = 152.625
    H[southern] = 2.0
    I[southern] = -75.0

    # Set all northern I values to 20.0
    # (the northern indices are the inverse (~) of
    # the southern indices)
    I[~southern] = 20.0

    I[(~southern) & (lon <= 0)] = 0.0

    bracket = (A + (C * np.sin(np.radians(D * (E + F))) + G *
                    np.sin(np.radians(H * (lon + I)))))

    sine_bit = np.sin(np.radians(B * lat))
    sine_bit = sine_bit ** 2

    result = J + (bracket * sine_bit)
    
    # We convert from dobson to atm-cm.
    result /= 1000

    return result   



def get_CDS_ozone_data(path, year, month=None):
    
    """
    This fucntion connects to the Climate Data Store (CDS) through its API and 
    downloads the ozone-column data from the database:
    'Ozone monthly gridded data from 1970 to present derived from satellite
    observations', for the requested timeframe. The downloaded data are .nc 
    files holding the monthly-average of ozone-column data (in m atm-cm, i.e,
    Dobson) of the whole globe, for the requested timeframe. Then, a new folder
    (whose path is specified by the user) is created to store the downloaded
    files.
    
    Parameters
    ----------
    path : path-str
        Path of the folder where one wishes to store the downloaded files.
    
    year : list of str
        List of years for which the ozone-column data is to be retieved. 
        The years must be of type str rather than int. 
        Eg.: year = ["2019", "2020", "2021"].
        
    month  : None or list of str
        If is None (default), all months for the selected years are retrieved.
        If not None, it must be list of months for which to retrieve the data.
        Said months must be of type str rather than int. 
        Eg.: month = ["01", "02", "11", "12"].
        

        
    Returns
    -------
    None
    
        
    Notes
    -----
    1) For this function to work, the user must have a Climate Data Store 
       account and be currently logged in. Furtheremore, the user's key and the
       API website link should be stored in a place, recognisable by the system 
       being used (see: https://cds.climate.copernicus.eu/api-how-to and 
       https://youtu.be/cVtiVTSVdlo).
       
    2) For more information on the specific databse used, see:
       https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-ozone-v1?tab=overview

    """
    
    zip_path = path + ".zip"
    
    
    if month is None:
        month_ = ["0" + str(i) for i in range(1,10)]
        month_ +=  [str(i) for i in range(10,13)]
        
    else: 
        month_ = month
    

    # The database being used here actually allows for the download of other 
    # variables. The settings used here are such as to only retrive ozone 
    # column data.
    
    c = cdsapi.Client()
    c.retrieve(
        'satellite-ozone-v1',
        {
            'processing_level'     : 'level_4',
            'variable'             : 'atmosphere_mole_content_of_ozone',
            'vertical_aggregation' : 'total_column',
            'sensor'               : 'msr',
            'year'                 : year,
            'month'                : month_,
            'version'              : 'v0024',
            'format'               : 'zip',
        }, 
        zip_path)
    
      
    with zipfile.ZipFile(zip_path, "r") as my_zip:
        my_zip.extractall(path = path)
        
    os.remove(zip_path)
        

    return None




def process_CDS_ozone_data(path, percentile = 0.5, interp_method = "linear"):
    
    """
    Process ozone data located in the local ozone database. 
    This function reads the .nc ozone-column files (files which were obatined
    via the 'get_CDS_ozone_data' function) stored at the directory
    specified by 'path' and then computes multiple useful quantities.
    
    Parameteres
    -----------
    path : path-str
        Path of the folder where the ozone column .nc files are stored.
        That is, the path to the local ozone database.        
        
    percentile : float
        Percentile for computing the 'percentile_data' and 'percentile_data_funcs'
        dictionaries. Must be a number between 0 and 1. Default is 0.5.
        
        
    interp_method : str
        The method of interpolation to perform when computing the 
        'raw_data_funcs', 'avg_data_funcs' and 'percentile_data_funcs'
        dictionaries. Supported methods are the same as supported by scipy's 
        RegularGridInterpolator. Default is "linear".
        
    Returns
    -------
    res : dict
        Dictionary of computed quantities. It has the following key-value
        pairs:
            
            Key : Value
            -----------
            "lat" : numpy.array of floats
                Array of latitude values (in degrees) used by the .nc files to 
                specify the locations at which the ozone column data is 
                reported.
                
            "lon" : numpy.array of floats
                Array of longitude values (in degrees) used by the .nc files to 
                specify the locations at which the ozone column data is 
                reported.
                
            "raw_data" : dict
                Dictionary containing the raw data of ozone-column values stored
                in the local ozone database. It has the following key-value
                pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.arrays of floats
                        The keys '(year, month)' specify the timpe period that 
                        the ozone column values belong to ('year' and 'month' 
                        are both of type int). There are as many keys as .nc
                        files in the local ozone database.
                        The actual ozone-column values (with unit of atm-cm)
                        are given as 2D numpy array of floats. The axis 0
                        accounts for the variation of ozone values with 
                        latitude, while the axis 1, accounts for the variation
                        of ozone values with longitude.
                        
            "raw_data_funcs" : dict 
                Dictionary containing the interpolating functions of the raw 
                data of ozone-column values stored in the local ozone database.
                It has the following key-value pair format:
                    
                Key : Value
                -----------
                    (year, month) : scipy.interpolate.RegularGridInterpolator object
                        The keys '(year, month)' specify the timpe period that 
                        the ozone column interpolating functions belong to 
                        ('year' and 'month' are both of type int). 
                        There are as many keys as .nc files in the local
                        ozone database.
                        The actual ozone-column interpolating functions 
                        (with unit of atm-cm) are a RegularGridInterpolator 
                        object which takes as input a value of latitude and
                        longitude and returns the ozone-column value expected
                        at that location.
                        
            "avg_data" : dict
                Dictionary containing the year-averagred data of ozone-column 
                values stored in the local ozone database. It has the following 
                key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        ozone column values belong to (month is of type int). 
                        The actual ozone-column values of averaged data across
                        different years but on the same month (with unit of
                        atm-cm), are given as a 2D numpy array of floats. 
                        The axis 0 accounts for the variation of ozone values
                        with latitude, while axis 1, accounts for the variation 
                        of ozone values with longitude.
                        
            "avg_data_funcs" : dict 
                Dictionary containing the interpolating functions of the year-
                averagred data of ozone-column values stored in the local ozone 
                database. It has the following key-value pair format:
                    
                Key : Value
                -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        ozone column interpolating functions belong to 
                        (month is of type int). The actual ozone-column 
                        interpolating functions (with unit of atm-cm) are a 
                        RegularGridInterpolator object which takes as input a 
                        value of latitude and longitude and returns the 
                        year-averaged ozone-column value expected
                        at that location.
                        
                        
            "percentile_data" : dict
                Dictionary containing the year-wise 'percentile'-th percentile
                of the ozone-column data values stored in the local ozone
                database. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        ozone column values belong to (month is of type int). 
                        The actual year-wise 'percentile'-th percentile values
                        (with unit of atm-cm), are given as a 2D numpy array of 
                        floats. The axis 0 accounts for the variation of ozone 
                        values with latitude, while axis 1, accounts for the
                        variation of ozone values with longitude.
                        
            "percentile_data_funcs" : dict 
                Dictionary containing the interpolating functions of the 
                year-wise 'percentile'-th percentile data of ozone-column 
                values stored in the local ozone database. It has the following 
                key-value pair format:
                    
                Key : Value
                -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        ozone column interpolating functions belong to 
                        (month is of type int). The actual ozone-column 
                        interpolating functions (with unit of atm-cm) are a 
                        RegularGridInterpolator object which takes as input a 
                        value of latitude and longitude and returns the 
                        year-wise 'percentile'-th percentile ozone-column value
                        expected at that location.
                        
 
    
    Notes
    -----
    1) The "lat" and "lon" arrays are shared across all the other output 
       dictionaries. 
    
    2) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
       
    3) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
       
    
    """
    
    #  ---- FILENAMES RETRIEVAL ----
    
    # We get all the filenames of the .nc files stored at the local ozone database.
    ncfile_names = [i for i in os.listdir(path) if ".nc" in i]
    
    
    # If the local ozone database is empty, we throw an error.    
    if len(ncfile_names) == 0:
        message = "Local ozone database is empty"
        message = f"{message}. No ozone .nc files to retrieve were found."
        raise Exception(message)

    
    #  ---- RAW DATA RETRIEVAL AND INTERPOLATION ----
    years, months = set(), set()
    
    # We read all the .nc files, convert the relevant variables into numpy
    # arrays and store everything into a user-friendlier format. We also
    # initialize the interpolating functions and store them.
    
    raw_data = {}
    raw_data_funcs = {}
    for i, ncfile_name in enumerate(ncfile_names):
        
        year  = int(ncfile_name[:4])
        month = int(ncfile_name[4:6])
        
        years.add(year)
        months.add(month)
        
        ncfile_path = os.path.join(path, ncfile_name)
        
        # Actual reading of .nc files
        ncfile = nc.Dataset(ncfile_path) 
        
        if i == 0:
            lat = np.array(ncfile.variables["latitude"][:])   
            lon = np.array(ncfile.variables["longitude"][:]) 
            
            
        # We convert the units from dobson to atm-cm.  
        raw_data[(year, month)] =\
        np.array(ncfile.variables["total_ozone_column"][0,:,:])/1000   
        
        
        raw_data_funcs[(year, month)] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = raw_data[(year, month)],
                                method = interp_method)
        
        ncfile.close()
        
        
    # If not all 12 months of the year are present in the local ozone database
    # we throw a warning.
    if len(months) < 12:
        message = "WARNING: Local ozone database lacks data for all 12 months"
        message = f"{message} of the year."
        warnings.warn(message)
        


    #  ---- AVG DATA COMPUTATION AND INTERPOLATION ----
    
    # We compute the year-averaged data and interpolating functions.
    avg_data = {}
    avg_data_funcs = {}
    for month in months:
        
        avg_data[month] = []
        for year in years:
            try: avg_data[month].append(raw_data[(year, month)])
            except KeyError: pass
        
        avg_data[month] = np.stack(avg_data[month], axis=2) 
        avg_data[month] = avg_data[month].mean(axis=2)
        
        avg_data_funcs[month] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = avg_data[month],
                                method = interp_method)
        
        
        
    #  ---- PERCENTILE DATA COMPUTATION AND INTERPOLATION ----        
        
    # We compute the year-wise 'perecentile'-th percentile data and 
    # interpolating functions.      
    percentile_data = {}
    percentile_data_funcs = {}
    for month in months:
        
        percentile_data[month] = []
        for year in years:
            try: percentile_data[month].append(raw_data[(year, month)])
            except KeyError: pass
        
        percentile_data[month] = np.stack(percentile_data[month], axis=2) 
        percentile_data[month] = np.percentile(percentile_data[month], 
                                               q=percentile, 
                                               axis=2)
        
        percentile_data_funcs[month] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = percentile_data[month],
                                method = interp_method)
        
        
    
    #  ---- RESULTS ----    
        
    # We store all the data and return it.
    res = {
    "lat" : lat,
    "lon" : lon,
    "raw_data" : raw_data,
    "raw_data_funcs" : raw_data_funcs,
    "avg_data" : avg_data,
    "avg_data_funcs" : avg_data_funcs,
    "percentile_data" : percentile_data,
    "percentile_data_funcs" : percentile_data_funcs }
        
        
        
    return res
        
        
        
        
    



#%%                              EXAMPLES

if __name__ == '__main__': 
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # We define the path of the folder where we want to save the retrieved ozone data.
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Local Ozone Column Database"
    
#%%
    # We retrieve the ozone data for all of the months of "2019", "2020" and 
    # "2021". Actually, for this level of data, this are the only years we
    # can retrieve.
    get_CDS_ozone_data(path = path, year =["2019", "2020", "2021"])
    
#%%
    # Having our local ozone databse well established, we process the retrieved data.
    res = process_CDS_ozone_data(path, percentile = 0.5, interp_method = "linear")
    
#%%
    # Let us plot some of the data.
    year, month = 2020, 5
    raw_data = res["raw_data"][(year, month)]
    avg_data = res["avg_data"][(month)]
    percentile_data = res["percentile_data"][(month)]
    
    lon, lat = res["lon"], res["lat"]
    Lon, Lat = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, raw_data, levels = np.linspace(np.nanmin(raw_data), np.nanmax(raw_data), 100) )
    cbar = plt.colorbar()
    plt.title(f"raw_data @ year = {year}, month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('ozone column [atm-cm]')
    plt.show()
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, avg_data, levels = np.linspace(avg_data.min(),avg_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"avg_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('ozone column [atm-cm]')
    plt.show()
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, percentile_data, levels = np.linspace(percentile_data.min(),percentile_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"percentile_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('ozone column [atm-cm]')
    plt.show()
    
#%%
    # And also interpolate it.
    lat_, lon_ = 6.230833, -75.590553 # Medellín coordinates
    raw_data_ = res["raw_data_funcs"][(year, month)]([lat_, lon_])
    avg_data_ = res["avg_data_funcs"][(month)]([lat_, lon_])
    percentile_data_ = res["percentile_data_funcs"][(month)]([lat_, lon_])
    
    print(f"raw_data @ year = {year}, month = {month} is {raw_data_}")
    print(f"avg_data @ month = {month} is {avg_data_}")
    print(f"percentile_data @ month = {month} is {percentile_data_}")

#%%
    # Let us also put van_Heuklon's model to the test and plotted.
    
    MONTH_DAYS =\
    {0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    if month < 10: month_ = f"0{month}"
    else: month_ = str(month)
        
        
    # So that the comparison be fair, we also compute the ozone monthly
    # average using van heuklen.
    van_heuklen_data = np.zeros(raw_data.shape)  
    for day in range(1, MONTH_DAYS[month]+1):
        
        timestamp = pd.Timestamp(f"{year}-{month_}-{day}")
        
        for i, lat_ in enumerate(lat):
                van_heuklen_data[i,:] += compute_van_Heuklon_ozone(lat_, lon, timestamp)
                
        print(f"Days computed: {day}")
    
    van_heuklen_data /= MONTH_DAYS[month]
    
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, van_heuklen_data, levels = np.linspace(van_heuklen_data.min(),van_heuklen_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"van_heuklen_data @ year ={year}, month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('ozone column [atm-cm]')
    plt.show()



