#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
computation and manipulation of the precipitable water column of a site.


#                  ---- CLIMATE DATA STORE ----

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
from scipy.interpolate import RegularGridInterpolator
from Ambience_Modelling import auxiliary_funcs as aux



#%%

def get_CDS_water_column_data(path, year, month=None):
    
    """
    This fucntion connects to the Climate Data Store (CDS) through its API and 
    downloads the monthly-averaged water column data from the database:'Monthly
    total column water vapour over land and ocean from 2002 to 2012 derived 
    from satellite observations', for the requested timeframe. The downloaded
    data are .nc files holding the Total Column Water Vapour (in kg/m²) of the
    whole globe, for the requested timeframe. Then, a new folder (whose path is 
    specified by the user) is created to store the downloaded files.
    
    Parameters
    ----------
    path : path-str
        Path of the folder where one wishes to store the downloaded files.
    
    year : list of str
        List of years for which the water-column data is to be retieved. 
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
       https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-total-column-water-vapour-land-ocean?tab=overview
       
    3) "Total Column Water Vapour (also called integrated Water Vapour (IWV) or
       Precipitable Water Vapour (PWV)) is the integrated mass of gaseous water 
       in the total column of the atmosphere over an area of 1 m²" and it is 
       usually given in kg/m². However, another common way of expressing this
       same variable, is as the height (in cm) that a column of x kilograms
       of water would have, when being bounded by a cube whose base has an area
       of 1 m². Using the average density of water (997 kg/m³ ≈ 1000 kg/m³), 
       we see that a Total column of water vapour of x kg/m² would roughly 
       equate to a x/10 cm water column.

    """
    
    zip_path = path + ".zip"
    
    
    if month is None:
        month_ = ["0" + str(i) for i in range(1,10)]
        month_ +=  [str(i) for i in range(10,13)]
        
    else: 
        month_ = month
    

    # The database being used here actually allows for the download of other 
    # variables. The settings used here are such as to only retrive water 
    # column data.
    
    c = cdsapi.Client()

    c.retrieve(
        'satellite-total-column-water-vapour-land-ocean',
        {
            'variable'               : 'all',
            'format'                 : 'zip',
            'horizontal_aggregation' : '0_5_x_0_5',
            'year'                   : year,
            'month'                  : month_,
        },
        zip_path)
    
      
    with zipfile.ZipFile(zip_path, "r") as my_zip:
        my_zip.extractall(path = path)
        
    os.remove(zip_path)
        

    return None




def fill_CDS_water_column_data_nans(path, iterations = 20000, show_progress = False):
    
    """
    The data retrieved from the Climate Data Store (CDS) database 'Monthly
    total column water vapour over land and ocean from 2002 to 2012 derived 
    from satellite observations' has a considerable amount of missing or
    defective values in its files. This is inconvenient for later computations.
    As such, this function reads all .nc files in the directory specified by 
    the 'path' variable (i.e, the local water column database), then uses the
    function 'auxiliary_funcs.fill_CDS_globe_nans_using_laplace' to fill each 
    file's NaN values with a suitable numeric approximation and then saves each
    modified file back to the same directory as .npy file. 
    
    Parameters
    ----------
    path : path-str
        Path of the folder where the .nc files, containing the water
        column information downloaded from the aforementioned database, are
        stored. The resulting .npy files will also be stored in this directory.
        
    iterations: int
        Number of iterations that the 'auxiliary_funcs.fill_CDS_globe_nans_using_laplace'
        function should use for computing the numerical approximation to the
        NaN values, before stopping (must be non-negative). The greater the 
        number of iterations, the greater the chance that convergence of the
        computed values has been reached. However, the time of computation also 
        increases. Default is 20000.
        
    show_progress : bool
        If True, after finishing processing each file, print how many files
        have been processed as of yet. If False, print nothing. Default is False.
        
    Returns
    -------
    None

    """
    
    years, months = set(), set()
    
    # We get all the filenames of the .nc files stored at the local water
    # column database.
    ncfile_names = [i for i in os.listdir(path) if ".nc" in i]
    
    # If the local water column database is empty, we throw an error.    
    if len(ncfile_names) == 0:
        message = "Local water database is empty"
        message = f"{message}. No water column .nc files to retrieve were found."
        raise Exception(message)

    
    # We read each .nc file, convert the column water information into a numpy 
    # array and the pass it to the 'auxilary_funcs.fill_CDS_globe_nans_using_laplace'
    # function in order to fill the missing values. Then we save the modifed 
    # array as a .npy file to the same directory.
    
    for i, ncfile_name in enumerate(ncfile_names):
        
        year  = int(ncfile_name[-9:-5])
        month = int(ncfile_name[-5:-3])
        
        years.add(year)
        months.add(month)
        
        ncfile_path = os.path.join(path, ncfile_name)
        ncfile = nc.Dataset(ncfile_path) # actual reading of .nc files
    
        # We convert the units from kg/m² to cm.  
        raw_data = np.array(ncfile.variables["tcwv"][0,:,:])/10 
        raw_data[raw_data < 0] = np.nan

        ncfile.close()
        
        # We use the function 'fill_CDS_globe_nans_using_laplace' to fill each
        # missing value with a suitable numerical approximation.
        filled_nans_data = aux.fill_CDS_globe_nans_using_laplace(raw_data, iterations)
        
        # We save each modified array to the same folder as .npy file.
        filled_nans_data_path = ncfile_name[:-3] + "_filled_NaNs.npy"
        filled_nans_data_path = os.path.join(path, filled_nans_data_path)
        np.save(filled_nans_data_path, filled_nans_data)
        
        
        if show_progress:
            print(f"Processed files: {i+1}")
        
        
    # If not all 12 months of the year are present in the local water column
    # database (with respect to the .nc files) we throw a warning.
    if len(months) < 12:
        message = "WARNING: Local water database lacks data for all 12 months"
        message = f"{message} of the year."
        warnings.warn(message)
        

    return None




def process_CDS_water_column_data(path, percentile = 0.5, interp_method = "linear"):
    
    """
    Process water column data located in the local water column database. 
    This function reads the .nc and filled_NaNs water column files (files which
    were obatined via the 'get_CDS_water_column_data' and  
    'fill_CDS_water_column_data_nans' functions) stored at the directory
    specified by 'path' and then computes multiple useful quantities.
    
    Parameteres
    -----------
    path : path-str
        Path of the folder where the water column .nc and filled_NaNs.npy files 
        are stored. That is, the path to the local water column database.        
        
    percentile : float
        Percentile for computing the 'percentile_data' and 'percentile_data_funcs'
        dictionaries. Must be a number between 0 and 1. Default is 0.5.
        
        
    interp_method : str
        The method of interpolation to perform when computing the 
        'filled_nans_data_funcs', 'avg_data_funcs' and 'percentile_data_funcs'
        dictionaries. Supported methods are the same as supported by scipy's 
        RegularGridInterpolator. Default is "linear".
        
    Returns
    -------
    res : dict
        Dictionary of computed quantities. It has the following key-value
        pairs:
            
            Key : Value
            ------------
            
            "lat" : numpy.array of floats
                Array of latitude values (in degrees) used by the .nc files to 
                specify the locations at which the water column data is 
                reported.
                
            "lon" : numpy.array of floats
                Array of longitude values (in degrees) used by the .nc files to 
                specify the locations at which the water column data is 
                reported.
                
            "raw_data" : dict
                Dictionary containing the raw data of water-column values stored
                in the local water column database. It has the following key-value
                pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.arrays of floats
                        The keys '(year, month)' specify the timpe period that 
                        the water column values belong to ('year' and 'month' 
                        are both of type int). There are as many keys as .nc
                        files in the local water database.
                        The actual water-column values (with unit of cm)
                        are given as 2D numpy array of floats. The axis 0
                        accounts for the variation of water values with 
                        latitude, while the axis 1, accounts for the variation
                        of water column values with longitude.
                        
            "filled_nan_data" : dict
                Dictionary containing the filled-NaN data of water-column values 
                stored in the local water column database. It has the following 
                key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.arrays of floats
                        The keys '(year, month)' specify the timpe period that 
                        the water column values belong to ('year' and 'month' 
                        are both of type int). 
                        The actual filled-NaN  water-column values (with unit 
                        of cm) are given as 2D numpy array of floats. The axis 0
                        accounts for the variation of water column values with 
                        latitude, while the axis 1, accounts for the variation
                        of water column values with longitude.
                        
            "filled_nan_data_funcs" : dict 
                Dictionary containing the interpolating functions of the 
                filled-NaN data of water-column values stored in the local 
                water column database. It has the following key-value pair format:
                    
                Key : Value
                -----------
                    (year, month) : scipy.interpolate.RegularGridInterpolator object
                        The keys '(year, month)' specify the timpe period that 
                        the water column interpolating functions belong 
                        to ('year' and 'month' are both of type int). 
                        The actual filled-NaN water-column interpolating functions 
                        (with unit of cm) are a RegularGridInterpolator 
                        object which takes as input a value of latitude and
                        longitude and returns the water-column value expected
                        at that location.
                        
            "avg_data" : dict
                Dictionary containing the year-averagred data of filled-NaN
                water-column values stored in the local water database.
                It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        water column values belong to (month is of type int). 
                        The actual water-column values of averaged data across
                        different years but on the same month (with unit of
                        cm), are given as a 2D numpy array of floats. 
                        The axis 0 accounts for the variation of water values
                        with latitude, while axis 1, accounts for the variation 
                        of water values with longitude.
                        
            "avg_data_funcs" : dict 
                Dictionary containing the interpolating functions of the year-
                averagred data of filled-NaN water-column values stored in the 
                local water column database. It has the following key-value pair 
                format:
                    
                Key : Value
                -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        water column interpolating functions belong to 
                        (month is of type int). The actual water-column 
                        interpolating functions (with unit of cm) are a 
                        RegularGridInterpolator object which takes as input a 
                        value of latitude and longitude and returns the 
                        year-averaged water-column value expected
                        at that location.
                        
                        
            "percentile_data" : dict
                Dictionary containing the year-wise 'percentile'-th percentile
                of the filled-NaN water-column data values stored in the local
                water database. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        water column values belong to (month is of type int). 
                        The actual filled-NaN year-wise 'percentile'-th 
                        percentile values (with unit of cm), are given as a 2D
                        numpy array of floats. The axis 0 accounts for the 
                        variation of water  values with latitude, while axis 1,  
                        accounts for the variation of water values with longitude.
                        
            "percentile_data_funcs" : dict 
                Dictionary containing the interpolating functions of the filled-NaN
                year-wise 'percentile'-th percentile data of water-column 
                values stored in the local water column database. It has the
                following key-value pair format:
                    
                Key : Value
                -----------
                    month : 2D numpy.arrays of floats
                        The keys 'month' specify the time period that the
                        water column interpolating functions belong to 
                        (month is of type int). The actual water-column 
                        interpolating functions (with unit of cm) are a 
                        RegularGridInterpolator object which takes as input a 
                        value of latitude and longitude and returns the 
                        year-wise 'percentile'-th percentile water-column value
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

    # We get all the filenames of the .nc files stored at the local water
    # column database.
    ncfile_names = [i for i in os.listdir(path) if ".nc" in i]
    filled_nans_file_names = [i for i in os.listdir(path) if "filled_NaNs.npy" in i]
    
    
    # If the local water column database is empty of .nc files, we throw an error.       
    if len(ncfile_names) == 0:
        message = "Local water column database is empty of .nc files"
        message = f"{message}. No column water .nc files to retrieve were found."
        raise Exception(message)
        
        
    # If the local water column database is empty of filled-NaN files, we 
    # throw an error.       
    if len(filled_nans_file_names) == 0:
        message = "Local water column database is empty of filled_NaNs files."
        message = f"{message}. No water column .npy filled_NaNs files to"
        message = f"{message} retrieve were found."
        raise Exception(message)
        
    
    
    # If a given .nc file does not have an accompanying filled_NaNs.npy
    # pair file, we warn the user that the data contained in the .nc file,
    # will not be included in the data processing process.
    
    check_list_nc = [i.split(".")[0] for i in ncfile_names]
    check_list_filled_nans = [i.split(".")[0] for i in filled_nans_file_names]    
        
    for string in check_list_filled_nans:
        pair_check = any(substring in string for substring in check_list_nc)
        
        ncfile_name = string.split("_")[0] + ".nc"
        if not pair_check:
            message = f"WARNING: '{ncfile_name}' file does not have an accompanying"
            message = f"{message} 'filled_NaNs.npy' file (or at least none that"
            message = f"{message} could be detected). If this is the case, be"
            message = f"{message} aware that the info contained in '{ncfile_name}'"
            message = f"{message} will be excluded from the data processing"
            message = f"{message} process, as this requires the use of data"
            message = f"{message} without missing values."
            warnings.warn(message)


    #  ---- RAW DATA RETRIEVAL ----
    years, months = set(), set()
    
    # We read all .nc files, convert the relevant variables into numpy
    # arrays and store everything into a user-friendlier format.
    
    raw_data = {}
    for i, ncfile_name in enumerate(ncfile_names):
        
        year  = int(ncfile_name[-9:-5])
        month = int(ncfile_name[-5:-3])
        
        years.add(year)
        months.add(month)
        
        ncfile_path = os.path.join(path, ncfile_name)
        
        # Actual reading of .nc files
        ncfile = nc.Dataset(ncfile_path) 
        
        if i == 0:
            lat = np.array(ncfile.variables["lat"][:])   
            lon = np.array(ncfile.variables["lon"][:]) 
            
            
        # We convert the units from dobson to cm.  
        raw_data[(year, month)] =\
        np.array(ncfile.variables["tcwv"][0,:,:])/10  
        
        raw_data[(year, month)][raw_data[(year, month)] < 0] = np.nan
        
        ncfile.close()
        
        
    # If not all 12 months of the year are present in the local water column
    # database (with respect to the .nc files) we throw a warning.
    if len(months) < 12:
        message = "WARNING: Local water column database lacks .nc data files"
        message = f"{message} for all 12 months of the year."
        warnings.warn(message)
        
        
    #  ---- FILLED_NaNS DATA RETRIEVAL AND INTERPOLATION ---- 
    
    years, months = set(), set()
    
    
    # We read all the filled_NaNs.npy files and store everything into a 
    # user-friendlier format. We also initialize the interpolating functions
    # and store them.
    
    filled_nans_data = {}
    filled_nans_data_funcs = {}
    for i, filled_nans_file_name in enumerate(filled_nans_file_names):
        
        year_month = filled_nans_file_name.split("_")[-3]
        year  = int(year_month[:4])
        month = int(year_month[4:])
        
        years.add(year)
        months.add(month)
        
        filled_nans_file_path = os.path.join(path, filled_nans_file_name)
        
        # Actual reading of filled_NaNs.npy files
        filled_nans_data[(year, month)] = np.load(filled_nans_file_path) 
        
        filled_nans_data_funcs[(year, month)] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = filled_nans_data[(year, month)],
                                method = interp_method)
        
        
   

    # If not all 12 months of the year are present in the local water column
    # database (with respect to the filled_NaNs.npy files) we throw a warning.
    if len(months) < 12:
        message = "WARNING: Local water column database lacks filled_NaNs.npy "
        message = f"{message} data files for all 12 months of the year."
        warnings.warn(message)
        
        
        
    #  ---- AVG DATA COMPUTATION AND INTERPOLATION ----    
        
    # We compute the year-averaged data and interpolating functions.
    avg_data = {}
    avg_data_funcs = {}
    for month in months:
        
        avg_data[month] = []
        for year in years:
            try: avg_data[month].append(filled_nans_data[(year, month)])
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
            try: percentile_data[month].append(filled_nans_data[(year, month)])
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
    "filled_nans_data" : filled_nans_data,
    "filled_nans_data_funcs" : filled_nans_data_funcs,
    "avg_data" : avg_data,
    "avg_data_funcs" : avg_data_funcs,
    "percentile_data" : percentile_data,
    "percentile_data_funcs" : percentile_data_funcs }
        
        
        
    return res



#%%                              EXAMPLES

if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    
    # We define the path of the folder where we want to save the retrieved water data.
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Local Water Column Database"
    
#%%
    # We retrieve the water data for all of the months of "2003", "2004" and
    # "2005" as an example.
    year = ['2003', '2004', '2005']
    get_CDS_water_column_data(path = path, year = year)
    
#%%
    # Let us compute the filled-NaN values of all .nc filles present in 
    # the local water databse. This may take a while if there are many.
    fill_CDS_water_column_data_nans(path = path, 
                                    iterations = 20000,
                                    show_progress=True)
    
#%%
    # Having our local ozone databse well established, we process the retrieved data.
    res = process_CDS_water_column_data(path, percentile = 0.5, interp_method = "linear")

#%%
    # Let us plot some of the data.
    
    year, month = 2006, 7    

    raw_data = res["raw_data"][(year, month)]
    filled_nans_data = res["filled_nans_data"][(year, month)]
    avg_data = res["avg_data"][month] 
    percentile_data = res["percentile_data"][month]
    
    
    lat = res["lat"]
    lon = res["lon"]
    # Let us plot some of the data.
    Lon, Lat = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, raw_data, levels = np.linspace(np.nanmin(raw_data), np.nanmax(raw_data), 100))
    cbar = plt.colorbar()
    plt.title(f"raw_data @ year = {year}, month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('water vapor column [cm]')
    plt.show()
    
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, filled_nans_data, levels = np.linspace(filled_nans_data.min(), filled_nans_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"filled_nans_data @ year = {year}, month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('water vapor column [cm]')
    plt.show()
    

    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, avg_data, levels = np.linspace(avg_data.min(),avg_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"avg_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('water vapor column [cm]')
    plt.show()
    
    
    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, percentile_data, levels = np.linspace(percentile_data.min(),percentile_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"percentile_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label('water vapor column [cm]')
    plt.show()
    
#%%
    # And also interpolate it.
    lat_, lon_ = 6.230833, -75.590553 # Medellín coordinates
    filled_nans_data_ = res["filled_nans_data_funcs"][(year, month)]([lat_, lon_])
    avg_data_ = res["avg_data_funcs"][(month)]([lat_, lon_])
    percentile_data_ = res["percentile_data_funcs"][(month)]([lat_, lon_])
    
    print(f"filled_nans_data @ year = {year}, month = {month} is {filled_nans_data_}")
    print(f"avg_data @ month = {month} is {avg_data_}")
    print(f"percentile_data @ month = {month} is {percentile_data_}")
















