#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains functions and methods related to reading and manipulating
Aerosol Optical Depth (AOD) Aeronet files.

More specifically, this module contains functions for the manipulation of 
'Aerosol Optical Depth (AOD) with Precipitable Water and Angstrom Parameter',
Level 2.0, aeronet files downloaded from one of the sites listed in:
https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3

"""

#%%                           IMPORTATION OF MODULES
import numpy as np
import pandas as pd
import Ambience_Modelling.auxiliary_funcs as aux

#%%                  DEFINITION OF CONSTANTS

# Number of days each month posesses.
MONTH_DAYS =\
{0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

MONTH_NAMES_TO_NUMBERS =\
{"JAN" : 1, "FEB" : 2, "MAR" : 3, "APR" : 4, "MAY" : 5, "JUN" : 6,
 "JUL" : 7, "AUG" : 8, "SEP" : 9, "OCT" : 10, "NOV" : 11, "DEC":12}

# Dict for timestamp to float conversion.
TIMESTAMP_HMS_TO_FLOAT_DICT = { "d" : [1/24, 1/1440, 1/86400], 
                                "h" : [1, 1/60, 1/3600], 
                                "m" : [60, 1, 1/60], 
                                "s" : [3600, 60, 1] }

#%%                DEFINITION OF LOACAL AUXILIARY FUNCTIONS

# The functions defined here actually originate from the 'Time' module.
# However, in order to avoid the risk of circular imports, we define them 
# again here for local use.

def timestamp_hms_to_float(timestamp, unit = "h"):
    
    """
    Convert Timestamp Hour:Minutes:Seconds information to float.
    Example: timestamp_hms_to_float(timestamp, unit = "h"), where
    timestamp = pd.Timestamp("2023-03-08 14:25:36") returns 14.426667. That is, 
    it turns the 14h 25min 36s of the timestamp to an equivalent number
    of hours. Had we used timestamp_hms_to_float(timestamp, unit = "s"),
    the result would have been 51936. That is, the equivalent of 14h 25min 36s
    in seconds.
    
    
    Parameters
    ----------
    timestamp : pandas.Timestamp object
        Timestamp to convert to float.
        
    unit : str, optional
        Time unit to which the timestamp is to be converted. It can either be
        'd' (day), 'h' (hour), 'm' (minute) or 's' (second). Default is 'h'.
        
    tz_name : str
        Time zone string accepted by pandas.
    

    Returns
    -------
    res : float
        timestamp converted to float to the specified unit.
    
    
    """
    
    conversion = TIMESTAMP_HMS_TO_FLOAT_DICT[unit]
    
    res = timestamp.hour*conversion[0]
    res += timestamp.minute*conversion[1]
    res += timestamp.second*conversion[2]
    
    return res


def utc_hour_to_tz_name(utc_hour):
    """
    Turns float representing the time zone into a string representing the time
    zone which is accepted by pandas.

    Parameters
    ----------
    utc_hour : float
        Timezone number. Must be anumber between -12 and 12.

    Returns
    -------
    tz_name : str
        Time zone string accepted by pandas.
        
    Notes
    -----
    1) For more information about the time zone strings accepted by pandas, see
       the link: https://pvlib-python.readthedocs.io/en/v0.3.0/timetimezones.html

    """
    utc_hour = int(utc_hour)
    tz_name = "Etc/GMT"    
    if(utc_hour <= 0):  tz_name = "".join([tz_name, "+"])
    tz_name = "".join([tz_name, str(-utc_hour)])

    return tz_name



#%%         DEFINITION OF FUNCTIONS FOR THE MANIPULATION OF AERONET DATA


def read_lev20_AOD_aeronet_file(path):
    
    """
    Read a level 2.0 (such files have '.lev20' extension) AERONET data file 
    of 'Aerosol Optical Depth (AOD) with Precipitable Water and Angstrom
    Parameter'  and transform it into a pandas.DataFrame object for
    easy manipulation.
    
    Parameters
    ----------
    path : path-str
        Path of the aeronet file from which the data is to be read.
        
    Returns
    -------
    aeronet_df : pandas.DataFrame
        Raw table of data read for the aeronet file.
    
    """
    
    file = open(path, "r")
    lines = file.readlines()
    
    cols = lines[6].split(",")
    data = [i.split(",") for i in lines[7:]]
    aeronet_df =\
    pd.DataFrame(index = range(len(data)), columns = cols)
    
    for i, data_ in enumerate(data):
        aeronet_df.iloc[i,:] = data_
    
    return aeronet_df




def compute_tmy_df_from_all_points_lev20_AOD_aeronet_df(aeronet_df, utc_hour, percentile = 0.5, full_index=True, interp=False):
    
    """
    Function for transforming the raw table DataFrame of level 2.0 
    'Aerosol Optical Depth (AOD) with Precipitable Water and Angstrom
    Parameter' (with 'All Points' data format) aeronet data, into a more
    suitable form for its utilization as TMY data.
    
    Parameters
    ----------
    aeronet_df : pandas.DataFrame
        Raw table of data read for the aeronet file.
        
    utc_hour : float
        Timezone number. Must be anumber between -12 and 12.
        
    percentile : float or None
        If float, it is the percentile used when calculating the hourly
        percentiles of the data. Must be a  umber between 0 a 1. If None,
        instead of 'percentile'-th percentile, the hourly mean is calculated
        for the data. Defualt is 0.5.
        
    full_index : bool
        If False, it only uses TMY indices for which there is data available.
        If True, the full index of TMY data is used and any extra-indices
        added are filled with NaNs. Default is True.
        
    interp : bool
        It is almost guranteed that the resulting 'tmy_df' will have missing
        values (ie, NaNs). If interp is set to True, these values are 
        filled using the average value of their non-NaN neighbours.
        If False, no interpolation takes place. Default is False.
        
        
    Returns
    -------
    tmy_df: pandas.DataFrame
        PVGIS TMY-like table of data, obtained from the AERONET file.
        
        
    Notes
    -----
    1) The units for the O3 and NO2 columns are atm-cm.
    
    2) The column "H2O" is the precipitable water column andhas units of cm.
    
    3) The "zen" column refers to the sun's zenith angle and is in degrees.
    
    4) If interp is True, the interpolation treats each column of the
       DataFrame as if it was a continuous array. That is, it has no regard
       for time continuity or time jumps in the data. 
       
    5) "_alpha" in any column is a stand-in for "Angstrom_Exponent".
      
    
    """

    tmy_df = aeronet_df.copy()
    
    
    # --- SET THE CORRECT DTYPE FOR COLUMNS ---
    
    not_float_cols =\
    ["Date(dd:mm:yyyy)",  "Time(hh:mm:ss)", "Data_Quality_Level", 
     "AERONET_Site_Name", "Last_Date_Processed"]
    
    float_cols = [col for col in tmy_df.columns if col not in not_float_cols]
    
    for col in float_cols:
        tmy_df[col] = tmy_df[col].astype(float) 
    

    # --- FILTER DATA ---

    # Drop all columns whose readings are non-sensical
    tmy_df[tmy_df==-999] = np.nan
    tmy_df = tmy_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    
    # Drop unrelevant columns.
    tmy_df.drop(columns=
    ["Day_of_Year", "Day_of_Year(Fraction)", "AERONET_Instrument_Number"], 
    inplace=True)
    
    
    # --- CREATE MONTH, DAY, HOUR COLUMNS ---

    tmy_df["Date(dd:mm:yyyy)"] =\
    tmy_df["Date(dd:mm:yyyy)"].apply(lambda x: "-".join(x.split(":")[::-1]))
    
    tmy_df["Date(dd:mm:yyyy)"] =\
    tmy_df["Date(dd:mm:yyyy)"] + " " + tmy_df["Time(hh:mm:ss)"]
    
    tmy_df["Date(dd:mm:yyyy)"] =\
    tmy_df["Date(dd:mm:yyyy)"].apply(lambda x:pd.Timestamp(x, tz="Etc/GMT+0"))
    
    tz_name = utc_hour_to_tz_name(utc_hour)
    
    tmy_df["Date(dd:mm:yyyy)"] =\
    tmy_df["Date(dd:mm:yyyy)"].apply(lambda x:x.tz_convert(tz_name))
    
    tmy_df.drop(columns=["Time(hh:mm:ss)"], inplace=True)
    
    tmy_df.rename(columns={"Date(dd:mm:yyyy)":"Date"}, inplace=True)
    
    tmy_df["Month"] = tmy_df["Date"].apply(lambda x:x.month)
    tmy_df["Day"]   = tmy_df["Date"].apply(lambda x:x.day)
    
    tmy_df["Hour"] =\
    tmy_df["Date"].apply(lambda x:round(timestamp_hms_to_float(x, unit="h")))
    
    
    
    # --- DROP ALL NON-NUMERIC COLUMNS ---
    cols_to_drop = ["Date","Data_Quality_Level", "AERONET_Site_Name", "Last_Date_Processed"]
    cols_to_keep = [col for col in tmy_df.columns if col not in cols_to_drop]
    tmy_df = tmy_df[cols_to_keep]

    # --- COMPUTE HOURLY MEANS/PERCENTILES OF VARIABLE VALUES ---
    if percentile is None:
        tmy_df = tmy_df.groupby(["Month", "Day", "Hour"]).mean(numeric_only=True)
    else:
        tmy_df = tmy_df.groupby(["Month", "Day", "Hour"]).quantile(q = percentile)
        
    # --- MODIFY DATA ---
    
    # Convert columns from Dobsons to atm-cm
    tmy_df["Ozone(Dobson)"] /= 1000
    tmy_df["NO2(Dobson)"] /= 1000
    
    tmy_df = tmy_df.rename(
    columns={"Ozone(Dobson)": "O3",
             "NO2(Dobson)"  : "NO2",
             "Precipitable_Water(cm)"        : "H2O",
             "440-870_Angstrom_Exponent"     : "440nm-870nm_alpha",
             "380-500_Angstrom_Exponent"     : "380nm-500nm_alpha",
             "440-675_Angstrom_Exponent"     : "440nm-675nm_alpha", 
             "500-870_Angstrom_Exponent"     : "500nm-870nm_alpha",
             "340-440_Angstrom_Exponent"     : "340nm-440nm_alpha",
             "Solar_Zenith_Angle(Degrees)"   : "zen",
             "Optical_Air_Mass"              : "rel_airmass",
             "Sensor_Temperature(Degrees_C)" : "T2m"})
    
    # Delete 29th of February if it exists.
    try: 
        for hour in tmy_df.loc[(2, 29),:].index:
            tmy_df.drop(index=(2,29,hour), inplace=True)
    except KeyError:
        pass
            
        
    # --- COMPUTE FULL TMY INDEX IF SPECIFIED ---
    if full_index:
        # The data present in tmy_df may not always include the full 365
        # days of a year. There may be a couple of days missing. If full
        # index is set to True, the full index is computed, not just the
        # days that are present. The added indices are filled with NaNs.
        
        new_index = []        
        for month in range(1, 13):
            for day in range(1, MONTH_DAYS[month]+1):
                for hour in range(24):
                    new_index.append((month, day, hour))
        
        new_index = pd.MultiIndex.from_tuples(new_index)
        new_tmy_df = pd.DataFrame(index=new_index, columns= tmy_df.columns)
        
        new_tmy_df.loc[tmy_df.index] = tmy_df
        tmy_df = new_tmy_df.apply(lambda x:x.astype(float))
        tmy_df.index.names = ["Month", "Day", "Hour"]
    
    
    if interp:
        # It is almost guranteed that the resulting data will have missing
        # values (ie, NaNs). If interp is set to True, these values are 
        # filled using the average value of their non-NaN neighbours.
        # Must be carefull though. These treats each column of the
        # DataFrame as if it was a continuous array. That is, it has no regard
        # for time continuity or time jumps. 
        tmy_df =\
        tmy_df.apply(lambda x: aux.fill_nans_using_laplace_1D(np.array(x), 5000))
        
        
    return tmy_df




def compute_tmy_df_from_month_avg_lev20_AOD_aeronet_df(aeronet_df, utc_hour, percentile = 0.5, full_index=True, interp=False):
    """
    Function for transforming the raw table DataFrame of level 2.0 
    'Aerosol Optical Depth (AOD) with Precipitable Water and Angstrom
    Parameter' (with 'Monthly Averages' data format) aeronet data, into a more
    suitable form for its utilization as TMY data.
    
    Parameters
    ----------
    aeronet_df : pandas.DataFrame
        Raw table of data read for the aeronet file.
        
    utc_hour : float
        Timezone number. Must be anumber between -12 and 12.
        
    percentile : float or None
        If float, it is the percentile used when calculating the hourly
        percentiles of the data. Must be a  umber between 0 a 1. If None,
        instead of 'percentile'-th percentile, the hourly mean is calculated
        for the data. Defualt is 0.5.
        
    full_index : bool
        If False, it only uses TMY indices for which there is data available.
        If True, the full index of TMY data is used and any extra-indices
        added are filled with NaNs. Default is True.
        
    interp : bool
        It is almost guranteed that the resulting 'tmy_df' will have missing
        values (ie, NaNs). If interp is set to True, these values are 
        filled using the average value of their non-NaN neighbours.
        If False, no interpolation takes place. Default is False.
        
        
    Returns
    -------
    tmy_df: pandas.DataFrame
        PVGIS TMY-like table of data, obtained from the AERONET file.
        
        
    Notes
    -----
    1) The units for the O3 and NO2 columns are atm-cm.
    
    2) The column "H2O" is the precipitable water column andhas units of cm.
    
    3) The "zen" column refers to the sun's zenith angle and is in degrees.
    
    4) If interp is True, the interpolation treats each column of the
       DataFrame as if it was a continuous array. That is, it has no regard
       for time continuity or time jumps in the data. 
       
    5) "_alpha" in any column is a stand-in for "Angstrom_Exponent".
      
    
    """
    tmy_df = aeronet_df.copy()
    
    # --- DROP DUPLICATE COLUMNS ---
    tmy_df = tmy_df.loc[:,~tmy_df.columns.duplicated()].copy()
    
    # --- DROP MOST NON-NUMERIC COLUMNS ---
    cols_to_drop = ["Data_Quality_Level", "AERONET_Site_Name", "Last_Date_Processed"]
    cols_to_keep = [col for col in tmy_df.columns if col not in cols_to_drop]
    tmy_df = tmy_df[cols_to_keep]
    
    # --- CONVERT MOTNH FROM STR TO NUMBER ---
    tmy_df["Month"] = tmy_df["Month"].apply(lambda x:MONTH_NAMES_TO_NUMBERS[x.split("-")[1]])
    
    # --- CONVERT ALL VALUES FROM STR TO FLOAT ---
    tmy_df = tmy_df.astype(float)
    
    # --- FILTER DATA ---
    # Drop all columns whose readings are non-sensical
    tmy_df[tmy_df<-500] = np.nan
    tmy_df = tmy_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    
    # --- CONVERT MONTH COLUMN TO STRING ---
    tmy_df["Month"] = tmy_df["Month"].astype(int)
    
    
    # --- COMPUTE HOURLY MEANS/PERCENTILES OF VARIABLE VALUES ---
    if percentile is None:
        tmy_df = tmy_df.groupby(["Month"]).mean(numeric_only=True)
    else:
        tmy_df = tmy_df.groupby(["Month"]).quantile(q = percentile)
        
        
    tmy_df = tmy_df.rename(
    columns={"Ozone(Dobson)": "O3",
             "NO2(Dobson)"  : "NO2",
             "Precipitable_Water(cm)"        : "H2O",
             "440-870_Angstrom_Exponent"     : "440nm-870nm_alpha",
             "380-500_Angstrom_Exponent"     : "380nm-500nm_alpha",
             "440-675_Angstrom_Exponent"     : "440nm-675nm_alpha", 
             "500-870_Angstrom_Exponent"     : "500nm-870nm_alpha",
             "340-440_Angstrom_Exponent"     : "340nm-440nm_alpha",
             "Solar_Zenith_Angle(Degrees)"   : "zen",
             "Optical_Air_Mass"              : "rel_airmass",
             "Sensor_Temperature(Degrees_C)" : "T2m"})
        
        
    if interp:
        # It is almost guranteed that the resulting data will have missing
        # values (ie, NaNs). If interp is set to True, these values are 
        # filled using the average value of their non-NaN neighbours.
        # Must be carefull though. These treats each column of the
        # DataFrame as if it was a continuous array. That is, it has no regard
        # for time continuity or time jumps. 
        tmy_df =\
        tmy_df.apply(lambda x: aux.fill_nans_using_laplace_1D(np.array(x), 3000))



    if full_index:
        # The data present in tmy_df may not always include the full 365
        # days of a year. There may be a couple of days missing. If full
        # index is set to True, the full index is computed, not just the
        # days that are present. The added indices are filled with NaNs.
        
        new_index = []        
        for month in range(1, 13):
            for day in range(1, MONTH_DAYS[month]+1):
                for hour in range(24):
                    new_index.append((month, day, hour))
        
        new_index = pd.MultiIndex.from_tuples(new_index)
        new_tmy_df = pd.DataFrame(index=new_index, columns= tmy_df.columns).astype(float)
        new_tmy_df.index.names = ["Month", "Day", "Hour"]
        
        for month in range(1, 13):
            new_tmy_df.loc[month,:] = np.array(tmy_df.loc[month,:]).astype(float)
            
        tmy_df = new_tmy_df.astype(float)
        
        
        return tmy_df







#%%          EXAMPLES 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from Ambience_Modelling.Site import Site
    from Ambience_Modelling import auxiliary_funcs as aux
    
    # ---LOAD SITE OBJ ---
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    Site_obj = aux.load_obj_with_pickle(path = path)
    
    # --- DEFINE PATHS OF STORED AERONET DATA --- 
    all_pts_path   = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Local Aeronet Database\20120101_20231231_Medellin_all_points.lev20"
    month_avg_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Local Aeronet Database\20120101_20231231_Medellin_avg_months.lev20"
    
    # --- READ STORED AERONET DATA ---
    all_pts_aeronet_df   = read_lev20_AOD_aeronet_file(all_pts_path)
    month_avg_aeronet_df = read_lev20_AOD_aeronet_file(month_avg_path)
    
    # --- COMPUTE TMY-LIKE DATAFRAME FOR AERONET DATA ---
    all_pts_tmy_df      = compute_tmy_df_from_all_points_lev20_AOD_aeronet_df(all_pts_aeronet_df,  utc_hour=-5, interp=True)
    month_avg_tmy_df    = compute_tmy_df_from_month_avg_lev20_AOD_aeronet_df(month_avg_aeronet_df, utc_hour=-5, interp=True)
    
    
#%%     PLOT MEAN, 25-TH, 50-TH, 75-TH PERCENTILES OF SITE_OBJ DATA AND AERONET DATA FOR COMPARISON
    num_pts = 50   
    hms_float_eval = np.linspace(6.5, 17.5, num_pts)
    
    data_of_site_list = []
    data_of_all_pts_aeronet_df_list = []
    data_of_month_avg_aeronet_df_list = []
    
    for year, month, day in Site_obj.time_data.keys():
        
        col = "AOD_500nm"
        data_of_site  = Site_obj.time_interpolate_variable(col = col, 
                                                           year = year, 
                                                           month = month,
                                                           day   = day,
                                                           new_hms_float = hms_float_eval)
        
        col = "AOD_500nm"
        data_of_all_pts_aeronet_df =\
        interp1d(x = range(24), y = all_pts_tmy_df.loc[(month, day), col])(hms_float_eval)
    
        
        data_of_month_avg_aeronet_df =\
        interp1d(x = range(24), y = month_avg_tmy_df.loc[(month, day), col])(hms_float_eval)
        
    
        data_of_site_list.append(data_of_site)
        data_of_all_pts_aeronet_df_list.append(data_of_all_pts_aeronet_df)
        data_of_month_avg_aeronet_df_list.append(data_of_month_avg_aeronet_df)
        
    data_of_site_list = np.stack(data_of_site_list, axis = 0)     
    data_of_all_pts_aeronet_df_list = np.stack(data_of_all_pts_aeronet_df_list, axis = 0)   
    data_of_month_avg_aeronet_df_list = np.stack(data_of_month_avg_aeronet_df_list, axis = 0)      
    
    
    p25_site  = np.percentile(data_of_site_list, q=0.25, axis=0)
    p50_site  = np.percentile(data_of_site_list, q=0.50, axis=0)
    p75_site  = np.percentile(data_of_site_list, q=0.75, axis=0)
    mean_site = data_of_site_list.mean(axis=0)
    
    p25_all_pts_aeronet  = np.percentile(data_of_all_pts_aeronet_df_list, q=0.25, axis=0)
    p50_all_pts_aeronet  = np.percentile(data_of_all_pts_aeronet_df_list, q=0.50, axis=0)
    p75_all_pts_aeronet  = np.percentile(data_of_all_pts_aeronet_df_list, q=0.75, axis=0)
    mean_all_pts_aeronet = data_of_all_pts_aeronet_df_list.mean(axis=0)
    
    p25_month_avg_aeronet  = np.percentile(data_of_month_avg_aeronet_df_list, q=0.25, axis=0)
    p50_month_avg_aeronet  = np.percentile(data_of_month_avg_aeronet_df_list, q=0.50, axis=0)
    p75_month_avg_aeronet  = np.percentile(data_of_month_avg_aeronet_df_list, q=0.75, axis=0)
    mean_month_avg_aeronet = data_of_month_avg_aeronet_df_list.mean(axis=0)
        
    

    fig = plt.figure(figsize = (16, 12))
    plt.plot(hms_float_eval, p25_site,              color="red",   linestyle="-.", label="TMY-like satelite: p25 ")
    plt.plot(hms_float_eval, p25_all_pts_aeronet,   color="green", linestyle="-.", label="TMY-like All points AERONET: p25 ")
    plt.plot(hms_float_eval, p25_month_avg_aeronet, color="blue",  linestyle="-.", label="TMY-like Avg-month AERONET: p25 ")
    
    plt.plot(hms_float_eval, p50_site,              color="red",   linestyle="-", label="TMY-like satelite: p50 ")
    plt.plot(hms_float_eval, p50_all_pts_aeronet,   color="green", linestyle="-", label="TMY-like All points AERONET: p50 ")
    plt.plot(hms_float_eval, p50_month_avg_aeronet, color="blue",  linestyle="-", label="TMY-like Avg-month AERONET: p50 ")
    
    plt.plot(hms_float_eval, p75_site,              color="red",   linestyle="--", label="TMY-like satelite: p75 ")
    plt.plot(hms_float_eval, p75_all_pts_aeronet,   color="green", linestyle="--", label="TMY-like All points AERONET: p75 ")
    plt.plot(hms_float_eval, p75_month_avg_aeronet, color="blue",  linestyle="--", label="TMY-like Avg-month AERONET: p75 ")
    
    plt.plot(hms_float_eval, mean_site,              color="red",   linestyle=":", label="TMY-like satelite: avg ")
    plt.plot(hms_float_eval, mean_all_pts_aeronet,   color="green", linestyle=":", label="TMY-like All points AERONET: avg ")
    plt.plot(hms_float_eval, mean_month_avg_aeronet, color="blue",  linestyle=":", label="TMY-like Avg-month AERONET: avg ")
    
    title = "Aerosol Optical Depth"
    plt.title(f"{title} | From {2022, 1, 1} - {2022, 12, 31}")
    plt.xlabel("Hour [24-h format]")
    plt.ylabel(col)
    plt.legend()
    plt.grid()
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    

            
            
        
        
        
    
