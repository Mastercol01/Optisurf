#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
computation and manipulation of the time for later simulations. It also
includes fucntions and methods for computing sunset and sunrise times.
"""

#%%                               IMPORTATION OF LIBRARIES 

import calendar
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from suntimes import SunTimes


#%%                              DEFINITION OF CONSTANTS

# Number of days each month posesses.
MONTH_DAYS =\
{0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

# Dict for timestamp to float conversion.
TIMESTAMP_HMS_TO_FLOAT_DICT = { "d" : [1/24, 1/1440, 1/86400], 
                                "h" : [1, 1/60, 1/3600], 
                                "m" : [60, 1, 1/60], 
                                "s" : [3600, 60, 1] }



#%%                           DEFINITION OF FUNCTIONS 

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



def process_month_tuple(months):
    
    """
    Verifies a tuple of months to see whether it is in the correct order and if
    it satisfies the appropiate bounds. If not, it modifies it to be in the 
    correct order and have the appropiate bounds.
    
    Parameters
    ----------
    months : 2-tuple of int
        Tuple of months.
        
    Returns
    -------
    (month0, month1) : 2-tuple of ints
        Tuple of months in the correct order and properly bounded.
    
    """
    
    month0 = min(max(1, months[0]), 12)
    month1 = min(max(1, months[1]), 12)
    month0, month1 = min(month0, month1), max(month0, month1)
    return (month0, month1)



def process_day_tuple(days, month):
    
    """
    Verifies a tuple of days to see whether it is in the correct order and if
    it satisfies the appropiate bounds. If not, it modifies it to be in the 
    correct order and have the appropiate bounds.
    
    Parameters
    ----------
    days : 2-tuple of int
        Tuple of days.
        
    month : int
        Month to which said days in 'days' belong to.
        
    Returns
    -------
    (day0, day1) : 2-tuple of ints
        Tuple of months in the correct order and properly bounded.
    
    """
    
    day0 = min(max(1, days[0]), MONTH_DAYS[month])
    day1 = min(max(1, days[1]), MONTH_DAYS[month])
    day0, day1 = min(day0, day1), max(day0, day1)
    return (day0, day1)



def process_hour_tuple(hours):
    
    """
    Verifies a tuple of hours to see whether it is in the correct order and if
    it satisfies the appropiate bounds. If not, it modifies it to be in the 
    correct order and have the appropiate bounds.
    
    Parameters
    ----------
    days : 2-tuple of int
        Tuple of days.
        
    Returns
    -------
    (hour0, hour1) : 2-tuple of ints
        Tuple of months in the correct order and properly bounded.
    
    """
    
    hour0 = min(max(0, hours[0]), 23)
    hour1 = min(max(0, hours[1]), 23)
    hour0, hour1 = min(hour0, hour1), max(hour0, hour1)
    return hour0, hour1
    


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



def compute_sunrise_sunset(lon, lat, alt, tz_name, year, month, day):
    
    """
    Compute the sunrise and sunset times of a site using the SunTimes 
    module.
    
    Parameters
    -----------
    lon : float
        Site's longitude in degrees. Must be a number between -180 and 180.
    
    lat : float
        Site's latitude in degrees. Must be a number between -90 and 90.
        
    alt : float
        Site's elevation above sea level in meters. Must be non-negative.
        
    tz_name : str
        Time zone of site, as given by a string accepted by pandas.
        
    year : int
        Year for which the calculation is to be performed. 
        
    month : int 
        Month of the year for which the calculation is to be performed.
        Must be a number between 1 and 12.
        
    day : int
        Day of the month for which the calculation is to be performed.
        Must be a number between 1 and 28, 30 or 31, depending on the month.
        

    Returns
    -------
    sunrise : pandas.Timestamp object, "PD" or "PN"
        Normally, it is equal to the Timestamp of the sunrise time. However
        at very high latitudes there may be times in which there is no sunrise,
        as the place is experiencing either "Polar Day" ("PD") or "Polar Night"
        ("PN").
        
    sunset : pandas.Timestamp object, "PD" or "PN"
        Normally, it is equal to the Timestamp of the sunset time. However
        at very high latitudes there may be times in which there is no sunset,
        as the place is experiencing either "Polar Day" ("PD") or "Polar Night"
        ("PN").
        
    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    2) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
        
    """
    
    datetime_obj =  datetime(year, month, day)    
    suntimes_obj = SunTimes(longitude=lon, latitude=lat, altitude=alt)
    
    sunrise = suntimes_obj.risewhere(datetime_obj, tz_name)
    sunset  = suntimes_obj.setwhere(datetime_obj, tz_name)
    
    if not isinstance(sunrise, str):
        sunrise = pd.Timestamp(sunrise)
    
    if not isinstance(sunset, str):
        sunset  = pd.Timestamp(sunset)
        
    
    return sunrise, sunset



def compute_min_sunrise_max_sunset(lon, lat, alt, tz_name, years, months, days):
    
    """
    Compute the earliest sunrise time and latest sunset time of a site
    in the period of time specified, using the SunTimes module.
    
    Parameters
    ----------
    lon : float
        Site's longitude in degrees. Must be a number between -180 and 180.
    
    lat : float
        Site's latitude in degrees. Must be a number between -90 and 90.
        
    alt : float
        Site's elevation above sea level in meters. Must be non-negative.
        
    tz_name : str
        Time zone of site, as given by a string accepted by pandas.
        
    year : int or tuple of int
        If tuple, first and last elements define the initial and final year
        of the time interval to be computed. If int it becomes a tuple
        of (year, year).
        
    month : int 
        If tuple, first and last elements define the initial and final month
        of the time interval to be computed. If int it becomes a tuple
        of (month, month).
        
    day : int
        If tuple, first and last elements define the initial and final day
        of the time interval to be computed. If int it becomes a tuple
        of (day, day).
        

    Returns
    -------
    sunrise : pandas.Timestamp object
        Timestamp of the earliest sunrise time in the defined interval.
        
    sunset : pandas.Timestamp object
        Timestamp of the latest sunset time in the defined interval.
        
    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    2) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
        
    """
    
    if isinstance(years, int): years = (years, years)
        
    if isinstance(months, int): months = (months, months)
        
    if isinstance(days, int): days = (days, days)  
        
    min_sunrise_h = np.inf
    max_sunset_h = -np.inf
    
    year0, year1 = min(years), max(years)
    month0, month1 = process_month_tuple(months)


    # We basically loop over each of the days in the day-range specified,
    # compute each of the sunrise and sunset times, and store the minimum and 
    # maximum values.
    for year_ in range(year0, year1+1):
        for month_ in range(month0, month1+1):
            
            day0, day1 = process_day_tuple(days, month_)
            
            for day_ in range(day0, day1):
                
                sunrise, sunset =\
                compute_sunrise_sunset(lon, lat, alt, tz_name, 
                                       year_, month_, day_)
                
                
                # In the case of a ploar day or a ploar night we through an
                # error, seeing as sunrise and sunset are not technically
                # defined for such cases.
                if isinstance(sunrise, str) or isinstance(sunset, str):
                    message = "Function is not defined for polar days/nights"
                    raise Exception(message)


                sunrise_h = timestamp_hms_to_float(sunrise, unit = "h")
                sunset_h  = timestamp_hms_to_float(sunset, unit = "h")
                
                
                
                
                if(sunrise_h < min_sunrise_h):
                    min_sunrise_h = sunrise_h
                    min_sunrise = sunrise
                    
                if(sunset_h > max_sunset_h):
                    max_sunset_h = sunset_h
                    max_sunset = sunset    
                    
                
                
    return min_sunrise, max_sunset   
              



def date_range(start_time, end_time, min_hms, max_hms, time_interval, UTC):
    
    """
    Creates a pandas.Series using pandas.date_range, given the input parameters.
    For the application in question, working directly with pandas.date_range 
    would have required a lot of pos-processing each time given. Hence this
    wrapper is meant to facilitate the process.
    
    Parameters
    ----------
    
    start_time : str
        This is the time from which we want to start the time series.
        That is, it is the first time period we consider. It must be a string 
        of the form "yyyy-mm-dd hh:mm:ss". 
        
    end_time : str
        This is the time we want the time series to end on. That is, it is
        the last time period we consider. It must be a string of the form
        "yyyy-mm-dd hh:mm:ss".  
        
    min_hms : str or None
        If str, this is is the minimum hms (hour-minute-second) value for a 
        Timestamp that is allowed within the time series. Any Timestamps whose
        hms values are strictly below this threshold, are removed from the time
        series. It must be a string of the form "hh:mm:ss". If None, this 
        condition gets ignored.
        
    max_hms : str or None
        If str, this is is the maximum hms (hour-minute-second) value for a 
        Timestamp that is allowed within the time series. Any Timestamps whose
        hms values are strictly above this threshold, are removed from the time 
        series. It must be a string of the form "hh:mm:ss". If None, this 
        condition gets ignored.
    
    time_interval : str
        This is the time we want to have in-between time samples. That
        is, how far apart each time sample is from one another. It must
        be in the following format: number-unit. Example: "10-s" means ten
        seconds, "20-min" means 20 minutes, "2-h" means two hours and 
        "5-d" means five days. Supported units are: "s", "min", "h" and "d".
        
    UTC : str
        Standard time zone we wish to consider. If positive, it should
        be given in the format: "hh:mm:ss" and if it is negative it should 
        be given as "-hh:mm:ss".
        
        
    Returns
    -------
    time_series : pandas.Series of pandas.Timestamp objs
        Time series from 'start_time' to 'end_time' of time period 
        equal to 'time_interval', with all Timestamps whose hms values 
        lie outside the closed interval ['min_hms', 'max_hms'], having
        been removed.
        
        
        
    Notes
    -----
    1) "yyyy-mm-dd hh:mm:ss" format means year-month-day hour:minute:second
       format. Eg.: "2020-10-05 14:30:25" means 5th of October of the year 2020,
       at 14 hours, 30 minutes and 25 seconds.
       
    2) "hh:mm:ss" format means hour:minute:second format. Eg.: "14:30:25"
       means 14 hours, 30 minutes and 25 seconds.
    
    
    """
    
    # We split the start/end_time into its component parts.
    start_date, start_hms = start_time.split(" ")
    end_date, end_hms = start_time.split(" ")
    
    # We compute the pandas acceptable time zone string.
    utc_hour = int(UTC.split(":")[0])
    tz_name = utc_hour_to_tz_name(utc_hour)
    
    # We compute the total time difference between the start_time and 
    # end_time, in days.
    time_diff = pd.Timestamp(end_time) - pd.Timestamp(start_time)
    time_diff_in_days = time_diff.days + time_diff.seconds/86400
    
    
    # Convert time of interval into unit days and compute the required 
    # number of periods for pandas.date_range.
    dt, dt_type = time_interval.split('-')
    dt = float(dt)
    
    if(dt_type=='s' or dt_type=='S'):
        periods = time_diff_in_days*(86400/dt)
        #dt_h = dt/3600
        
    elif(dt_type=='min' or dt_type=='T'):
        periods = time_diff_in_days*(1440/dt)
        #dt_h = dt/60
        
    elif(dt_type=='h'or dt_type=='H'):
        periods = time_diff_in_days*(24/dt)
        #dt_h = dt
        
    elif(dt_type=='d' or dt_type=='D'):
        periods = time_diff_in_days*(1/dt)
        #dt_h = 24*dt
        
        

    # We overshoot the period just a little bit. Anything that exceeds the 
    # the end_time by more than is required, gets removed.
    periods = np.ceil(periods) + 1
    freq = str(dt) + dt_type
    
    # We create a pandas.Series, starting in start_time, up to 'periods'
    # periods using the frequency specified. 
    time_series = pd.Series(
    pd.date_range(start_time, periods = periods, freq = freq, tz = tz_name))     
    
    
    
    # In the case that min_hms and max_hms are both simultaneously not None, we
    # prepare to drop all indices that donot line within the closed interval
    # [min_hms, max_hms].
    if (min_hms is not None) or (max_hms is not None):
        
        # We create a list of indices to be dropped. 
        time_series_index_drop = []
    
        # We convert the min_hms into units its equivalent in hours.
        min_hms_float = timestamp_hms_to_float(pd.Timestamp(min_hms), unit="h")
    
        # We convert the max_hms into units its equivalent in hours.
        max_hms_float = timestamp_hms_to_float(pd.Timestamp(max_hms), unit="h")
    
        # We loop over each entry of the time series DataFrame and check
        # whether the hms of each entry is within the start_hms and the end_hms
        # range. If not, it is added to the list of indices to drop.
        
        for i, timestamp in enumerate(time_series): 
            hms_to_check = timestamp_hms_to_float(timestamp, unit="h")
            
            if min_hms is not None:
                logic1 = hms_to_check < min_hms_float
            else:
                logic1 = False
                
            if max_hms is not None:
                # logic2 = (hms_to_check > max_hms_float) and\
                #           (hms_to_check - max_hms_float >= 0.999*dt_h)
                logic2 = hms_to_check > max_hms_float
            else:
                logic2 = False
            
            
            if logic1 or logic2:
                time_series_index_drop.append(i)
                     
                
        # We drop the indices appended.
        time_series = time_series.drop(time_series_index_drop)
        time_series = time_series.reset_index(drop=True)
        
        
    # We assume that the maximum timestamp of the time series is either 
    # equal to or greater than the end_time timestamp. As such we remove all
    # timestamps that are bigger than requiered. 
    
    end_time_timestamp = pd.Timestamp(end_time, tz = tz_name)
    
    if any(time_series == end_time_timestamp):
        time_series = time_series[time_series <= end_time_timestamp]
        
    else:
        idx = len(time_series[time_series < end_time_timestamp])
        time_series = time_series.iloc[:idx+1]

    # We save some relevant variables as attributes.    
    return time_series





def geo_date_range(lat, lon, alt,
                   start_time, end_time, 
                   min_hms, max_hms, 
                   time_interval, UTC, 
                   skip_polar_nights = True,
                   time_delta = "5-min"):
    
    """
    Creates a pandas.Series using 'date_range' function, for each day of the 
    specified interval, given the input parameters, and returns all the 
    computed time series as a dictionary. This function is similar to
    the 'date_range' function, but this fucntion includes a new functionality,
    which is the ability to compute the values of sunrsie and 
    sunset, and subsequently use them as the values for min_hms and max_hms.
    Hence why each time_series needs to be stored independently.
    
    Parameters
    ----------
    
    start_time : str
        This is the time from which we want to start to compute the time data.
        That is, it is the first time period we consider. It must be a string 
        of the form "yyyy-mm-dd hh:mm:ss". 
        
    end_time : str
        This is the time we want the time data to end on. That is, it is
        the last time period we consider. It must be a string of the form
        "yyyy-mm-dd hh:mm:ss".  
        
    min_hms : str, None or "sunrise"
        If str, this is is the minimum hms (hour-minute-second) value for a 
        Timestamp that is allowed within the each day's time series. Any 
        Timestamps whose hms values are strictly below this threshold, are
        removed from said time series. It must be a string of the form 
        "hh:mm:ss". If None, this condition gets ignored. If "sunrise", the
        sunrise time for the location specified is computed and used as the 
        value for min_hms.
        
    max_hms : str, None or "sunset"
        If str, this is is the maximum hms (hour-minute-second) value for a 
        Timestamp that is allowed within the each day's time series. Any 
        Timestamps whose hms values are strictly above this threshold, are
        removed from said time series. It must be a string of the form 
        "hh:mm:ss". If None, this condition gets ignored. If "sunset", the
        sunset time for the location specified is computed and used as the 
        value for max_hms.
    
    time_interval : str
        This is the time we want to have in-between time samples of each day.
        That is, how far apart each time sample is from one another. It must
        be in the following format: number-unit. Example: "10-s" means ten
        seconds, "20-min" means 20 minutes, "2-h" means two hours and 
        "5-d" means five days. Supported units are: "s", "min", "h" and "d".
        
    UTC : str
        Standard time zone we wish to consider. If positive, it should
        be given in the format: "+hh:mm:ss" and if it is negative it should 
        be given as "-hh:mm:ss".
        
    skip_polar_nights : bool
        If True, errase all NaN values from 'time_data' variable, that 
        correspond to polar nights. If False, these NaN values are not
        removed. This applies only when min_hms == "sunrise" or 
        max_hms == "sunset". Default is True.
        
    time_delta : str
        This is how much time of leeway we want to give the code, so that it
        is able to catch the actual sunrise and sunset events within
        each day's time series. It must be in the following format: number-unit. 
        Example: "10-s" means ten seconds, "20-min" means 20 minutes, "1-h" 
        means one hour. Supported units are: "s", "min" and "h". This applies 
        only when min_hms == "sunrise" or max_hms == "sunset". Default is True.
        Default is "5-min".
        
        
    Returns
    -------
    time_data : dict of pandas.Series of pandas.Timestamp objs
        Dictionary whose keys are (year, month, day), where the range
        of year, month and day are defined by the start_time and end_time
        variables. Each key contains a time series of period 
        equal to 'time_interval', with all Timestamps whose hms values 
        lie outside the closed interval ['min_hms', 'max_hms'], having
        been removed.
        
        
        
    Notes
    -----
    1) "yyyy-mm-dd hh:mm:ss" format means year-month-day hour:minute:second
       format. Eg.: "2020-10-05 14:30:25" means 5th of October of the year 2020,
       at 14 hours, 30 minutes and 25 seconds.
       
    2) "hh:mm:ss" format means hour:minute:second format. Eg.: "14:30:25"
       means 14 hours, 30 minutes and 25 seconds.
       
       
    Warnings
    --------
    1) "WARNING: Using a 'time_delta' of less than 5 minutes runs the risk of 
        not catching the actual sunset and sunrise events within each day's
        time series."
        
    2) "WARNING: POLAR NIGHTS DETECTED" (Happens if skip_polar_nights is False)
    
    3) "WARNING: POLAR NIGHTS DETECTED. Polar Nights will be removed from
       the time_data dict. Hence, some key-value pairs may be missing for
       certain dates." (Happens if skip_polar_nights is True)
       
    4) "WARNING: Data does not increase monotonically from day to day. This may
        happen because of 3 reasons. Either: (1) 'time_interval' variable is 
        too big, (2) 'time_delta' variable is too big or (3) the site of 
        computation lies near or inside a polar circle."
    
    """
    
    # We compute the pandas acceptable time zone string.
    utc_hour = int(UTC.split(":")[0])
    tz_name = utc_hour_to_tz_name(utc_hour)
    
    if utc_hour >= 0:
        sep_symbol = "+"
    else: 
        sep_symbol = "-"
    
    
    
    
    #       ----------- DESCRIPTION -------- 
    
    # As sunrise and sunset times change with time of the year, we need to
    # compute an independent time series for each simulated day.
    # We store all of these different time series using a dictionary.
    
    
    # ----------- COMPUTE KEYS OF DICTIONARY -------- 
    
    # We compute the keys for storing each time series. Each
    # key is made up of 3 levels: Year, Month, Day. As such,
    # Year = 2023, Month = 3, Day = 8 will contain the specific time series 
    # for the 8th of March of the year 2023.
    # For computing the keys as discussed above, we need a list of
    # (year, month, day) tuples for all the dates to evaluate.
    
    start_date = start_time.split(" ")[0]
    end_date   = end_time.split(" ")[0]
    
    start_year, start_month, start_day =\
    [int(i) for i in start_date.split("-")]
        
    end_year, end_month, end_day =\
    [int(i) for i in end_date.split("-")]
     
    
    day_list, month_list, year_list = [], [], []
    day, month, year = start_day, start_month, start_year
    
    
    while True:
        
        # This helps us keep track of leap years.
        leap_year_int = int(calendar.isleap(year) and month == 2)
        
        if(day > MONTH_DAYS[month] + leap_year_int):
            day = 1
            month += 1
            
        if(month > 12):
            month = 1
            year += 1
            
        year_list.append(year)
        month_list.append(month)
        day_list.append(day)
    
    
        if([day, month, year]==[end_day, end_month, end_year]): 
            break
        
        day += 1
        
        
    
    # We initialize the time delta as a pandas.Timedelta obj.
    TIMEDELTA = pd.Timedelta(float(time_delta.split("-")[0]), 
                             time_delta.split("-")[1])
    
    # We warn the that they meay be choosing very low value for 'time_delta'.
    if TIMEDELTA < pd.Timedelta(5, "min") and (min_hms == "sunrise" or max_hms == "sunset"):
        message = "WARNING: Using a 'time_delta' of less than 5 minutes"
        message = f"{message} runs the risk of not catching the actual sunset"
        message = f"{message} and sunrise events within each day's time series."
        warnings.warn(message)
        
        
        
    # --- COMPUTE INDIVIDUAL TIME SERIES AND STORE THEM ----
            
    # We compute a time series for each day specified in the 
    # initialization. As such, if "sunrise" and/or "sunset" are 
    # specified, we ought to compute the sunrise and/or sunset times 
    # for each day in order to create the specific time series for
    # that particular day.
    
    
    time_data = {}
    polar_nights_warning = False
    
    for year, month, day in list(zip(year_list, month_list, day_list)): 
        
        fill_with_NaN = False
        day_str, month_str, year_str = str(day), str(month), str(year)
         
        if(day < 10): day_str = f"0{day_str}"
        if(month < 10): month_str = f"0{month_str}" 
            
        date = "-".join([year_str, month_str, day_str])
        min_hms_, max_hms_ = min_hms, max_hms
        
        
        
        # --------- WE COMPUTE THE SUNRISE AND SUNSET TIMES IF SPECIFIED ----------
        
        if min_hms == "sunrise" or max_hms == "sunset" :
            
            sunrise, sunset =\
            compute_sunrise_sunset(lon = lon,  lat = lat,  alt = alt, 
                                   tz_name = tz_name, year = year, 
                                   month = month,  day = day)
            
            
            # At high latitudes (i.e, near the poles), it is perfectly 
            # possible for the sun to not set for days or months at a time, 
            # as well as for it to not rise for days and months at a time. 
            # These are called "Polar Days" ("PD") and "Polar Nights" 
            # ("PN"), respectively. These possibilities need to be taken 
            # into account.
            
            local_day_start = pd.Timestamp(date, tz = tz_name) 
            local_day_end   = local_day_start + pd.Timedelta(1, "day")
            local_day_end  -= pd.Timedelta(1, "ms")
           
            
            if min_hms == "sunrise" and sunrise == "PD":
                sunrise    = local_day_start
                min_hms_ = str(sunrise).split(" ")[-1].split(sep_symbol)[0]
                
            elif min_hms == "sunrise" and sunrise == "PN":
                fill_with_NaN = True 
                
            elif min_hms == "sunrise":
                sunrise    = sunrise - TIMEDELTA
                min_hms_   = str(sunrise).split(" ")[-1].split(sep_symbol)[0]
                
                
                
            if max_hms == "sunset" and sunset == "PD":
                sunset   = local_day_end
                max_hms_ = str(sunset).split(" ")[-1].split(sep_symbol)[0]
                
            elif max_hms == "sunset" and sunset == "PN":
                fill_with_NaN = True 
                
            elif max_hms == "sunset":
                sunset   = sunset + TIMEDELTA
                max_hms_ = str(sunset).split(" ")[-1].split(sep_symbol)[0]
                
    
            # The reason why we compute sunrise and sunset (when sunrise
            # and sunset are not equal to "PD" or "PN") as:
                
            #   sunrise = sunrise - TIMEDELTA
            #   sunset  = sunset + TIMEDELTA
            
            # Is because we would like to include the actual sunrise and sunset
            # events within the time series. You see, while the function 
            # 'compute_sunrise_sunset' is fairly accurate, it is not infinitely
            # accurate. In fact, its predicted sunrise and sunset times may vary 
            # by a few minutes when compared to something like NREL's algorithm 
            # for tracking the position of the sun. Why not use that algorithm
            # instead then? Well, because determining sunrise and sunset that 
            # way would be quite painstaking. We'll actually do just that, 
            # but for another application. FOR THIS APPLICATION we only need a
            # good estimate. More specifically, we require that actual sunrise
            # and sunset times lie somewhere within the time samples of the each
            # day's time series. Hence why we substract a 'time_delta' amount of
            # time to the computed sunrise and add a 'time_delta' amount of time
            # to the computed sunset. 
            

                
        # We store each individual time series everything in the time_data
        # dictionary.
        
        # --------- WE COMPUTE THE TIME SERIES FOR EACH DAY ----------
        
        if fill_with_NaN:
            # In the case we have a ploar night, we set that value of the 
            # dictonary to NaN.
            time_data[(year, month, day)] = np.nan
            polar_nights_warning = True 
            

        else: 
            
            # When the sunset of day actually spills over to the next day,
            # we ecounter errors when setting date_range(max_hms = sunset).
            # It actually ends up errasing most of the time series for those 
            # cases. Hence it is better to simply do 
            # date_range(start_time = sunrise, end_time = sunset, min_hms=None,
            # max_hms = None). This will have the same effect as initially
            # desired. For the rest of cases, when "sunrise" or "sunset" are
            # not selected, we return to the default use.
            
            
            if date == start_date: 
                start_time_ = start_time
            
            elif min_hms == "sunrise":
                start_time_    = str(sunrise).split(" ")
                start_time_[1] = start_time_[1].split(sep_symbol)[0]
                start_time_ = " ".join(start_time_)
                
                # This condition is necessary so that we dont have accidentally get
                # loose trail ends sometimes, when end_date = "yyyy-mm-dd 23:59:59"
                # and we still have max_hms = sunset.
                if date != end_date:
                    min_hms_ = None

            else:
                start_time_ = date + " 00:00:00"
                
                
                
            if date == end_date: 
                end_time_ = end_time
                
            elif max_hms == "sunset":
                
                end_time_ = str(sunset).split(" ")
                end_time_[1] = end_time_[1].split(sep_symbol)[0]
                end_time_ = " ".join(end_time_)
                max_hms_ = None
                
            else:
                end_time_ = date + " 23:59:59"
                

            
            local_day_time_series =\
            date_range(start_time = start_time_,
                       end_time = end_time_,
                       min_hms = min_hms_,
                       max_hms = max_hms_,
                       time_interval = time_interval,
                       UTC = UTC)
            
            
            # ----- We store the data ----
            time_data[(year, month, day)] = local_day_time_series
            
            
            
    #              --- DEALING WITH POLAR NIGHTS ---
    
    # We warn the user if any Polar Nights are detected, and errase them if the
    # user specified so. Althouth, tbh, If the whole day is dark, what energy
    # are you going to produce with solar arrays at that time? Better to writte
    # them off.
    time_data_without_NaNs =\
    {key:val for key,val in time_data.items() if not isinstance(val, float)}
    
    if skip_polar_nights and polar_nights_warning:
        message = "WARNING: POLAR NIGHTS DETECTED"
        message = f"{message}. Polar Nights will be removed from"
        message = f"{message} the time_data dict. Hence, some key-value pairs"
        message = f"{message} may be missing for certain dates."
        warnings.warn(message)
        time_data = time_data_without_NaNs.copy()
        
    elif not skip_polar_nights and polar_nights_warning:
        message = "WARNING: POLAR NIGHTS DETECTED"
        warnings.warn(message)
        
        
        
    # --- THE POSSIBILITY OF NON MONOTONIC-INCREASING TIME SERIES ---
    
    # At higher and higher latitudes it happens that, for certain times
    # of the year, the sunset time of a place, may not lie within the 
    # same day that sunset corresponds to. An example is Reykjavík, 
    # Iceland, whose sunrise and sunset times for the 24th of June of
    # 2023 are (around) "2023-06-24 02:56:00" and "2023-06-25 00:04:00"
    # (in local time: UTC = "00:00:00"), respectively. This is not a 
    # problem and can actually be taken account by the code for most 
    # places. However, the closer we get to a porlar circle, without 
    # entering said polar circle, the more probable it is that we may 
    # get the case that the sunset corresponding to one day ocurrs after 
    # the sunrise of the next day because of the +/- TIMEDELTA procedure 
    # explained above. So, you could theoretically have the case where, 
    # for example, the sunset of a place on the 24th of June of 2023 is
    # "2023-06-25 00:57:00", while the sunrise of that same place on
    # the 25th of June of 2023 is "2023-06-25 00:54:00". It is
    # completely possible, though rare. Indeed, this error should only 
    # happen at very high lattitudes, during vey short periods of the 
    # year. That is, for a couple of weeks at the very most, during the
    # whole year for very high latitudes. The code will make sure to 
    # warn the user about this error when it happens.
    
    # Finally, take Note that this error may also be due to the fact 
    # that the 'time_interval' selected is too large or 'time_delta'
    # is too large rather than being the result from being at very 
    # high latitude. In this cases, try to select a smaller time interval.
    
        
    concatd_time_series_without_NaNs =\
    pd.concat([series for series in time_data_without_NaNs.values()], ignore_index=True)
    
    if not concatd_time_series_without_NaNs.is_monotonic_increasing:
        message = "WARNING: Data does not increase monotonically from day to day"
        message = f"{message}. This may happen because of 3 reasons"
        message = f"{message}. Either: (1) 'time_interval' variable is too big"
        message = f"{message}, (2) 'time_delta' variable is too big or"
        message = f"{message}  (3) the site of computation lies near or inside"
        message = f"{message} a polar circle."
        
        
        
    return time_data



#%%                      EXAMPLES


if __name__ == '__main__':
    
    
    # Example of date_range where min_hms and max_hms are None and
    # the time series ends exactly on the specified value.
    date_range_1 = date_range(start_time = "2023-01-01 06:00:00" , 
                              end_time   = "2024-05-15 18:30:00" , 
                              min_hms    = None, 
                              max_hms    = None,
                              time_interval = "10-min", 
                              UTC = "-05:00:00")
    
    # Example of date_range where min_hms is not None and max_hms is None, and
    # the time series ends exactly on the specified value.
    date_range_2 = date_range(start_time = "2023-01-01 06:00:00" , 
                              end_time   = "2024-05-15 18:30:00" , 
                              min_hms    = "05:00:00", 
                              max_hms    = None,
                              time_interval = "10-min", 
                              UTC = "-05:00:00")    
    
    # Example of date_range where min_hms is None and max_hms is not None, and
    # the time series ends exactly on the specified value.
    date_range_3 = date_range(start_time = "2023-01-01 06:00:00" , 
                              end_time   = "2024-05-15 18:30:00" , 
                              min_hms    = None, 
                              max_hms    = "18:30:00",
                              time_interval = "10-min", 
                              UTC = "-05:00:00")
    
    # Example of date_range where min_hms is not None and max_hms is not None,
    # and the time series does not end exactly on the specified value, but 
    # exactly on the next possible value.
    date_range_4 = date_range(start_time = "2023-01-01 06:07:00" , 
                              end_time   = "2024-05-15 17:54:00" , 
                              min_hms    = "06:30:00", 
                              max_hms    = "19:00:00",
                              time_interval = "37-min", 
                              UTC = "-05:00:00")
    
    
    # Example where we show some of the less obvious differences between 
    # date_range and geo_date_range. The time series for date_range is computed
    # as a continuum, while the time series for geo_date_range is computed 
    # individually for each day. This may sometimes create differences between
    # both time series (mostly when the start_time, end_time and time_interval
    # are really uneven/arbitray), as while date_range has to mantain continuity
    # of time-spacing, globally, geo_date_range needs only to mantain it 
    # daily. Hence, the difference. However, for most applications, and when 
    # using time_intervals which are not as uneven as, say, 37min, we can say
    # that the differences are very insignificant in practice.
    
    # Finally, not that we could have used any values for the lon, lat, alt
    # variables here and nothing about the result, in this particular case, 
    # would've changed
    geo_date_range_1 = geo_date_range(lon = -75.590553, 
                                      lat = 6.230833, 
                                      alt = 1500,
                                      start_time = "2023-01-01 06:07:00", 
                                      end_time   = "2024-05-15 17:54:00", 
                                      min_hms    = "06:30:00", 
                                      max_hms    = "19:00:00", 
                                      time_interval = "37-min",
                                      UTC = "-05:00:00")
    
    
    # Example of geo_date_range using "sunrise" and "sunset" for Medellín, Colombia.
    geo_date_range_2 = geo_date_range(lon = -75.590553, 
                                      lat = 6.230833, 
                                      alt = 1475,
                                      start_time = "2023-01-01 00:00:00", 
                                      end_time   = "2023-12-31 23:59:59", 
                                      min_hms    = "sunrise", 
                                      max_hms    = "sunset", 
                                        time_interval = "5-min",
                                        UTC = "-05:00:00",
                                        skip_polar_nights = True,
                                        time_delta = "5-min")
    
    
    # Example of geo_date_range using "sunrise" and "sunset" for Reikiavik, Iceland.
    geo_date_range_3 = geo_date_range(lon = -21.827774, 
                                      lat = 64.128288, 
                                      alt = 0,
                                      start_time = "2023-01-01 00:00:00", 
                                      end_time   = "2023-12-31 23:59:59", 
                                      min_hms    = "sunrise", 
                                      max_hms    = "sunset", 
                                      time_interval = "5-min",
                                      UTC = "00:00:00",
                                      skip_polar_nights = True,
                                      time_delta = "5-min")
    
    
    # Example of geo_date_range using "sunrise" and "sunset" for Svalbard, Norway.
    # (We get Polar Nights Warning).
    geo_date_range_4 = geo_date_range(lon = 15.86426, 
                                      lat = 78.15706, 
                                      alt = 462,
                                      start_time = "2023-01-01 00:00:00", 
                                      end_time   = "2023-12-31 23:59:59", 
                                      min_hms    = "sunrise", 
                                      max_hms    = "sunset", 
                                      time_interval = "5-min",
                                      UTC = "01:00:00",
                                      skip_polar_nights = False,
                                      time_delta = "5-min")

        
    

    
   

    
    
    
    
    
    
    
  

    
    
    
    
    