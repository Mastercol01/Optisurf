import pvlib as pv
import pandas as pd
import numpy as np
import datetime 
import scipy as sc




class Site:
    
    def __init__(self, name='Magangué', lat=9.25055, long=-74.7661, alt=19, UTC='-05:00:00' ):
        self.name=name
        self.lat=lat
        self.long=long
        self.alt=alt
        self.utc=UTC
    
        tmy_data=pv.iotools.get_pvgis_tmy(lat=self.lat,
                                       lon=self.long, 
                                       startyear=2005, 
                                       endyear=2015, 
                                       outputformat='csv' )[0]
        
        
        utc_hour=int((self.utc).split(':')[0][0:3])
        

        
        tmy_data=tmy_data.reindex(index=np.roll(tmy_data.index, utc_hour))
        
        MONTH_DAYS={0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 
                    8:31, 9:30, 10:31, 11:30, 12:31}
        
        new_indices=[]
        
        for month in range(1,13):
            for day in range(1,MONTH_DAYS[month]+1):
                for hour in range(0,24):
                    new_indices.append((month, day, hour))
                
        tmy_data['TMY(m,d,h)']=new_indices
        tmy_data=tmy_data.set_index('TMY(m,d,h)')

 
        self.tmy_data=tmy_data
        
        
    def get_tmy_units(attribute='Gb(n)'):
           UNITS={
           'T2m':'[°C]',
           'RH':'[%]',
           'G(h)':'[W/m^2]',
           'Gb(n)':'[W/m^2]',
           'Gd(h)':'[W/m^2]',
           'IR(h)':'[W/m^2]',
           'WS10m':'[m/s]',
           'WD10m':'[°]',
           'SP':'[Pa]'
           }
           return UNITS[attribute]
    
        
        
    def get_tmy_values(self, attribute='Gb(n)', month=7, day=20, hour=(0,23)):
        
        MONTH_DAYS={0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 
                    8:31, 9:30, 10:31, 11:30, 12:31}
        
        if(type(month)==int):
            start_month=month
            final_month=month
        else:
            start_month, final_month=month
            
        if(type(day)==int):
            start_day=day
            final_day=day
        else:
            start_day, final_day=day                      
            
        if(type(hour)==int):
            start_hour=hour
            final_hour=hour
        else:
            start_hour, final_hour=hour
                        
        index_evaluate=[]
        
        for i in range(start_month,final_month+1):
            for j in range(start_day,final_day+1):
                if(j==MONTH_DAYS[i]+1):
                    break
                for k in range(start_hour, final_hour+1):
                    index_evaluate.append((i, j, k))
            
        return self.tmy_data.loc[index_evaluate, attribute]      
    
    
    def tmy_interpolate(self, Time_object):
        
        time_series=Time_object.time_series.copy()
        length_time_series=Time_object.length
        start_date=str(time_series[0])[0:10]
        
        num_elem_days=0
        
        try:
            while start_date==str(time_series[num_elem_days])[0:10]:
                num_elem_days+=1
        except:
            num_elem_days==1
            
            
        start_hms=Time_object.start_hms
        end_hms=Time_object.end_hms
        
        start_hms_int=[float(i) for i in start_hms.split(':')]
        start_hms_int=int(np.round(sum([a*b for a,b in zip(start_hms_int,[1, 1/60, 1/3600])])))
        
        end_hms_int=[float(i) for i in end_hms.split(':')]
        end_hms_int=int(np.round(sum([a*b for a,b in zip(end_hms_int,[1, 1/60, 1/3600])])))

        hour_range_interp=np.linspace(start_hms_int, end_hms_int, num_elem_days)
        hour_range_orig=np.arange(start_hms_int, end_hms_int+1)

        hour_indeces=[num_elem_days*i for i in range(0, int(length_time_series/num_elem_days)+1)]
        
        COLUMNS_NAMES=['T2m', 'RH', 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)','WS10m', 'WD10m', 'SP']
        
        tmy_interpolated=pd.DataFrame(index=time_series, columns=COLUMNS_NAMES)
        
        for hour in hour_indeces[1:]:
            for name in COLUMNS_NAMES:
                
                orig_y=self.get_tmy_values( attribute=name, 
                                            month=time_series[hour-1].month, 
                                            day=time_series[hour-1].day, 
                                            hour=(start_hms_int, end_hms_int) )
                
                interp_func=sc.interpolate.interp1d(hour_range_orig, orig_y, kind='cubic') 
                
                interp_y=interp_func(hour_range_interp)
                
                tmy_interpolated.loc[hour-num_elem_days:hour, name]=interp_y
                
                T2m_copy=tmy_interpolated['T2m'].copy()
                
                tmy_interpolated[tmy_interpolated<0]=0
                
                tmy_interpolated['T2m']=T2m_copy
            
        self.tmy_interpolated=tmy_interpolated  
        
        
    def get_tmy_interpolated_time_range(self, lower_bound, upper_bound, inclusive=1):
        
        tmy_interpolated=self.tmy_interpolated.copy()
        UTC=self.utc
        
        if(type(lower_bound)==str):
            lower_bound=pd.Timestamp(lower_bound + UTC.replace(":", "")[0:5])
            
        if(type(upper_bound)==str):
            upper_bound=pd.Timestamp(upper_bound + UTC.replace(":", "")[0:5])
            
        if(upper_bound==None and lower_bound!=None):
            
            if(inclusive==1):
                tmy_interpolated2=tmy_interpolated.loc[tmy_interpolated.index>=lower_bound]
            else: 
                tmy_interpolated2=tmy_interpolated.loc[tmy_interpolated.index>lower_bound]
                
        if(upper_bound!=None and lower_bound==None):
            
            if(inclusive==1):
                tmy_interpolated2=tmy_interpolated.loc[tmy_interpolated.index<=upper_bound]
            else: 
                tmy_interpolated2=tmy_interpolated.loc[tmy_interpolated.index<upper_bound]
                
        if(upper_bound!=None and lower_bound!=None):
            
            if(inclusive==1):
                tmy_interpolated2=tmy_interpolated.loc[(tmy_interpolated.index>=lower_bound)
                                              & (tmy_interpolated.index<=upper_bound)]
            else: 
                tmy_interpolated2=tmy_interpolated.loc[(tmy_interpolated.index>lower_bound)
                                              & (tmy_interpolated.index<upper_bound)]
                
        self.tmy_interpolated2=tmy_interpolated2    
    
            
            
            

            
 #%%
 
 
class Time:
    
    def __init__(self, start_date='2021-01-01', end_date='2021-12-31',
                 start_hms='06:00:00', end_hms='18:00:00',
                 time_interval='30-min', UTC='-05:00:00' ):
        
        self.start_time=start_date + '-' + start_hms
        self.end_time=end_date + '-' + end_hms
        self.start_hms=start_hms
        self.end_hms=end_hms
        self.time_interval=time_interval
        self.utc=UTC

        
        start_time=self.start_time
        end_time=self.end_time
        utc_=self.utc

                    
        FMT='%Y-%m-%d'
        time_initial = datetime.datetime.strptime(start_time[0:10], FMT)
        time_final = datetime.datetime.strptime(end_time[0:10], FMT)
        
        time_initial = time_initial.timetuple()
        julian_date_initial = time_initial.tm_yday      #jdate is julian date
        
        time_final = time_final.timetuple()
        julian_date_final = time_final.tm_yday          #julian dates are expressed in days.
                
        year_initial=float(start_time[0:4])  
        
        hour_initial_in_days=(float(start_time[11:13]))/24        #Hours, minutes and seconds are converted to their day equivalents
        minute_initial_in_days=(float(start_time[14:16])/60)/24   #i.e: 1 hour=1/24 day. Years remain unchanged.
        second_initial_in_days=(float(start_time[17:19])/3600)/24
        
        
        year_final=float(end_time[0:4])  
        
        hour_final_in_days=(float(end_time[11:13]))/24
        minute_final_in_days=(float(end_time[14:16])/60)/24
        second_final_in_days=(float(end_time[17:19])/3600)/24
        
        days_initial=julian_date_initial + hour_initial_in_days + minute_initial_in_days + second_initial_in_days
        days_final=julian_date_final + hour_final_in_days + minute_final_in_days + second_final_in_days
        
        start=year_initial
        end=year_final
        
        leap_days=0
        
        while start <= end:
           if start % 4 == 0 and start % 100 != 0:
               leap_days +=1
           if start % 100 == 0 and start % 400 == 0:
               leap_days +=1
           start +=1
           
           
        year_diff_in_days= 365*(year_final-year_initial) + (1+np.sign(leap_days-2))*(leap_days/2 -1)
        
        total_diff_days= year_diff_in_days - days_initial + days_final
        
        total_diff_days=np.round(total_diff_days,5)
        
        
        dt,dt_type=time_interval.split('-')
        dt=float(dt)
        
        if(dt_type=='s' or dt_type=='S'):
            periods1= total_diff_days*(86400/dt)
        elif(dt_type=='min' or dt_type=='T'):
            periods1= total_diff_days*(1440/dt)
        elif(dt_type=='h'or dt_type=='H'):
            periods1= total_diff_days*(24/dt)
        elif(dt_type=='d' or dt_type=='D'):
            periods1= total_diff_days*(1/dt)
            
        periods1 +=1
        freq1=str(dt)+dt_type
        
        time_series1=pd.Series(
        pd.date_range(start_time, periods=periods1, freq=freq1, tz=utc_)
        )      
        
        time_series1_index_drop=[]
        
        start_hms_float=[float(i) for i in start_hms.split(':')]
        start_hms_float=sum([a*b for a,b in zip(start_hms_float,[1, 1/60, 1/3600])])
        
        end_hms_float=[float(i) for i in end_hms.split(':')]
        end_hms_float=sum([a*b for a,b in zip(end_hms_float,[1, 1/60, 1/3600])])
        
        for i in range(len(time_series1)): 
            hms_check=str(time_series1[i])[11:19].split(':')
            hms_check=[float(i) for i in hms_check]
            hms_check=sum([a*b for a,b in zip(hms_check,[1, 1/60, 1/3600])])
           
            if (hms_check>end_hms_float or hms_check<start_hms_float):
                time_series1_index_drop.append(i)
        
        time_series1=time_series1.drop(time_series1_index_drop)
        time_series1=time_series1.reset_index(drop=True)
        
        self.freq=freq1
        self.periods=periods1
        self.length=len(time_series1)
        self.time_series=time_series1
        
    def get_time_range(self, lower_bound, upper_bound, inclusive=1):
        
        time_series=self.time_series.copy()
        UTC=self.utc
        
        if(type(lower_bound)==str):
            lower_bound=pd.Timestamp(lower_bound + UTC.replace(":", "")[0:5])
            
        if(type(upper_bound)==str):
            upper_bound=pd.Timestamp(upper_bound + UTC.replace(":", "")[0:5])
            
        if(upper_bound==None and lower_bound!=None):
            
            if(inclusive==1):
                time_series2=time_series.loc[time_series.iloc[:]>=lower_bound]
            else: 
                time_series2=time_series.loc[time_series.iloc[:]>lower_bound]
                
        if(upper_bound!=None and lower_bound==None):
            
            if(inclusive==1):
                time_series2=time_series.loc[time_series.iloc[:]<=upper_bound]
            else: 
                time_series2=time_series.loc[time_series.iloc[:]<upper_bound]
                
        if(upper_bound!=None and lower_bound!=None):
            
            if(inclusive==1):
                time_series2=time_series.loc[(time_series.iloc[:]>=lower_bound)
                                             & (time_series.iloc[:]<=upper_bound)]
            else: 
                time_series2=time_series.loc[(time_series.iloc[:]>lower_bound)
                                             & (time_series.iloc[:]<upper_bound)]
                
        self.time_series2=time_series2
        

                
            
        

 

 #%% 

class Sun:    
    
    def __init__(self, time_series, Site_object, temp=25, press=101325):
                
        SLOPE=0
        AZIMUTH=0
                
        solar_data=pv.solarposition.get_solarposition(time_series, 
                                                 latitude=Site_object.lat, 
                                                 longitude=Site_object.long, 
                                                 altitude=Site_object.alt,
                                                 pressure=press,
                                                 method='nrel_numpy', 
                                                 temperature=temp)  
        
        #NOTE: temp and press can either be floats or numpy arrays whose length
        #      is equal to that of 'time_series'.
        
        apparent_zenith=solar_data.loc[:,'apparent_zenith']
        sun_azimuth=solar_data.loc[:,'azimuth']
    

       
        plane_aoi=pv.irradiance.aoi(surface_tilt=SLOPE, 
                               surface_azimuth=AZIMUTH, 
                               solar_zenith=apparent_zenith, 
                               solar_azimuth=sun_azimuth)
        
        solar_data['plane_aoi']=pd.DataFrame(plane_aoi)
        
        solar_data.index=pd.to_datetime(solar_data.index)
        
        self.solar_data=solar_data
        self.apzen=solar_data.loc[:, 'apparent_zenith']
        self.zen=solar_data.loc[:, 'zenith']
        self.apel=solar_data.loc[:, 'apparent_elevation']
        self.el=solar_data.loc[:, 'elevation']
        self.az=solar_data.loc[:, 'azimuth']
        self.plane_aoi=solar_data.loc[:, 'plane_aoi']
        self.utc=Site_object.utc
        
        
        
    def get_solar_time_range(self, lower_bound, upper_bound, inclusive=1):
        
        solar_data=self.solar_data.copy()
        UTC=self.utc
        
        if(type(lower_bound)==str):
            lower_bound=pd.Timestamp(lower_bound + UTC.replace(":", "")[0:5])
            
        if(type(upper_bound)==str):
            upper_bound=pd.Timestamp(upper_bound + UTC.replace(":", "")[0:5])
            
        if(upper_bound==None and lower_bound!=None):
            
            if(inclusive==1):
                solar_data2=solar_data.loc[solar_data.index>=lower_bound]
            else: 
                solar_data2=solar_data.loc[solar_data.index>lower_bound]
                
        if(upper_bound!=None and lower_bound==None):
            
            if(inclusive==1):
                solar_data2=solar_data.loc[solar_data.index<=upper_bound]
            else: 
                solar_data2=solar_data.loc[solar_data.index<upper_bound]
                
        if(upper_bound!=None and lower_bound!=None):
            
            if(inclusive==1):
                solar_data2=solar_data.loc[(solar_data.index>=lower_bound)
                                              & (solar_data.index<=upper_bound)]
            else: 
                solar_data2=solar_data.loc[(solar_data.index>lower_bound)
                                              & (solar_data.index<upper_bound)]
                
        self.solar_data2=solar_data2
        



#%%



def calc_solar_vector(Sun_object, Site_object):
    
    theta=np.deg2rad(Sun_object.apzen)  
    phi=np.deg2rad(Sun_object.az)
    
    sunvec=np.zeros((len(theta), 4))
    
    sunvec[:,0]=np.array(Site_object.tmy_interpolated.loc[:,'Gb(n)'], np.float64)
    
    sunvec[:,1]=np.array(np.cos(phi)*np.sin(theta), np.float64)
    sunvec[:,2]=np.array(np.sin(phi)*np.sin(theta), np.float64)
    sunvec[:,3]=np.array(np.cos(theta), np.float64)
    
    return sunvec
     

#%%


# site1=Site()
# tmy_data1=site1.tmy_data
# #%%
# values=np.array(site1.get_tmy_values(month=6, day=(1,31), hour=(0,23)))
 
# #%%   
    
# time1=Time()

# time_series1=time1.time_series

# sun1=Sun(time1.time_series, site1)   

# solar_data=sun1.solar_data

# sun1.get_solar_time_range('2021-06-12 00:00:00', '2021-06-12 23:00:00')
# print(sun1.solar_data2)

# #%%

# site1.tmy_interpolate(time1)

# tmy_interpolated1=site1.tmy_interpolated

# #%%

# time1.get_time_range(lower_bound='2021-06-12 00:00:00', upper_bound='2021-06-12 23:00:00')
