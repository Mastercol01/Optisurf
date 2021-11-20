import Genetic_Algorithim.GenAlgth as ga
import Mesh_Generation.MeshGen as mg
import Initial_Population_Creation.VarZ_initpop_Gen as vzipg
import Mesh_Generation.DomainDef as dodef
import Fitness_Function.fit_calc2 as fitc
import Fitness_Function.pvpowlib.geoloc as pvgeo
import Initial_Population_Creation.VarZ_funcs as vzf
import numpy as np
import time as tm
import pandas as pd



#%%                      DEFINITION OF SITE AND TIME OBJECTS


Magangue=pvgeo.Site(name='Magangué', 
                    lat=9.25055, 
                    long=-74.7661, 
                    alt=19, 
                    UTC='-05:00:00' )


time=pvgeo.Time(start_date='2021-01-01', 
                end_date='2021-01-31',
                start_hms='06:00:00',
                end_hms='18:00:00',
                time_interval='20-min', 
                UTC='-05:00:00')


#interpolation of tmy data so it maches up with the simulation time.
Magangue.tmy_interpolate(Time_object=time)  


sun=pvgeo.Sun(time_series=time.time_series, 
              Site_object=Magangue,
              temp=np.array(Magangue.tmy_interpolated.loc[:,'T2m'], np.float64), 
              press=np.array(Magangue.tmy_interpolated.loc[:,'SP'], np.float64) )


#%%                 CALCULATION OF SUN'S MAGNITUDE VECTOR

sunvec=pvgeo.calc_solar_vector(sun, Magangue)
#%%            GENERATION OF THE INITIAL POPULATION OF SURFACES SURFACES

xy_plane=dodef.RectDomain(dx=0.05, dy=0.05, dims=(1,1,1))


population=vzipg.create_init_pop(xy_plane)


#%%               CREATION OF MESHES TO BE COMPARED

surf_plane=vzf.build_plane(xy_plane.x_vals, 
                           xy_plane.y_vals, 
                           xy_plane.x_dim, 
                           xy_plane.y_dim, 
                           xy_plane.dx, 
                           xy_plane.dy, 
                           rad=1, 
                           ang_freq=1, 
                           ampl=1)

surf_plane_mesh=mg.Mesh(x_vals=xy_plane.x_vals, 
                        y_vals=xy_plane.y_vals,
                        z_vals=surf_plane)

best_vertices=pd.read_excel(r'C:/Users/andre/Desktop/Eafit/8vo Semestre/Monitoría Panel Solar/OptiSurf/Fitness_Function/best_vertices.xlsx')
best_faces=pd.read_excel(r'C:/Users/andre/Desktop/Eafit/8vo Semestre/Monitoría Panel Solar/OptiSurf/Fitness_Function/best_faces.xlsx')

best_surf_mesh=mg.Mesh(vertices=np.array(best_vertices), 
                       faces=np.array(best_faces))



#%%

t=tm.time()
results1=fitc.power_per_surf2(surf_plane_mesh.areas, 
                              surf_plane_mesh.normals, 
                              sunvec)
dt1=tm.time()-t


t=tm.time()
results2=fitc.power_per_surf2(surf_plane_mesh.areas, 
                             surf_plane_mesh.normals, 
                             sunvec)
dt2=tm.time()-t




#%%              CALCULATION OF GRAPHS

surf_plane_powers=np.zeros(36*31)
best_surf_powers=np.zeros(36*31)
time_series=time.time_series  

c=-1
cc=0
for i in range(1,len(sunvec)):
    
    
    if(str(time_series[i])[11:19]=='06:00:00'):
        cc+=1
    else:
        c+=1
        sub_sunvec=sunvec[i-1:i+1,:]
    
        surf_plane_powers[c]=fitc.power_per_surf2(surf_plane_mesh.areas, 
                                                   surf_plane_mesh.normals, 
                                                   sub_sunvec)
        
        best_surf_powers[c]=fitc.power_per_surf2(best_surf_mesh.areas, 
                                                  best_surf_mesh.normals, 
                                                  sub_sunvec)
    
    
#%%    

c=-1  
cc=0
k=-1
plane_surf_mean_power=np.zeros(36)
best_surf_mean_power=np.zeros(36)
for i in range(1, len(time_series)):
    k+=1
    
    if(str(time_series[i])[11:19]=='06:00:00'):
        c=-1
        cc+=1
        k=-1
    else:
        c+=1
        plane_surf_mean_power[k]+=surf_plane_powers[c]
        best_surf_mean_power[k]+=best_surf_powers[c]
        
plane_surf_mean_power=plane_surf_mean_power/cc
best_surf_mean_power=best_surf_mean_power/cc
        

#%%
import matplotlib.pyplot as plt

t_axis=np.linspace(6.333, 18, 36)
plt.plot(t_axis, plane_surf_mean_power, label='reference plane surface')    
plt.plot(t_axis, best_surf_mean_power, label='best surface')    
plt.xlabel('time [h]')
plt.ylabel('Mean wattage generated over the last 20-min')
plt.suptitle('Reference plane surface vs. Best surface')
plt.title('Estimation of generated mean wattage over january 2021, Magangué')
plt.legend()
plt.show()
    
 

