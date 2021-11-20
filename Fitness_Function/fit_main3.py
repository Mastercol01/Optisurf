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
                end_date='2021-06-30',
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

best_vertices=pd.read_excel(r'C:\Users\andre\Desktop\Eafit\8vo Semestre\Monitoría Panel Solar\OptiSurf\Fitness_Function\best_vertices.xlsx')
best_faces=pd.read_excel(r'C:\Users\andre\Desktop\Eafit\8vo Semestre\Monitoría Panel Solar\OptiSurf\Fitness_Function\best_faces.xlsx')

best_surf_mesh=mg.Mesh(vertices=np.array(best_vertices), 
                       faces=np.array(best_faces))


#%%


surf_obj=best_surf_mesh


# --- PARAMS DEF ---
faces=surf_obj.faces
vertices=surf_obj.vertices
areas=surf_obj.areas
normals=surf_obj.normals
max_z=max(vertices[:,2])
min_z=max(vertices[:,2])
num_faces=len(normals)
num_times=len(sunvec)

# --- CALC POW PER FACE PER TIME WITHOUT SHADING ---
area_normals=np.einsum('i,ij->ij', areas, normals)
irrad_vec=np.einsum('i,ij->ij', sunvec[:,0], sunvec[:,1:])
pow_per_face_per_time=np.einsum('kj,ij->ik', area_normals, irrad_vec)

pow_per_face_per_time[pow_per_face_per_time<0]=0



#%%





# --- CALC DIRECT FACING FACE INDICES PER TIME ---
num_faces=len(normals)
num_times=len(sunvec)

pow_per_face_in_a_time=np.arange(num_faces)

direct_facing_face_indices_per_time=\
[pow_per_face_in_a_time[row>0] for row in pow_per_face_per_time]


# --- CALC TEST POINTS PER FACE ---

test_points_per_face=np.zeros((num_faces,3,5))

uv_vals=[(0.5, 0.5),
         (0.05, 0.05),
         (0.05, 0.95),
         (0.95, 0.05),
         (0.95, 0.95)]


i=-1
for face_vertices in faces:
    i+=1
    A=vertices[face_vertices[0]]
    B=vertices[face_vertices[1]]
    C=vertices[face_vertices[2]] 
    
    j=-1
    for uv in uv_vals:
        j+=1
        test_points_per_face[i,:,j]=(1-uv[0]-uv[1])*A + uv[0]*B +uv[1]*C
        
        
#--- CALC FRACTIONS OF RECIEVED RAYS PER FACE PER TIME ---   

frac_of_recieved_rays_per_face_per_time=np.zeros(pow_per_face_per_time.shape)

T_PARM=20

c=-1
for indeces in direct_facing_face_indices_per_time: 
    c+=1
    sun_vec=sunvec[c,:]
    
    for index in indeces:  
        
        #---CALC OF THE TEST RAYS ORIGINS PER FACE---
        rays_orig_per_face_per_time=\
        test_points_per_face[300,:,:].transpose() + \
        np.tile(T_PARM*sun_vec[1:], (5,1))
        
        
        #---CALC INTSERCTION COORDS OF UPPER MAX PLANE---
        max_z_plane_intersects=rays_orig_per_face_per_time.copy()
    
        max_z_plane_intersects[:,0]=max_z_plane_intersects[:,0] + \
        (max_z-max_z_plane_intersects[:,2])*(sun_vec[1]/sun_vec[3])
    
        max_z_plane_intersects[:,0]=max_z_plane_intersects[:,1] + \
        (max_z-max_z_plane_intersects[:,2])*(sun_vec[2]/sun_vec[3])
        
        max_z_plane_intersects[:,2]=max_z
        
        
        #---CALC INTSERCTION COORDS OF LOWER MIN PLANE---
        min_z_plane_intersects=rays_orig_per_face_per_time.copy()
    
        min_z_plane_intersects[:,0]=min_z_plane_intersects[:,0] + \
        (min_z-min_z_plane_intersects[:,2])*(sun_vec[1]/sun_vec[3])
    
        min_z_plane_intersects[:,0]=min_z_plane_intersects[:,1] + \
        (min_z-min_z_plane_intersects[:,2])*(sun_vec[2]/sun_vec[3])
        
        min_z_plane_intersects[:,2]=min_z
        
        
        
    
    
    
    
    
    
    
    

    
    
#%%
    

    
    
    
    
    
    
    

