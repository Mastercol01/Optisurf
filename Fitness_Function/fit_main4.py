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


# --- DEFINITION OF PARAMS ---
faces=surf_obj.faces
vertices=surf_obj.vertices
areas=surf_obj.areas
normals=surf_obj.normals
dx=0.05
dy=0.05
max_z=max(vertices[:,2])
min_z=max(vertices[:,2])
num_faces=len(normals)
num_times=len(sunvec)

# --- CALC POW PER FACE PER TIME WITHOUT SHADING ---
area_normals=np.einsum('i,ij->ij', areas, normals)
irrad_vec=np.einsum('i,ij->ij', sunvec[:,0], sunvec[:,1:])
pow_perf_pert=np.einsum('kj,ij->ik', area_normals, irrad_vec)
pow_perf_pert[pow_perf_pert<0]=0



# --- CALC DIRECT FACING FACE INDICES PER TIME ---
num_faces=len(normals)
num_times=len(sunvec)

pow_perf_in_a_time=np.arange(num_faces)

direct_facing_face_indices_pert=\
[pow_perf_in_a_time[row>0] for row in pow_perf_pert]


# --- CALC TEST POINTS PER FACE ---

uv_vals=np.array([ [0.5, 0.5],
                   [0.05, 0.05],
                   [0.05, 0.95],
                   [0.95, 0.05],
                   [0.95, 0.95] ])

faces_coords=np.zeros((num_faces,3,3))

i=-1
for face_vertices in faces:
    i+=1
    faces_coords[i,0,:]=vertices[face_vertices[0]]
    faces_coords[i,1,:]=vertices[face_vertices[1]]
    faces_coords[i,2,:]=vertices[face_vertices[2]]    


test_points_coords_perf=np.zeros((num_faces, 5, 3))

i=0
for face_coords in faces_coords:
    j=0
    for uv in uv_vals:
        test_points_coords_perf[i,j,:]= (1-uv[0]-uv[1])*face_coords[0,:] +  \
        uv[0]*face_coords[1,:] +uv[1]*face_coords[2,:]
        j+=1
    i+=1
 
    
# --- CALC RAY ORIGS PER FACE PER TIME ---
        
rays_origs_perf_pert=np.zeros((num_times, num_faces, 5, 3))
T_PARAM=20

for i in range(num_times):
    
    sum_mat_sunpos=np.tile(T_PARAM*sunvec[i, 1:], (5,1))
    
    for j in range(num_faces):
        
        rays_origs_perf_pert[i,j,:,:]= \
        test_points_coords_perf[j,:,:] + sum_mat_sunpos
    
        
#%%    
    
#--- CALC FRACTIONS OF RECIEVED RAYS PER FACE PER TIME ---   

frac_recieved_rays_perf_pert=np.zeros(pow_perf_pert.shape)


t=-1
for indeces in direct_facing_face_indices_pert: 
    t+=1
    sun_vec=sunvec[t,1:]
    
    for index in indeces:  
        
        #--TEST RAYS ORGIGS FOR A FACE IN A TIME---
        rays_origs=rays_origs_perf_pert[t, index, :, :]
        
        #--- CALC SEARCH POINTS 1,2 (UPPER, LOWER) ---
        search_points1=np.zeros((5,2))
        search_points2=np.zeros((5,2))
        
        search_points1[:,0]=rays_origs[:,0] + \
        (max_z-rays_origs[:,2])*(sun_vec[0]/sun_vec[2])
        search_points1[:,1]=rays_origs[:,0] + \
        (max_z-rays_origs[:,2])*(sun_vec[1]/sun_vec[2])
        
        search_points2[:,0]=rays_origs[:,0] + \
        (min_z-rays_origs[:,2])*(sun_vec[0]/sun_vec[2])
        search_points2[:,1]=rays_origs[:,0] + \
        (min_z-rays_origs[:,2])*(sun_vec[1]/sun_vec[2])
        
        #--- CALC OF EDGE CUADRILATERAL-ELEMENTS ---
        
        
        
        cuadri_elems1_coords=search_points1
        cuadri_elems1_coords[:,0]=dx*(cuadri_elems1_coords[:,0]//dx)
        

        

        
        
        
    
    
    
    
    
    
    
    

    
    
#%%
    

    
    
    
    
    
    
    

