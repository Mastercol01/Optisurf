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
num_faces=normals.shape[0]
num_times=sunvec.shape[0]
num_coords=vertices.shape[1]
num_vertices_perf=faces.shape[1]

# --- CALC POW PER FACE PER TIME WITHOUT SHADING ---
area_normals=np.einsum('i,ij->ij', areas, normals)
irrad_vec=np.einsum('i,ij->ij', sunvec[:,0], sunvec[:,1:])
pow_perf_pert=np.einsum('kj,ij->ik', area_normals, irrad_vec)
pow_perf_pert[pow_perf_pert<0]=0

t1=tm.time()

#%%
# --- CALC DIRECT FACING FACE INDICES PER TIME ---
pow_perf_in_a_time=np.arange(num_faces)

direct_facing_face_indices_pert=\
[pow_perf_in_a_time[row>0] for row in pow_perf_pert]

#%%

# --- CALCULATION OF A NEW MORE COMPLETE ARRAY OF FACES WITH ITS COORDS---     
def create_fv(face, vertices=vertices):
    
    x=np.zeros((num_vertices_perf, num_coords))
    x[0,:]=vertices[face[0],:]
    x[1,:]=vertices[face[1],:]
    x[2,:]=vertices[face[2],:]
    
    return x

fv_array=np.apply_along_axis(create_fv, 1, faces)

# --- CALC ORIGS RAYS PER FACE ---

uv_vals=np.array([ [0.5, 0.5],
                   [0.05, 0.05],
                   [0.05, 0.95],
                   [0.95, 0.05],
                   [0.95, 0.95] ])

num_tpoints_perf=uv_vals.shape[0]

ray_origs=np.ones((num_faces, num_tpoints_perf, num_coords))

for i in range(uv_vals.shape[0]):
    ray_origs[:,i,:]= uv_vals[i,0]*fv_array[:,1,:] + \
                      uv_vals[i,1]*fv_array[:,2,:] + \
                      (1-uv_vals[i,0]-uv_vals[i,1])*fv_array[:,0,:]
                        
ray_origs=np.ones((num_times, num_faces,
                   num_tpoints_perf, num_coords))*ray_origs      

def adapt1(x):
    x=np.broadcast_to(x, (num_tpoints_perf, num_coords))
    x=np.broadcast_to(x, (num_faces, num_tpoints_perf, num_coords))
    return x

# def show(x):
#     print(x)

T_PARAM=30
ray_origs=ray_origs + np.apply_along_axis(adapt1, 1, T_PARAM*sunvec[:,1:])


def adapt2(y):
    y=np.broadcast_to(y, (num_tpoints_perf, num_coords))
    return y

Vo=np.apply_along_axis(adapt2, 1, fv_array[:,0,:])
Vo=np.broadcast_to(Vo, (num_times, num_faces, num_tpoints_perf, num_coords))

vecssT=ray_origs-Vo

#%%

vecsA=fv_array[:,1,:]-fv_array[:,0,:]
vecsB=fv_array[:,2,:]-fv_array[:,0,:]

frac_recieved_rays_perf_pert=np.zeros((num_times, num_faces))
vec_num_faces=np.arange(num_faces)

for t in range(num_times):
        
    if(len(direct_facing_face_indices_pert[t])==0):
        continue
    
    direct_facing_face_indices_in_a_t=\
    np.array(direct_facing_face_indices_pert[t])
    vecsd=np.broadcast_to(-sunvec[t,1:], (num_faces, num_coords))
    
    for f in direct_facing_face_indices_in_a_t:
        for r in range(num_tpoints_perf):            
            vecsT=np.broadcast_to(vecssT[t,f,r,:], (num_faces, num_coords))
            
            vecsP=np.cross(vecsd, vecsB)
            vecsQ=np.cross(vecsT, vecsA)
            
            divider=1./np.einsum('ij,ij->i', vecsP, vecsA)
            
            t_vec=divider*np.einsum('ij,ij->i', vecsQ, vecsB)
            u_vec=divider*np.einsum('ij,ij->i', vecsP, vecsT)
            v_vec=divider*np.einsum('ij,ij->i', vecsQ, vecsd)
            
            uv_vec=u_vec+v_vec

            grl_vec=np.column_stack((vec_num_faces, t_vec, u_vec, 
                                    v_vec, uv_vec))
            
            grl_vec=grl_vec[grl_vec[:,2]>=0]
            grl_vec=grl_vec[grl_vec[:,3]>=0]
            grl_vec=grl_vec[grl_vec[:,4]<=1]
            
            if(len(grl_vec)==0):
                continue

            index= grl_vec[ np.where(grl_vec[:,1] == np.amin(grl_vec[:,1])), 0]
            
            if(index==f):
                frac_recieved_rays_perf_pert[t,f]+=1
                
frac_recieved_rays_perf_pert=\
frac_recieved_rays_perf_pert/num_tpoints_perf            
            
dt=tm.time()-t1

            










        
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    

