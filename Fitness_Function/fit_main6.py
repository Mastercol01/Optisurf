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



#%%
# --- CALC DIRECT FACING FACE INDICES PER TIME ---
pow_perf_in_a_time=np.arange(num_faces)

direct_facing_face_indices_pert=\
[pow_perf_in_a_time[row>0] for row in pow_perf_pert]


#%%


A = vertices[faces[:,0],:]
B = vertices[faces[:,1],:]
C = vertices[faces[:,2],:]


#%%

E1 = B-A
E1 = np.array([E1]*num_times)
E1 = np.moveaxis(E1, 0, -1)

E2 = C-A
E2 = np.array([E2]*num_times)
E2 = np.moveaxis(E2, 0, -1)

uv_list=[(0.45, 0.45),
         (0.1, 0.1),
         (0.1, 0.8),
         (0.8, 0.1) ]

#%%
T_list=[]

D=-sunvec[:, 1:]
D=np.moveaxis(D, 0, -1)
D=np.array([D]*num_faces)

DIST_PARAM = 5

for u, v in uv_list:
    O = (1-u-v)*A + u*B + v*C
    T = O - A
    T = np.array([T]*num_times)
    T = np.moveaxis(T, 0, -1) 
    T = T - DIST_PARAM*D
    
    T_list.append(T)
    

#%%

def adapt(x,y):

    print(x)
    print(y)

np.apply_over_axes(adapt, T, [1,2])















#%%

a = np.arange(180).reshape(5,3,12)

b = np.arange(15).reshape(5,3)

c = np.arange(36).reshape(12,3)


bb = np.array( [b]*12)
bb = np.moveaxis(bb, 0, -1)
r1 = a-bb


cc = np.array( [c]*5)
cc = np.moveaxis(cc, 1, -1)


r2 = np.cross(a,bb, axisa=1, axisb=1)
r2 = np.moveaxis(r2, 1, -1)
    
test= np.zeros((5,3,12))

for i in range(5):
 for k in range(12):
     
     test[i,:,k]=np.cross(a[i,:,k], b[i,:])
     
    
    
# P = np.cross(D, E2, axisa=1, axisb=1)
# P = np.moveaxis(P, 1, -1)

# Q = np.cross(T, E1, axisa=1, axisb=1)
# Q = np.moveaxis(Q, 1, -1)


# divider = 1/np.sum(P*E1, axis=1)

# t_term = np.sum(Q*E2, axis=1)
# u_term = np.sum(P*T, axis=1)
# v_term = np.sum(Q*D, axis=1)    


    
    
    