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


columns=['t','f','']



#%%

A = vertices[faces[:,0],:]
B = vertices[faces[:,1],:]
C = vertices[faces[:,2],:]


E1 = B-A
E2 = C-A

E1 = np.array([E1]*num_times)
E2 = np.array([E2]*num_times)


uv_list=[(0.45, 0.45),
         (0.1, 0.1),
         (0.1, 0.8),
         (0.8, 0.1) ]

#%%
T_list=[]

D=-sunvec[:, 1:]
D=np.array([D]*num_faces)
D=np.moveaxis(D, 0, 1)

DIST_PARAM = 5

for u, v in uv_list:
    O = (1-u-v)*A + u*B + v*C
    T = O - A
    T = np.array([T]*num_times)
    T = T - DIST_PARAM*D
    
    T_list.append(T)
    
#%%    

time_idx = np.arange(num_times*num_faces)//num_faces
faces_idx = np.array([np.arange(num_faces)]*num_times)
faces_idx = faces_idx.reshape(num_times*num_faces)
# index = list(zip(time_idx, faces_idx))
index = np.arange(num_times*num_faces)

Tt = T.reshape((num_times*num_faces,3))
Dd = D.reshape((num_times*num_faces,3))
E11 = E1.reshape((num_times*num_faces,3))
E22 = E2.reshape((num_times*num_faces,3))
super_array = np.concatenate([Tt, Dd, E11, E22], axis=1)

columns = ['Tx','Ty','Tz',
           'Dx','Dy','Dz',
           'E1x','E1y','E1z',
           'E2x','E2y','E2z']

df = pd.DataFrame(super_array, 
                  columns=columns,
                  index=index)

df['time']=time_idx
df['faces']=faces_idx


del time_idx
del faces_idx
del index
del Tt
del Dd
del E11
del E22
del super_array


#%%

def ray_trace1(df):

    
    D = np.array(df.loc[:, ['Dx','Dy','Dz']])
    E1 = np.array(df.loc[:, ['E1x','E1y','E1z']])
    E2 = np.array(df.loc[:, ['E2x','E2y','E2z']])
    
    result = df.groupby(['faces']).apply(ray_trace2, E1=E1, 
                                         E2=E2, D=D)
    print(result)
    
    return result

    

def ray_trace2(df, D=D, E1=E1, E2=E2):

    T = np.array(df.loc[:, ['Tx','Ty','Tz']])
    T = np.array([T]*len(D))
    face = int(df.at[0, 'faces'])
    
    P = np.cross(D, E2)
    Q = np.cross(T, E1)
    
    divider = 1/np.sum(P*E1, axis=1)
    divider = np.array([divider]*3)
    divider = np.moveaxis(divider, 0, -1)
    
    t_vec = divider*np.sum(Q*E2, axis=1)
    u_vec = divider*np.sum(P*T, axis=1)
    v_vec = divider*np.sum(Q*D, axis=1)
    
    uv_vec = u_vec + v_vec

    grl_vec=np.column_stack((np.arange(len(D)), t_vec,
                             u_vec, v_vec, uv_vec))
    
    grl_vec=grl_vec[grl_vec[:,2]>=0]
    grl_vec=grl_vec[grl_vec[:,3]>=0]
    grl_vec=grl_vec[grl_vec[:,4]<=1]

    
    if(len(grl_vec)==0):
        return 0
        
    index= grl_vec[ np.where(grl_vec[:,1] == np.amin(grl_vec[:,1])), 0]
    
    if(index==face):
        return 1
    
    
#%%

df2=df.groupby(['time']).apply(ray_trace1)

    
    
    