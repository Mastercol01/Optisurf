from copy import deepcopy
import Mesh_Generation.MeshGen as mg
import Initial_Population_Creation.VarZ_funcs as vzf
import numpy as np
#import time as tm
#%%

def create_init_pop(RectDomain_object):

#    t1=tm.time()
        
    x_vals=RectDomain_object.x_vals
    y_vals= RectDomain_object.y_vals
    x_dim=RectDomain_object.x_dim
    y_dim=RectDomain_object.y_dim
    dx=RectDomain_object.dx
    dy=RectDomain_object.dy
    rad= 6.5 
    ang_freq=50.0
    ampl=1.0  
    
    
    
    dispatcher={0: vzf.build_plane,
                1: vzf.build_double_sine,
                2: vzf.build_convex_sphere1,
                3: vzf.build_convex_sphere2,        
                4: vzf.build_concave_sphere1, 
                5: vzf.build_concave_sphere2 }
    
    
    
    
    
    list_rads=[2,4]
    list_ang_freqs=[60]
    list_ampls=[0.05]
    
    rot_array=vzf.calc_rot_arrays(3)
    rot_angles=[17]
    
    total_combinations=len(list_rads)*len(list_ang_freqs)*len(list_ampls)
    combination_array=np.zeros((total_combinations,3))
    
    i=-1
    for ampl in list_ampls:
        for ang_freq in list_ang_freqs:
            for rad in list_rads:
                i+=1
                combination_array[i,:]=[rad, ang_freq, ampl]
                
                
               
                
    init_population={}
    
    surf_num=-1
    for i in [1,2,4]:
        for j in range(total_combinations):
            
            surf_num+=1
            
            rad=combination_array[j,0]
            ang_freq=combination_array[j,1]
            ampl=combination_array[j,2]
    
            init_population[surf_num]=dispatcher[i](x_vals, y_vals, x_dim, y_dim, 
                                                    dx, dy, rad, ang_freq, ampl)
    
    
    for i in [0,3,5]:
        surf_num+=1
        init_population[surf_num]=dispatcher[i](x_vals, y_vals, x_dim, y_dim, 
                                                dx, dy, rad, ang_freq, ampl)
        
        
    for i in init_population:
        init_population[i]=mg.Mesh(x_vals, y_vals, init_population[i])
        
    
    
    for i in range(len(init_population)):
        for rot_angle in rot_angles:
            for rot_axis in rot_array:
                surf_num+=1
                init_population[surf_num]=deepcopy(init_population[i])
                init_population[surf_num].rotate(rot_axis, rot_angle)
                
                
#    print(tm.time()-t1)            
    return init_population   
    
    

    
    