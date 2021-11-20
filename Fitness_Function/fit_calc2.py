import numpy as np

# import ray 
# import random, logging

# ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)


def power_per_surf2(areas, normals, sunvec):
    
    area_normals=np.einsum('i,ij->ij', areas, normals)
    irrad_vec=np.einsum('i,ij->ij', sunvec[:,0], sunvec[:,1:])
    pow_per_face_per_time=np.einsum('kj,ij->ik', area_normals, irrad_vec)
    
    pow_per_face_per_time[pow_per_face_per_time < 0] = 0
    
    pow_per_surf_per_time=np.sum(pow_per_face_per_time, axis=1)
    
    mean_pow_per_surf= np.sum(pow_per_surf_per_time[1:] + 
                              pow_per_surf_per_time[:-1])
    
    return 0.5*mean_pow_per_surf/(len(sunvec)-1)


    


# @ray.remote
# def power_per_surf2_ray(key, areas, normals, sunvec):
    
#     area_normals=np.einsum('i,ij->ij', areas, normals)
#     irrad_vec=np.einsum('i,ij->ij', sunvec[:,0], sunvec[:,1:])
#     pow_per_face_per_time=np.einsum('kj,ij->ik', area_normals, irrad_vec)
    
#     pow_per_face_per_time[pow_per_face_per_time < 0] = 0
    
#     pow_per_surf_per_time=np.sum(pow_per_face_per_time, axis=1)
    
#     mean_pow_per_surf= np.sum(pow_per_surf_per_time[1:] + 
#                               pow_per_surf_per_time[:-1])
    
#     mean_pow_per_surf=0.5*mean_pow_per_surf/(len(sunvec)-1)
    
#     return key, mean_pow_per_surf


    
   
    
    
#%%


# ars=[10,11,12]

# a=np.array([[1, 2, 3],
#             [2, 2, 3],
#             [1, 2, 3]])

# z=np.einsum('i,ij->ij', ars,a)


# b=np.array([[1, 2, 3],
#             [3, 4, 5],
#             [5, 4, 3],
#             [1, 1, 2]])



# #power_per_face_per_time
# cos_aoi=np.einsum('kj,ij->ik', a,b) # j-axis: face/area axis
#                                     # i-axis: time/axis
                                    
# #power_per_surface_per_time                                    
# cos_aoi2=np.einsum('kj,ij->i', a,b) # i-axis: time/axis

    
   