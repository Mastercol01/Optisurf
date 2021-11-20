import numpy as np
from numba import njit
from numba.types import float64



@njit(float64(float64, float64[:], float64[:]), parallel=True)
def power_per_face_per_time(area, normal, sun_vec):
    
    cos_aoi=np.dot(normal, sun_vec[1:])
    
    if(cos_aoi>0):
        pow_=sun_vec[0]*cos_aoi*area
    else:
        pow_=0
        
    return pow_



@njit(float64(float64[:], float64[:,:], float64[:]))
def power_per_surf_per_time(areas, normals, sun_vec):
    
    total_pow=0
    
    for i in range(len(areas)):
        
        total_pow+=power_per_face_per_time(areas[i], 
                                            normals[i,:], 
                                            sun_vec)
        
    return total_pow  
 

@njit(float64(float64[:], float64[:,:], float64[:,:]))
def power_per_surf(areas, normals, sunvec):
    
    mean_pow=0
    len_units_time=len(sunvec)
    
    for i in range(1, len_units_time):
        
        pow1=power_per_surf_per_time(areas, normals, sunvec[i-1,:])
        pow2=power_per_surf_per_time(areas, normals, sunvec[i,:])
        
        mean_pow+=(pow1+pow2)
        
    mean_pow=0.5*mean_pow/(len_units_time-1)
        
    return mean_pow

    
    
def rank_surf_population(population, sunvec):
    surf_ranking=[]          
    
    for key in population:
        
        fitness_score=power_per_surf(population[key].areas, 
                                      population[key].normals, 
                                      sunvec)
        
        surf_ranking.append((key, fitness_score))
        print(key)
        
        
    return sorted(surf_ranking, key=lambda x: x[1], reverse=1) 
    
    

    
    
   