import Mesh_Generation.DomainDef as dodef
import Mesh_Generation.MeshGen as mg
from scipy.spatial import Delaunay
#from numba import njit
#from numba.types import float64
import numpy as np
import random

#%%

#@njit(float64[:,:](float64[:,:], float64[:,:], float64, float64))
def procreate_vertices(mom_vertices, dad_vertices, tolx, toly):
    
    child_vertices=np.zeros((len(mom_vertices),3))    
    
    i=-1
    for mom_vertex in mom_vertices:
        i+=1
        
        try:
            dad_rows=dodef.rect_c2v(mom_vertex[0],
                                    mom_vertex[1],
                                    tolx,
                                    toly,
                                    dad_vertices[:,0:2])
            
            dad_vertex=[abs(dad_vertices[i][2]-mom_vertex[2]) for i in dad_rows]
            index=dad_vertex.index(min(dad_vertex))
            dad_vertex=dad_vertices[ dad_rows[index], : ] 
            
            child_vertices[i,2]=np.mean(dad_vertex[2], mom_vertex[2])
        except:
            child_vertices[i,2]=mom_vertex[2]
            
        child_vertices[i,0]=mom_vertex[0]
        child_vertices[i,1]=mom_vertex[1]
            
        child_vertices[i,2]+=np.random.normal(0,0.005)
        
    return child_vertices



def evolve(population, rank_list):
    
    dx=dy=0.05
    tolx=0.99*dx/2
    toly=0.99*dy/2
    
    num_surfs=len(population)
    new_population={}
    
    percentil=0.12
    num_top_surfs=int(dodef.njit_round(percentil*num_surfs, 0))
    
    num_nrp_pts=np.ones(num_top_surfs)*(num_surfs//num_top_surfs)
    num_nrp_pts[0]+=num_surfs%num_top_surfs
    num_nrp_pts=[int(i) for i in num_nrp_pts]


    key=0

    
    for i in range(num_top_surfs):
        surf1=population[rank_list[i][0]]
        
        for j in range(1, num_nrp_pts[i]):
            surf2=population[rank_list[i+j][0]]

            sort_list=[ ( surf1, len(surf1.areas) ), \
                        ( surf2, len(surf2.areas) ) ]
                
            sort_list=sorted(sort_list, key=lambda x: x[1])
            
            mom_surf=sort_list[0][0]
            dad_surf=sort_list[1][0]
            
            child_vertices=procreate_vertices(mom_surf.vertices, 
                                              dad_surf.vertices,
                                              tolx, 
                                              toly)
            
            tri = Delaunay(child_vertices[:,0:2])
            child_faces=tri.simplices     
            new_population[key]=mg.Mesh(vertices=child_vertices, 
                                        faces=child_faces) 
            
            key+=1
    
    return new_population
        

    






