import Genetic_Algorithim.GenAlgth as ga
import Initial_Population_Creation.VarZ_initpop_Gen as vzipg
import Mesh_Generation.DomainDef as dodef
import Fitness_Function.fit_calc2 as fitc
import Fitness_Function.pvpowlib.geoloc as pvgeo
import Fitness_Function.fit_calc as fitcc
import pandas as pd
import numpy as np
import time as tm
#import ray 
import random, logging
#ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
#from numba import njit
# import Mesh_Generation.MeshGen as mg
# from scipy.spatial import Delaunay


#%%                      DEFINITION OF SITE AND TIME OBJECTS
t1=tm.time()


Magangue=pvgeo.Site(name='Magangué', 
                    lat=9.25055, 
                    long=-74.7661, 
                    alt=19, 
                    UTC='-05:00:00' )


time=pvgeo.Time(start_date='2021-01-01', 
                end_date='2021-03-30',
                start_hms='06:00:00',
                end_hms='18:00:00',
                time_interval='30-min', 
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



#%%           CALCULATION OF THE FITNESS SCORE OF EACH SURFACE AND SORTING FOR MULTIPLE GENERATIONS

generations={}
t1=tm.time()

for i in range(5):
    
    surf_ranking=fitcc.rank_surf_population(population, sunvec)
    
    generations[i]=(population, surf_ranking)
    
    population=ga.evolve(population, surf_ranking)
    
    print(i)
    print(tm.time()-t1)

dt1=tm.time()-t1


#%%           CALCULATION OF THE FITNESS SCORE OF EACH SURFACE AND SORTING FOR MULTIPLE GENERATIONS

generations2={}
t2=tm.time()
population=vzipg.create_init_pop(xy_plane)

for i in range(5):

    futures=[(surf,fitc.power_per_surf2(population[surf].areas, 
                                  population[surf].normals, 
                                  sunvec)) for surf in population]
    
    #surf_ranking=ray.get(futures)
    surf_ranking=futures
    surf_ranking=sorted(surf_ranking, key=lambda x: x[1], reverse=1)
    
    generations2[i]=(population, surf_ranking)
    population=ga.evolve(population, surf_ranking)
    
dt2=tm.time()-t2

#%%                EXPORT BEST SURFACE

# best_mesh=population[surf_ranking[0][0]]

# best_vertices=pd.DataFrame(best_mesh.vertices)
# best_faces=pd.DataFrame(best_mesh.faces)

# best_vertices.to_excel(r'C:/Users/andre/Desktop/Eafit/8vo Semestre/Monitoría Panel Solar/OptiSurf/Fitness_Function/best_vertices.xlsx', index = False)
# best_faces.to_excel(r'C:/Users/andre/Desktop/Eafit/8vo Semestre/Monitoría Panel Solar/OptiSurf/Fitness_Function/best_faces.xlsx', index = False)

