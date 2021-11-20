import Mesh_Generation.DomainDef as dodef
import numpy as np
from numba import njit


#%%                DEFINITION OF BASE SURFACE-BUILDING FUNCTIONS


#@njit
def build_plane(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    
    z_vals=np.ones((len(x_vals), len(y_vals)))*0.5
    return z_vals

@njit
def build_double_sine(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    
   z_vals=np.zeros((len(x_vals), len(y_vals)))
   for x in x_vals:
       for y in y_vals:
           i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
           z_vals[i,j]=ampl *np.sin(ang_freq*x/(2*np.pi)) \
                            *np.sin(ang_freq*y/(2*np.pi)) + ampl
   return z_vals


@njit
def build_convex_sphere1(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    height=-np.sqrt( rad**2 - x_dim**2/4 -y_dim**2/4 )
    z_vals=np.zeros((len(x_vals), len(y_vals)))
    
    for x in x_vals:
        for y in y_vals:
            i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
            z_vals[i,j]=np.sqrt(rad**2-x**2-y**2)+height
    return z_vals      

#@njit
def build_convex_sphere2(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    rad=min(x_dim, y_dim)/2
    z_vals=np.zeros((len(x_vals), len(y_vals)))
    
    for x in x_vals:
        for y in y_vals:
            i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
            z_vals[i,j]=np.sqrt(rad**2-x**2-y**2)
            if(np.isnan(z_vals[i,j])):
                z_vals[i,j]=-1
    return z_vals 
      
@njit    
def build_concave_sphere1(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    z_vals=np.zeros((len(x_vals), len(y_vals)))
    for x in x_vals:
        for y in y_vals:
            i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
            z_vals[i,j]=-np.sqrt(rad**2-x**2-y**2) + rad
    return z_vals

#@njit
def build_concave_sphere2(x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl):
    rad=min(x_dim, y_dim)/2
    z_vals=np.zeros((len(x_vals), len(y_vals)))
    for x in x_vals:
        for y in y_vals:
            i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
            z_vals[i,j]=-np.sqrt(rad**2-x**2-y**2) + rad
            if(np.isnan(z_vals[i,j])):
                z_vals[i,j]=-1
    return z_vals


@njit
def calc_rot_arrays(length):
    rot_arrays=np.zeros((length, 3))
    angles=np.linspace(0, 2*np.pi, length)
    
    for i in range(length):
        rot_arrays[i,0]=np.cos(angles[i])
        rot_arrays[i,1]=np.sin(angles[i])
        
    return rot_arrays
        
    
    







#%%

# xy_plane=dodef.RectDomain(dx=0.1, dy=0.1, dims=(1,1,1))

# normal=np.float64(np.array([0,0.364,1]))
# x_vals=xy_plane.x_vals
# y_vals=xy_plane.y_vals
# x_dim=xy_plane.x_dim
# y_dim=xy_plane.y_dim
# dx=xy_plane.dx
# dy=xy_plane.dy
# rad=6.5
# ang_freq=50 
# ampl=1




# dispatcher={0: build_plane,
#             1: build_double_sine,
#             2: build_convex_sphere1,
#             3: build_convex_sphere2,        
#             4: build_concave_sphere1, 
#             5: build_concave_sphere2 }


# z_vals=dispatcher[5](x_vals, y_vals, x_dim, y_dim, dx, dy, rad, ang_freq, ampl)


# surfvis.quickplot3D(x_vals, y_vals, z_vals)

# mesh1=mg.Mesh(x_vals, y_vals, z_vals)

# mesh1.rotate([1,1,0],20)

# mesh1.visualize()

