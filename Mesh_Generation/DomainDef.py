import numpy as np
#import time as tm
from numba import njit
from numba.types import float64, uint64


class RectDomain:
    
    def __init__(self, dx, dy, dims):
        
        self.dx = dx
        self.dy = dy
        
        self.dims=dims
        self.x_dim=dims[0]
        self.y_dim=dims[1]
        self.z_dim=dims[2]
        
        self.j_dim=int(dims[0]/dx)+1
        self.i_dim=int(dims[1]/dx)+1
        self.nodes_dims=(self.i_dim, self.j_dim)
        self.num_vertices=dims[0]*dims[1]
        
        self.x_vals=np.linspace(-dims[0]/2, dims[0]/2, self.j_dim)
        self.y_vals=np.linspace(-dims[1]/2, dims[1]/2, self.i_dim)
        
        def i2c(self, i, j):
            y=-self.y_dim/2 +i*self.dy
            x=-self.x_dim/2 +j*self.dx
            return x,y
        
        def c2i(self, x, y):
           j=round( (x + self.x_dim/2 )/self.dx )
           i=round( (y + self.y_dim/2 )/self.dy )
           return i,j 
            





@njit(float64(float64, uint64))
def njit_round(x, n):
    x=np.array([x])
    out = np.empty_like(x)
    return np.round(x, n, out)[0]
    
    
@njit((uint64, uint64, float64, float64, float64, float64))
def rect_i2c(i, j, x_dim, y_dim, dx, dy):
    y=-y_dim/2 +i*dy
    x=-x_dim/2 +j*dx
    return x,y
    
@njit((float64, float64, float64, float64, float64, float64))
def rect_c2i(x, y, x_dim, y_dim, dx, dy):
    
    j=njit_round((x + x_dim/2 )/dx, 0)
    i=njit_round((y + y_dim/2 )/dy, 0)
           
    return int(i), int(j)


@njit((float64[:], float64[:]))
def extract_rect_params(x_vals, y_vals):
    x_dim=max(x_vals)-min(x_vals)
    y_dim=max(y_vals)-min(y_vals)
        
    dx=njit_round(x_vals[1]-x_vals[0], 10)
    dy=njit_round(y_vals[1]-y_vals[0], 10)
    
    return [x_dim, y_dim, dx, dy]



@njit((float64,float64,float64,float64, float64[:,:]))
def rect_c2v(x,y,tolx,toly,vertices):
    
    x_min, x_max= x-tolx, x+tolx
    y_min, y_max= y-toly, y+toly
    
    vertex0=set(np.where( np.logical_and( vertices>x_min , vertices<x_max) )[0])
    vertex1=set(np.where( np.logical_and( vertices>y_min , vertices<y_max) )[0])
    
    vertex_num=list(vertex0.intersection(vertex1))
    
    return vertex_num

#%%
# x=np.linspace(-1,1,20)
# y=np.linspace(-1,1,20)
# x_dim=y_dim=1
# dx=0.01
# dy=0.01
# x1=0.5
# y1=0.5
# b=extract_rect_params(x,y)
# a=rect_c2i(x1, y1, x_dim, y_dim, dx, dy)
#a=njit_round(3.6,0)
# t1=tm.time()
# rect_i2c(10, 10, 1, 1, 0.01, 0.01)
# t2=tm.time()

# print((t2-t1)*1000)

