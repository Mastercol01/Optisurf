import Surfaces_Visualization.surface_vis as surfvis
import Mesh_Generation.DomainDef as dodef
from scipy.spatial import Delaunay
from numba import njit
import numpy as np
import quaternion

#%%
  
def create_mesh(x_vals, y_vals, z_vals):
    
    vertices={}
    
    params=dodef.extract_rect_params(x_vals, y_vals)
    x_dim, y_dim, dx, dy=params

    vertex=0
    for y in y_vals:
        for x in x_vals:
            i,j=dodef.rect_c2i(x, y, x_dim, y_dim, dx, dy)
            vertices[vertex]=(x,y,z_vals[i,j])
            vertex+=1
            
    if(z_vals.min()<0):
        vertices_copy=vertices.copy()
        for vertex in vertices:
            if(vertices[vertex][2]<0):
                del vertices_copy[vertex]
                
        vertices=vertices_copy.copy()
#        del vertices_copy     
    
    vertices_array=np.zeros([len(vertices),3])
    
    new_vertex=-1
    for vertex in vertices:
        new_vertex+=1
        vertices_array[new_vertex, :]=vertices[vertex]
        
    vertices=vertices_array.copy()
#    del vertices_array
    
    
    tri = Delaunay(vertices[:,0:2])
    faces=tri.simplices

    return vertices, faces
 
  
   
def rotate_point(point_coords, rot_axis, angle):
    
    
#    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.deg2rad(angle)/2
    
    q=np.quaternion(np.cos(angle),
                     np.sin(angle)*rot_axis[0],
                     np.sin(angle)*rot_axis[1],
                     np.sin(angle)*rot_axis[2])
    
    
    qc=np.quaternion(np.cos(angle),
                     -np.sin(angle)*rot_axis[0],
                     -np.sin(angle)*rot_axis[1],
                     -np.sin(angle)*rot_axis[2])
    
    p1=np.quaternion(0, point_coords[0], 
                    point_coords[1], point_coords[2])
    
    p2=q*p1*qc

    p2=quaternion.as_float_array(p2)[1:]
    p2=p2.round(9)
    
    return p2
    

def rotate_mesh(vertices, rot_axis, angle):
    
    i=-1
    for vertex in vertices:
        i+=1
        vertices[i,:]=rotate_point(vertex, rot_axis, angle)
        
    min_z=vertices[:,-1].min()
    if(min_z<0):
        vertices[:,-1]+=abs(min_z)
        
    return vertices

@njit
def calc_mesh_area_normals(vertices, faces):
    
    areas=np.zeros(len(faces), np.float64)
    normals=np.zeros((len(faces), 3), np.float64)
    
    counter=-1
    for face in faces:
        
        counter+=1
    
        p0=vertices[face[0],:]
        p1=vertices[face[1],:]
        p2=vertices[face[2],:]

        v0=p1-p0
        v1=p2-p0
        
        area_normal=0.5*np.cross(v0,v1)
        
        area=np.linalg.norm(area_normal)
        normal=area_normal/area
        
        areas[counter]=area
        normals[counter,:]=normal
        
    return areas, normals
        

class Mesh:
    
    def __init__(self, x_vals=None, y_vals=None, z_vals=None, vertices=None, faces=None):
        
        if(vertices is None and faces is None):
            self.vertices, self.faces= create_mesh(x_vals, y_vals, z_vals)
        else:
            self.vertices=vertices
            self.faces=faces
            
        self.areas, self.normals= calc_mesh_area_normals(self.vertices, self.faces)
        
    def rotate(self, rot_axis, angle):
        
        self.vertices= rotate_mesh(self.vertices, rot_axis, angle)
        self.areas, self.normals= calc_mesh_area_normals(self.vertices, self.faces)
        
    def visualize(self, axis_view=(25,30)):
        surfvis.vismesh1(self.vertices, 
                         self.faces, 
                         xlabel='X', 
                         ylabel='Y', 
                         zlabel='Z', 
                         colors=None, 
                         axis_view=axis_view)
        
    




