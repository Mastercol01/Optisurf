#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions and classes related to construction and 
manipulation of a surface mesh.

"""

#%%                 IMPORTATION OF LIBRARIES

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import art3d
import Surface_Modelling.self_shading as shading


#%%                  DEFINITION OF FUNCTIONS

def create_mesh(X, Y, Z, mode=0):
    
    """
    Function for creating a mesh from a point cloud in a rectangular domain.
    
    Parameters
    ----------
    
    X : 2D numpy.array of floats
        2D array of x-values defined over the rectangular domain (i.e, x
        meshgrid values).
        
    Y : 2D numpy.array of floats
        2D array of y-values defined over the rectangular domain (i.e, y
        meshgrid values).
        
    Z : 2D numpy.array of floats
        2D array of values defined over the rectangular domain. Each entry
        of this array is the z coordinate of a point defined in the xy plane.
        
    mode : int
        It defines what is to be done with points which have negative
        z-coordinates. If mode equals 0 (default), nothing special happens,
        the point cloud is meshed normally. If mode equals 1, the point 
        cloud is elevated such that there are no longer points with
        negative z-coordinates. If mode equals 2 we drop all points with
        negative z-coordinates.
        
        
    Returns
    --------

    vertices : 2D numpy.array of floats
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points in ther mesh). The number of each row
        defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    faces : 2D numpy.array of ints
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the number of the verticies the face is made 
        out of (the faces are triangular elements). For example:
        faces[6,:] = [1,20,17] are the vertices which make up the face number 4.
        The order here is important, the conventioned followed here is that of
        a right hand rule, thus, in face , vertex 1 is connected to vertex 20,
        vertex 20 is connected to vertex 17 and vertex 17 is connected to vertex
        1.
        
    
    """

    # We store the points (vertices) that make up the point cloud as a list,
    # associating each point's z coordinate with its corresponding x, y
    # coordinate. 
    vertices = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    # If mode equals 1, we elevate the surface so there are no points with
    # negative z-coordinates.
    if mode == 1:
        Zmin = Z.min()
        if Zmin < 0: 
            vertices[:,2] += abs(Zmin)
        
    # If mode equals 2, we drop all points with negative z-coordinates.
    elif mode == 2:
        vertices = vertices[vertices[:,2]>=0]

    # We use Delaunay triangulation in order to mesh all points/vertices toguether.
    tri_ = Delaunay(vertices[:,0:2])
    faces = tri_.simplices
    
    return vertices, faces




def rotate_point(point_coords, rot_axis, angle):
    
    """
    This function takes in the coordinates of a point, the components of a
    vector indicating a rotation axis and an angle of rotation and then
    returns the coordinates of the rotated point around the specified 
    axis.
    
    Parameters
    ----------
    
    point_coords : list-like
        List, tuple or array of 3 floats, indicating the coordinates of the point
        that one wants to rotate. It should have the form: (x,y,z).
        
    rot_axis : list-like
        List, tuple or array of 3 floats, indicating the components of a vector 
        parallel to the wanted rotation axis. It should have the form:
        (vx,vy,vz). Components should be normalized to unity.
        
    angle : float
        Angle in sexagesimal degrees specifying by how much one whishes to
        rotate the point by the specified axis.
    
    
    """
    
    # We convert the angle to radians and divide by two because of quaternion 
    # reasosns.
    angle    = np.deg2rad(angle)
    r_vec    = np.array(point_coords).astype(float)
    rot_axis = np.array(rot_axis).astype(float)
    
    r_vec_parallel = np.dot(r_vec, rot_axis)*rot_axis
    r_vec_perpendicular = r_vec - r_vec_parallel 
    
    rotated_r_vec  = r_vec_parallel 
    rotated_r_vec += r_vec_perpendicular*np.cos(angle)    
    rotated_r_vec += np.cross(rot_axis, r_vec_perpendicular)*np.sin(angle) 
    
    return rotated_r_vec
    



def rotate_mesh(vertices, rot_axis, angle, elevate = False):
    
    """
    This function rotates an entire mesh of points around an specified rotation
    axis by the specified angle.
    
    Parameters
    ----------
    
    vertices : 2D numpy.array of floats
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points in ther mesh). The number of each row
        defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    rot_axis : list-like
        List, tuple or array of 3 floats, indicating the components of a vector 
        parallel to the wanted rotation axis. It should have the form:
        (vx,vy,vz).
        
    angle : float
        Angle in sexagesimal degrees specifying by how much one whishes to 
        the specified point by the specified axis.
        
    elevate : bool
        If True, in the case that after the rotation some points of the mesh
        have negative z-coordinates, then all vertices of the mesh are elevated 
        such that there are no longer points with negative z-coordinates.
        
    Returns
    -------
    
    vertices : 2D numpy.array of floats
        Verticies array but rotated.
    
    
    """
    
    
    vertices_parallel = (vertices*rot_axis).sum(axis=1)
    vertices_parallel = vertices_parallel.reshape(len(vertices_parallel), 1)
    vertices_parallel = vertices_parallel*rot_axis.reshape(1,3)
    vertices_perpendicular = vertices - vertices_parallel
    
    rotated_vertices  = vertices_parallel
    rotated_vertices += vertices_perpendicular*np.cos(angle)
    rotated_vertices += np.cross(rot_axis, vertices_perpendicular)*np.sin(angle)      
    
    if elevate:
        rotated_vertices[:,2] -= rotated_vertices[:,2].min()
        

    return rotated_vertices




def calc_mesh_area_normals(vertices, faces):
    
    """
    Function for calculating the normal vectors and areas of each the 
    faces of a mesh.
    
    Parameters
    ----------
    
    vertices : 2D numpy.array of floats
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points in ther mesh). The number of each row
        defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    faces : 2D numpy.array of ints
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the number of the verticies the face is made 
        out of (the faces are triangular elements). For example:
        faces[6,:] = [1,20,17] are the vertices which make up the face number 4.
        The order here is important, the conventioned followed here is that of
        a right hand rule, thus, in face , vertex 1 is connected to vertex 20,
        vertex 20 is connected to vertex 17 and vertex 17 is connected to vertex
        1.
        
    Returns
    -------
    
    areas : 1D numpy.array
         Array of areas of faces. Each row's position specifies the number of 
         the face each area belongs to. E.g: areas[3] = 0.0123 specifies the
         area of the face of number 3.
        
    
    normals : 2D numpy.array
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the coordinates of the normal vector to
        each face of each. For example: normals[6,:] = [0.45,0.5,0.643]
        indicates the coordinates of the normal vector to the face number 6.
    
    """
    
    # Position vectors of the points that make up each face.
    A = vertices[faces[:, 0], :]
    B = vertices[faces[:, 1], :]
    C = vertices[faces[:, 2], :]
    
    # Vectors of 2 of the sides that make up each face.
    E1 = B - A
    E2 = C - A
    
    # Normal of each face scaled by their area.
    area_normals = 0.5*np.cross(E1, E2)
    
    # Area of each face.
    areas = np.linalg.norm(area_normals, axis=1)
    
    # Normal vector to each face.
    normals = np.zeros(area_normals.shape)
    normals[:,0] = area_normals[:,0]/areas
    normals[:,1] = area_normals[:,1]/areas
    normals[:,2] = area_normals[:,2]/areas
    
    return areas, normals



def calc_curvature(Zx, Zy, Zxx, Zyy, Zxy):
    
    """
    Computes, numerically, the Mean and Gaussian curvatures of a surface of the
    form Z = f(x,y), evaluated over the domain.
    
    Parameters
    ----------
    
    Zx : 2D numpy.array of floats
        First derivative with respect to x of the function Z, evaluated over
        the domain.
        
    Zy : 2D numpy.array of floats
        First derivative with respect to y of the function Z, evaluated over
        the domain.
        
    Zxx : 2D numpy.array of floats
        Second derivative with respect to x of the function Z, evaluated over
        the domain.
        
    Zyy : 2D numpy.array of floats
        Second derivative with respect to y of the function Z, evaluated over
        the domain.
        
    Zxy : 2D numpy.array of floats
        Mixed derivative with respect to x and y of the function Z, evaluated 
        over the domain.
    
        
    Returns
    -------
    
    H : 2D numpy.array of floats
        Mean curvature of the polynomial, evaluated over the domain.
        
    K : 2D numpy.array of floats
        Gaussian curvature of the polynomial, evaluated over the domain.
        
    """
    
    # We compute the Mean (H) and Gaussian (K) curvatures using the 
    # 'Monge patch' formula. See: https://mathworld.wolfram.com/MongePatch.html
    # This formula is just a special case of the general case we already know, 
    # when r_vec = [x, y, f(x,y)].
    
    K = (Zxx*Zyy - Zxy**2)/(1 + Zx**2 + Zy**2)**2
    
    num_H = (1 + Zy**2)*Zxx - 2*Zx*Zy*Zxy + (1 + Zx**2)*Zyy
    dem_H = 2*(1 + Zx**2 + Zy**2)**1.5
    H = num_H/dem_H
    
    return H, K



def vismesh(vertices, faces, facevalues = None, config = None):
    
    """
    Visualize surface mesh.
    
    Parameters
    ----------
    vertices : 2D numpy.array of floats
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points the mesh is made out of). The number of each
        row defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    faces : 2D numpy.array of ints
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the number of the verticies the face is made 
        out of (the faces are triangular elements). For example:
        faces[6,:] = [1,20,17] are the vertices which make up the face number 4.
        The order here is important, the conventioned followed here is that of
        the right hand rule, thus, in face , vertex 1 is connected to vertex 20,
        vertex 20 is connected to vertex 17 and vertex 17 is connected to vertex
        1; in that order.
        
    facevalues : numpy.array of floats or None
        If None (the default), it is not used. Else, it must be a numpy array 
        with length f, where f is the number of faces/elements. It stores the
        numeric values of a scalar field, defined over the entire meshed surface, 
        associated with each faces f, of said surface. 
        
    config : dict or None
        Dict of plot configuration options. If None (the default), it uses
        the default confifuration plot options. If dict, it should include
        one or more of the following key-value pairs:
            
        Keys : Values
        -------------
        "title" : str or None
            Title of plot. Default is None.
            
        "cbar_title" : str or None
            Title of colorbar (only applies if 'facevalues' is not None).
            Default is None.
            
        "cbar_label" : str or None
            label of colorbar (only applies if 'facevalues' is not None).
            Default is None.
            
        "xlabel" : str or None
            X-axis label of plot. Default is: 'X [m] (↑ == N, ↓ == S)'.
            
        "ylabel" : str or None
            Y-axis label of plot. Default is: 'Y [m] (↑ == E, ↓ == W)'.
            
        "zlabel" : str or None
            Z-axis label of plot. Default is: 'Z [m]'.
            
        "figsize" : 2-tuple of int
            Figure size. Default is (13,13).
            
        "edges" : bool
            If True, draw face edges of meshed surface. If False, don't draw
            them. Default is False.
            
        "xlims" : 2-tuple of float
            Tuple containing the x-limits bounds (lower_bound, upper_bound).
            Default is (vertices[:,0].min(), vertices[:,0].max()).
        
        "ylims" : 2-tuple of float
            Tuple containing the y-limits bounds (lower_bound, upper_bound).
            Default is (vertices[:,1].min(), vertices[:,1].max()).
        
        "zlims" : 2-tuple of float
            Tuple containing the z-limits bounds (lower_bound, upper_bound).
            Default is (vertices[:,2].min(), vertices[:,2].max()).
            
        "vmin" : float or None
            Minimum value for the colorbar (only applies if 'facevalues' 
            is not None). Default is None.
            
        "vmax" : float or None
            Maximum value for the colorbar (only applies if 'facevalues' 
            is not None). Default is None.
            
        "figsize" : 2-tuple of int
            Figure size of plot.
            
        "axis_view" : 2-tuple of int
            Elevation, azimuth of plot camara in degrees. Default is (25, 30).
            
        "cmap_name" : str
            Name of the matplotlib cmap to use in order to plot 'facevalues'
            (this only applies if facevalues is not None). Default is 'hot'.
            
        "show_plot" : bool
            If True, it means that show plot. If False, don't show plot inmediately. 
            
    Returns
    -------
    None
    
    """
    
    # Default plot configuration.
    config_ = {"title"     : None,
               "cbar_title": None,
               "cbar_label": None,
               "xlabel"    : 'X [m] (↑ == N, ↓ == S)',
               "ylabel"    : 'Y [m] (↑ == E, ↓ == W)',
               "zlabel"    : 'Z [m]',
               "edges"     : True,
               "xlims"     : None,
               "ylims"     : None,
               "zlims"     : None, 
               "vmin"      : None,
               "vmax"      : None,
               "axis_view" : (25, 30),
               "figsize"   : (16, 16),
               "cmap_name" : 'hot'}
    
    # User configuration overwrites default configuration.
    if isinstance(config, dict):
        for key, val in config.items():
            config_[key] = val
                
            
    # Initialize figure.
    fig = plt.figure(figsize = config_["figsize"])
    ax = fig.add_subplot(projection="3d")
    # Plot meshed surface and scalar field defined over surface.
    if facevalues is not None:
        norm                 = plt.Normalize(config_["vmin"] , config_["vmax"] )
        cmap                 = plt.get_cmap(name = config_["cmap_name"])
        norm_facevalues_cmap = cmap(norm(facevalues))
        
        # Draw edges.
        if config_["edges"]:
            pc = art3d.Poly3DCollection(vertices[faces], 
                                        facecolors = norm_facevalues_cmap,
                                        edgecolor = "black")
        # Don't Draw edges.
        else:
            pc = art3d.Poly3DCollection(vertices[faces], 
                                        facecolors = norm_facevalues_cmap)
            
        # Initialize scalar field colorbar.
        scalar_mappable = plt.cm.ScalarMappable(cmap = cmap)        
        scalar_mappable.set_array(facevalues)
        cbar = plt.colorbar(scalar_mappable, ax = ax) 
        
        # Set scalar field colorbar titles and labels.
        cbar.ax.set_title(config_["cbar_title"])
        cbar.set_label(config_["cbar_label"], rotation=270)
       
        
    # Plot meshed surface.
    else:
        # Draw edges.
        if config_["edges"]:
            pc = art3d.Poly3DCollection(vertices[faces], 
                                        edgecolor = "black")
        # Don't Draw edges.
        else:
            pc = art3d.Poly3DCollection(vertices[faces])
            
    ax.add_collection(pc)  
    
    
      
    # Set axis limits.
    if config_["xlims"] is None:
        min_x = vertices[:,0].min()
        max_x = vertices[:,0].max()
        ax.set_xlim(min_x, max_x)
    else:
        ax.set_xlim(config_["xlims"])
        
    if config_["ylims"] is None:
        min_y = vertices[:,1].min()
        max_y = vertices[:,1].max()
        ax.set_ylim(min_y, max_y)
    else:
        ax.set_ylim(config_["ylims"])
        
    if config_["zlims"] is None:
        min_z = vertices[:,2].min()
        max_z = vertices[:,2].max()
        ax.set_zlim(min_z, max_z)
    else:
        ax.set_zlim(config_["zlims"])
        
    # Set titles and labels.
    ax.set_title( config_["title"])
    ax.set_xlabel(config_["xlabel"])
    ax.set_ylabel(config_["ylabel"])
    ax.set_zlabel(config_["zlabel"])
    ax.view_init(elev = config_["axis_view"][0], 
                 azim = config_["axis_view"][1])
    
    
    plt.show()
    
    
    return None
    



def read_stl(path, dec=6, scale_factor=1):
    
    """
    Read .stl file of a surface Mesh obj and obtain the vertices and faces
    arrays for remeshing.
    
    Parameters
    ----------
    
    path : path-str
        Path of the .stl file to be read.
        
    dec: int or None
        Number of decimals for rounding. If int, must be positive; default is 6.
        If None, no rounding is performed.
        
    scale_factor : float
        Number by which to scale the vertex coordinates of the file. Default
        is 1 (i.e, no scaling).
        
    Returns
    -------
    vertices : 2D numpy.array
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points the mesh is made out of). The number of each
        row defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    faces : 2D numpy.array
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the number of the verticies the face is made 
        out of (the faces are triangular elements). For example:
        faces[6,:] = [1,20,17] are the vertices which make up the face number 4.
        The order here is important, the conventioned followed here is that of
        the right hand rule, thus, in face , vertex 1 is connected to vertex 20,
        vertex 20 is connected to vertex 17 and vertex 17 is connected to vertex
        1; in that order.
        
        
    """
    
    
    # We open the file, read it as a str and split said string into a list
    # of substrings, each contaning the info of each face. We then close 
    # the file
    file = open(path, "r")
    file_str = file.read()
    file_str_list = file_str.split("facet normal")[1:]
    file.close()
    
    # The length of file_str_list is equal to the number of faces in the mesh.
    # We also initialize the arrays where we will store the unit normals and 
    # the face points.
    num_faces = len(file_str_list)
    N = np.zeros((num_faces, 3))
    A = np.zeros((num_faces, 3))
    B = np.zeros((num_faces, 3))
    C = np.zeros((num_faces, 3))
    
    # We loop over each element of file_str_list and we extract the info 
    # corresponding to the unit normals and points of each face.
    for nf, face_str in enumerate(file_str_list):
        
        split_face_str = face_str.split("\n")
        
        # Unit normal components of face.
        Nnf = split_face_str[0].strip().split(" ")
        Nnf = [float(i) for i in Nnf]
        
        # Coordinates of first point in the triangle element/face.
        Anf = split_face_str[2].strip().split("vertex")[-1].split()
        Anf = [float(i) for i in Anf]
        
        # Coordinates of second point in the triangle element/face.
        Bnf = split_face_str[3].strip().split("vertex")[-1].split()
        Bnf = [float(i) for i in Bnf]
        
        # Coordinates of third point in the triangle element/face.
        Cnf = split_face_str[4].strip().split("vertex")[-1].split()
        Cnf = [float(i) for i in Cnf]
        
        N[nf,:] = Nnf
        A[nf,:] = Anf
        B[nf,:] = Bnf
        C[nf,:] = Cnf
        
    # We scale the point coordinates and round all stored numbers to the value 
    # specified.
    A *= scale_factor
    B *= scale_factor  
    C *= scale_factor  
    
    if dec is not None:
        N = np.around(N, dec)
        A = np.around(A, dec)
        B = np.around(B, dec)
        C = np.around(C, dec)
    

    # We construct the vertices array by vertically stacking the A,B,C arrays
    # and eliminating all repeated rows. In this way, we end up only with a list
    # of all the unique points.
    vertices = np.unique(np.vstack((A, B, C)), axis=0)
    faces = np.zeros((num_faces, 3)).astype(int)
    
    # Vertex by vertex we figure out in which faces each vertex appears, in 
    # order to construct the faces array, which details the conectivity
    # of each face with each vertex.
    for i, vertex in enumerate(vertices):
        
        condA = (A == vertex).all(axis=1)
        condB = (B == vertex).all(axis=1)
        condC = (C == vertex).all(axis=1)
        
        if condA.any():
            faces[:, 0] += i*condA.astype(int)
        
        if condB.any():
            faces[:, 1] += i*condB.astype(int)
            
        if condC.any():
            faces[:, 2] += i*condC.astype(int)
            
    return vertices, faces

    




#%%                  DEFINITION OF CLASSSES

class Mesh:
    
    """
    This class contains all info and methods to create a surface Mesh of an
    explicit function.
    
    Parameters
    ----------
    
    
    X : 2D numpy.array of floats
        2D array of x-values defined over the rectangular domain (i.e, x
        meshgrid values).
        
    Y : 2D numpy.array of floats
        2D array of y-values defined over the rectangular domain (i.e, y
        meshgrid values).
        
    Z : 2D numpy.array of floats
        2D array of values defined over the rectangular domain. Each entry
        of this array is the z coordinate of a point defined in the xy plane.
        
    mode : int
        It defines what is to be done with points which have negative
        z-coordinates. If mode equals 0 (default), nothing special happens,
        the point cloud is meshed normally. If mode equals 1, the point 
        cloud is elevated such that there are no longer points with
        negative z-coordinates. If mode equals 2 we drop all points with
        negative z-coordinates.


    vertices : 2D numpy.array of floats
        Numpy array with dimensions vx3, where v is the number of vertices
        (i.e, the number of points the mesh is made out of). The number of each
        row defines the number of each vertex, while the 3 entries hold the 
        x, y and z coordinates of the vertex in question. For example:
        vertices[4,:] = [0.1,-0.2, 0.375] are the x,y,z coordinates of the 
        vertex number 4. 
    
    faces : 2D numpy.array of ints
        Numpy array with dimensions fx3, where f is the number of 
        faces/elements. The number of each row defines the number of each face,
        while the 3 entries hold the number of the verticies the face is made 
        out of (the faces are triangular elements). For example:
        faces[6,:] = [1,20,17] are the vertices which make up the face number 4.
        The order here is important, the conventioned followed here is that of
        the right hand rule, thus, in face , vertex 1 is connected to vertex 20,
        vertex 20 is connected to vertex 17 and vertex 17 is connected to vertex
        1; in that order.
        
   Raises
   ------
   1) Exception : "All entries cannot be None"
   2) Exception: "Incorrect set of information inputted. To \
                   initialize an instance of class 'Mesh', input\
                   either 'x_vals', 'y_vals','z_vals' or input \
                   'vertices', 'faces'."
     
    
    
    """
    
    
    def __init__(self, X=None, Y=None, Z=None, vertices=None, faces=None, mode=0):
        
        """
        Constructor function. 
        """
        
        cklist = [X, Y, Z, vertices, faces]
        cklist = [x is None for x in cklist]
        
        if all(cklist):
           raise Exception("All entries cannot be None")
           
        elif (not all(cklist[:-2])) and all(cklist[3:]) :
            
            self.vertices, self.faces = create_mesh(X, Y, Z, mode = mode)
                
        elif all(cklist[:-2]) and (not all(cklist[3:])):
            
            self.vertices = vertices
            self.faces = faces
            
        else: 
            raise Exception("Incorrect set of information inputted. To \
                            initialize an instance of class 'Mesh', input\
                            either 'X', 'Y','Z' or input 'vertices', 'faces'.")
            
            
        # We store the number of vertices and faces of the mesh.
        self.num_faces    = len(self.faces)    
        self.num_vertices = len(self.vertices)
        
        # We compute the list of indexes for the faces and the vertices.
        self.list_faces = np.arange(self.num_faces)
        self.list_vertices = np.arange(self.num_vertices)

        # We compute the areas and normals of the mesh.
        self.areas, self.normals =\
        calc_mesh_area_normals(self.vertices, self.faces)
        self.total_area = self.areas.sum()
            
        # We compute the position vectors of each of the points forming each
        # face.
        self.A = self.vertices[self.faces[:,0],:]
        self.B = self.vertices[self.faces[:,1],:]
        self.C = self.vertices[self.faces[:,2],:]
        
        # We compute the barycenters of each face.
        self.face_centers = (self.A + self.B + self.C)/3

        # We compute the meshgrid representation of the surface.
        self.sqlen = round(np.sqrt(len(self.vertices)))
        self.X = self.vertices[:,0].reshape((self.sqlen, self.sqlen))
        self.Y = self.vertices[:,1].reshape((self.sqlen, self.sqlen))
        self.Z = self.vertices[:,2].reshape((self.sqlen, self.sqlen))
        
        # We compute the x and y spacing.
        self.dx = np.mean(self.X[0, 1:] - self.X[0, :-1])
        self.dy = np.mean(self.Y[1:, 0] - self.Y[:-1, 0])
        
        
        # ---- CREATE DATABASE FOR FACES INFORMATION ----
        self.faces_data = {}
        for face_num in range(self.num_faces):
            self.faces_data[face_num] =\
            {"vertices"        : self.faces[face_num,:],
             "vertices_coords" : self.vertices[self.faces[face_num,:]],
             "center"          : self.face_centers[face_num,:],
             "area"            : self.areas[face_num],
             "unit_normal"     : self.normals[face_num]
             }
            
            
        

        
    def rotate(self, rot_axis, angle, elevate = False):
        
        """
        This function rotates the entire mesh of points around an specified 
        rotation axis by the specified angle.
        
        Parameters
        ----------
        
        rot_axis : list-like
            List, tuple or array of 3 floats, indicating the components of a vector 
            parallel to the wanted rotation axis. It should have the form:
            (vx,vy,vz).
            
        angle : float
            Angle in sexagesimal degrees specifying by how much one whishes to 
            the specified point by the specified axis.
            
        elevate : bool
            IF True, it elevates the rotated mesh so no vertex has a negative z
            coordinate. If Flase, it does nothing. Default is False.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.vertices : 2D numpy.array of floats
            Rotated array of vertices. This new array overwrites the previously
            existing one.
            
        self.areas : 2D numpy.array of floats
            Recomputed array of areas. This new array overwrites the previously
            existing one.
        
        self.normals : 2D numpy.array of floats
            Recomputed array of face normals. This new array overwrites the previously
            existing one.
            
        """
        
        self.vertices = rotate_mesh(self.vertices, rot_axis, angle)
        self.areas, self.normals = calc_mesh_area_normals(self.vertices, 
                                                          self.faces,
                                                          int(elevate))
        
        
    def get_neighbours(self, num, kind="vv"):
        
        """
        This method allows us to compute the immediate spatial neighbours of a
        vertex or a face of the mesh.
        
        Parameters
        ----------
        num : int
            Number of the vertex or face whose neighbours we want to find.
            
        kind : str
            Can be either: "vv", "ff", "vf". In the first case we find 
            the vertices neighbouring the vertex defined by'num'. In the
            second case we find the faces neighboruing the face defined by
            'num'. In the thrid case, we find the faces neighbouring the 
            vertex defined by 'num'.
            
        Returns
        -------
        neighbours : np.array of ints
            Array of numbers, identifying the neighbours.
            
        Notes
        -----
        1) this method is quite computationally expensive for large meshes.
        
        """
        
        
        if(kind=="vv"):
            
            # We get vertices which belong to all the faces which have as 
            # vertex, the vertex defined  by 'num'.
            group0 = self.faces[self.faces[:,0]==num].flatten()
            group1 = self.faces[self.faces[:,1]==num].flatten()
            group2 = self.faces[self.faces[:,2]==num].flatten()
            
            # We perform the set union of each group (this keeps only one copy
            # of any repeated values) and then earrase 'num' from the set.
            neighbours = np.union1d(group0, np.union1d(group1, group2))
            neighbours = np.delete(neighbours, np.where(neighbours == num))
            neighbours = np.array(neighbours).astype(int)
            
            
        elif(kind=="vf"):
            
            # We simply get the index of the faces which contain num as a 
            # vertex.
            group0 = self.list_faces[self.faces[:,0]==num].flatten()
            group1 = self.list_faces[self.faces[:,1]==num].flatten()
            group2 = self.list_faces[self.faces[:,2]==num].flatten()
            
            # We perform the set union of each group,this keeps only one copy
            # of any repeated values.
            neighbours = np.union1d(group0, np.union1d(group1, group2))
            neighbours = np.array(neighbours).astype(int)
            
            
        elif(kind=="ff"):
            
            # We check self.faces row by row to see if said row contains 
            # 2 or more of the vertices that the face defined by num contains.
            logic = np.sum(np.isin(self.faces, self.faces[num, :]), axis=1) > 1
            
            # Then, we get the face indexes which satisfy the aforementioned
            # and errase "num" from de list of neighbourgs.
            neighbours = self.list_faces[logic]
            neighbours = np.delete(neighbours, np.where(neighbours == num))
            
        return neighbours
    
    
    def calc_curvature(self):
        """
        Computes, numerically, the Mean and Gaussian curvatures of the surface
        mesh.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        
        Produces
        --------
        
        self.H : 2D numpy.array of floats
            Mean curvature of the polynomial, evaluated over the domain.
            
        self.K : 2D numpy.array of floats
            Gaussian curvature of the polynomial, evaluated over the domain.

        """
        
        Zy, Zx = np.gradient(self.Z, self.dx, self.dy, edge_order=2)
        Zyx, Zxx = np.gradient(Zx, self.dx, self.dy, edge_order=2)
        Zyy, Zxy = np.gradient(Zy, self.dx, self.dy, edge_order=2)
        
        self.H, self.K = calc_curvature(Zx, Zy, Zxx, Zyy, Zxy)
        
        return self.H, self.K
    

            
        
    def visualize(self, facevalues = None, config = None):
        
        """
        Visualize surface mesh.
        
        Parameters
        ----------

        facevalues : numpy.array of floats or None
            If None (the default), it is not used. Else, it must be a numpy array 
            with length f, where f is the number of faces/elements of the surface. 
            It stores the numeric values of a scalar field, defined over the
            entire meshed surface, associated with each faces f, of said surface. 
            
        config : dict or None
            Dict of plot configuration options. If None (the default), it uses
            the default confifuration plot options. If dict, it should include
            one or more of the following key-value pairs:
                
            Keys : Values
            -------------
            "title" : str or None
                Title of plot. Default is None.
                
            "cbar_title" : str or None
                Title of colorbar (only applies if 'facevalues' is not None).
                Default is None.
                
            "cbar_label" : str or None
                label of colorbar (only applies if 'facevalues' is not None).
                Default is None.
                
            "xlabel" : str or None
                X-axis label of plot. Default is: 'X [m] (↑ == N, ↓ == S)'.
                
            "ylabel" : str or None
                Y-axis label of plot. Default is: 'Y [m] (↑ == E, ↓ == W)'.
                
            "zlabel" : str or None
                Z-axis label of plot. Default is: 'Z [m]'.
                
            "figsize" : 2-tuple of int
                Figure size. Default is (13,13).
                
            "edges" : bool
                If True, draw face edges of meshed surface. If False, don't draw
                them. Default is False.
                
            "xlims" : 2-tuple of float
                Tuple containing the x-limits bounds (lower_bound, upper_bound).
                Default is (vertices[:,0].min(), vertices[:,0].max()).
            
            "ylims" : 2-tuple of float
                Tuple containing the y-limits bounds (lower_bound, upper_bound).
                Default is (vertices[:,1].min(), vertices[:,1].max()).
            
            "zlims" : 2-tuple of float
                Tuple containing the z-limits bounds (lower_bound, upper_bound).
                Default is (vertices[:,2].min(), vertices[:,2].max()).
                
            "figsize" : 2-tuple of int
                Figure size of plot.
                
            "axis_view" : 2-tuple of int
                Elevation, azimuth of plot camara in degrees. Default is (25, 30).
                
            "cmap_name" : str
                Name of the matplotlib cmap to use in order to plot 'facevalues'
                (this only applies if facevalues is not None). Default is 'hot'.
                
        Returns
        -------
        None
        
        Produces
        --------
        None
        
        """

        vismesh(vertices   = self.vertices, 
                faces      = self.faces, 
                facevalues = facevalues,
                config     = config)
        
        
        return None
    
    
    
    def compute_directions_logic(self, dvecs, rad = 0.1, u = 1/3, v = 1/3, lmbda = 3.0): 
        
        """
        Given an array of unit ray directions, compute which faces are able
        to receive radiation from each direction. This method consider's
        self-shading effects as well as incident angle values.
        
        dvecs : numpy.array of floats with shape (ndvec, 3)
            Unit ray-direction vectors. 'dvecs[i,:]' encodes the unit vector, in 
            cartesian coordinates, of the i-th ray/ direction that is to be 
            considered for ray tracing.
            
        rad : float
            Cylinder radius for considering faces. From each ray-facing face center
            imagine an axis passing through that face center having the same
            direction as the ray. We then compute the closest distance from each of the 
            face centers of the surface to each of face centers, in general.
            If this distance is supirior to 'rad', the face element is not
            considered for ray-tracing. If it inferior, it is considered.
            Default is 0.1 .
            
        u : float
            Barycentric u-coordinate of the position used within the each face
            element to determine the origin vector of each ray. Default is 1/3.
        
        v : float
            Barycentric v-coordinate of the position used within the each face
            element to determine the origin vector of each ray. Default is 1/3.
        
        lmbda : float 
            Distance up to which to extend each of the rays. It should be greater
            than the dimensions of the surface. Default is 3.0 .
            
        Returns
        -------
        None
        
        Produces
        --------
        self.ray_facing_logic : numpy.array of bools with shape (self.num_faces, ndvecs)
            'ray_facing_logic[i,j]' tells us whether the i-th face/element of a
            meshed explicit surface could get energy from the j-th direction.
        
           If 'self.ray_facing_logic[i,j]' is False:
               It means that the normal vector of the i-th face toguether with the
               j-th ray-direction vector make an angle greater than 90 degrees. Therefore
               the i-th face cannot recieve energy from the j-th direction.
        
           If 'self.ray_facing_logic[i,j]' is True:
               It means that the normal vector of the i-th face toguether with the
               j-th ray-direction vector make an angle equal or less than 90 degrees.
               Therefore the i-th face can indeed recieve energy from the j-th direction.
            
        self.domain_logic : numpy.array of bools with shape (ndvec, self.num_faces, self.num_faces)
            'self.domain_logic[i,j,k]' tells us whether the k-th face/element of a meshed 
            explicit surface is close enough to a ray, originating from the j-th 
            face/element of that same surface, with direction given by the i-th 
            direction vector, in order to be worth performing ray-tracing on that 
            element.
    
            If 'self.domain_logic[i,j,k]' is False:
                It means that the center of face/element k of the surface is not close 
                enough to a ray originating from the center of face/element j, and with 
                direction i, to merit any computation of ray tracing.
    
            If 'self.domain_logic[i,j,k]' is True:
                It means that the center of face/element k of the surface is close 
                enough to a ray originating from the center of face/element j, and with 
                direction i, to actually justify performing a ray-tracing computation.
            
            
        self.self_shading_logic : numpy.array of bools with shape (self.num_faces, ndvecs)
            'self.self_shading_logic[i,j]' tells whether a ray originating from the face/element i
            and with direction given by the ray-direction vector j, will be cut
            off from its path to the sky by another face element of the surface.
            
            If 'self.self_shading_logic[i,j]' is False:
                It means that a ray originating from the face/element i
                and with direction given by the ray-direction vector j, will indeed
                be cut off from its path to the sky by another face element of the 
                surface.
    
            If 'self.self_shading_logic[i,j]' is True:
                It either means that the face element was not considered in the 
                ray tracing calculation or it can mean that the it was considered
                and a ray originating from said face/element i, with direction a
                given by the ray-direction vector j, will, in fact, not be cut off
                from its path to the sky by another face element of the surface.
                    
                    
        self.directions_logic : numpy.array of bools with shape (self.num_faces, ndvecs)
                'self.directions_logic[i,j]' tells us wether the i-th face is
                able to receive radiant power from direction j.
                
            If 'self.self_shading_logic[i,j]' is False:
                It means that face/element i cannot recieve power or radiant
                energy from direction j. Either because it experiences self
                shading or because the angle of incidence is equal or greater
                than 90 degrees.
    
            If 'self.self_shading_logic[i,j]' is True:
                It means that face/element i can indeed recieve power or radiant
                energy from direction j. Becasue it does not experience self
                shading and because the angle of incidence is less than 90
                degrees.
                
                
        Notes
        -----
        1) For more information on how these method works, please check the
           Surface_Modelling.self_shading module.  
            
        
        """
        
        self.domain_logic, self.ray_facing_logic = \
        shading.reduce_ray_tracing_domain(
        rad          = rad, 
        dvecs        = cp.array(dvecs), 
        list_faces   = cp.array(self.list_faces), 
        face_centers = cp.array(self.face_centers),
        face_normals = cp.array(self.normals)
        )
        
        self.domain_logic, self.ray_facing_logic =\
        self.domain_logic.get(), self.ray_facing_logic.get()
        
        
        self.self_shading_logic =\
        shading.ray_tracing(
        A = self.A, 
        B = self.B,
        C = self.C, 
        dvecs = dvecs,
        logic = self.domain_logic,
        u = u,
        v = v, 
        lmbda = lmbda
        )
        
        self.directions_logic = np.logical_and(self.self_shading_logic,
                                               self.ray_facing_logic)
        
        
        return None
    
    
    
    def compute_absorbed_incident_energy(self, absorbance_function, dvecs, time_integrated_spectral_irradiance_magnitudes, wavelengths):
        
        """
        This is a method computes the absorbed incident energy and absorbed 
        incident spectral energy for each face of a meshed surface, using given
        input parameters. 
        
        Parameters
        ----------
        absorbance_function : callable
            A function that takes in a numpy.array of floats with shape (n,2).
            We call this input numpy array 'eval_pts'. The first column of this
            array contains the angles of incidence, while the second column
            contains the wavelengths for which the fraction of absorbed incident 
            energy is to be calculated.
            
        dvecs : numpy.array of floats with shape (n, 3)
            Unit ray-direction vectors. 'dvecs[i,:]' encodes the unit vector, in 
            cartesian coordinates, of the i-th ray-direction that is to be 
            considered for the computation of the absorption of incident energy.

        time_integrated_spectral_irradiance_magnitudes : numpy.array of floats with shape (n, m)
            Array containing the Magnitude of the Spectral Irradiance vector 
            used for the computation of absorbed spectral energy.
            'time_integrated_spectral_irradiance_magnitudes[i,:]' encodes
            the magnitude of time-integrated irradiance by wavelength for
            direction 'dvecs[i,:]'. Each row should have units of Wh/m^2/nm.
        
        wavelengths : numpy.array of floats with shape (m,) 
            Wavelengths at which the time_integrated_spectral_irradiance_magnitudes
            are defined. It should have units of nanometers.
            
        Returns
        -------
        None
        
        Produces
        --------
        
        self.faces_data[face_num]["absorbed_incident_energy"] : float
            Total energy absorbed by the face 'face_num' from the Sky-Vault.
            It has units of Wh.
            
        self.faces_data[face_num]["absorbed_incident_spectral_energy"] : numpy.array of floats with shape (m,)
            Total energy absorbed by the face 'face_num' from the Sky-Vault,
            discretised by wavelength. It has units of Wh/nm.
            
        self.absorbed_incident_energy : numpy.array of floats with shape (self.num_faces,)
            Array containing the energy absorbed by each face from the Sky-Vault.
            'self.absorbed_incident_energy[face_num]' contains the energy absorbed by
            face 'face_num'. It has units of wh.
            
        self.absorbed_incident_energy : numpy.array of floats with shape (self.num_faces,)
            Array containing the energy absorbed by each face from the Sky-Vault.
            'self.absorbed_incident_energy[face_num]' contains the energy absorbed by
            face 'face_num'. It has units of wh.
            
        self.absorbed_incident_spectral_energy : numpy.array of floats with shape (self.num_faces, m)
            Array containing the energy absorbed by each face from the Sky-Vault,
            discretised by wavelength. 'self.absorbed_incident_energy[face_num,:]' 
            contains the energy absorbed by face 'face_num', for each
            wavelength in wavelengths. It has units of wh/nm.
            
        self.total_absorbed_incident_energy : float
            Total energy absorbed by the whole surface from the Sky-Vault.
            It has units of Wh.
            
        self.total_absorbed_incident_energy : numpy.array of floats with shape (m,)
            Total energy absorbed by the whole surface from the Sky-Vault,
            discretised by wavelngth. It has units of Wh/nm.
            
        self.wavelengths : numpy.array with shape (m,)
            Wavelengths at which all spectral quantites are defined. It
            has units of nanometers.
            
            
        Notes
        -----
        1) This method requires that the 'self.directions_logic' attribute
           already be defined. If it is not, please check the 
           "self.compute_directions_logic" method.
            
            
        """
        
        num_wavelengths = len(wavelengths)
        self.absorbed_incident_energy = np.zeros(self.num_faces)
        self.absorbed_incident_spectral_energy = np.zeros((self.num_faces, num_wavelengths))
        
        for face_num in self.list_faces:
            
            # Get the direction indices from which the current face is
            # able to receive radiant power.
            face_directions_logic = self.directions_logic[face_num,:]
            
            # Count how many directions contribute to the absorbed
            # energy of the current face.
            num_allowed_directions = face_directions_logic.sum()
            
            # If there are no directions that contibute any energy,
            # just skip the current face.
            if num_allowed_directions  == 0:
                self.faces_data[face_num]["absorbed_incident_energy"] = 0
                self.faces_data[face_num]["absorbed_incident_spectral_energy"] =\
                np.zeros(num_wavelengths)
                continue
            
            # Get area and unit normal of current face.
            face_area        = self.areas[face_num]
            face_unit_normal = self.normals[face_num]
            
            # Get directions from which the current face is
            # able to receive radiant power
            allowed_dvecs = dvecs[face_directions_logic]
            
            # Get the time-integrated spectral irradiance magnitudes 
            # for the direction that actually contribute to the current face.
            allowed_time_integrated_spectral_irradiance_magnitudes =\
            time_integrated_spectral_irradiance_magnitudes[face_directions_logic]
            
            # Compute the incident angle between those directions and 
            # current face normal as well as the cosines.
            allowed_incident_angle_cosines =\
            (face_unit_normal.reshape(1,3)*allowed_dvecs).sum(axis=1)
            
            allowed_incident_angles =\
            np.rad2deg(np.arccos(allowed_incident_angle_cosines))
            
            # Compute the fraction of absorbed incident energy as a function of
            # incident angles and wavelength, and evaluate them at the
            # incident angles and wavlengths of the current allowed face directions.
            Wvs, All_incident_angs = np.meshgrid(wavelengths, allowed_incident_angles)
            Wvs, All_incident_angs = Wvs.flatten(), All_incident_angs.flatten()
            eval_pts = np.stack([All_incident_angs, Wvs], axis=1)
            
            face_absorbance_arr = absorbance_function(eval_pts)
            face_absorbance_arr = face_absorbance_arr.reshape(num_allowed_directions,
                                                              num_wavelengths)
            
            # Compute the incident spectral energy on the current surface,
            # from all of the contributing directions.
            incident_spectral_energies =\
            allowed_incident_angle_cosines.reshape(num_allowed_directions, 1)*\
            allowed_time_integrated_spectral_irradiance_magnitudes*face_area
            
            
            # Compute the absorbed incident spectral energy by the current surface,
            # from all of the contributing directions.
            absorbed_incident_spectral_energies =\
            incident_spectral_energies*face_absorbance_arr
            
            # Compute the total absorbed incident spectral energy by the
            # current surface.
            absorbed_incident_spectral_energy =\
            absorbed_incident_spectral_energies.sum(axis=0)
            
            # Compute the total absorbed energy by the current surface.
            absorbed_incident_energy =\
            simpson(y = absorbed_incident_spectral_energy, 
                    x = wavelengths)
            
        
            # Save all of the computed data.
            self.faces_data[face_num]["absorbed_incident_energy"] = absorbed_incident_energy
            self.faces_data[face_num]["absorbed_incident_spectral_energy"] = absorbed_incident_spectral_energy 
            
            self.absorbed_incident_energy[face_num] = absorbed_incident_energy
            self.absorbed_incident_spectral_energy[face_num,:] = absorbed_incident_spectral_energy 
            
            
        # Save the wavelengths for which the spectral energies are defined.
        self.wavelengths = wavelengths
        
        # Compute total energy absorbed. 
        self.total_absorbed_incident_energy = self.absorbed_incident_energy.sum()
        self.total_absorbed_incident_spectral_energy = self.absorbed_incident_spectral_energy.sum(axis=0)
            
        return None    
            
            

 
    def export_as_stl(self, path, dec=6, scale_factor=1.0, format_type="dec", unit="m"):
        
        """
        Export surface  frame (vertices and faces) as a .stl file.
        
        Parameters
        ----------
        
        path : path-str
            Path/Name to which the .stl file will be saved.
            
        dec: int or None
            Number of decimals for rounding. If int, must be positive; 
            default is 6. If None, no rounding is performed.
            
        scale_factor : float
            Number by which to scale the vertex coordinates of the file. 
            Default is 1 (i.e, no scaling).
            
        format_type : "dec" or "exp":
            Number format in which the .stl should be exported. If equal to
            "dec" (Default), numbers are exported in decimal format. If equal
            to "exp" numbers are exported in exponent format.
            
        unit : str
            Length unit used in file. Default is meters [m].
            
        Returns
        -------
        None
        
        
        Produces
        --------
        None

                        
        """
        
        if(path[-4:]!=".stl"):
            raise Exception("Wrong format. File path must end with .stl")
            
        
        N = self.normals
        A = scale_factor*self.A
        B = scale_factor*self.B
        C = scale_factor*self.C
        
        if dec is not None:
            N = np.around(N, dec)
            A = np.around(A, dec)
            B = np.around(B, dec)
            C = np.around(C, dec)
        
    
        tab1 = "    "
        tab2 = "        "
        
        file = open(path, "w")
        file.write(f"solid surface | Unit={unit}\n")

        for i in range(len(N)):
            
            if format_type == "dec":
                nx, ny, nz = [format(j, f".{dec}f") for j in N[i]]
                ax, ay, az = [format(j, f".{dec}f") for j in A[i]]
                bx, by, bz = [format(j, f".{dec}f") for j in B[i]]
                cx, cy, cz = [format(j, f".{dec}f") for j in C[i]]
                
            elif format_type == "exp":
                nx, ny, nz = [f"{j:e}" for j in N[i]]
                ax, ay, az = [f"{j:e}" for j in A[i]]
                bx, by, bz = [f"{j:e}" for j in B[i]]
                cx, cy, cz = [f"{j:e}" for j in C[i]]
                

            
            file.write(f"facet normal {nx} {ny} {nz}\n")
            
            file.write(f"{tab1} outer loop\n")
            file.write(f"{tab2} vertex {ax} {ay} {az}\n")
            file.write(f"{tab2} vertex {bx} {by} {bz}\n")
            file.write(f"{tab2} vertex {cx} {cy} {cz}\n")
            file.write(f"{tab1} endloop\n")
            
            file.write("endfacet\n")
            
        file.write("endsolid surface\n")
        file.close()
        
        
        
#%%        EXAMPLES

if __name__ == '__main__':
    import Surface_Modelling.Domain as dom
    from Ambience_Modelling.Sky import Sky
    from Ambience_Modelling.Site import Site
    import Surface_Modelling.taylor_funcs as tayf
    import Ambience_Modelling.auxiliary_funcs as aux
    from scipy.interpolate import RegularGridInterpolator
    
    
    # Define domain over which the explicit surface to mesh is defined.
    xy_plane = dom.RectDomain(dx = 0.05, dy = 0.05, dims = (1,1,1))
    
    # Compute surface point cloud.
    surf_pts = np.sin(3*xy_plane.Y)*np.cos(3*xy_plane.X) 
    
    # Compute surface mesh from point cloud.
    Mesh_obj = Mesh(X=xy_plane.X, Y = xy_plane.Y, Z = surf_pts)
    
    # Visualize surface mesh.
    config = {"title" : "Meshed Surface"}
    Mesh_obj.visualize(config = config)
    
    # Visualize scalar field defined over the meshed surface. In particular,
    # let us visualize the different sizes of each face element of the 
    # meshed surface.
    config = {"title" : "Area of face elements", "cbar_title" : "Area [cm^2]"}
    Mesh_obj.visualize(facevalues = 10000*Mesh_obj.areas, config = config)
    
    # Let us have a look at some relevant and useful attributes.
    vertices, faces = Mesh_obj.vertices, Mesh_obj.faces
    faces_data = Mesh_obj.faces_data
    
    # --- Discretise Sky-Vault ---
    # Let us discretise the Sky-Vault into 400 different directions in
    # order to test the self.compute_directions_logic function.
    Phi, Theta = np.meshgrid(np.linspace(0, 2*np.pi, 20),
                             np.linspace(0, np.pi/2, 20))
    
    Phi, Theta = Phi.flatten(), Theta.flatten()
    
    # Compute the ray-directions vectors. These are unit vectors
    # that describe the oposite directions from which rays from the 
    # sky vault would fall upon the surface.
    dvecs = np.stack([np.cos(Phi)*np.sin(Theta),
                      np.sin(Phi)*np.sin(Theta),
                      np.cos(Theta)], axis = 1)
    
    Mesh_obj.compute_directions_logic(dvecs = dvecs, 
                                      rad   = 0.1, 
                                      u     = 1/3,
                                      v     = 1/3, 
                                      lmbda = 3.0)
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    for j in range(len(dvecs)):
        config = {"title" : f"Radiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (90 - np.rad2deg(Theta[j]), np.rad2deg(Phi[j]))}
        
        Mesh_obj.visualize(facevalues = Mesh_obj.directions_logic[:,j], 
                           config = config)
        
    # --- EXPORT SURFACE ----
    # Finally, let us export the surface as .stl file.
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\Mesh_obj.stl"
    Mesh_obj.export_as_stl(path = path)
    
    
    
#%%   --- COMPUTATION OF ABSORBED INCIDENT ENERGY 1---

    # Let us compute how much energy is aborbed by the surface across a year.
    
    # Load absorbance array of Silicon.
    absorbance_arr_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\silicon_absorbance.npy" 
    absorbance_arr = np.load(absorbance_arr_path)
    
    # We interpolate the absorbance array in order to generate the absorbance function.
    
    absorbance_function = RegularGridInterpolator(points = (absorbance_arr[:,0,2],
                                                            absorbance_arr[0,:,1]), 
                                                  values =  absorbance_arr[:,:,0])
    
    # Test absorbance function for 45° at 500 nm.
    print(absorbance_function([45, 500]))
    
    
    # Load Sky_obj.
    Sky_obj_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Sky_obj_Medellin_2022.pkl"  
    Sky_obj = aux.load_obj_with_pickle(Sky_obj_path)
    
    # Extract the direction vectors from the Sky_obj.
    dvecs = Sky_obj.time_integrated_spectral_irradiance_res["spectrally_averaged_unit_global"]
    
    # Extract the direction vectors and time-integrated irradiance magnitudes from the Sky_obj.
    time_integrated_spectral_irradiance_magnitudes =\
    Sky_obj.time_integrated_spectral_irradiance_res["magnitude_global"]
    
    #Extract the wavelengths form the Sky_obj.
    wavelengths = Sky_obj.time_integrated_spectral_irradiance_res["wavelengths"]
    
    # Compute new directions logic.
    Mesh_obj.compute_directions_logic(dvecs = dvecs, 
                                      rad   = 0.1, 
                                      u     = 1/3,
                                      v     = 1/3, 
                                      lmbda = 3.0)
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    for j in range(len(dvecs)):
        config = {"title" : f"Radiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (35, -75)}
        
        Mesh_obj.visualize(facevalues = Mesh_obj.directions_logic[:,j], 
                           config = config)
    
    
    # Compute absorbed energy by each face of the surface as well as the total.
    Mesh_obj.compute_absorbed_incident_energy(
    absorbance_function = absorbance_function, 
    dvecs               = dvecs,
    time_integrated_spectral_irradiance_magnitudes =\
    time_integrated_spectral_irradiance_magnitudes, 
    wavelengths = wavelengths
    )
        
        
    # Plot Energy absorbed by face.
    config = {"title" : "Absorbed Energy by Face",
              "cbar_title" : "[Wh]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = Mesh_obj.absorbed_incident_energy, config = config)
    
    print("----- ABSORBED INCIDEN ENERGY 1 -----")
    print(f"TOTAL ABSORBED ENERGY : {Mesh_obj.total_absorbed_incident_energy/1000} [kWh]")
    
    
    
    # Plot Energy per unit area absorbed by face.
    config = {"title" : "Absorbed Energy per unit area by Face",
              "cbar_title" : "[kWh/m^2]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = 0.001*Mesh_obj.absorbed_incident_energy/Mesh_obj.areas, 
                       config = config)
    
    print(f"EQUIVALENT ABSORBED ENERGY PER UNIT AREA : {0.001*Mesh_obj.total_absorbed_incident_energy/Mesh_obj.total_area} [kWh/m^2]")
    
    
    # Let us also plot the absorbed spectral energy by a face of Mesh_obj
    face_num = 0
    fig = plt.figure(figsize=(16,12))
    data = Mesh_obj.absorbed_incident_spectral_energy[face_num,:]
    plt.plot(wavelengths, data)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel("Wavelengths [nm]")
    plt.ylim(data.min(), data.max())
    plt.ylabel("Absorbed Incident Spectral Energy [Wh/nm]")
    plt.title(f"Absorbed Incident Spectral Energy by face {face_num}.")
    plt.grid()
    plt.show()
    
    # And also th total:
    fig = plt.figure(figsize=(16,12))
    data = Mesh_obj.total_absorbed_incident_spectral_energy
    plt.plot(wavelengths, data)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel("Wavelengths [nm]")
    plt.ylim(data.min(), data.max())
    plt.ylabel("Total Absorbed Incident Spectral Energy [Wh/nm]")
    plt.title("Total Absorbed Incident Spectral Energy by the surface.")
    plt.grid()
    plt.show()
    
    
        
#%%    --- COMPUTATION OF ABSORBED INCIDENT ENERGY 1m x 1m HORIZONTAL PLANE---

    # --- DEFINITION OF SURFACE ---

    triangle1 = np.array([[0,0,0],
                          [1,0,0],
                          [0,1,0]]).astype(float)
    
    triangle2 = np.array([[1,0,0],
                          [0,1,0],
                          [1,1,0]]).astype(float)
    
    vertices = np.stack([triangle1[0], 
                         triangle1[1], 
                         triangle1[2],
                         triangle2[2]], axis=0)
    
    faces = np.array([[0,1,2],
                      [1,3,2]])
    
    # --- INITIALIZATION OF MESH OBJ ---
    Mesh_obj = Mesh(vertices=vertices, faces = faces)
    Mesh_obj.visualize()
    
    
    
    # --- COMPUTE SELF SHADING AND DIRECTIONS LOGIC ---
    Mesh_obj.compute_directions_logic(dvecs = dvecs, 
                                      rad   = 0.1, 
                                      u     = 1/3,
                                      v     = 1/3, 
                                      lmbda = 3.0)
    
    
    
    # --- VISUALIZE SELF SHADING AND DIRECTIONS LOGIC ---
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    for j in range(len(dvecs)):
        config = {"title" : f"Radiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (35, -75)}
        
        Mesh_obj.visualize(facevalues = Mesh_obj.directions_logic[:,j], 
                           config = config)
        


    # --- DEFINE COMPONENT OF RADIATION TO USE FOR THE COMPUTATION ---
    time_integrated_spectral_irradiance_magnitudes =\
    Sky_obj.time_integrated_spectral_irradiance_res["magnitude_global"]
    
    
    # --- COMPUTATION OF ABSORBED ENERGY ---
    # Compute absorbed energy by each face of the surface as well as the total.
    absorbance_function2 = lambda x:1
    
    Mesh_obj.compute_absorbed_incident_energy(
    absorbance_function = absorbance_function, 
    dvecs               = dvecs,
    time_integrated_spectral_irradiance_magnitudes =\
    time_integrated_spectral_irradiance_magnitudes, 
    wavelengths = wavelengths
    )
        

    #---  PLOT ENERGY ABSORBED BY FACE ---
    config = {"title" : "Absorbed Energy by Face",
              "cbar_title" : "[Wh]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = Mesh_obj.absorbed_incident_energy, config = config)
    
    
    #---  PRINT TOTAL ENERGY ABSORBED BY SURFACE ---
    print("----- ABSORBED INCIDEN ENERGY -----")
    print(f"TOTAL ABSORBED ENERGY : {Mesh_obj.total_absorbed_incident_energy/1000} [kWh]")
    
    
    # Plot Energy per unit area absorbed by face.
    config = {"title" : "Absorbed Energy per unit area by Face",
              "cbar_title" : "[kWh/m^2]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = 0.001*Mesh_obj.absorbed_incident_energy/Mesh_obj.areas, 
                       config = config)
    

    #---  PRINT EQUIVALENT ENERGY ABSORBED PER UNIT AREA ---
    print(f"EQUIVALENT ABSORBED ENERGY PER UNIT AREA : {0.001*Mesh_obj.total_absorbed_incident_energy/Mesh_obj.total_area} [kWh/m^2]")
        


#%%    --- COMPUTATION OF ABSORBED INCIDENT ENERGY 1m x sec(l) INCLINED PLANE---

    l = np.deg2rad(0)
    # --- DEFINITION OF SURFACE ---

    triangle1 = np.array([[0,  0, 0],
                          [0, -1, 0],
                          [1, -1, np.tan(l)]]).astype(float) + np.array([-0.5,0.5,0])
    
    triangle2 = np.array([[1, -1, np.tan(l)],
                          [1,  0, np.tan(l)],
                          [0,  0, 0]]).astype(float)  + np.array([-0.5,0.5,0])
    
    vertices = np.stack([triangle1[0], 
                         triangle1[1], 
                         triangle1[2],
                         triangle2[1]], axis=0) 
    
    faces = np.array([[0,1,2],
                      [2,3,0]])
    
    # --- INITIALIZATION OF MESH OBJ ---
    Mesh_obj = Mesh(vertices=vertices, faces = faces)
    
    
    config = {"title" : "Horizontal Plane",
              "axis_view" : (35,80),
              "figsize" : (10,10)}
    Mesh_obj.visualize(config=config)
    
    
    
    # --- COMPUTE SELF SHADING AND DIRECTIONS LOGIC ---
    Mesh_obj.compute_directions_logic(dvecs = dvecs, 
                                      rad   = 0.1, 
                                      u     = 1/3,
                                      v     = 1/3, 
                                      lmbda = 3.0)
    
    
    
    # --- VISUALIZE SELF SHADING AND DIRECTIONS LOGIC ---
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    for j in range(len(dvecs)):
        config = {"title" : f"Radiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (35, -75)}
        
        Mesh_obj.visualize(facevalues = Mesh_obj.directions_logic[:,j], 
                           config = config)
        


    # --- DEFINE COMPONENT OF RADIATION TO USE FOR THE COMPUTATION ---
    time_integrated_spectral_irradiance_magnitudes =\
    Sky_obj.time_integrated_spectral_irradiance_res["magnitude_global"]
    
    
    # --- COMPUTATION OF ABSORBED ENERGY ---
    # Compute absorbed energy by each face of the surface as well as the total.
    absorbance_function2 = RegularGridInterpolator(points = (absorbance_arr[:,0,2],
                                                             absorbance_arr[0,:,1]), 
                                                    values = 0*absorbance_arr[:,:,0] + 1)
    
    Mesh_obj.compute_absorbed_incident_energy(
    absorbance_function = absorbance_function2, 
    dvecs               = dvecs,
    time_integrated_spectral_irradiance_magnitudes =\
    time_integrated_spectral_irradiance_magnitudes, 
    wavelengths = wavelengths
    )
        

    #---  PLOT ENERGY ABSORBED BY FACE ---
    config = {"title" : "Absorbed Energy by Face",
              "cbar_title" : "[Wh]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = Mesh_obj.absorbed_incident_energy, config = config)
    
    
    #---  PRINT TOTAL ENERGY ABSORBED BY SURFACE ---
    print("----- ABSORBED INCIDEN ENERGY -----")
    print(f"TOTAL ABSORBED ENERGY : {Mesh_obj.total_absorbed_incident_energy/1000} [kWh]")
    
    
    # Plot Energy per unit area absorbed by face.
    config = {"title" : "Absorbed Energy per unit area by Face",
              "cbar_title" : "[kWh/m^2]",
              "axis_view" : (35, -75)}
        
    Mesh_obj.visualize(facevalues = 0.001*Mesh_obj.absorbed_incident_energy/Mesh_obj.areas, 
                       config = config)
    

    #---  PRINT EQUIVALENT ENERGY ABSORBED PER UNIT AREA ---
    print(f"EQUIVALENT ABSORBED ENERGY PER UNIT AREA : {0.001*Mesh_obj.total_absorbed_incident_energy/Mesh_obj.total_area} [kWh/m^2]")
        

#%%   --- COMPUTATION OF ABSORBED INCIDENT ENERGY FOR POLY1---



    # poly = np.array([[ 0.62401339, -0.93556008,  -0.43358944],
    #                  [ 0.05675702,  0.07461949,   0.],
    #                  [-0.00321045,  0.,           0.]])
    
    
    
    poly = np.array([[ 5.05654171e-01, -9.87507192e-01,  -7.87618204e-03],
                     [ 6.26737011e-04,  2.24602935e-02,   0.],
                     [-1.60123516e-02,  0.,           0.]])
    
    
    
    
    Z = tayf.polyeval(poly, xy_plane.XYpows)
    
    Mesh_obj = Mesh(X=xy_plane.X, Y=xy_plane.Y, Z=Z)
    
    
    
    config = {"title" : "Horizontal Plane",
              "axis_view" : (35,80),
              "figsize" : (10,10)}
    Mesh_obj.visualize(config=config)
    
    
    
    # --- COMPUTE SELF SHADING AND DIRECTIONS LOGIC ---
    Mesh_obj.compute_directions_logic(dvecs = dvecs, 
                                      rad   = 0.1, 
                                      u     = 1/3,
                                      v     = 1/3, 
                                      lmbda = 3.0)
    
    
    
    # --- VISUALIZE SELF SHADING AND DIRECTIONS LOGIC ---
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    for j in range(len(dvecs)):
        config = {"title" : f"Radiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (35, 75)}
        
        Mesh_obj.visualize(facevalues = Mesh_obj.directions_logic[:,j], 
                           config = config)
    


    # --- DEFINE COMPONENT OF RADIATION TO USE FOR THE COMPUTATION ---
    time_integrated_spectral_irradiance_magnitudes =\
    Sky_obj.time_integrated_spectral_irradiance_res["magnitude_global"]
    
    
    # --- COMPUTATION OF ABSORBED ENERGY ---
    # Compute absorbed energy by each face of the surface as well as the total.
    
    Mesh_obj.compute_absorbed_incident_energy(
    absorbance_function = absorbance_function, 
    dvecs               = dvecs,
    time_integrated_spectral_irradiance_magnitudes =\
    time_integrated_spectral_irradiance_magnitudes, 
    wavelengths = wavelengths
    )
        
    
    #---  PLOT ENERGY ABSORBED BY FACE ---
    config = {"title" : "Absorbed Energy by Face",
              "cbar_title" : "[Wh]",
              "axis_view" : (35, 75)}
        
    Mesh_obj.visualize(facevalues = Mesh_obj.absorbed_incident_energy, config = config)
    
    
    #---  PRINT TOTAL ENERGY ABSORBED BY SURFACE ---
    print("----- ABSORBED INCIDEN ENERGY -----")
    print(f"TOTAL ABSORBED ENERGY : {Mesh_obj.total_absorbed_incident_energy/1000} [kWh]")
    
    
    # Plot Energy per unit area absorbed by face.
    config = {"title" : "Absorbed Energy per unit area by Face",
              "cbar_title" : "[kWh/m^2]",
              "axis_view" : (35, 75)}
        
    Mesh_obj.visualize(facevalues = 0.001*Mesh_obj.absorbed_incident_energy/Mesh_obj.areas, 
                       config = config)
    
    
    #---  PRINT EQUIVALENT ENERGY ABSORBED PER UNIT AREA ---
    print(f"EQUIVALENT ABSORBED ENERGY PER UNIT AREA : {0.001*Mesh_obj.total_absorbed_incident_energy/Mesh_obj.total_area} [kWh/m^2]")
    
    
    # And also th total:
    fig = plt.figure(figsize=(16,12))
    data = Mesh_obj.total_absorbed_incident_spectral_energy
    plt.plot(wavelengths, data)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel("Wavelengths [nm]")
    plt.ylim(data.min(), data.max())
    plt.ylabel("Total Absorbed Incident Spectral Energy [Wh/nm]")
    plt.title("Total Absorbed Incident Spectral Energy by the surface.")
    plt.xlim(0, 2000)
    plt.grid()
    plt.show()



