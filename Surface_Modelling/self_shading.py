#%%         MODULE DESCRIPTION AND/OR INFO

"""
The self-shading module provides a method for simulating self-shading on a 
surface, given the directions of the incomming rays. This module only works
for surfaces discretised using triangular face elements.


       --- DETAILED DESCRIPTION OF THE METHOD USED ---
In real life, rays can be thought of as coming from multiple directions of
the sky vault and falling on a surface. In reality, the number of directions
is infinite and it spans the whole hemisphere. However, having discretised
the sky vault (see 'Ambience_Modelling.Sky' module, for example) we are 
able to reduce these directions to a finite number with which to work with.
Now, yet still, from each of these directions an infinitude of rays are
still falling on the surface. So we still have to reduce the number of these
rays to a finite value so we can work with it. Now, usually for this kind 
of stuff, a montecarlo simulation would be in order, but that would still
require a very large number of rays just to make sure that the simulation is
representative and that self-shading is accurately calculated. What we shall 
do instead is reverse the situation. We set the number of rays to be exactly
equal to the number of faces of the surface (with a condition that we'll explain
below). And instead of thinking that the rays come from the sky and impinge 
upon a face, we take a point within each face as being the origin of each ray 
which is then directed in the exact oposite direction from which the original
ray it was supposed to come. Now, we test whether said 'fired' impacts on
another surface on its way to the sky. If it doesn't, that means that face is
exposed to radiation and there is no self-shading. If it does, however, that
means that the face is not exposed to radiation, since it is currently being 
blocked by another face element. 

Now, not all faces of the surface can originate rays. Only those faces whose
normals make an angle of less than 90 degrees with the ray diraction must be 
considered as 'ray-origins'. This is becasue, a ray coming from the oposite 
direction and impinging on that surface would not contribute to the total energy
recieved by that surface as that it would be facing that ray the wrong way
(i.e, the ray instead of the ray impinging on the front, it impinges on the 
face's back). As such, only points lying within faces whose normals an angle
of less than 90 degrees with the ray diraction must be considered as
'ray-origins'. However, for the purposes of detecting self shading, we do need
to check all faces, no matter their orientation with respect to the rays-direction,
as even if a ray impinges on the back of a face, this ray is still cut-off
from it path way, which means that it may not reach another face that actually
had the correct orientation. In short, face's backs can still account
for self-shading.

But wait a minute. Let us revisit something said in the previous paragraph: "...for 
the purposes of detecting self shading, we do need to check all faces...". Is that
really True though? I highly doubt that just any face on any part of the surface
could block a ray that's heading for/coming from another face on the other side of the 
surface, when a given direction is fixed. And that is true, in reality, we can 
think of each ray being 'fired' form each face as forming a semi-infinite cylinder
whose axis matches that the ray direction and whose bottom's center lies at
the ray-origin of the face who fired it. As such, only those faces who find
themselves within this semi-infinite cylinder could ever hope to block said
ray. As such, we can reduce the domain of search and only tests those faces which
lie sufficiently close to the axis of the fired ray and lie further along the
ray's path than from where it was fired, as posible candidates fro self shading 
that need to be tested in order to check wether or not they do block the fired
ray.

After having reduced the domain of search, the actual algorthim for testing wether
the path of a ray is cut off by another face is the one explained in 
https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/bool_intersection.html
which utilizes various signed volumes of a polygon to figure out whether 
an intersection between a line segment and a triangle has been made.

       --- EXECUTIVE CHAT GPT SUMMARY ---
The text discusses how to calculate self-shading on a surface, which involves
reducing the number of rays to a finite value in order to work with it. 
Instead of using a Monte Carlo simulation, the number of rays is set to be
exactly equal to the number of faces of the surface. Points within each face
are taken as the origin of each ray, which is then directed in the exact 
opposite direction from which the original ray was supposed to come. Only
those faces whose normals make an angle of less than 90 degrees with the ray 
direction are considered as "ray-origins". For detecting self-shading, the domain
of search is reduced to those faces that lie within a semi-infinite cylinder 
whose axis matches that of the ray direction and whose bottom's center lies at
the ray-origin of the face that fired it. The algorithm for testing whether the
path of a ray is cut off by another face is explained in
https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/bool_intersection.html.

"""

#%%        IMPORTATION OF LIBRARIES

import time as tm
import cupy as cp
import numpy as np
from numba import njit, bool_, float64

#%%                 DEFINITION OF FUNCTIONS



def reduce_ray_tracing_domain(rad, dvecs, list_faces, face_centers, face_normals):
    
    """
    Computing ray-tracing for a meshed surface with many faces/elements and to
    do so for multiple directions is computationally expensive. As such, we
    first reduce the number of faces for which the ray-traicing procedure has
    to catually be applied. With this, we can focus on performing ray-tracing
    to faces which actually have a chance of blocking a ray and generating 
    self-shading for the surface. This function works more or less as follows:
        
        1) Loop over each of the specified ray directions and for each ray
           direction, do:
    
        2) Compute which face centers can be considered allowable ray origins.
    
        3) Compute the closest distance of all faces to each of the axes defined
           by each of the rays being fired by each of their corresponding
           ray orgins, for the given ray direction.
           
        4) If the distance of a face to a given ray axis is small enough,
           the face is considered for the ray-tracing procedure. otherwise it
           is not.
           
           
    rad : float
        Cylinder radius for considering faces. From each ray-facing face center
        imagine an axis passing through that face center having the same
        direction as the ray. We then compute the closest distance from each of the 
        face centers of the surface to each of face centers, in general.
        If this distance is supirior to 'rad', the face element is not
        considered for ray-tracing. If it inferior, it is considered.
        
    dvecs : cupy.array of floats with shape (ndvec, 3)
        Unit ray-direction vectors. 'dvecs[i,:]' encodes the unit vector, in 
        cartesian coordinates, of the i-th ray-direction that is to be 
        considered for ray tracing. 
        
    list_faces : cupy.array with shape (nf,)
        Array of face indices.
        
    face_centers : cupy.array of floats with shape (nf, 3)
        Barycenters of each of the faces/element that make up the 
        meshed surface. 'face_centers[i,:]' contains the position vector
        of the barycenter of the i-th face/element of the surface.
        
    face_normals : cupy.array of floats with shape (nf, 3)
        Unit normals of each of the faces/element that make up the 
        meshed surface. 'face_normals[i,:]' contains the unit normal vector
        of the i-th face/element of the surface.

    Returns
    -------
    whole_logic : cupy.array of bools with shape (ndvec, nf, nf)
        'whole_logic[i,j,k]' tells us whether the k-th face/element of a meshed 
        explicit surface is close enough to a ray, originating from the j-th 
        face/element of that same surface, with direction given by the i-th 
        direction vector, in order to be worth performing ray-tracing on that 
        element.

        If 'whole_logic[i,j,k]' is False:
            It means that the center of face/element k of the surface is not close 
            enough to a ray originating from the center of face/element j, and with 
            direction i, to merit any computation of ray tracing.

        If 'whole_logic[i,j,k]' is True:
            It means that the center of face/element k of the surface is close 
            enough to a ray originating from the center of face/element j, and with 
            direction i, to actually justify performing a ray-tracing computation.
            
    ray_facing_logic : numpy.array of bools with shape (nf, ndvecs)
        'ray_facing_logic[i,j]' tells us whether the i-th face/element of a
        meshed explicit surface could get energy from the j-th direction.
    
       If 'ray_facing_logic[i,j]' is False:
           It means that the normal vector of the i-th face toguether with the
           j-th ray-direction vector make an angle greater than 90 degrees. Therefore
           the i-th face cannot recieve energy from the j-th direction.
    
       If 'ray_facing_logic[i,j]' is True:
           It means that the normal vector of the i-th face toguether with the
           j-th ray-direction vector make an angle equal or less than 90 degrees.
           Therefore the i-th face can indeed recieve energy from the j-th direction.
           

        
    """
    
    # Get total number of directions.
    ndvec = int(dvecs.shape[0])
    
    # Get total number of faces.
    nf    = int(list_faces[-1] + 1)
    
    
    # Initialize ray-facing logic array.
    ray_facing_logic = cp.zeros((ndvec, nf)).astype(bool)

    # Initialize whole-logic logic array.
    whole_logic = cp.zeros((ndvec, nf, nf)).astype(bool)
    
    # Note: 'ray-facing faces' means faces that can be considered 'ray-origins'
    # beacause their normals and the ray-directions make an angle of less than
    # 90 degrees.
    
    
    # --- ACTUAL COMPUTATION OF LOGIC ---

    # Loop over each of the ray directions.
    for i, dvec in enumerate(dvecs):
    
        # Get current ray direction and copy it as many times as there are faces.
        d = cp.array([dvec]*nf)
        
        # Compute what faces have normals that can actually recieve energy 
        # from the current direction.
        ray_facing_logic[i,:] = cp.einsum("ij,ij->i", d, face_normals) > 0        
        
        # Get the indices/id numbers of the aforementioned faces.
        ray_facing_faces = list_faces[ray_facing_logic[i,:]]
    
        # Get the number of ray-facing faces.
        nrff = len(ray_facing_faces)
        
        try:
            # Get the face centers of all faces which are able to recieve 
            # energy from the current ray direction (i.e, ray-facing faces),
            # and copy the information for as many times as there are faces 
            # in total. It has shape (nrff, nf, 3).
            O = cp.swapaxes(cp.array([face_centers[ray_facing_faces]]*nf), 0, 1)
            
            
            # Get the face centers of all faces. And copy the information
            # for as many times as there are ray-facing faces. It has shape 
            # (nrff, nf, 3).
            P = cp.array([face_centers]*nrff)
            
            # We compute the vector going from each of the ray-facing face centers
            # to all possible face centers and dot each one of the resulting vectors 
            # with the current ray-direction vector. What we basically get is 
            # the projection of each of these vectors onto the ray-diretcion
            # vector. We may safely ignore face centers for which 't' is negative
            # as this implies that the ray origin lies further along the ray
            # axis than said face and, as such, there is no risk of blocking said 
            # ray and thus ray-tracing computation for that face is not 
            # required.
            t = cp.einsum("ij,kij->ki", d, P-O)
            
            d = cp.array([d]*nrff)
            t = cp.swapaxes(cp.swapaxes(cp.array([t]*3), 0, 2), 0, 1)  
            
            
            # From each ray-facing face center imagine an axis passing through
            # that face center having the same direction as the current ray-direction
            # vector. We then compute the closest distance from each of the 
            # face centers of the surface to each of face centers, in general.
            # That is 'dist'. I.e, it is the closest distance from each of the
            # ray-facing face center points (i.e, ray-origin points) to each
            # of the face centers of the surface.
            dist = cp.linalg.norm(P - t*d - O, axis=-1)
            
            # Finally, for a face to possibly block a ray originating from
            # a ray-origing point, said face must lie within a semi-cylinder.
            # As such, we evaluate this condition. 
            logic = cp.logical_and(dist<=rad, t[:,:,0]>0)
            
        except ValueError as error:
            if nrff<1: continue
            else: raise ValueError(error)
        
        # We store the whole calculated logic for the current ray9direction vector
        # and repeat the process for all ray directions.
        whole_logic[i,ray_facing_faces,:] = logic
        
        # whole_logic = (ray directions, ray origins, faces that are closed to a given ray origin)
    return whole_logic, ray_facing_logic.T




#%%

@njit(bool_[:,:](float64[:,:], float64[:,:], float64[:,:], float64[:,:], bool_[:,:,:], float64, float64, float64))
def ray_tracing(A, B, C, dvecs, logic, u=1/3, v=1/3, lmbda=3.0):
    
    """
    Perform ray-tracing on the specified faces of the meshed surface.
    The algorithm hereby implemented for ray-tracing is known as segura's
    algorithim. More information can be found in:
        
    1) https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/bool_intersection.html
    2) https://www.researchgate.net/publication/221546368_Algorithms_to_Test_Ray-Triangle_Intersection_Comparative_Study
    3) https://core.ac.uk/download/pdf/82190425.pdf
    4) https://www.sciencedirect.com/science/article/abs/pii/S0097849398000648
    
    
    Parameters
    ----------
    A : numpy.array of floats with shape (nf,3)
        Array of vectors detailing the position of each of the first
        points forming each face of the meshed surface. 'A[i,:]' encodes
        the position vector of the first point that makes up the i-th
        face of the surface.
    
    B : numpy.array of floats with shape (nf,3)
        Array of vectors detailing the position of each of the second
        points forming each face of the meshed surface. 'B[i,:]' encodes
        the position vector of the second point that makes up the i-th
        face of the surface.
    
    C : numpy.array of floats with shape (nf,3)
        Array of vectors detailing the position of each of the third
        points forming each face of the meshed surface. 'C[i,:]' encodes
        the position vector of the third point that makes up the i-th
        face of the surface.
    
    dvecs : cupy.array of floats with shape (ndvec, 3)
        Unit ray-direction vectors. 'dvecs[i,:]' encodes the unit vector, in 
        cartesian coordinates, of the i-th ray-direction that is to be 
        considered for ray tracing.
        
        
    logic : numpy.array of bools with shape (ndvec, nf, nf)
        'logic[i,j,k]' tells us whether the k-th face/element of a meshed 
        explicit surface is close enough to a ray, originating from the j-th 
        face/element of that same surface, with direction given by the i-th 
        direction vector, in order to be worth performing ray-tracing on that 
        element.

        If 'logic[i,j,k]' is False:
            It means that the center of face/element k of the surface is not close 
            enough to a ray originating from the center of face/element j, and with 
            direction i, to merit any computation of ray tracing.

        If 'logic[i,j,k]' is True:
            It means that the center of face/element k of the surface is close 
            enough to a ray originating from the center of face/element j, and with 
            direction i, to actually justify performing a ray-tracing computation.
            
    u : float
        Barycentric u-coordinate of the position used within the each face
        element to determine the origin vector of each ray. Default is 1/3.
    
    v : float
        Barycentric v-coordinate of the position used within the each face
        element to determine the origin vector of each ray. Default is 1/3.
    
    lmbda : float 
        Distance up to which to extend each of the rays. It should be greater
        than the dimensions of the surface. Default is 3.
        
    Returns
    -------
    res : numpy.array of bools with shape (nf, ndvec)
        'res[i,j]' tells whether a ray originating from the face/element i
        and with direction given by the ray-direction vector j, will be cut
        off from its path to the sky by another face element of the surface.
        
        If 'res[i,j]' is False:
            It means that a ray originating from the face/element i
            and with direction given by the ray-direction vector j, will indeed
            be cut off from its path to the sky by another face element of the 
            surface.

        If 'res[i,j]' is True:
            It either means that the face element was not considered in the 
            ray tracing calculation or it can mean that the it was considered
            and a ray originating from said face/element i, with direction a
            given by the ray-direction vector j, will, in fact, not be cut off
            from its path to the sky by another face element of the surface.
        
        
    Notes
    -----
    1) This ray-tracing algorithim does not include any bounces. So it can only
       be used to determine self-shading.
        
    """
    
    # Number of unit-vector directions.
    nd = logic.shape[0]
    
    # Number of face/ elements of the surface.
    nf = logic.shape[-1]
    
    # Initialize results array.
    res = np.zeros((nf, nd)).astype(bool_)
    
    
    # Loop over each of the ray-directions.
    for j in range(nd):
        
        # Loop over each of the surface's faces.
        for i in range(nf):
            
            # Get the number of face elements for which the 
            # ray-tracing computation is justified.
            nl = int(logic[j,i,:].sum())
            
            if(nl==0):
                continue
            
            # Initialize array for containing the intersection logic.
            intersection_logic = np.zeros(nl).astype(bool_)
            
            
            # --- (1) COMPUTE RAYS ----
    
            # Compute rays origins.
            Q = u*A[i,:] + v*B[i,:] + (1-u-v)*C[i,:]
            
            # Compute rays endings.
            Qp = Q + lmbda*dvecs[j,:]
            
            # Copy the info in Q and Qp for as many times as 
            # there are ray-facing faces:
            
            # --- Not Numba-compatible implementation ---
            # Q  = np.array([Q]*nl)
            # Qp = np.array([Qp]*nl)

            # --- Numba-compatible implementation ---
            Q  = Q.repeat(nl).reshape((-1, nl)).T
            Qp = Qp.repeat(nl).reshape((-1, nl)).T
            # --------------------------------------
            
            
            #           ---- MATHEMATICAL BACKGROUND ----
            # For each face, a ray origin and a ray ending toguether with
            # the 3 points that make up a triangular face element, we can define
            # four 3D simplexes. These four simplexes are (as expressed by the
            # union of their points): QABC, QABQp, QBCQp and QCAQp. By computing 
            # the signed volumes of each of these simplexes, and applying some
            # logic, we can figure out wether the segment QQp intersects
            # the triangle ABC. In particular, it can be proven that:
                
            # When sgn(|QABC|) > 0, AND:
            # sgn(|QABQp|) ≥ 0 AND sgn(|QBCQp|) ≥ 0 AND sgn(|QCAQp|)≥ 0
            
            #                           OR
            
            # When sgn(|QABC|) < 0, AND:
            # sgn(|QABQp|) ≤ 0 AND sgn(|QBCQp|) ≤ 0 AND sgn(|QCAQp|) ≤ 0
            
            # The segment QQp intersects the triangle ABC. Where |WXYZ| is the
            # signed volume of the 3D simplex defined by the points W, X, Y and 
            # Z and sgn() is the sign function.
            
            
            
            
            
            # --- (2) INITIALIZE DETERMINANT MATRIX FOR EACH SIMPLEX ---
            
            # We initialize the determinant matrix for each simplex.
            # We create a determinant matrix for all faces for whcih the 
            # ray-tracing procedure has been deemed worth while.
            QABC  = np.ones((nl, 4, 4), dtype=np.float64)
            QABQp = np.ones((nl, 4, 4), dtype=np.float64)
            QBCQp = np.ones((nl, 4, 4), dtype=np.float64)
            QCAQp = np.ones((nl, 4, 4), dtype=np.float64)
            
            
            # --- (3) COMPUTE DETERMINANT MATRIX OF EACH SIMPLEX ---
            
            QABC[:,0,:3] = Q
            QABC[:,1,:3] = A[logic[j,i,:]]
            QABC[:,2,:3] = B[logic[j,i,:]]
            QABC[:,3,:3] = C[logic[j,i,:]]
            
            QABQp[:,0,:3] = Q
            QABQp[:,1,:3] = A[logic[j,i,:]]
            QABQp[:,2,:3] = B[logic[j,i,:]]
            QABQp[:,3,:3] = Qp
            
            QBCQp[:,0,:3] = Q
            QBCQp[:,1,:3] = B[logic[j,i,:]]
            QBCQp[:,2,:3] = C[logic[j,i,:]]
            QBCQp[:,3,:3] = Qp
            
            QCAQp[:,0,:3] = Q
            QCAQp[:,1,:3] = C[logic[j,i,:]]
            QCAQp[:,2,:3] = A[logic[j,i,:]]
            QCAQp[:,3,:3] = Qp
            
            
            
            # --- (4) COMPUTE SIGNED VOLUME OF EACH SIMPLEX ---
            # Unfortunately numba still doesn't support many numpy functions.
            # The computation of the signed volumes would literally take
            # 4 lines of code if it were not for this caveat. 
            # Instead, we have to painstakingly compute each of the 4x4 
            # determinants in the old-fashioned way. I.e, explicetly.
            # After that, the sign of the computed volumes can be readily
            # computed for all pertinent faces using numpy.sign function.
            
            
            # --- Not Numba-compatible implementation ---    
            # sgn_QABC  = np.sign(np.linalg.det(QABC))
            # sgn_QABQp = np.sign(np.linalg.det(QABQp))
            # sgn_QBCQp = np.sign(np.linalg.det(QBCQp))
            # sgn_QCAQp = np.sign(np.linalg.det(QCAQp))
            
            
            # --- Numba-compatible implementation ---
            A11, A12, A13, A14  = QABC[:,0,0], QABC[:,0,1], QABC[:,0,2], QABC[:,0,3]
            A21, A22, A23, A24  = QABC[:,1,0], QABC[:,1,1], QABC[:,1,2], QABC[:,1,3]
            A31, A32, A33, A34  = QABC[:,2,0], QABC[:,2,1], QABC[:,2,2], QABC[:,2,3]
            A41, A42, A43, A44  = QABC[:,3,0], QABC[:,3,1], QABC[:,3,2], QABC[:,3,3]
            
            det_QABC = \
            A11*( A22*(A33*A44 - A34*A43) + A23*(A34*A42 - A32*A44) + A24*(A32*A43 - A33*A42) ) -\
            A21*( A12*(A33*A44 - A34*A43) + A13*(A34*A42 - A32*A44) + A14*(A32*A43 - A33*A42) ) +\
            A31*( A12*(A23*A44 - A24*A43) + A13*(A24*A42 - A22*A44) + A14*(A22*A43 - A23*A42) ) -\
            A41*( A12*(A23*A34 - A24*A33) + A13*(A24*A32 - A22*A34) + A14*(A22*A33 - A23*A32) )
            
            sgn_QABC = np.sign(det_QABC)
            
            
            
            A11, A12, A13, A14  = QABQp[:,0,0], QABQp[:,0,1], QABQp[:,0,2], QABQp[:,0,3]
            A21, A22, A23, A24  = QABQp[:,1,0], QABQp[:,1,1], QABQp[:,1,2], QABQp[:,1,3]
            A31, A32, A33, A34  = QABQp[:,2,0], QABQp[:,2,1], QABQp[:,2,2], QABQp[:,2,3]
            A41, A42, A43, A44  = QABQp[:,3,0], QABQp[:,3,1], QABQp[:,3,2], QABQp[:,3,3]
            
            det_QABQp = \
            A11*( A22*(A33*A44 - A34*A43) + A23*(A34*A42 - A32*A44) + A24*(A32*A43 - A33*A42) ) -\
            A21*( A12*(A33*A44 - A34*A43) + A13*(A34*A42 - A32*A44) + A14*(A32*A43 - A33*A42) ) +\
            A31*( A12*(A23*A44 - A24*A43) + A13*(A24*A42 - A22*A44) + A14*(A22*A43 - A23*A42) ) -\
            A41*( A12*(A23*A34 - A24*A33) + A13*(A24*A32 - A22*A34) + A14*(A22*A33 - A23*A32) )
            
            sgn_QABQp = np.sign(det_QABQp)
            
            
            
            A11, A12, A13, A14  = QBCQp[:,0,0], QBCQp[:,0,1], QBCQp[:,0,2], QBCQp[:,0,3]
            A21, A22, A23, A24  = QBCQp[:,1,0], QBCQp[:,1,1], QBCQp[:,1,2], QBCQp[:,1,3]
            A31, A32, A33, A34  = QBCQp[:,2,0], QBCQp[:,2,1], QBCQp[:,2,2], QBCQp[:,2,3]
            A41, A42, A43, A44  = QBCQp[:,3,0], QBCQp[:,3,1], QBCQp[:,3,2], QBCQp[:,3,3]
            
            det_QBCQp = \
            A11*( A22*(A33*A44 - A34*A43) + A23*(A34*A42 - A32*A44) + A24*(A32*A43 - A33*A42) ) -\
            A21*( A12*(A33*A44 - A34*A43) + A13*(A34*A42 - A32*A44) + A14*(A32*A43 - A33*A42) ) +\
            A31*( A12*(A23*A44 - A24*A43) + A13*(A24*A42 - A22*A44) + A14*(A22*A43 - A23*A42) ) -\
            A41*( A12*(A23*A34 - A24*A33) + A13*(A24*A32 - A22*A34) + A14*(A22*A33 - A23*A32) )
            
            sgn_QBCQp = np.sign(det_QBCQp)
            
            
            
            A11, A12, A13, A14  = QCAQp[:,0,0], QCAQp[:,0,1], QCAQp[:,0,2], QCAQp[:,0,3]
            A21, A22, A23, A24  = QCAQp[:,1,0], QCAQp[:,1,1], QCAQp[:,1,2], QCAQp[:,1,3]
            A31, A32, A33, A34  = QCAQp[:,2,0], QCAQp[:,2,1], QCAQp[:,2,2], QCAQp[:,2,3]
            A41, A42, A43, A44  = QCAQp[:,3,0], QCAQp[:,3,1], QCAQp[:,3,2], QCAQp[:,3,3]
            
            det_QCAQp = \
            A11*( A22*(A33*A44 - A34*A43) + A23*(A34*A42 - A32*A44) + A24*(A32*A43 - A33*A42) ) -\
            A21*( A12*(A33*A44 - A34*A43) + A13*(A34*A42 - A32*A44) + A14*(A32*A43 - A33*A42) ) +\
            A31*( A12*(A23*A44 - A24*A43) + A13*(A24*A42 - A22*A44) + A14*(A22*A43 - A23*A42) ) -\
            A41*( A12*(A23*A34 - A24*A33) + A13*(A24*A32 - A22*A34) + A14*(A22*A33 - A23*A32) )
            
            sgn_QCAQp = np.sign(det_QCAQp)
            
            # ---------------------------------------------------------------
            
            
            # ----- (4) COMPUTE LOGIC OF INTERSECTIONS -----
            
            # We separate the computation into 2 cases. One when the simplex
            # has positive orientation and one where it has negative orientation.
            simplex_has_pos_orientation = sgn_QABC > 0
            simplex_has_neg_orientation = sgn_QABC < 0
            
            # For those simplexes that have positive orientation, we apply
            # the specific condition for triangle intersection when the 
            # orientation is positive.
            intersection_logic[simplex_has_pos_orientation] =\
            np.logical_and(np.logical_and(sgn_QABQp[simplex_has_pos_orientation] >= 0,
                                          sgn_QBCQp[simplex_has_pos_orientation] >= 0),
                                          sgn_QCAQp[simplex_has_pos_orientation] >= 0)
            
            # For those simplexes that have neagtive orientation, we apply
            # the specific condition for triangle intersection when the 
            # orientation is negative.
            intersection_logic[simplex_has_neg_orientation] =\
            np.logical_and(np.logical_and(sgn_QABQp[simplex_has_neg_orientation] <= 0,
                                          sgn_QBCQp[simplex_has_neg_orientation] <= 0),
                                          sgn_QCAQp[simplex_has_neg_orientation] <= 0)
            

        
            # If any intersection happened, that means that the ray was blocked
            # on its way to the sky. Hence, the face element from which it
            # originated expiriences self-shading for that particular 
            # ray direction.
            res[i,j] = intersection_logic.any()
            
            
        
    res = np.logical_not(res)    
    return res


#%%               EXAMPLES 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import Surface_Modelling.Mesh as mg
    import Surface_Modelling.Domain as dom
    from mpl_toolkits.mplot3d import art3d
    

    # --- TEST OF RAY-TRACING FOR JUST 2 TRIANGLES ---
    
    # Define triangle 1.
    triangle1 = np.array([[0.,0.,0.],      
                          [1.,0.,0.],
                          [0.,1.,0.]]) 
    
    # Define triangle 2.
    triangle2 = np.array([[0.,0.,0.],
                          [0.,1.,0.],
                          [0.,0.,1.]])
    
    # Define vertices.
    vertices = np.array([[0.,0.,0.],
                         [1.,0.,0.],
                         [0.,1.,0.],
                         [0.,0.,1.]])
    
    # Define faces.
    faces = np.array([[0,1,2],
                      [0,1,3]])
    
    # Mesh surfaces.
    Mesh_obj = mg.Mesh(vertices = vertices, faces = faces)
        
    
    # Visualize surface.
    config = {"xlims"     : (-0.2, 1.2), "ylims"     : (-0.2, 1.2),
              "zlims"     : (0,  1), "axis_view" : (25, 30),
              "title"     : "Test surfaces"}
    
    Mesh_obj.visualize(config = config)
    
    
    # --- DEFINE DIRECTIONS FOR WHICH TO TEST RAY-TRACING ---
    dvecs = np.array([[0., 0.,  1.],
                      [0., 1.,  1.],
                      [1., 0.,  1.],
                      [0., -1., 1.]])

    dvecs /= np.linalg.norm(dvecs, axis=1).reshape(dvecs.shape[0], 1)
    
    
    # Visualize the selected directions.
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    
    for dvec in dvecs:
        plt.quiver(Mesh_obj.face_centers[0,0], 
                   Mesh_obj.face_centers[0,1],
                   Mesh_obj.face_centers[0,2], 
                   dvec[0], dvec[1], dvec[2], 
                   color = "black")
        
    ax.set_xlim((-0.2, 1.2))
    ax.set_ylim((-0.2, 1.2))
    ax.set_zlim((0, 1))
    ax.set_title("Rays fired from face 0")
    ax.set_xlabel("X [m] (↑ == N, ↓ == S)")
    ax.set_ylabel("Y [m] (↑ == E, ↓ == W)")
    ax.set_ylabel("Z [m]")
    
    # --- COMENTARY ---
    # We simulate 4 rays being fired from face 0 in 4 different directions.
    # From the graphs above it is clear that the only fired ray whose
    # path should be cut-off is that of the last ray fired, since face 1
    # lies there completely blocking it. Let us see if the ray-tracing code 
    # actually predicts that. 
    
    # --- PERFORM RAY TRACING ----
    
    # We compute the logic array in such a way as to only test for the intersection
    # of rays fired from face 0 with other faces. That is, we ignore only focus on
    # the rays fired from face 0 and not any other faces.
    logic = np.zeros((len(dvecs), faces.shape[0], faces.shape[0])).astype(bool)
    logic[:,0,1] = True
    
    # Compute ray tracing.
    res =\
    ray_tracing(A     = Mesh_obj.A, 
                B     = Mesh_obj.B,
                C     = Mesh_obj.C, 
                dvecs = dvecs,
                logic = logic,
                u     = 1/3,
                v     = 1/3,
                lmbda = 2.0)
    
    # If we check 'res' we'll see that it says that face number 0 does not
    # recieve radiation from direction 3. Which is what was expected.
    
    for j, dvec in enumerate(dvecs):
        config = {"title" : f"Radiation received by face form direction {j} : {dvec}",
                  "vmin"  : -0.1,
                  "vmax"  :  1,
                  "cbar_title" : "Radiation Recieved"}
        
        Mesh_obj.visualize(facevalues = res[:,j], config=config)
        
    # This corroborates that the ray tracing works.
    
    
    
    
    
    # --- TEST OF SELF-SHADING FOR MULTIIPLE TRIANGLES---

    # Initialize RectDomain object.
    xy_plane = dom.RectDomain(dx = 0.05, dy = 0.05, dims = (1,1,1))
    
    # Compute surface point cloud.
    Z = np.sin(3*xy_plane.Y)*np.cos(3*xy_plane.X)
    
    # Compute surface mesh from point cloud and plot it.
    Mesh_obj = mg.Mesh(xy_plane.X, xy_plane.Y, Z)
    config = {"axis_view" : (25,-45),
              "title"     :  "Test surface"}
    Mesh_obj.visualize(config = config)
    
    
    # Let us discretise the Sky-Vault into 400 different directions in
    # order to test the described functions.
    Phi, Theta = np.meshgrid(np.linspace(0, 2*np.pi, 20),
                             np.linspace(0, np.pi/2, 20))
    
    Phi, Theta = Phi.flatten(), Theta.flatten()
    
    # Compute the ray-directions vectors. These are unit vectors
    # that describe the oposite directions from which rays from the 
    # sky vault would fall upon the surface.
    dvecs = np.stack([np.cos(Phi)*np.sin(Theta),
                      np.sin(Phi)*np.sin(Theta),
                      np.cos(Theta)], axis = 1)
    
    
    
    t = tm.time()
    
    # Compute domain reduction logic.
    whole_logic, ray_facing_logic = \
    reduce_ray_tracing_domain(rad = 0.1, 
                              dvecs = cp.array(dvecs), 
                              list_faces = cp.array(Mesh_obj.list_faces), 
                              face_centers = cp.array(Mesh_obj.face_centers),
                              face_normals = cp.array(Mesh_obj.normals))
    
    whole_logic, ray_facing_logic = whole_logic.get(), ray_facing_logic.get()
    
    
    # Compute ray tracing logic.
    res =\
    ray_tracing(A = Mesh_obj.A, 
                B = Mesh_obj.B,
                C = Mesh_obj.C, 
                dvecs = dvecs,
                logic = whole_logic,
                u=1/3, 
                v=1/3, 
                lmbda=2.0)

    
    # Combine both logics in order to get an array detailing what faces do
    # get exposed to radiation coming from a certaing direction. That is,
    # what faces do the rays coming from a certain direction of the sky-vault,
    # manage to fall upon?
    facevalues = np.logical_and(ray_facing_logic, res)
    dt = tm.time() - t
    
    
    # In white we plot the faces that recieve radiation for a certain direction.
    # In brown we plot the faces that do not recieve radiation for a certain direction.
    # in the title we write the ray-direction unit vector.
    vertices, faces = Mesh_obj.vertices, Mesh_obj.faces
    for j in range(res.shape[1]):
        config = {"title" : f"Radtiation recieved from direction {j} = {dvecs[j]}",
                  "vmin" : -0.1,
                  "vmax" : 1,
                  "axis_view" : (90 - np.rad2deg(Theta[j]), np.rad2deg(Phi[j]))}
        
        Mesh_obj.visualize(facevalues = facevalues[:,j], config = config)
    
    








