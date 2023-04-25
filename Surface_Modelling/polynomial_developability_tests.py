#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module is not intended for use per se. This module is just an exploration
on the developability conditions of polynomial surfaces of different orders.
"""


#%%                    IMPORTATION OF LIBRARIES

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import Surface_Modelling.Domain as dom
import Surface_Modelling.taylor_funcs as tayf


#%%              DEFINITION OF DOMAIN

# We create the rectangular over which a surface will be defined. 
xy_plane = dom.RectDomain(dx = 0.01, dy = 0.01, dims = (1,1,1))
X,Y    = xy_plane.X, xy_plane.Y
XYpows = xy_plane.XYpows
    
#%%         TESTS WITH SHAPE (3,3)

# We create a symbolic polynomial.
sym_z = tayf.create_symbolic_matrix((3,3), symbol="C")

print("--------- ORIGINAL SYMBOLIC POLYNOMIAL ----------")
print(sym_z)

# We compute the derivatives of the symbolic polynomial.
sym_z_Dxx = tayf.sym_polydiff(sym_z, order=(2,0))
sym_z_Dyy = tayf.sym_polydiff(sym_z, order=(0,2))
sym_z_Dxy = tayf.sym_polydiff(sym_z, order=(1,1))

# We compute the symbolic polynomial of the Gaussian Curvature condition.
print("---------Kcond SYMBOLIC POLYNOMIAL ----------")
Kcond = tayf.sym_polymul(sym_z_Dxx, sym_z_Dyy) - tayf.sym_polymul(sym_z_Dxy, sym_z_Dxy)
print(Kcond)



# We print out each of the coefficients of this new Gaussian-Curvature-Condition
# polynomial. Equating each one of these to zero defines a set of nonlinear
# algebraic equations that, when solved, gives us the condition that each of
# the coefficients of the original polynomial should obey in order to describe
# a developable surface (i.e, Gaussian curvature equal to zero).
print("---------ORIGINAL SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond[i,j] != 0:
            print(f"{Kcond[i,j]}")
    
            
            
# As we can see the code spits out a lot of equations. However, a few 
# observations and study of these will make our life easier:
    
# 1) Coefficients C00, C01 and C10 do not appear (and never will) in Kcond. 
#    As such, these coeffcients may be chosen with complete freedom without
#    affecting the Gaussian Curvature.
#
# 2) Some of the resulting equations are quite easy to solve. They say, 
#    unequivocally, that one coefficient should be zero. Then, if we make said
#    coefficient equal to zero and substitute it back into the system of
#    equations, the latter simplifies and we obatin a new easier system.
#    In turn, this new system may contain another easy-to-solve equation that 
#    tells us that another coefficient should be zero. We can then repeat this 
#    process until no further simplification is possible. 
#
#    Having done this for multiple cases, we see that a pattern emerges: For a
#    polynomial surface of order (n, n), the resulting equations for Kcond are 
#    such that require that: Cij = 0, for i + j >= n.


# Thus, further simplifying the sistem of equations yields:
sym_z2 = sym_z.copy()        
Kcond2 = Kcond.copy() 
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        Kcond2[i,j] = Kcond2[i,j].subs("C22",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C21",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C12",0)

        
print("---------SIMPLIFIED SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond2[i,j] != 0:
            print(f"{Kcond2[i,j]}")
        
    
# In the case of an surface polynomial of (3,3) order, we see that the 
# simplified system of equations is a second-degree underdetermined system.
# As such, we decide to parametrize coefficients C11 and C20, and compute C02
# from them both using the last relation 4*C02*C20 - C11**2 = 0.


for C20 in np.linspace(-5, 5, 20):
    for C11 in np.linspace(-5, 5, 20):
        
        C02 = 0.25*(C11**2/C20)
        
        poly_1 = np.array([ [0,    -1,    C02,],
                            [1,    C11,   0,],
                            [C20,  0,     0,  ]])
        
        
        
        
        Z = tayf.polyeval(poly_1, XYpows)
        
        
        # We may re-scale the polynomial surface so its range of values does not
        # exceed 1. With developable surfaces, the effect that this scaling 
        # has on the surface should be negligible.
        Z_range = Z.max() - Z.min()
        if Z_range > 1:
            poly_1 /= Z_range
            
        Z = tayf.polyeval(poly_1, XYpows)
        
        
        # ------ PLOTS --------
        fig = plt.figure(figsize =(14, 9))
        ax = plt.axes(projection ='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_title("Original Surface Z")
        plt.show()
        
        
        #We compute the curvatures of original polynomial.
        H, K = tayf.poly_curvature(poly_1, XYpows)
        
        plt.contourf(X,Y, abs(K))
        plt.title("Abs of Gaussian curvature")
        plt.colorbar()
        plt.show()
        
        
        # plt.contourf(X,Y, abs(H))
        # plt.title("Abs of Mean curvature")
        # plt.colorbar()
        # plt.show()





#%%              TESTS WITH SHAPE (4,4)

# We create a symbolic polynomial.
sym_z = tayf.create_symbolic_matrix((4,4), symbol="C")

print("--------- ORIGINAL SYMBOLIC POLYNOMIAL ----------")
print(sym_z)

# We compute the derivatives of the symbolic polynomial.
sym_z_Dxx = tayf.sym_polydiff(sym_z, order=(2,0))
sym_z_Dyy = tayf.sym_polydiff(sym_z, order=(0,2))
sym_z_Dxy = tayf.sym_polydiff(sym_z, order=(1,1))

# We compute the symbolic polynomial of the Gaussian Curvature condition.
print("---------Kcond SYMBOLIC POLYNOMIAL ----------")
Kcond = tayf.sym_polymul(sym_z_Dxx, sym_z_Dyy) - tayf.sym_polymul(sym_z_Dxy, sym_z_Dxy)
print(Kcond)


# We print out each of the coefficients of this new Gaussian-Curvature-Condition
# polynomial. Equating each one of these to zero defines a set of nonlinear
# algebraic equations that, when solved, gives us the condition that each of
# the coefficients of the original polynomial should obey in order to describe
# a developable surface (i.e, Gaussian curvature equal to zero).
print("---------ORIGINAL SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond[i,j] != 0:
            print(f"{Kcond[i,j]}")
    
            
            
# As we can see the code spits out a lot of equations. However, a few 
# observations and study of these will make our life easier:
    
# 1) Coefficients C00, C01 and C10 do not appear (and never will) in Kcond. 
#    As such, these coeffcients may be chosen with complete freedom without
#    affecting the Gaussian Curvature.
#
# 2) Some of the resulting equations are quite easy to solve. They say, 
#    unequivocally, that one coefficient should be zero. Then, if we make said
#    coefficient equal to zero and substitute it back into the system of
#    equations, the latter simplifies and we obatin a new easier system.
#    In turn, this new system may contain another easy-to-solve equation that 
#    tells us that another coefficient should be zero. We can then repeat this 
#    process until no further simplification is possible. 
#
#    Having done this for multiple cases, we see that a pattern emerges: For a
#    polynomial surface of order (n, n), the resulting equations for Kcond are 
#    such that require that: Cij = 0, for i + j >= n.


# Thus, further simplifying the sistem of equations yields:
sym_z2 = sym_z.copy()        
Kcond2 = Kcond.copy() 
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        Kcond2[i,j] = Kcond2[i,j].subs("C33",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C32",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C31",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C23",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C22",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C13",0)
        
        # Kcond2[i,j] = Kcond2[i,j].subs("C11","2*(C02*C20)**0.5")
        # Kcond2[i,j] = Kcond2[i,j].subs("C12","2*(C03*C21)**0.5")
        
print("---------SIMPLIFIED SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond2[i,j] != 0:
            print(f"{Kcond2[i,j]}")
        

        
# In the case of an surface polynomial of (4,4) order, we see that the 
# simplified system of equations is a first-degree underdetermined system. 
# As such, we decide to parametrize coefficient C11 and compute the rest of the 
# coefficients from the equations, numerically (using fsolve). 

# Now, it looks like this system of nonlinear equations exhibits multiple
# valid solutions for a single value of C11 (unlike with the one obtained from
# the (3,3) order system, whose solution was uniquely specified by C11 and C20).
# That means, that, even for the same value of C11, there may be more than one
# valid solution. We may attempt to find such solutions numerically by varying 
# the 'initial guess' parameter of the fsolve function.

for C11 in [-0.5]:

    def equations(p):
        C02, C20, C21, C03, C12, C30 = p
        
        eq1 = 4*C02*C20 - C11**2
        eq2 = 4*C02*C21 + 12*C03*C20 - 4*C11*C12
        eq3 = 12*C03*C21 - 4*C12**2
        eq4 = 12*C02*C30 - 4*C11*C21 + 4*C12*C20
        eq5 = 36*C03*C30 - 4*C12*C21
        eq6 = 12*C12*C30 - 4*C21**2
        
        return (eq1, eq2, eq3, eq4, eq5, eq6)
    
    C02, C20, C21, C03, C12, C30 =  fsolve(equations, (1, 1, 1, 1, 1, 1))
    
    print((C02, C20, C21, C03, C12, C30))
    #print(equations((C02, C20, C21, C03, C12, C30)))

    
    poly_1 = np.array([ [0,    -1,  C02,    C03],
                        [1,    C11,   C12,    0],
                        [C20,  C21,   0,      0], 
                        [C30,  0,     0,      0]])
    
    
    
    
    Z = tayf.polyeval(poly_1, XYpows)
    
    
    # We may re-scale the polynomial surface so its range of values does not
    # exceed 1. With developable surfaces, the effect that this scaling 
    # has on the surface should be negligible.
    # Z_range = Z.max() - Z.min()
    # if Z_range > 1:
    #     poly_1 /= Z_range
        
    # Z = tayf.polyeval(poly_1, XYpows)
    
    
    # ------ PLOTS --------
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_title("Original Surface Z")
    plt.show()
    
    
    # We compute the curvatures of original polynomial.
    H, K = tayf.poly_curvature(poly_1, XYpows)
    
    plt.contourf(X,Y, abs(K))
    plt.title("Abs of Gaussian curvature")
    plt.colorbar()
    plt.show()
    
    
    # plt.contourf(X,Y, abs(H))
    # plt.title("Abs of Mean curvature")
    # plt.colorbar()
    # plt.show()




#%%              TESTS WITH ORDER (5,5)

# We create a symbolic polynomial.
sym_z = tayf.create_symbolic_matrix((5,5), symbol="C")

print("--------- ORIGINAL SYMBOLIC POLYNOMIAL ----------")
print(sym_z)

# We compute the derivatives of the symbolic polynomial.
sym_z_Dxx = tayf.sym_polydiff(sym_z, order=(2,0))
sym_z_Dyy = tayf.sym_polydiff(sym_z, order=(0,2))
sym_z_Dxy = tayf.sym_polydiff(sym_z, order=(1,1))

# We compute the symbolic polynomial of the Gaussian Curvature condition.
print("---------Kcond SYMBOLIC POLYNOMIAL ----------")
Kcond = tayf.sym_polymul(sym_z_Dxx, sym_z_Dyy) - tayf.sym_polymul(sym_z_Dxy, sym_z_Dxy)
print(Kcond)


# We print out each of the coefficients of this new Gaussian-Curvature-Condition
# polynomial. Equating each one of these to zero defines a set of nonlinear
# algebraic equations that, when solved, gives us the condition that each of
# the coefficients of the original polynomial should obey in order to describe
# a developable surface (i.e, Gaussian curvature equal to zero).
print("---------ORIGINAL SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond[i,j] != 0:
            print(f"{Kcond[i,j]}")
    
            
            
# As we can see the code spits out a lot of equations. However, a few 
# observations and study of these will make our life easier:
    
# 1) Coefficients C00, C01 and C10 do not appear (and never will) in Kcond. 
#    As such, these coeffcients may be chosen with complete freedom without
#    affecting the Gaussian Curvature.
#
# 2) Some of the resulting equations are quite easy to solve. They say, 
#    unequivocally, that one coefficient should be zero. Then, if we make said
#    coefficient equal to zero and substitute it back into the system of
#    equations, the latter simplifies and we obatin a new easier system.
#    In turn, this new system may contain another easy-to-solve equation that 
#    tells us that another coefficient should be zero. We can then repeat this 
#    process until no further simplification is possible. 
#
#    Having done this for multiple cases, we see that a pattern emerges: For a
#    polynomial surface of order (n, n), the resulting equations for Kcond are 
#    such that require that: Cij = 0, for i + j >= n.


# Thus, further simplifying the sistem of equations yields:
sym_z2 = sym_z.copy()        
Kcond2 = Kcond.copy() 
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        Kcond2[i,j] = Kcond2[i,j].subs("C44",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C43",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C42",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C41",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C34",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C33",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C24",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C32",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C23",0)
        Kcond2[i,j] = Kcond2[i,j].subs("C14",0)

        

        
print("---------SIMPLIFIED SYSTEM OF EQUATIONS ----------")
for i in range(Kcond.shape[0]):
    for j in range(Kcond.shape[1]):
        if Kcond2[i,j] != 0:
            print(f"{Kcond2[i,j]}")
        
#%%
        
# In the case of an surface polynomial of (5,5) order, we see that the 
# simplified system of equations is a third-degree overdetermined system. }

# As such.... we are not really sure what to do.

for C11 in np.linspace(0, 30, 100):

    def equations(p):
        C20, C30, C40, C21, C31, C02, C12, C22, C03, C13, C04 = p
        
        eq1 = 4*C02*C20 - C11**2
        eq2 = 4*C02*C21 + 12*C03*C20 - 4*C11*C12
        eq3 = 4*C02*C22 + 12*C03*C21 + 24*C04*C20 - 6*C11*C13 - 4*C12**2
        eq4 = 12*C03*C22 + 24*C04*C21 - 12*C12*C13
        eq5 = 24*C04*C22 - 9*C13**2
        eq6 = 12*C02*C30 - 4*C11*C21 + 4*C12*C20
        eq7 = 12*C02*C31 + 36*C03*C30 - 8*C11*C22 - 4*C12*C21 + 12*C13*C20
        eq8 = 36*C03*C31 + 72*C04*C30 - 12*C12*C22
        eq9 = 72*C04*C31 - 12*C13*C22
        eq10 = 24*C02*C40 - 6*C11*C31 + 12*C12*C30 + 4*C20*C22 - 4*C21**2
        eq11 = 72*C03*C40 + 36*C13*C30 - 12*C21*C22


        
        return (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11)
    
    C20, C30, C40, C21, C31, C02, C12, C22, C03, C13, C04 =  fsolve(equations, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    
    print((C20, C30, C40, C21, C31, C02, C12, C22, C03, C13, C04))
    print(equations((C20, C30, C40, C21, C31, C02, C12, C22, C03, C13, C04)))
    
    eq12 = 144*C04*C40 + 18*C13*C31 - 12*C22**2
    eq13 = 24*C12*C40 - 12*C21*C31 + 12*C22*C30
    eq14 = 72*C13*C40 - 12*C22*C31
    eq15 = 24*C22*C40 - 9*C31**2
    
    print(eq12, eq13, eq14, eq15)

    
    poly_1 = np.array([[0, 3, C02, C03, C04],
                       [-0.5, C11, C12, C13, 0],
                       [C20, C21, C22, 0, 0],
                       [C30, C31, 0, 0, 0],
                       [C40, 0, 0, 0, 0]])






    Z = tayf.polyeval(poly_1, XYpows)
    
    # We may re-scale the polynomial surface so its range of values does not
    # exceed 1. With developable surfaces, the effect that this scaling 
    # has on the surface should be negligible.
    # Z_range = Z.max() - Z.min()
    # if Z_range > 1:
    #     poly_1 /= Z_range
        
    # Z = tayf.polyeval(poly_1, XYpows)
    
    # ------ PLOTS --------
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_title("Original Surface Z")
    plt.show()
    
    
    # We compute the curvatures of original polynomial.
    H, K = tayf.poly_curvature(poly_1, XYpows)
    
    Kmax = H + np.sqrt(H**2-K)
    Kmin = H - np.sqrt(H**2-K)
    
    plt.contourf(X,Y, abs(K))
    plt.title("Abs of Gaussian curvature")
    plt.colorbar()
    plt.show()
    
    # plt.contourf(X,Y, abs(H))
    # plt.title("Abs of Gaussian curvature")
    # plt.colorbar()
    # plt.show()
    

    







