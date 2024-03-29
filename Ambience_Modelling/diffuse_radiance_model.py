#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions related to the computation of the 
spatial distribution of diffuse irradiance, also called radiance. This is 
accomplished by the radiance model developed by Iagawa et al. 
in their 2004 and 2014 papers. The implementation here is mainly based off the
2014 paper, but making use of some expressions that only appeared in the 2004 
paper.

"""

#%%                    IMPORTATION OF LIBRARIES
import numpy as np
import pandas as pd
import Ambience_Modelling.auxiliary_funcs as aux

#%%              DEFINITION OF IMPROVED ALL SKY MODEL CONSTANTS

_IMPROVED_ALL_SKY_MODEL_COLS  = ["a", "b", "c", "d", "e"]
_IMPROVED_ALL_SKY_MODEL_INDEX = ["A", "B", "C", "D", "E", "F", "G", "H"]

IMPROVED_ALL_SKY_MODEL_CTS =\
pd.DataFrame(index   = _IMPROVED_ALL_SKY_MODEL_INDEX,
             columns = _IMPROVED_ALL_SKY_MODEL_COLS).astype(float)


IMPROVED_ALL_SKY_MODEL_CTS.loc["A",:] = -1.0193, -0.3646, -3.3246, -3.8472,  -0.6370 
IMPROVED_ALL_SKY_MODEL_CTS.loc["B",:] = -0.0955,  0.8806,  1.8413,  2.1573,   0.5995
IMPROVED_ALL_SKY_MODEL_CTS.loc["C",:] = -0.0823,  1.6503,  0.8436, -0.5050,   1.0259
IMPROVED_ALL_SKY_MODEL_CTS.loc["D",:] =  0.4530,  0.3319,  0.3009,  0.6257,   1.3334
IMPROVED_ALL_SKY_MODEL_CTS.loc["E",:] = -0.1294, -0.6525,  8.3642,  61.0275, -0.0022
IMPROVED_ALL_SKY_MODEL_CTS.loc["F",:] = -0.2876, -0.2681,  0.8183, -3.2725,   1.0765
IMPROVED_ALL_SKY_MODEL_CTS.loc["G",:] =  0.3169,  0.5434,  0.5424,  1.2096,   0.7066
IMPROVED_ALL_SKY_MODEL_CTS.loc["H",:] =  6.4046, -12.3328, 9.1901,  31.1039,  0.5187


#%%                  DEFINITION OF LzEd CONSTANTS

_MULTIINDEX = []
for _k in range(5, -1, -1):
    for _j in range(6, -1, -1):
        _MULTIINDEX.append((_k, _j))
    
_COLUMNS = [i for i in range(5,-1,-1)]
_MULTIINDEX = pd.MultiIndex.from_tuples(_MULTIINDEX)
LZED_CTS = pd.DataFrame(index = _MULTIINDEX, columns = _COLUMNS).astype(float) 

LZED_CTS.loc[(5, 6),:] =  5.6146, -29.4046,  47.2024, -43.8510,   8.2509, -0.9358
LZED_CTS.loc[(5, 5),:] = -17.9921, 93.4316, -142.8905, 130.9200, -17.7456, 2.6364          
LZED_CTS.loc[(5, 4),:] =  20.0121,-103.1918, 142.9116,-130.0067,  3.1167, -3.7005
LZED_CTS.loc[(5, 3),:] = -12.0503, 55.2228, -58.2657,  49.5379,   14.3877, 3.5037
LZED_CTS.loc[(5, 2),:] =  8.2042, -28.2605,  23.5534, -13.0987,  -9.0805, -2.2572
LZED_CTS.loc[(5, 1),:] = -2.2514,  7.3074,  -5.7338,   2.4593,    2.3038,  1.2745
LZED_CTS.loc[(5, 0),:] =  0.4774, -1.2853,   0.8565,  -0.2806,   -0.1641, -0.7447

LZED_CTS.loc[(4, 6),:] = -17.2129,  85.8973, -129.4606,  125.4744, -16.6675, -1.7011
LZED_CTS.loc[(4, 5),:] =  63.0588, -298.9370, 420.7243, -391.1156,  25.7323,  8.4401
LZED_CTS.loc[(4, 4),:] = -86.5230,  382.9478,-477.7507,  419.8383,  28.0500, -10.4232
LZED_CTS.loc[(4, 3),:] =  64.5195, -250.6187, 249.3821, -189.4251, -70.2059,  1.0365
LZED_CTS.loc[(4, 2),:] = -36.9118,  122.2518,-103.4001,  56.5677,   38.5437,  4.9664
LZED_CTS.loc[(4, 1),:] =  8.3944,  -26.3761,  19.1065,  -8.7967,   -9.4755,  -3.6080
LZED_CTS.loc[(4, 0),:] = -1.6652,   4.5943,  -3.1165,    1.4959,    0.5221,   1.9573

LZED_CTS.loc[(3, 6),:] =  21.5603, -98.3234,  133.2000, -134.7364, 5.7213,   7.9890
LZED_CTS.loc[(3, 5),:] = -88.8005,  376.6700, -473.6141, 443.8715, 15.9462, -31.5361
LZED_CTS.loc[(3, 4),:] =  140.5464,-549.7882, 617.7442, -524.2791, -92.1837, 41.4865
LZED_CTS.loc[(3, 3),:] = -115.2602, 408.1553, -389.1329, 279.5759, 121.5988,-18.9449
LZED_CTS.loc[(3, 2),:] =  58.4325, -188.1080, 158.1039, -90.2370, -60.4685, -0.8295
LZED_CTS.loc[(3, 1),:] = -12.5318,  38.1286, -26.3229,   14.5404,  13.3797,  2.5300
LZED_CTS.loc[(3, 0),:] =  1.7622,  -5.0850,   2.9477,   -2.1838,  -0.5745,  -1.2611

LZED_CTS.loc[(2, 6),:] = -16.1603, 62.0261,  -68.6303,  66.7874,    9.3995,  -8.0240
LZED_CTS.loc[(2, 5),:] =  68.1074, -249.5476, 263.2480, -233.4506, -51.2836, 30.4587
LZED_CTS.loc[(2, 4),:] = -110.3658, 384.7705, -376.5734, 301.1853, 105.3289, -41.6451
LZED_CTS.loc[(2, 3),:] =  88.4298, -291.6143, 255.1865, -180.4192, -100.9524, 24.4274
LZED_CTS.loc[(2, 2),:] = -39.1455, 122.2380, -95.2499, 60.1343, 43.8912, -5.8629 
LZED_CTS.loc[(2, 1),:] =  8.5411, -25.5973, 17.1831, -11.9369, -7.4727, 0.8271
LZED_CTS.loc[(2, 0),:] = -0.5530, 1.8213, -0.3930, 1.0051, 0.2158, -0.0791

LZED_CTS.loc[(1, 6),:] =  5.6538, -18.5946, 15.3888, -15.0642, -6.8261, 2.4525
LZED_CTS.loc[(1, 5),:] = -22.4881, 72.5977, -58.6626, 54.7188, 28.0338, -9.9369
LZED_CTS.loc[(1, 4),:] =  34.5496, -109.0127, 83.4590, -75.1759, -45.1168, 15.8059
LZED_CTS.loc[(1, 3),:] = -26.0768, 80.1132, -55.9029, 49.8447, 34.7254, -12.6379
LZED_CTS.loc[(1, 2),:] =  10.1609, -30.7499, 19.0722, -17.7449, -11.9372, 5.3456
LZED_CTS.loc[(1, 1),:] = -1.4801, 4.7414, -1.9300, 2.6996, 1.2676, -1.0207
LZED_CTS.loc[(1, 0),:] =  0.0550, -0.2373, -0.0316, -0.0642, 0.0032, -0.0227

LZED_CTS.loc[(0, 6),:] = -0.8791, 3.2070, -2.8856, 3.0796, 0.2823, 0.1061
LZED_CTS.loc[(0, 5),:] =  2.7495, -10.1893, 8.5197, -10.6148, -1.0694, 0.2046
LZED_CTS.loc[(0, 4),:] = -3.0179, 11.6684, -8.6199, 14.0185, 1.3755, -1.7036
LZED_CTS.loc[(0, 3),:] =  1.1932, -5.4566, 3.0029, -8.7173, -0.5736, 2.7262
LZED_CTS.loc[(0, 2),:] = -0.0024, 0.7879, -0.0560, 2.4222, -0.1517, -1.4338
LZED_CTS.loc[(0, 1),:] =  0.0089, -0.1344, 0.1890, -0.1446, 0.1348, -0.1598
LZED_CTS.loc[(0, 0),:] = -0.0018, 0.0124, -0.0062, -0.0134, -0.0078, 0.4086


#%%        DEFINITION OF FUNCTIONS FOR COMPUTING THE CLOUDLESS INDEX CLE

def compute_cloud_ratio_Ce(Gdh, Gh):
    
    """
    Compute Igawa's 'cloud ratio' (Ce) parameter.
    
    Parameters
    ----------
    Gdh : float or numpy.array of floats
        Diffuse horizontal irradiance. Must be a non-negative number or
        array of numbers. If it is a numpy.array, it should be the same shape
        as 'Gdh'.
        
    Gh : float or numpy.array of floats
        Global horizontal irradiance. Must be a positive number or array of 
        numbers. If it is a numpy.array, it should be the shape as 'Gh'. 
    
    Returns
    ---------
    Ce : float or numpy.array of floats 
        Igawa's 'cloud ratio' (Ce) parameter. It is float if params are float
        and a numpy.array of equal shape if params are numpy.arrays.
    
    """
    Ce = Gdh/Gh
    return Ce


def compute_standard_cloud_ratio_Ces(sun_apel):
    """
    Compute Igawa's 'standard cloud ratio' (Ces) parameter.
    
    Parameters
    ----------
    sun_apel : float or numpy.array of floats
        Sun's apparent elevation in degrees. Must be a number or array of 
        numbers between 0 and 90.
        
    Returns
    ---------
    Ces : float or numpy.array of floats 
        Igawa's 'standard cloud ratio' (Ces) parameter. It is float if params
        are float and a numpy.array of equal shape if params are numpy.arrays.
        
    Notes
    -----
    1) Ces must be a number between 0 and 1. As such, any values above 1 are
    clipped down to 1. It is float if params are float and a numpy.array of 
    equal shape if params are numpy.arrays.
    
    """
    
    sun_apel_ = np.deg2rad(sun_apel)
    
    Ces  = 0.08302 
    Ces += 0.5358*np.exp(-17.3*sun_apel_)
    Ces += 0.3818*np.exp(-3.2899*sun_apel_)
    Ces[Ces>1] = 1
    
    return Ces


def compute_cloudless_index_Cle(Ce, Ces):
    """
    Compute Igawa's 'cloudless index' (Cle) parameter.
    
    Parameters
    ----------
    Ce : float or numpy.array of floats 
        Igawa's 'cloud ratio' (Ce) parameter. Must be a number or array of
        numbers between 0 and 1. If it is a numpy.array, it should
        be the same shape as 'Ces'. 
        
    Ces : float or numpy.array of floats 
        Igawa's 'standard cloud ratio' (Ces) parameter. Must be a number or
        array of numbers between 0 and 1. If it is a numpy.array, it should be 
        the same shape as 'Ce'. 
        
    Returns
    ---------
    CLe : float or numpy.array of floats 
        Igawa's 'cloudless index' (Cle) parameter. It is float if params are
        float and a numpy.array of equal shape if params are numpy.arrays.
        
    Notes
    -----
    1) Values of Cle above 1.2 are clipped down to 1.2. The reason is that
       in Igawa's 2014 paper, all values reported for Cle were at or below 1.2,
       which makes me believe that a Cle of 1.2 is the maximum 
       physically-sensible value for Cle.
       
    """
    
    Cle = (1 - Ce)/(1 - Ces)
    Cle[Cle>1.2] = 1.2
    return Cle
    
    
#%%     DEFINITION OF FUNCTIONS FOR COMPUTING THE CLEAR SKY INDEX KC

def compute_standard_global_irradiance_Ghs(extra_Gbn, rel_airmass):
    
    """
    Compute Igawa's 'standard global irradiance' (Ghs) parameter.
    
    Parameters
    ----------
    extra_Gbn : float or numpy.array of floats 
        Extraterrestrial normal irradiance [W/m^2]. Must be a non-negative
        number or array of numbers. If it is a numpy.array, it should be the 
        same shape as 'rel_airmass'.
        
        
    rel_airmass : float or numpy.array of floats 
        Relative airmass [adimensional]. Must be a positive number or array of 
        numbers. If it is a numpy.array, it should be the same shape as 
        'extra_Gbn'.
        
    Returns
    ---------
    Ghs : float or numpy.array of floats 
        Igawa's 'standard global irradiance' (Ghs) parameter. It is float if
        params are float and a numpy.array of equal shape if params are numpy.arrays.
    """
    Ghs = 0.84*(extra_Gbn/rel_airmass)*np.exp(-0.054*rel_airmass)
    return Ghs



def compute_clear_sky_index_Kc(Gh, Ghs):
    """
    Compute Igawa's 'clear sky index' (Kc) parameter.
    
    Parameters
    ----------
    Gh : float or numpy.array of floats 
       Global horizontal irradiance [W/m^2]. Must be a non-negative number or array
       of numbers. If it is a numpy.array, it should be the same shape as 'Ghs'.
        
        
    Ghs : float or numpy.array of floats 
        Standard global irradiance [W/m^2]. Must be a positive number or array
        of numbers. If it is a numpy.array, it should be the same shape as 
        'Gh'.
        
    Returns
    ---------
    Kc : float or numpy.array of floats 
        Igawa's 'clear sky index' (Kc) parameter. It is float if params are float
        and a numpy.array of equal shape if params are numpy.arrays.
        
    Notes
    -----
    1) Values of Kc above 1.2 are clipped down to 1.2. The reason is that
       in Igawa's 2014 paper, all values reported for Kc were at or below 1.2,
       which makes me believe that a Cle of Kc is the maximum 
       physically-sensible value for Kc.
       
    """
    
    Kc = Gh/Ghs
    Kc[Kc>1.2] = 1.2
    return Kc


#%%    DEFINITION OF FUNCTIONS FOR COMPUTING THE SKY INDEX Siv

def compute_sky_index_Siv(Kc, Cle):
    
    """
    Compute Igawa's 'Sky index' (Siv) parameter used for classifying types of
    sky.
    
    Parameters
    ----------
    Kc : float or numpy.array of floats 
       Igawa's 'clear sky index' (Kc) parameter. Must be a number of array 
       of numbers between 0 and 1.2. If it is a numpy.array, it should be the
       same shape as Cle.
        
    Cle : float or numpy.array of floats 
        Igawa's 'cloudless index' (Cle) parameter. Must be a number of array 
        of numbers between 0 and 1.2. If it is a numpy.array, it should be the
        same shape as Kc.    
        
    Returns 
    -------
    Siv : float or numpy.array of floats
        Igawa's 'Sky index' (Siv) parameter. It is float if params are float and a 
        numpy.array of equal shape if params are numpy.arrays.
    
    """
    

    
    Siv  = (1.0 - Kc)**2
    Siv += (1.0 - np.sqrt(Cle))**2
    Siv = np.sqrt(Siv)
    
    return Siv




#%%   DEFINITION OF FUNCTIONS FOR COMPUTING THE IMPROVED ALL SKY MODEL COEFFICIENTS


def compute_improved_all_sky_model_coeffs(coeff_name, Kc, Cle):
    
    """
    Compute Igawa's improved all sky model coefficients.
    
    Parameters
    ----------
    coeff_name : str
        Name of the coefficient to compute. 
        Supported are: "a", "b", "c", "d", "e".
        
    Kc : float or numpy.array of floats 
       Igawa's 'clear sky index' (Kc) parameter. Must be a number of array 
       of numbers between 0 and 1.2. If it is a numpy.array, it should be the
       same shape as Cle.
        
    Cle : float or numpy.array of floats 
        Igawa's 'cloudless index' (Cle) parameter. Must be a number of array 
        of numbers between 0 and 1.2. If it is a numpy.array, it should be the
        same shape as Kc.    
        
    Returns
    -------
    coeff_val : float or numpy.array of floats 
        Value of the specified sky model coefficients. It is float if params 
        are float and a numpy.array of equal shape if params are numpy.arrays.
    
    
    """
        
    
    A, B, C, D, E, F, G, H =\
    IMPROVED_ALL_SKY_MODEL_CTS[coeff_name]
    
    Gkc  = ((Kc - C)/D)**2
    Gcle = ((Cle - F)/G)**2
    
    coeff_val  = A
    coeff_val += B*np.exp(-Gkc/2)
    coeff_val += E*np.exp(-Gcle/2)
    coeff_val += H*np.exp(-(Gkc + Gcle)/2)
    
    
    try:
        coeff_val_len = len(coeff_val)
    except TypeError:
        coeff_val_len = 0
        coeff_val = np.array([coeff_val])
        
        
    if coeff_name == "b":
        coeff_val[coeff_val > 0] = 0
    
    elif coeff_name in ["c", "e"]:
        coeff_val[coeff_val < 0] = 0
        
    if coeff_val_len == 0:
        coeff_val = coeff_val[0]
        

    return coeff_val


#%%  DEFINITION OF FUNCTIONS FOR COMPUTING THE INVERSE OF THE INTEGRATION VALUE OF RELATIVE SKY RADIANCE DISTRIBUTION LzEd

def compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_using_constants_LzEd(Kc, Cle, sun_apel):
    """
    Compute the inverse of the integration value of relative sky radiance 
    distribution, according to Igawa's 2014 paper, using the non-linear
    regression constants proposed by the author.
    
    Parameters
    ----------
    Kc : float or numpy.array of floats 
        Igawa's 'clear sky index' (Kc) parameter. Must be a number of array 
        of numbers between 0 and 1.2. If it is a numpy.array, it should be the
        same shape as sun_apel.
        
    Cle : float or numpy.array of floats 
        Igawa's 'cloudless index' (Cle) parameter. Must be a number of array 
        of numbers between 0 and 1.2. If it is a numpy.array, it should be the
        same shape as sun_apel.   
        
    sun_apel : float or numpy.array of floats
        Sun's apparent elevation in degrees. Must be a number or array of 
        numbers between 0 and 90.

    Returns
    -------
    LzEd : float or numpy.array of floats
        Inverse of the integration value of relative sky radiance distribution.
        It is float if params are float and a numpy.array of equal shape if
        params are numpy.arrays.

    """
    
    sun_apel_ = np.deg2rad(sun_apel)
    
    LzEd =\
    sum([sum([sum([LZED_CTS.loc[(k,j),i]*sun_apel_**i for i in range(6) ])*Cle**j for j in range(6) ])*Kc**k for k in range(6)])

    return LzEd    




def compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_numerically_LzEd(Le, El, dAz, dEl):
    """
    Compute the inverse of the integration value of relative sky radiance 
    distribution, according to Igawa's 2014 paper, by numerically computing the 
    integral of the relative sky radiance distribution using the trapezoidal 
    rule.
    
    Parameters
    ----------
    Le : numpy.array of floats with shape (E,A,T)
        Un-normalized relative sky radiance distribution in W/m^2/sr,
        across time. 'Le[n,m,t]' holds the radiance of a sky element with
        elevation specified by n, azimuth specified by m on a time specified by 
        t. In other words, axis 0 of 'Le' takes account of the variation of 
        radiance across the sky with respect to elevation, axis 1 accounts for 
        the variation with respect to azimuth and axis 2 accounts for the
        variation with respect to time. Values of 'le' must be non-negative.
        Values must be non-negative.
        
    El : numpy.array of floats with shape (E,A,1)
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of 'LzEd'. The values of 'El' should vary along axis 0.
       Values must lie between 0 and 90 (inclusive).
       
    dAz : float
        Angular resolution of azimuth values in degrees.
        Values must be positive.
        
    dEl : float
        Angular resolution of elevation values in degrees.
        Values must be positive.
        

        

    Returns
    -------
    LzEd : float or numpy.array of floats with shape (1,1,T)
        Inverse of the integration value of relative sky radiance distribution.

    """
    
    El_ = np.deg2rad(El)
    dAz_ = np.deg2rad(dAz)
    dEl_ = np.deg2rad(dEl)
    
    
    integrand = Le*np.sin(El_)*np.cos(El_)
    
    LzEd =\
    0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])

    
    LzEd = LzEd.sum(axis=(0,1))*dEl_*dAz_
    LzEd = (1/LzEd).reshape(1,1,Le.shape[-1])
    
    return LzEd
    




#%%         DEFINITION OF THE GRADATION FUNCTION

def gradation_function(El, a, b):
    
    """
    Compute Igawa's gradation function.
    
    Parameters
    ----------
    
    El : float or numpy.array of floats with shape (E,A,1)
       If float it is the elevation (in degrees) of the sky element for which
       the gradation at that point is to be calculated. Else, it is an
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of 'gradation'. The values of 'El' should vary along axis 0.
       Values of must lie between 0 and 90 (inclusive).
       
    a : float or numpy.array of floats with shape (1,1,T)
      Igawa's improved all sky model coefficient 'a'.
    
    b : float or numpy.array of floats with shape (1,1,T)
      Igawa's improved all sky model coefficient 'b'.
        
    Returns
    -------
    gradation : float or numpy.array of floats with shape (E,A,T)
        Values of the gradation function, at a time or across multiple times.
        
    """
    
    El_ = np.deg2rad(El)
    gradation = 1 + a*np.exp(b/np.sin(El_))
    return gradation
        

#%%       DEFINITION OF FUNCTIONS FOR THE COMPUTATION OF THE DIFFUSION INDICATRIX

def compute_angular_distance_between_sun_and_sky_element_Zeta(Az, El, sun_az, sun_apel):
    
    """
    Compute the angular distance between the sun and one or multiple sky 
    elements on the sky.
    
    Parameters
    ----------
    Az : float or numpy.array of floats with shape (E,A,1)
       If float it is the azimuth (in degrees) of the sky element for which
       its angular distance to the sun is to be calculated. Else, it is an
       Azimuth array of meshgrid of Azimuth, Elevation values. It contains
       the azimuth (in degrees) of each sky element to be considered in 
       the calculation of 'Zeta'. The values of 'Az' should vary along axis 1.
       Values of 'Az' must lie between 0 and 360 (inclusive).
    
    El : float or numpy.array of floats with shape (E,A,1)
       If float it is the elevation (in degrees) of the sky element for which
       its angular distance to the sun is to be calculated. Else, it is an
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of 'Zeta'. The values of 'El' should vary along axis 0.
       Values of 'El' must lie between 0 and 90 (inclusive).
      
    sun_az : float or numpy.array of floats with shape (1,1,T)
        Azimuth coordinate elevation of the sun (in degrees) across time.
        Values of must lie between 0 and 360 (inclusive).
    
    sun_apel : float or numpy.array of floats with shape (1,1,T)
        Apparent elevation coordinate of the sun (in degrees) acorss time.
        Values of must lie between 0 and 90 (inclusive).
    

        
    Returns
    -------
    Zeta : float or numpy.array of floats with shape (E,A,T)
        Angular distance between the sun and all sky element(s) (in degrees)
        at a time or across multiple times.
        
    """
    
    
    
    Az_       = np.deg2rad(Az)
    El_       = np.deg2rad(El)
    sun_az_   = np.deg2rad(sun_az)
    sun_apel_ = np.deg2rad(sun_apel)
    
    
    Zeta_  = np.cos(np.absolute(sun_az_ - Az_))
    Zeta_ *= np.cos(sun_apel_)*np.cos(El_)
    Zeta_ += np.sin(sun_apel_)*np.sin(El_)
    Zeta_  = np.arccos(Zeta_)
    Zeta   = np.rad2deg(Zeta_)
    
    return Zeta




def diffusion_indicatrix_function(Zeta, c, d, e):
    """
    Compute Igawa's diffusion indicatrix function.
    
    Parameters
    ----------
    
    Zeta : float or numpy.array of floats with shape (E,A,T)
        Angular distance between the sun and_sky element(s) (in degrees).
        If a numpy.array, axis0 and axis1 take account of the spatial 
        varaition of 'Zeta' with respect to the Elevation and the Azimuth,
        respectively, while axis 2 takes account of its temporal variation.
    
    c : float or numpy.array of floats with shape (1,1,T)
      Igawa's improved all sky model coefficient 'c'.
    
    d : float or numpy.array of floats with shape (1,1,T)
      Igawa's improved all sky model coefficient 'd'.
      
    e : float or numpy.array of floats with shape (1,1,T)
      Igawa's improved all sky model coefficient 'e'.
        
    Returns
    -------
    diffusion_indicatrix : float or numpy.array of floats with shape (E,A,T)
        Values of the diffusion indicatrix at a time or across multiple times.
    """
    
    Zeta_ = np.deg2rad(Zeta)
    
    diffusion_indicatrix  = 1
    diffusion_indicatrix += c*( np.exp(d*Zeta_) - np.exp(d*np.pi/2) )
    diffusion_indicatrix += e*np.cos(Zeta_)**2
    
    return diffusion_indicatrix

    


#%%          DEFINITION OF FUNCTION FOR COMPUTING THE SKY RADIANCE DISTRIBUTION


def compute_diffuse_radiance(Az, El, dAz, dEl, Gh, Gdh, extra_Gbn, sun_az, sun_apel, rel_airmass, num_iterations=500):
    
    """
    Compute diffuse sky radiance using Igawa's 2014 paper detailed in "Improving the All Sky Model 
    for the luminance and radiance distributions of the sky".
    
    Parameters
    ----------
    Az : numpy.array of floats with shape (E,A,1)
       Azimuth array of meshgrid of Azimuth, Elevation values. It contains
       the azimuth (in degrees) of each sky element to be considered in 
       the calculation of diffuse sky radiance. The values of 'Az' should vary 
       along axis 1. Values should be between 0 and 360.
    
    El : numpy.array of floats with shape (E,A,1)
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of diffuse sky radiance. The values of 'El' should vary 
       along axis 0. Values should be between 0 and 90.
       
    dAz : float
        Angular resolution of 'Az' in degrees.
        
    dEl : float
        Angular resolution of 'El' in degrees.
       
    Gh : numpy.array of floats with shape (1,1,T)  
       Global horizontal irradiance [W/m^2] across time. Must be a
       non-negative array of numbers. 
       
    Gdh : numpy.array of floats with shape (1,1,T) 
        Diffuse horizontal irradiance[W/m^2] across time. Must be a non-negative
        array of numbers.
        
    extra_Gbn : numpy.array of floats with shape (1,1,T) 
        Extraterrestrial normal irradiance [W/m^2]. Must be a non-negative
        array of numbers. 
        
    sun_apel : numpy.array of floats with shape (1,1,T) 
        Sun's apparent elevation (in degrees) across time. Values must lie 
        between 0 and 90 (inclusive).
        
    sun_az : numpy.array of floats with shape (1,1,T) 
        Suns's azimuth (in degrees) across time. Values must lie 
        between 0 and 360 (inclusive).
        
    rel_airmass : numpy.array of floats with shape (1,1,T) 
        Relative airmass [unitless] across time. Values must be non-negative.
        
    num_iterations : int
        Number of iterations to use when filling NaN data. Default is 500.
        
    
    Returns
    -------
    res : dict
        Dictionary containing result variables. It has the following Key-Value
        pairs:
            
            Keys : Values
            -------------
            Siv   : numpy.array of floats with shape (1,1,T)   
                Igawa's 'Sky Index' parameter across time.
            
            "Kc"  : numpy.array of floats with shape (1,1,T) 
                Igawa's 'Clear Sky Index' parameter across time.
                
            "Cle" : numpy.array of floats with shape (1,1,T) 
                Igawa's 'Cloudless Index' parameter across time.
            
                
            "Lea" : numpy.array of floats with shape (E,A,T) 
                Diffuse sky radiance distribution [W/m^2/sr] across time.
                
            "Le" : numpy.array of floats with shape (E,A,T) 
                Relative sky radiance distribution [adimensional] across time.
                
            "Lez" : numpy.array of floats with shape (1,1,T) 
                Zenith radiance distribution [W/m^2/sr] across time.
                
            "LzEd" : numpy.array of floats with shape (1,1,T) 
                Inverse of the integration value of relative sky radiance 
                distribution.
                
    Notes
    -----
    1) This function computes 'LzEd' numerically. I tried using the shortcut
       method proposed by the authors but the resulting values were quite bad.
       I Should  this.
       

    """
    
    
    # ---- COMPUTATION OF CLE ----- 
    
    Ce  = compute_cloud_ratio_Ce(Gdh, Gh)
    Ces = compute_standard_cloud_ratio_Ces(sun_apel)
    Cle = compute_cloudless_index_Cle(Ce, Ces)
    
    # We interpolate any NaN values.
    if np.isnan(Cle).any():
        try:
            Cle[0,0,:] =\
            aux.fill_nans_using_laplace_1D(Cle[0,0,:], num_iterations = num_iterations)
        except IndexError: 
            pass
            
     
    # ---- COMPUTATION OF KC ----- 
    Ghs = compute_standard_global_irradiance_Ghs(extra_Gbn, rel_airmass)
    Kc  = compute_clear_sky_index_Kc(Gh, Ghs)
    
    # We interpolate any NaN values.
    if np.isnan(Kc).any():
        try:
            Kc[0,0,:] =\
            aux.fill_nans_using_laplace_1D(Kc[0,0,:], num_iterations = num_iterations)
        except IndexError: 
            pass
        
        
    # ---- COMPUTATION OF Siv ----- 
    Siv = compute_sky_index_Siv(Kc, Cle)
    
    # We interpolate any NaN values.
    if np.isnan(Siv).any():
        try:
            Siv[0,0,:] =\
            aux.fill_nans_using_laplace_1D(Siv[0,0,:], num_iterations = num_iterations)
        except IndexError: 
            pass
        
    
    # ---- COMPUTATION OF IMPROVED -MODEL CONSTANTS ----- 
    a = compute_improved_all_sky_model_coeffs("a", Kc, Cle)
    b = compute_improved_all_sky_model_coeffs("b", Kc, Cle)
    c = compute_improved_all_sky_model_coeffs("c", Kc, Cle)
    d = compute_improved_all_sky_model_coeffs("d", Kc, Cle)
    e = compute_improved_all_sky_model_coeffs("e", Kc, Cle)
    
    
    # ---- COMPUTATION OF ANGULAR DISTANCES BETWEEN THE SUN AND ALL THE SKY ELEMENTS ----- 
    Zeta = compute_angular_distance_between_sun_and_sky_element_Zeta(Az, El, sun_az, sun_apel)
    
    # ---- COMPUTATION OF RELATIVE SKY RADIANCE DISTRIBUTION ----- 
    Le  = gradation_function(El, a, b)
    Le *= diffusion_indicatrix_function(Zeta, c, d, e)
    
    if isinstance(Le, int) or isinstance(Le, float):
        Le /= gradation_function(90, a, b)
        Le /= diffusion_indicatrix_function(90 - sun_apel, c, d, e)
    else:
        arr_90 = np.full(shape = Le.shape, fill_value = 90)
        Le /= gradation_function(arr_90, a, b)
        Le /= diffusion_indicatrix_function(arr_90 - sun_apel, c, d, e)
        
    # ---- COMPUTATION OF INVERSE OF THE INTEGRATION VALUE OF RELATIVE SKY RADIANCE DISTRRIBUTION ----- 
    LzEd = compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_numerically_LzEd(Le, El, dAz, dEl)
    #LzEd = compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_LzEd(Kc, Cle, sun_apel)
    
    
    # ---- COMPUTATION OF ZENITH RADIANCE ----- 
    Lez = Gdh*LzEd
    
    # ---- COMPUTATION OF SKY RADIANCE DISTRIBUTION ----- 
    Lea = Lez*Le
    
    # Results 
    res = {"Siv":Siv, "Kc":Kc, "Cle":Cle, "Lea":Lea, "Le":Le, "Lez":Lez, "LzEd":LzEd}
    
    return res




#%%                             EXAMPLES       

if __name__ == '__main__':
    
    # We import libraries 
    import time as tm
    import matplotlib.pyplot as plt
    from Ambience_Modelling.Site import Site
    from Ambience_Modelling.auxiliary_funcs import load_obj_with_pickle
    
    # ---- DEFINE EVALUATION POINTS FOR SKY RADIANCE DISTRIBUTION ----
    angular_resolution = 0.5 #[degrees]
    num_pts_el = round(90/angular_resolution) + 1
    num_pts_az = 4*(num_pts_el - 1) + 1
    
    az = np.linspace(0, 360, num_pts_az)
    el = np.linspace(0,  90, num_pts_el)
    
    Az, El = np.meshgrid(az, el)
    
    # Reshape Az, El into the required shapes.
    Az = Az.reshape(list(Az.shape) + [1])
    El = El.reshape(list(El.shape) + [1])
    
    
    # ---- LOAD SITE_OBJ ----
    # As a test, we get the atmospheric data required for this module from
    # precomputed Site obj.
    
    path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Ambience_Modelling\Site_obj_Medellin_2022.pkl"
    Site_obj = load_obj_with_pickle(path = path)
    
    # Scpecify year, day and month of the data.
    # year, month, day = 2022, 1, 1
    year, month, day = 2022, 2, 1
    
    Gh        = Site_obj.site_data[(year, month, day)]["G(h)"]
    Gdh       = Site_obj.site_data[(year, month, day)]["Gd(h)"]
    extra_Gbn = Site_obj.site_data[(year, month, day)]["extra_Gb(n)"]
    sun_apel    = Site_obj.sun_data[(year, month, day)]["apel"]
    sun_az      = Site_obj.sun_data[(year, month, day)]["az"]
    rel_airmass = Site_obj.sun_data[(year, month, day)]["rel_airmass"]
    
    index = Gh.index
    # Transform data from pandas.Series to numpy.arrays
    
    Gh          = np.array(Gh)
    Gdh         = np.array(Gdh)
    extra_Gbn   = np.array(extra_Gbn)
    sun_apel    = np.array(sun_apel)
    sun_az      = np.array(sun_az)
    rel_airmass = np.array(rel_airmass)
    
    # Reshape data into the required shapes.
    
    Gh           = Gh.reshape(1, 1, len(Gh))
    Gdh          = Gdh.reshape(1, 1, len(Gdh))
    extra_Gbn    = extra_Gbn.reshape(1, 1, len(extra_Gbn))
    sun_apel     = sun_apel.reshape(1, 1, len(sun_apel))
    sun_az       = sun_az.reshape(1, 1 ,len(sun_az))
    rel_airmass  = rel_airmass.reshape(1, 1, len(rel_airmass))
    
    # Plot Irradiances
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, Gdh.flatten(), label = "Diffuse Horizontal Irradiance")
    plt.plot(index, Gh.flatten(), label = "Global Horizontal Irradiance")
    plt.title(f"Irradiances for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Irradiance [W/m^2]")
    plt.xlabel("Time [month-day hour]")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
#%%       ----- COMPUTE DIFFUSE SKY RADIANCE DISTRIBUTION ----

    
    t = tm.time()
    res = compute_diffuse_radiance(Az              = Az,  
                                   El              = El,
                                   dAz     = angular_resolution,
                                   dEl     = angular_resolution,
                                   Gh              = Gh,
                                   Gdh             = Gdh, 
                                   extra_Gbn       = extra_Gbn, 
                                   sun_az          = sun_az,
                                   sun_apel        = sun_apel,
                                   rel_airmass     = rel_airmass,
                                   num_iterations  = 500)
    
    dt = tm.time() - t
    
#%%    ---- PLOT SKY RADIANCE DISTRIBUTION FOR ALL TIMES ----

    
    Phi = np.deg2rad(Az).reshape(Az.shape[0], Az.shape[1])
    Theta = np.deg2rad(90 - El).reshape(El.shape[0], El.shape[1]) 
    
    X = np.cos(Phi)*np.sin(Theta)
    Y = np.sin(Phi)*np.sin(Theta)
    Z = np.cos(Theta)

    for nt in range(Gdh.shape[-1]):
        
        Color = res["Lea"][:,:,nt]
        
        max_ = np.max(res["Lea"][:,:,nt])
        if(max_ > 0):
            Color_ = Color/max_
        else:
            Color_ = Color
        
         
        title = f"{Site_obj.name}: Diffuse Radiance at time: {index[nt]}"
        
        
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.view_init(45, 180)
        ax.set_xlabel("X (↑ == N, ↓ == S)")
        ax.set_ylabel("Y (↑ == E, ↓ == W)")
        ax.set_title(title)
        ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
        
        m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
        m.set_array(Color)
        cbar = plt.colorbar(m)
        cbar.ax.set_title('W/m^2/sr')
        
        plt.show()
        
#%%           ---- CHECKING CONSERVATION OF DIFFUSE IRRADIANCE -----   
      
    # Energy must be conserved. As such, the diffuse irradiance on a horizontal
    # plane should be constant. In other words, we should be able to recompute 
    # Gdh from Lea and still get the exact same values.

    Lea = res["Lea"]
    integrand = Lea*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)
    
    recomputed_Gdh =\
    0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])
    
    ang_res = np.deg2rad(angular_resolution)
    recomputed_Gdh = recomputed_Gdh.sum(axis=(0,1))*ang_res*ang_res
    
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, Gdh.flatten(), label = "Original")
    plt.plot(index, recomputed_Gdh, label = "Recomputed")
    plt.title(f"Diffuse horizontal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Diffuse horizontal irradiance [W/m^2]")
    plt.legend()
    plt.grid()
    plt.show()
    

#%%      ---- CHECKING UNFITNESS OF CONSTANTS AS A METHOD FOR COMPUTING LzED 1------

    # Let us compute LzEd numerically and via Igawa's non linear regression in 
    # order to compare them.

    # ------ NUMERICAL ------
    Le = res["Le"]
    integrand = Le*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)
    numerical_LzEd =\
    0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])
    
    ang_res = np.deg2rad(angular_resolution)
    numerical_LzEd = numerical_LzEd.sum(axis=(0,1))*ang_res*ang_res
    numerical_LzEd = 1/numerical_LzEd
    
    # ------ REGRESSION ------
    
    # Plotting
    cts_LzEd = compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_using_constants_LzEd(res["Kc"], res["Cle"], sun_apel)
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, cts_LzEd.flatten(), label = "Igawa's non-linear Regression")
    plt.plot(index, numerical_LzEd, label = "Numerical Computation")
    plt.title(f"LzEd for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("LzEd")
    plt.legend()
    plt.grid()
    plt.show()

#%%      ---- CHECKING UNFITNESS OF CONSTANTS AS A METHOD FOR COMPUTING LzED 2------
    
    # Let's take the last excercise a step further and check that using Igawa's
    # regresio Gdh is not conserved.
    
    new_Lea = res["Le"]*Gdh*cts_LzEd
    integrand = new_Lea*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)
    recomputed_Gdh =\
    0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])
    
    ang_res = np.deg2rad(angular_resolution)
    recomputed_Gdh = recomputed_Gdh.sum(axis=(0,1))*ang_res*ang_res
    
    fig = plt.figure(figsize=(16,12))
    plt.plot(index, Gdh.flatten(), label = "Original")
    plt.plot(index, recomputed_Gdh, label = "Recomputed, using Igawa's non-linear regression")
    plt.title(f"Diffuse horizontal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Diffuse horizontal irradiance [W/m^2]")
    plt.legend()
    plt.grid()
    plt.show()
        

#%%           ------ PLOT OF SKY CONDITIONS -------

    fig = plt.figure(figsize=(16,12))
    plt.plot(index, res["Siv"].flatten())
    plt.axhline(y = 1.15, color = 'gray', linestyle = ':', label="Intermidiate Overcast limit")
    plt.axhline(y = 0.9, color = 'gray', linestyle = '-.', label="Intermidiate Sky limit")
    plt.axhline(y = 0.3, color = 'gray', linestyle = '--', label="Intermidiate Clear Sky limit")
    plt.axhline(y = 0.15, color = 'gray', linestyle = '-', label="Clear Sky limit")
    plt.title(f"Sky Index for {Site_obj.name} on the {year}, {month}, {day}")
    plt.ylabel("Sky Index")
    plt.xlabel("Time [month-day hour]")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    