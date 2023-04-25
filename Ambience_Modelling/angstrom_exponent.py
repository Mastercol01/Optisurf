#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the Angstrom Turbidity Exponent of a site.
"""

#%%                  IMPORTATION OF LIBRARIES

import numpy as np
import pandas as pd

#%%                 DEFINITION OF CONSTANTS

# We define the constants of the Shettel & Fenn Aerosol model for computing the 
# angstrom turbidity exponent, as described in the paper "SMARTS2, a simple 
# model of the atmospheric radiative transfer of sunshine: algorithms and 
# performance assessment" pages 16-18.

INDEX = ["Rural", "Urban", "Maritime"]
COLUMNS = ["C1", "C2", "C3", "D1", "D2", "D3", "D4"]

ANGSTROM_EXP_COEFFS = pd.DataFrame(index = INDEX , columns = COLUMNS).astype(float)
ANGSTROM_EXP_COEFFS.loc["Rural",:]    = [0.581, 16.823, 17.539, 0.8547, 78.696, 0, 54.416]
ANGSTROM_EXP_COEFFS.loc["Urban",:]    = [0.2595, 33.843, 39.524, 1.0, 84.254, -9.1, 65.458]
ANGSTROM_EXP_COEFFS.loc["Maritime",:] = [0.1134, 0.8941, 1.0796, 0.04435, 1.6048, 0, 1.5298]


#%%               DEFINITION OF FUNCTIONS 


def compute_angstrom_exponent_using_SF(RH, wavelength = 500, model = "Urban"):
    
    """
    Compute the Ansgtrom turbidity exponent suing the Shettel and Fenn model,
    as detailed in the paper "SMARTS2, a simple  model of the atmospheric 
    radiative transfer of sunshine: algorithms and performance assessment".
    
    Parameters
    ----------
    
    RH : float or numpy.array of floats
      Relative Humidity of the air in %. Must be a non-negative number or array
      of numbers between 0 and 100.
      
    Wavelength : float
       Wavelength in nanometers for which the Angstrom turbidity exponent
       is to be computed. Must be a non-ngeative number. Default is 500.
       
    model : str
       Model to be used in the computation of the Angstrom exponent. Supported 
       are “Rural”, “Urban” and "Maritime".
       
    Returns
    -------
    alpha : float or numpy.array of floats
        Angstrom turbidity coefficient.
    
    
    """
    
    C1, C2, C3, D1, D2, D3, D4 = ANGSTROM_EXP_COEFFS.loc[model, :]
    
    
    Xrh = np.cos(np.deg2rad(0.9*RH))
    
    # According to the Shettel & Fenn model, alpha is computed differently, 
    # depending on the specific spectral region it is intended for.
    
    if wavelength < 500:
        alpha = (C1 + C2*Xrh) / (1 + C3*Xrh)
    
    else:
        alpha = (D1 + D2*Xrh + D3*Xrh**2) / (1 + D4*Xrh)
        
    return alpha


#%%                              EXAMPLES

if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    
    # Let us check the variation of the Angstrom exponent with respected to
    # relative humidity, for both spectral ranges (below and above 500nm), for 
    # all models.
    RHs = np.linspace(0, 100, 101)
    
    for model in ["Rural", "Urban", "Maritime"]:

        fig = plt.figure(figsize=(15,10))
        alpha1 = compute_angstrom_exponent_using_SF(RHs, 499, model=model)
        alpha2 = compute_angstrom_exponent_using_SF(RHs, 500, model=model)
        plt.plot(RHs, alpha1, label="λ < 500 nm")
        plt.plot(RHs, alpha2, label="λ ≥ 500 nm")
        plt.grid()
        plt.xlim(0,100)
        plt.xlabel("RH [%]")
        plt.ylabel("Angstrom Exponent")
        plt.title(f"Model = {model}")
        plt.legend(title = "Spectral Range")
        plt.show()
        
    
    
    

