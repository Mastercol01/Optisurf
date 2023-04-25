#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module is intended as a stand-alone file for use. It is used to
compute the absorption function of Silicon in a MAXEON PV solar cell (at least
to the best of the ability of the model being employed). This module may serve
as a guide for the user so they can develop their own module utilizing the 
functionality of the 'solcore' package. In any case, the purpose of this module 
is to generate an abosrption function for the Silicon wafer inside a solar cell,
that depends both on the angle of incidence as well as the wavelength. This 
function can then be utilized in the Mesh module in order to compute the
total energy absorbed by the Silicon.
"""

#%%                   IMPORTATION OF LIBRAIRES

import os
import time as tm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solcore import si, material
from scipy import constants as ct
from solcore.structure import Layer, Structure
from solcore.absorption_calculator import calculate_rat
from solcore.material_system.create_new_material import create_new_material
           

#%%            DEFINITION OF OPTICAL PARAMETERS FOR SIMULATION

angles      = np.linspace(0, 89.999999999, 91)
angles  = np.array([0, 30, 60, 75, 80, 84, 87, 89])
wavelengths = np.linspace(300, 4000, 741)

# We compute some useful quantities.
len_wl = len(wavelengths)
delta_angle = (angles[1:] - angles[:-1]).mean()
delta_wl = (wavelengths[1:] - wavelengths[:-1]).mean()

#
#%%          COMPUTATION OF EXPOXY RESIN'S OPTICAL PROPERTIES

# The following implementation for the computation of Epoxy Resin optical
# properties comes from paper titled: "Optical Properties of Modified Epoxy 
# Resin with Various Oxime Derivatives in the UV-ViS Spectral Region""


# We encapuslate all necessary logic into a single function easy to call.
# One thing to note is that the angular frequencies must be in Tera-Herz for 
# the code to properly work. That is something the paper omitted to mention.
def compute_epoxy_resin_optical_params(df_params, wavelengths_nm):
    
    """
    Compute optical parameters of expoxy resin by following the procedure 
    detailed in: "Optical Properties of Modified Epoxy Resin with Various Oxime 
    Derivatives in the UV-ViS Spectral Region".
    
    df_params : pandas.DataFrame of float
        DataFrame containig all required parameters for the computation
        of the epoxy resin properties. 
        
    wavelengths_nm : numpy.array of floats
        Wavelengths at which the epoxy resin optical properties are to be
        computed in nanometers.
        
    Returns
    -------
    res : dict of numpy.arrays
        Dict containing the results of the computation. It has the following 
        key-value pairs:
            
            Keys : Values 
            -------------
            "lambda_nm" : numpy array of floats
                Wavelengths at which the epoxy resin optical properties have been
                computed, in nanometers.
                
            "lambda_m" : numpy array of floats
                Wavelengths at which the epoxy resin optical properties have been
                computed, in meters.
                
            "angular_freqs_Tradsec" : numpy array of floats
                Angurlar frequencies at which the epoxy resin optical properties 
                have been computed, in Tera radians per second.
                
            "eps_complex" : numpy array of complex
                Complex dialectric constant of the epoxy resin at each wavelength.
                
            "eps_real" : numpy array of floats
                Real part of the complex dialectric constant of the epoxy resin 
                at each wavelength.
                
            "eps_imag" : numpy array of floats
                Imaginary part of the complex dialectric constant of the epoxy
                resin at each wavelength.
                
            "n" : numpy.array of floats
                Refractive index of the of the epoxy resin at each wavelength.
                
            "k" : numpy.array of floats
                Extinction coefficient of the of the epoxy resin at each wavelength.
                

    """
    
    wavelengths_m = (wavelengths_nm)*10**-9
    angular_freqs_Tradsec = (2*np.pi*ct.c/wavelengths_m)*10**-15
   
    
    eps_inf = df_params.loc[0, "Eps Inf"]
    S = np.array(df_params.loc[:, "S"])
    W = np.array(df_params.loc[:, "Omega"])
    F = np.array(df_params.loc[:, "F"])
    n = len(S)
    
    imag_dielectric_ct = 0
    real_dielectric_ct = eps_inf
    
    for i in range(n):
        
        W_ratio = angular_freqs_Tradsec/W[i]
        
        imag_numerator = S[i]*F[i]*W_ratio
        real_numerator = S[i]*( 1 - W_ratio**2 )
        denominator = ( 1 - W_ratio**2 )**2 + (F[i]*W_ratio)**2
        
        imag_dielectric_ct += imag_numerator/denominator
        real_dielectric_ct += real_numerator/denominator
        
                
        
    complex_dielectric_ct = real_dielectric_ct.astype(complex) + 1j*imag_dielectric_ct.astype(complex)
    
    refractive_index =\
    0.5*np.sqrt( 2*real_dielectric_ct + 2*np.sqrt(real_dielectric_ct**2 + imag_dielectric_ct**2) )
    
    extinction_coef = 0.5*imag_dielectric_ct/refractive_index
    
    res = {"lambda_nm" : wavelengths_nm,
           "lambda_m" : wavelengths_m,
           "angular_freqs_Tradsec" : angular_freqs_Tradsec,
           "eps_complex" : complex_dielectric_ct,
           "eps_real" : real_dielectric_ct,
           "eps_imag" : imag_dielectric_ct,
           "n" : refractive_index,
           "k" : extinction_coef}
    
    return res


# --- IMPORT EPOXY RESIN PROPERTIES ----
# We import all necessary parameters for the computation of the resin's optical
# properties, from the local optical properties database.
Epoxy_params_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\Local Optical Properties Database\Epoxy Resin Coefs.xlsx"
Epoxy_resin_params = pd.read_excel(Epoxy_params_path)

# --- COMPUTE EPOXY RESIN OPTICAL PROPERTIES ----
res = compute_epoxy_resin_optical_params(df_params = Epoxy_resin_params,
                                         wavelengths_nm = wavelengths)

# We store all relevant data into a DataFrame
Epoxy_optical_data = pd.DataFrame(columns=["WL (m)", "n", "k"], 
                                 index=range(len(wavelengths))).astype(float)

Epoxy_optical_data["WL (m)"] = si(wavelengths, "nm")
Epoxy_optical_data["n"] = res["n"]
Epoxy_optical_data["k"] = res["k"]

# --- SAVE EPOXY RESIN OPTICAL PROPERTIES ----
# We save the epoxy resin's computed optical properties  in the Local Optical Properties Database, 
# in case we ever want to use them in another project. They are already computed.
Epoxy_n_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\Local Optical Properties Database\Epoxy_n.txt"
Epoxy_k_path = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\Local Optical Properties Database\Epoxy_k.txt"
Epoxy_optical_data[["WL (m)", "n"]].to_csv(Epoxy_n_path, sep=" ", header=False, index=False)
Epoxy_optical_data[["WL (m)", "k"]].to_csv(Epoxy_k_path, sep=" ", header=False, index=False)

#%%             DEFINITION OF MATERIALS FOR SIMULATION


# We initialize all the materials that make up our optical model
# for the MAXEON PV cell. For this we utilize solcore's material class.

# --- (1) EPOXY RESIN ---
# Define the Epoxy resin as a new material, using the previously computed
# optical properties.
create_new_material(mat_name = 'Epoxy', n_source = Epoxy_n_path, k_source = Epoxy_k_path)

# Initialize instance of Epoxy Resin material and compute refractive  
# index and and extinction coefficient, for all wavelengths.
Epoxy   = material("Epoxy")(T=300)
Epoxy_n = Epoxy.n(si(wavelengths, "nm"))
Epoxy_k = Epoxy.k(si(wavelengths, "nm"))


# --- (2) TITANIUM DIOXIDE ---
# Initialize instance of Titanium dioxide material and compute refractive  
# index and and extinction coefficient, for all wavelengths.
TiO2   = material("TiO2")(T=300)
TiO2_n = TiO2.n(si(wavelengths, "nm"))
TiO2_k = TiO2.k(si(wavelengths, "nm"))


# --- (3) SILICON ---
# Initialize instance of Silicon material and compute refractive  
# index and and extinction coefficient, for all wavelengths.
Silicon   = material("Si")(T=300)
Silicon_n = Silicon.n(si(wavelengths, "nm"))
Silicon_k = Silicon.k(si(wavelengths, "nm"))


# --- (3) SILICON DIOXIDE ---
# Initialize instance of Silicon dioxide material and compute refractive  
# index and and extinction coefficient, for all wavelengths.
SiO2   = material("SiO2")(T=300)
SiO2_n = SiO2.n(si(wavelengths, "nm"))
SiO2_k = SiO2.k(si(wavelengths, "nm"))


# --- (3) ALUMINUM ---
# Initialize instance of Aluminum  material and compute refractive  
# index and and extinction coefficient, for all wavelengths.
Aluminum   = material("Al")(T=300)
Aluminum_n = Aluminum.n(si(wavelengths, "nm"))
Aluminum_k = Aluminum.k(si(wavelengths, "nm"))


# --- VISUALIZATION ---

# PLOT: Refractive Index of Layers vs. Wavelength 
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(12,10))
plt.plot(wavelengths, Epoxy_n, label="Epoxy Resin")
plt.plot(wavelengths, TiO2_n, label="TiO2")
plt.plot(wavelengths, Silicon_n, label="Mono. Silicon")
plt.plot(wavelengths, SiO2_n, label="SiO2")
plt.plot(wavelengths, Aluminum_n, label="Aluminum")
plt.title("Refractive index (n) vs. Wavelength", fontsize=20)
plt.xlabel("Wavelength [nm]", fontsize=20)
plt.ylabel("n [adm]", fontsize=20)
plt.xlim((300, 4000))
plt.legend(prop={"size":17})
plt.grid()
plt.show()

# PLOT: Extinction Coeff. of Layers vs. Wavelength 
fig = plt.figure(figsize=(12,10))
plt.plot(wavelengths, Epoxy_k, label="Epoxy Resin")
plt.plot(wavelengths, TiO2_k, label="TiO2")
plt.plot(wavelengths, Silicon_k, label="Mono. Silicon")
plt.plot(wavelengths, SiO2_k, label="SiO2")
plt.plot(wavelengths, Aluminum_k, label="Aluminum")
plt.title("Extinction coefficient (κ) vs. Wavelength", fontsize=20)
plt.xlabel("Wavelength [nm]", fontsize=20)
plt.ylabel("κ [adm]", fontsize=20)
plt.xlim((300, 4000))
plt.legend(prop={"size":17})
plt.grid()
plt.show()


#%%         DEFINITION OF STRUCTURES/OPTISTACKS FOR SIMULATION
# Let us now define the Optistacks for simulation. Here is where we construct
# the optical equivalent of the Maxeon PVcell.


# Data is dict containing multiple structures to be simulated. 
data = {"structures":{}}

# PV module using MAXEON solar cells with ARC.
data["structures"][0] = Structure([Layer(width = si(500, "um"), material = Epoxy),
                                   Layer(width = si(75,  "nm"), material = TiO2), 
                                   Layer(width = si(140, "um"), material = Silicon),
                                   Layer(width = si(95,  "nm"), material = SiO2),
                                   Layer(width = si(133, "nm"), material = Aluminum)])

# PV MODULE, using MAXEON solar cells without ARC.
data["structures"][1] = Structure([Layer(width = si(500, "um"), material = Epoxy),
                                   Layer(width = si(140, "um"), material = Silicon),
                                   Layer(width = si(95,  "nm"), material = SiO2),
                                   Layer(width = si(133, "nm"), material = Aluminum)])

# Bare Silicon wafer.
data["structures"][2] = Structure([Layer(width = si(500, "um"), material = Epoxy)])




#%%             1) OPTICAL SIMULATION: FULLY COHERENT
# We perform a simulation for each optistack, while treating all layers
# of each optistack as fully coherent.


RAT_dicts_c = {key:{"R":[], "A":[], "T":[], "A_per_layer":[]} for key in data["structures"]}


for angle in angles:
    for key1 in RAT_dicts_c:
                
        rat_data = calculate_rat(structure = data["structures"][key1],
                                 wavelength = wavelengths, 
                                 angle = angle, 
                                 pol = 'u', 
                                 coherent = True,
                                 no_back_reflection = False)
        
        for key2 in RAT_dicts_c[key1]:
            RAT_dicts_c[key1][key2].append(rat_data[key2])
            
            
#%%         2) OPTICAL SIMULATION: PARTIALLY COHERENT
# We perform a simulation for each optistack, while treating some layers
# as fully coherent and others as fully incoherent.

RAT_dicts_ci = {key:{"R":[], "A":[], "T":[], "A_per_layer":[]} for key in data["structures"]}
coherency_dict = {0:["i", "c", "i", "c", "c"], 1:["i", "i", "c", "c"], 2:["i"]}


for angle in angles:
    for key1 in RAT_dicts_ci:
                
        rat_data = calculate_rat(structure = data["structures"][key1],
                                 wavelength = wavelengths, 
                                 angle = angle, 
                                 pol = 'u', 
                                 coherent = False,
                                 coherency_list = coherency_dict[key1],
                                 no_back_reflection = False)
        
        for key2 in RAT_dicts_ci[key1]:
            RAT_dicts_ci[key1][key2].append(rat_data[key2])
            
            
#%%         3) OPTICAL SIMULATION: FULLY INCOHERENT 1
# We perform a simulation for each optistack, while treating all layers
# as fully incoherent.


RAT_dicts_i = {key:{"R":[], "A":[], "T":[], "A_per_layer":[]} for key in data["structures"]}
coherency_dict = {0:["i", "i", "i", "i", "i"], 1:["i", "i", "i", "i"], 2:["i"]}


for angle in angles:
    for key1 in RAT_dicts_i:
                
        rat_data = calculate_rat(structure = data["structures"][key1],
                                 wavelength = wavelengths, 
                                 angle = angle, 
                                 pol = 'u', 
                                 coherent = False,
                                 coherency_list = coherency_dict[key1],
                                 no_back_reflection = False)
        
        for key2 in RAT_dicts_i[key1]:
            RAT_dicts_i[key1][key2].append(rat_data[key2])
    

  
#%%         4) OPTICAL SIMULATION: FULLY INCOHERENT 2
# Incoherent simulation is achieved by performing multiple coherent simulations
# with some random noise added into the input parameters, and then averaging 
# out all the results. 

t1 = tm.time()
RAT_dicts_ii = {key:{"R":[], "A":[], "T":[], "A_per_layer":[]} for key in data["structures"]}


n = 300
for angle in angles:
    for key1 in RAT_dicts_ii:
        for i in range(n):
            
            new_wavelengths = wavelengths.copy()
            new_wavelengths += np.random.normal(scale = delta_wl/6, size = len_wl)
            new_angle = angle + np.random.normal(scale = delta_angle/6)
            if new_angle < 0 : new_angle = 0
    
    
    
            if i == 0:
                rat_data = calculate_rat(structure = data["structures"][key1],
                                         wavelength = new_wavelengths, 
                                         angle = new_angle, 
                                         pol = 'u', 
                                         coherent = True,
                                         no_back_reflection = False)
                
                for key in rat_data.keys():
                    try: rat_data[key] /= n
                    except TypeError: pass
                
            
            else:
                new_rat_data = calculate_rat(structure = data["structures"][key1],
                                             wavelength = new_wavelengths, 
                                             angle = new_angle, 
                                             pol = 'u', 
                                             coherent = True, 
                                             no_back_reflection = False)
                

                    
                for key, val in rat_data.items():
                    try: rat_data[key] += new_rat_data[key]/n
                    except TypeError: pass
                    
                
        for key2 in RAT_dicts_ii[key1]:
            RAT_dicts_ii[key1][key2].append(rat_data[key2])
            
    print(angle)
    
dt1 = (tm.time() - t1)/60
print(f"Ellapsed time: {dt1} [min]")



#%%        ANALYSIS OF RELFECTANCE, ABSORPTION AND TRANSMITTANCE OF PV CELL

# We plot the results and analyse the Rflectance, Transmittance and absorbance
# of the each optistack as a whole, depending on the mode of simulation.

# --PARAMS--
coherency = "ci"
structure = 0
variable  = "A"

title_dict  = {"R":"Reflectance", "A":"Absorption", "T":"Transmittance"}
ylabel_dict = {"R":"reflected", "A":"absorbed", "T":"transmitted"}

# Get Data by simulation type.
if coherency == "c":
    local_data = np.array(RAT_dicts_c[structure][variable])
    
elif coherency == "ci": 
    local_data = np.array(RAT_dicts_ci[structure][variable])
    
elif coherency == "i": 
    local_data = np.array(RAT_dicts_i[structure][variable])
    
elif coherency == "ii": 
    local_data = np.array(RAT_dicts_ii[structure][variable])
    

# Plot data for all wavelengths and for each angle.
plt.rcParams.update({'font.size': 24})

fig = plt.figure(figsize=(12,10))
for i, vals in enumerate(local_data):
    plt.plot(wavelengths, vals, label = str(round(angles[i], 2)) + "°")
    
if len(local_data) < 10:
    plt.legend(title = "Incident Angle", prop={'size': 17})
    
plt.xlabel("Wavelength [nm]", fontsize = 24)
plt.ylabel(f"Fraction of incident power that is {ylabel_dict[variable]} ", fontsize = 24)
plt.title(f"{title_dict[variable]} of PV surface", fontsize = 24)

plt.xlim(300, 4000)
plt.ylim(0, 1.05*local_data.max())
#plt.ylim(0, 1)
plt.grid()
plt.show()


    
#%%                 ABSORBANCE BY PV CELL LAYER

# UNDERSTANDING "A_per_layer" matrix of rat_data
# axis 1: wavelength
# axis 0:
#           0 : Transmitance into the Optistack
#           1 : Absorption of the 1st layer per wavelength.
#           2 : Absorption of the 2nd layer per wavelength.
#           3 : Absorption of the 3rd layer per wavelength.
#           .
#           .
#           n - 1 : Absorption of the nth layer per wavelength.
#           n : Transmitance out of the Optistack  


# Let us now focus on analysing the absorbance behaviour of the optistack
# layer by layer. In particular, we really care about how much of the incident
# power gets absorbed by the Silicon.


# --PARAMS--
coherency = "ci"
structure = 0
layer = 5

# Get Data by simulation type.
if coherency == "c":
    local_data = RAT_dicts_c[structure]["A_per_layer"]
    
elif coherency == "ci": 
    local_data = RAT_dicts_ci[structure]["A_per_layer"]
    
elif coherency == "i": 
    local_data = RAT_dicts_i[structure]["A_per_layer"]
    
elif coherency == "ii": 
    local_data = RAT_dicts_ii[structure]["A_per_layer"]
    
    
# Define titles and labels.  
if structure == 0:
    layer_dict = {1:"Epoxy Resin", 2:"TiO2 ARC", 3:"Silicon", 4:"SiO2 layer", 5:"Aluminum backside coating"}

elif structure == 1:
    layer_dict = {1:"Epoxy Resin", 2:"Silicon", 3:"SiO2 layer", 4:"Aluminum backside coating"}
    
elif structure == 2:
    layer_dict = {1:"Epoxy Resin"}
    


# Plot data for all wavelengths and for each angle.
plt.rcParams.update({'font.size': 24})    

fig = plt.figure(figsize=(12,10))
for i, vals in enumerate(local_data):
    plt.plot(wavelengths, vals[layer, :], label = str(round(angles[i], 2)) + "°")
    

plt.xlabel("Wavelength [nm]", fontsize = 24)
plt.ylabel("Fraction of incident power that is absorbed", fontsize =24)
plt.suptitle(f"Absorption due to {layer_dict[layer]} inside the PV surface", fontsize =24)

plt.xlim(300, 4000)
plt.ylim(0, 1.05*np.array([i[layer,:].max() for i in local_data]).max())
#plt.ylim(0, 0.8)
#plt.axvline(1102.55, color="r", linestyle="--")
plt.grid()

if len(local_data) < 10:
    plt.legend(title = "Incident Angle", prop={'size': 17})
plt.show()

#%%      EXPORT SILICON ABSORPTION ARRAY
# Let us export the array of Silicon Absorbance. This array will be crucial later
# as it will be used in other modules for computing the actual total amount of energy 
# received by a surface.

# In particular, let us save the results of the partially coherent simulation for
# the PV model including the ARC, as these are the ones that come closest to 
# the situation we expect in real life.

coherency = "ci"
structure = 0
layer = 3


silicon_absorbance = np.zeros((len(angles), len(wavelengths)))

for i in range(len(angles)):
    silicon_absorbance[i, :] = RAT_dicts_ci[structure]["A_per_layer"][i][layer,:]

    
    
# Finally, let us also include in the array that we are about to save,
# a copy of the meshgrid values used for its definition, since we'll need
# these values later in order to create the function that interpolates the
# silicon absorbance.
Wavelengths, Angles = np.meshgrid(wavelengths, angles)

arr_to_save        = np.zeros((len(angles), len(wavelengths), 3))
arr_to_save[:,:,0] = silicon_absorbance
arr_to_save[:,:,1] = Wavelengths
arr_to_save[:,:,2] = Angles


np.save(file = r"C:\Users\andre\Desktop\Eafit\10mo Semestre\Optisurf_v2\Surface_Modelling\silicon_absorbance.npy",
        arr = arr_to_save)

#%%

vals = []
for val in RAT_dicts_ci[0]["A_per_layer"]:
    vals.append(val[1:-1,:])
    

total_vals   = sum(vals)/len(vals)
wv = wavelengths <= 1200

a  =  100*total_vals.sum(axis=1)/total_vals.sum(axis=1).sum()
aa =  100*total_vals[:,:182].sum(axis=1)/total_vals[:,:182].sum(axis=1).sum()

#total_vals = total_vals/total_vals.sum(axis=0)




#%%         COMPUTING IMPROVEMENT DUE TO ARC
# Out of curiosity, let us analyse the effect of the TiO2 ARC coating on the 
# solar Cell. How does the ARC coating improve absorbance.

Si_absorbance_with_ARC_ci = np.array([i[3,:] for i in RAT_dicts_ci[0]["A_per_layer"]])
Si_absorbance_without_ARC_ci = np.array([i[2,:] for i in RAT_dicts_ci[1]["A_per_layer"]])

change_in_abs =  Si_absorbance_with_ARC_ci - Si_absorbance_without_ARC_ci 
percentage_change_in_abs = 100*change_in_abs/Si_absorbance_without_ARC_ci 


fig = plt.figure(figsize=(12,10))
for i, vals in enumerate(change_in_abs):
    plt.plot(wavelengths, vals, label = str(round(angles[i], 2)) + "°")

plt.xlabel("Wavelength [nm]")
plt.ylabel("Absolute Change [fraction of absorbed incident power]")
plt.title("Absolute Change in Silicon Absorption due to ARC")
plt.rcParams.update({'font.size': 20})
plt.xlim(300, 1200)
plt.ylim(1.05*change_in_abs.min(), 1.05*change_in_abs.max())

if len(local_data) < 10:
    plt.legend(title = "Incident Angle", prop={'size': 13})
plt.grid()
#plt.axhline(y = change_in_abs.mean(), color = 'r', linestyle = '--')
plt.show()

print(f"Average Absolute Change in Silicon Absorbance due to ARC: {change_in_abs.mean()}")


fig = plt.figure(figsize=(12,10))
for i, vals in enumerate(percentage_change_in_abs):
    plt.plot(wavelengths, vals, label = str(round(angles[i], 2)) + "°")

plt.xlabel("Wavelength [nm]")
plt.ylabel("Relative Change [%]")
plt.title("Relative changle in Silicon Absorbance due to ARC")
plt.rcParams.update({'font.size': 20})
plt.xlim(300, 1500)
plt.ylim(1.05*percentage_change_in_abs.min(), 1.05*percentage_change_in_abs.max())

if len(local_data) < 10:
    plt.legend(title = "Incident Angle", prop={'size': 13})
plt.grid()
plt.axhline(y = percentage_change_in_abs.mean(), color = 'r', linestyle = '--')
plt.show()

print(f"Average Relative Change in Silicon Absorbance due to ARC: {percentage_change_in_abs.mean()} %")



    
        
        

