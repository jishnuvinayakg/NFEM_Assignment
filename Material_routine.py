import numpy as np

# This routine will calulate Material features : element level Over stress, 
# Material tangent stifness matrix Ct and element level stress

def compute_over_stress(sigma_ov_int,strain_next,strain_int,dt,T,Q):
    ''' Method to compute over stress. 
    Input parameters include over stress and strain from previous time step'''

    temp_v1 = 1/(1+(dt/T))
    del_strain = strain_next-strain_int
    sigma_ov_next = temp_v1*(sigma_ov_int + (Q/3)*np.dot(np.array([[2,-1],[-1,2]]),del_strain))
    return sigma_ov_next

def compute_Ct(E,v,Q,dt,T):
    '''Method to compute the Material Tangent Stifness Matrix.'''

    val_1 = E/((1+v)*(1-2*v))
    temp_C = np.array([[1-v,v],[v,1-v]])
    C = val_1*temp_C
    val_2 = Q/(1+(dt/T))
    temp_val_3 = np.array([[2.0/3,-1.0/3],[-1.0/3,2.0/3]])
    Ct = C+ (val_2*temp_val_3)

    return Ct
    
def compute_sigma(strain_next,strain_int,sigma_ov_int,E,v,dt,T,Q):
    '''Method to calculate element level stress. Uses the function which computes the overstress'''

    val_1 = E/((1+v)*(1-2*v))
    temp_C = np.array([[1-v,v],[v,1-v]])
    C = val_1*temp_C
    sigma_ov = compute_over_stress(sigma_ov_int,strain_next,strain_int,dt,T,Q)
    sigma_next = np.dot(C,strain_next) + sigma_ov
    return sigma_next,sigma_ov


def compute_C(E,v):
    '''This will calculate and return the isotropic stifness matrix C'''
    val_1 = E/((1+v)*(1-2*v))
    temp_C = np.array([[1-v,v],[v,1-v]])
    C = val_1*temp_C
    return C
