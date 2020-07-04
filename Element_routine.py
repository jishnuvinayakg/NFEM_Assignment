import numpy as np
from Material_routine import compute_Ct,compute_sigma,compute_C

#This routine will calculate all element level entities.
#Shape function, B matrix, Ke_t, element internal force,external force and stress/strain

def shape_fn(Zta):
    '''Returns shape function N'''

    N1 = 1/2*(1-Zta)
    N2 = 1/2*(1+Zta)
    return np.array([[N1],[N2]])


def configure_B_matrix(Le,Zta,re1,re2):
    '''Returns B matrix'''

    B11 = -1/Le
    B12 = 1/Le
    r_zta = 0.5*(re1+re2) +  0.5*(Zta*Le)
    B21 = 0.5*(1-Zta)/r_zta
    B22 = 0.5*(1+Zta)/r_zta
    B = np.array([[B11,B12],[B21,B22]])
    return B

def element_external_force():
    '''Method for external force'''

    val = 0
    return np.array([[val],[val]])

def elemental_strain(ue,Le,Zta,re1,re2):
    '''Calculates the element strain using B and element displacement.
    Calls B matrix fuction'''

    B = configure_B_matrix(Le,Zta,re1,re2)
    strain = B@ue
    return strain

def elemental_internal_force(ue,Le,w,Zta,re1,re2,strain_int,sigma_ov_int,E,v,dt,Q,T):
    '''Calculates element internal forces 
    Calls strain and stress functions from material routine'''

    B = configure_B_matrix(Le,Zta,re1,re2)
    r_zta = 0.5*(re1+re2) +  0.5*(Zta*Le)# calculating r_zta from nodal points
    strain_next= elemental_strain(ue,Le,Zta,re1,re2)
    sigma,sigma_ov = compute_sigma(strain_next,strain_int,sigma_ov_int,E,v,dt,T,Q)
    Fe_int = w*(np.dot(B.T,sigma)*r_zta*(Le/2))
    return Fe_int,sigma_ov

def gauss_quadrature_Ke_t(w,Le,E,v,Q,dt,T,Zta,re1,re2):
    '''Finding Ket using gauss quadrature scheme, with one gauss point zta =0,weight w= 2.
    Calls Function to calculate Ct from material routine'''

    r_zta = 0.5*(re1+re2) +  0.5*(Zta*Le)
    B = configure_B_matrix(Le,Zta,re1,re2)
    C_t = compute_Ct(E,v,Q,dt,T)
    Ke_t = w*(((B.T)@C_t)@B)*(r_zta * (Le/2))
    return Ke_t

def elemental_stress(strain_e,sigma_ov_e,E,v):
    '''Used to calculate stress as post processing using the obtained displacemnt values and overstress'''
    
    C = compute_C(E,v)
    stress_e = C@strain_e + sigma_ov_e
    return stress_e

