import Element_routine
import numpy as np 

#Global routine assembles Kt,F_int,F_ext using methods from element routine and uses Newton Raphsons method
#for solving the non linear equation

def assignment_matrix(element_i,number_nodes):
    '''Return the assignment matrix based on number of elements'''

    A = np.zeros((2,number_nodes))
    start_index = element_i - 1
    A[0][start_index] = 1
    A[1][element_i] = 1
    return A
    
def assemble_Kt(num_elements,element_lengths,E,v,Q,T,dt,r_nodes):

    '''Method to assemble Kt using the Ket calculated by element routine'''

    #Weight and zta for gauss quadrature(1 point)
    w =2
    Zta = 0
    node = len(r_nodes)
    Kt = np.zeros((node,node))
    for i,Le in zip(range(1, num_elements+1),element_lengths):
        re1 = r_nodes[i-1]
        re2 = r_nodes[i]
        Ae = assignment_matrix(i,node)
        Ket = Element_routine.gauss_quadrature_Ke_t(w,Le,E,v,Q,dt,T,Zta,re1,re2)
        tempK1 = (Ae.T)@Ket
        tempK2 = tempK1@Ae
        Kt = np.add(Kt,tempK2)
    return Kt

def assemble_Fext(num_elements,element_lengths,p,a):
    '''Method to assemble external force. Applies the boundary condition of internal pressure to first node'''

    node = num_elements+1
    F_ext = np.zeros((node,1))
    for i in range(1, num_elements+1):
        Fe = Element_routine.element_external_force()
        Ae = assignment_matrix(i,node)
        tempK1 = (Ae.T)@Fe
        F_ext = np.add(F_ext,tempK1)
    F_ext[0][0] = p*a #Boundary condition
    return F_ext

def assemble_Fint(U,strain,sigma_ov_int,E,v,dt,Q,T,r_nodes,element_lengths):
    
    w =2
    Zta = 0
    node = len(r_nodes)
    num_elements = node-1
    F_int = np.zeros((node,1))
    sigma_ov = np.array([[],[]])
    
    for i,Le in zip(range(1, num_elements+1),element_lengths):
        re1 = r_nodes[i-1]
        re2 = r_nodes[i]
        ue = np.array([U[i-1],U[i]])
        strain_int = np.array([[strain[:,i-1][0]],[strain[:,i-1][1]]])
        sigma_ov_int_n = np.array([[sigma_ov_int[:,i-1][0]],[sigma_ov_int[:,i-1][1]]])
        rn = r_nodes[i-1]
        Ae = assignment_matrix(i,node)
        Fi,sigma_ov_next = Element_routine.elemental_internal_force(ue,Le,w,Zta,re1,re2,strain_int,sigma_ov_int_n,E,v,dt,Q,T)
        temp_val = np.dot(Ae.T,Fi)
        F_int = np.add(F_int,temp_val)
        sigma_ov = np.append(sigma_ov,sigma_ov_next,axis=1)

    return F_int,sigma_ov

def Newton_Raphson_method(K_max,tolerence,num_elements,element_lengths,E,v,Q,T,dt,p,a,strain_int,sigma_ov_int,r_nodes,U_int,t):

    Kt = assemble_Kt(num_elements,element_lengths,E,v,Q,T,dt,r_nodes)
    F_ext = assemble_Fext(num_elements,element_lengths,p,a)
    k =0
    Kt_inv = np.array([[],[]],dtype=float)
    def Infinity_norm(y):
        result = np.linalg.norm(y,ord=np.inf)
        return result.item()
    while True:
        F_int,sigma_ov = assemble_Fint(U_int,strain_int,sigma_ov_int,E,v,dt,Q,T,r_nodes,element_lengths)
        detKt = np.linalg.det(Kt)
        if( detKt != 0):
            Kt_inv = np.linalg.inv(Kt)
        else:
            print('Singular Matrix error')
        R = -(F_int - F_ext)
        del_u = Kt_inv@R
        U_next = U_int + del_u
        F_int_next,sigma_ov = assemble_Fint(U_next,strain_int,sigma_ov_int,E,v,dt,Q,T,r_nodes,element_lengths)
        R = -(F_int_next-F_ext)

        norm_R = Infinity_norm(R)
        norm_delta_U = Infinity_norm(del_u)
        norm_Fint = Infinity_norm(F_int_next)
        norm_U = Infinity_norm(U_next)

        if k == K_max:
            print(f'Not converging {k}')
        if ((k>=K_max) or (norm_R <tolerence*norm_Fint or norm_delta_U <tolerence*norm_U)):
            print(f'Numer of NR runs : {k+1} @ time step {t}')
            return U_next,sigma_ov
        U_int = U_next
        k+=1

def assemble_strain(U,number_elements,element_lengths,r_nodes):
    Zta = 0
    Strain = np.array([[],[]])

    for i,Le in zip(range(1, number_elements+1),element_lengths):
        re1 = r_nodes[i-1]
        re2 = r_nodes[i]
        ue = np.array([U[i-1],U[i]])
        B = Element_routine.configure_B_matrix(Le,Zta,re1,re2)
        strain = np.dot(B,ue)
        Strain = np.append(Strain,strain,axis =1)
    return  Strain

def assemble_stress(strain,sigma_ov,E,v,number_elements):

    Stress = np.array([[],[]])
    for i in range(1,number_elements+1):
        strain_e = np.array([[strain[0][i-1]],[strain[1][i-1]]])
        sigma_ov_e = np.array([[sigma_ov[0][i-1]],[sigma_ov[1][i-1]]])
        stress = Element_routine.elemental_stress(strain_e,sigma_ov_e,E,v)
        Stress = np.append(Stress,stress,axis=1)

    return Stress

def non_linear_fem_solver(num_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f):

    time = np.arange(0,t_f+1,dt)
    time = time[time<=10.0]
    U = []
    U_int = np.zeros((len(r_nodes),1))
    strain = np.zeros((2,num_elements))
    stress_ov = np.zeros((2,num_elements))
    U_tL = np.array([[],[]])
    p =0

    #print(f'Time : {time}')
    print(f'Number of elements : {num_elements}')
    for t in time[1:]:
        if( t <= t_l):
            p+= 25*dt
        K_max = 25
        tolerence = 0.005
        U_next,sigma_ov_updated = Newton_Raphson_method(K_max,tolerence,num_elements,element_lengths,E,v,Q,T,dt,p,a,strain,stress_ov,r_nodes,U_int,t)
        U.append(U_next.flatten())
        stress_ov = sigma_ov_updated
        strain = assemble_strain(U_next,num_elements,element_lengths,r_nodes)
        #print(f'Strain in NR : {strain}')
        #print(f'Overstress : {stress_ov}')
        U_int = U_next
        U_tL = U_next
    
    #compute stress and strain for last time step

    strain_L = assemble_strain(U_tL,num_elements,element_lengths,r_nodes)
    #print(f'Strain last {strain_L}')
    stress_L = assemble_stress(strain_L,stress_ov,E,v,num_elements)

    return U,stress_L
