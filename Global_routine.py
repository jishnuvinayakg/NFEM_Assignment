import Element_routine
import numpy as np 
def assignment_matrix(element_i,number_nodes):
        A = np.zeros((2,number_nodes))
        start_index = element_i - 1
        A[0][start_index] = 1
        A[1][element_i] = 1
        return A
    
def assemble_Kt(num_elements,element_lengths,E,v,Q,T,dt,r_nodes):

    w =2
    Zta = 0
    node = len(r_nodes)
    Kt = np.zeros((node,node))

    for i,Le in zip(range(1, num_elements+1),element_lengths):
        re1 = r_nodes[i-1]
        re2 = r_nodes[i]
        #rn = 0.5*(r1+r2)
        Ae = assignment_matrix(i,node)
        Ket = Element_routine.gauss_quadrature_Ke_t(w,Le,E,v,Q,dt,T,Zta,re1,re2)
        #tempK1 = np.dot((Ae.T),Ket)
        tempK1 = (Ae.T)@Ket
        #tempK2 = np.dot(tempK1,Ae)
        tempK2 = tempK1@Ae
        Kt = np.add(Kt,tempK2)
    return Kt

def assemble_Fext(num_elements,element_lengths,p,a):

    node = num_elements+1
    F_ext = np.zeros((node,1))
    for i in range(1, num_elements+1):
        Fe = Element_routine.element_external_force()
        Ae = assignment_matrix(i,node)
        #tempK1 = np.dot(Ae.T,Fe)
        tempK1 = (Ae.T)@Fe
        F_ext = np.add(F_ext,tempK1)
    F_ext[0][0] = p*a
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
        #rn = 0.5*(r1+r2)
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
    #np.set_printoptions(suppress=True)
    k =1
    Kt_inv = np.array([[],[]],dtype=float)
    def Euclidean_norm(y):
        result = np.sqrt(y.T@y)
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
        norm_R = Euclidean_norm(R)
        norm_delta_U = Euclidean_norm(del_u)
        norm_Fint = Euclidean_norm(F_int)
        norm_U = Euclidean_norm(U_next)
        if k == K_max:
            print(f'Not converging {k}')
        if ((k>=K_max) or (norm_R >tolerence*norm_Fint or norm_delta_U >tolerence*norm_delta_U)):
        #if(not(norm_R > tolerence or norm_delta_U > tolerence) and (not k < K_max)):
            print(f'Numer of NR runs : {k} @ time step {t}')
            return U_next,sigma_ov
        U_int = U_next
        k+=1

def assemble_strain(U,number_elements,element_lengths,r_nodes):
    Zta = 0
    Strain = np.array([[],[]])

    def r_zta(Zta,Le,rn):
        return (Le/2)*(1+Zta) + rn

    for i,Le in zip(range(1, number_elements+1),element_lengths):
        re1 = r_nodes[i-1]
        re2 = r_nodes[i]
        ue = np.array([U[i-1],U[i]])
        B = Element_routine.configure_B_matrix(Le,Zta,re1,re2)
        strain = np.dot(B,ue)
        Strain = np.append(Strain,strain,axis =1)
    return  Strain

def non_linear_fem_solver(num_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f):

    time = np.arange(0,t_f,dt)
    U = []
    U_int = np.zeros((len(r_nodes),1))
    strain = np.zeros((2,num_elements))
    stress_ov = np.zeros((2,num_elements))
    p =0

    for t in time[1:]:
        if( t <= t_l):
            p+= 25*dt

        K_max = 25
        tolerence = 0.005
        U_next,sigma_ov_updated = Newton_Raphson_method(K_max,tolerence,num_elements,element_lengths,E,v,Q,T,dt,p,a,strain,stress_ov,r_nodes,U_int,t)
        U.append(U_next.flatten())
        stress_ov = sigma_ov_updated
        strain = assemble_strain(U_next,num_elements,element_lengths,r_nodes)
        U_int = U_next
    return U,strain
