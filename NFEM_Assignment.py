import numpy as np 
import matplotlib.pyplot as plt 

def meshing(r_inner,r_outer,meshrefinementfactor,number_elements):

    q = meshrefinementfactor**(1.0/(number_elements-1))
    dr=(r_outer-r_inner)*(1-q)/(1-meshrefinementfactor*q)
    rnode=r_inner
    rnodes = np.array([])
    rnodes = np.append(rnodes,rnode)

    for i in range(number_elements):
        rnode=rnode+dr
        rnodes = np.append(rnodes,rnode)
        dr=dr*q
    
    element_lengths = np.array([])
    for index,node in enumerate( rnodes[:-1]):
        element_length = rnodes[index+1]-rnodes[index]
        element_lengths = np.append(element_lengths,element_length)
    
    return rnodes,element_lengths

def material_routine():

    def compute_over_stress(sigma_ov_int,strain_next,strain_int,dt,T,Q):

        temp_v1 = 1/(1+(dt/T))
        del_strain = strain_next-strain_int
        sigma_ov_next = temp_v1*(sigma_ov_int + (Q/3)*np.dot(np.array([[2,-1],[-1,2]]),del_strain))
        return sigma_ov_next

    def compute_Ct(E,v,Q,dt,T):

        val_1 = E/((1+v)*(1-2*v))
        temp_C = np.array([[1-v,v],[v,1-v]])
        C = val_1*temp_C
        val_2 = Q/(1+(dt/T))
        temp_val_3 = np.array([[2.0/3,-1.0/3],[-1.0/3,2.0/3]])
        Ct = C+ (val_2*temp_val_3)
        return Ct
    
    def compute_sigma(strain_next,strain_int,sigma_ov_int,E,v,dt,T,Q):
        val_1 = E/((1+v)*(1-2*v))
        temp_C = np.array([[1-v,v],[v,1-v]])
        C = val_1*temp_C
        sigma_ov = compute_over_stress(sigma_ov_int,strain_next,strain_int,dt,T,Q)
        sigma_next = np.dot(C,strain_next) + sigma_ov

        return sigma_next,sigma_ov
    
    
    return compute_Ct,compute_sigma

def element_routine():

    compute_Ct,compute_sigma = material_routine()

    def shape_fn(Zta):
         N1 = 1/2*(1-Zta)
         N2 = 1/2*(1+Zta)
         return np.array([[N1],[N2]])

    
    def configure_B_matrix(Le,Zta,r):
        B11 = -1/Le
        B12 = 1/Le
        B21 = (0.5*(1-Zta))/((Le/2)*(1+Zta)+r)
        B22 = (0.5*(1+Zta))/((Le/2)*(1+Zta)+r)
        B = np.array([[B11,B12],[B21,B22]])
        return B
    
    def element_external_force():
        val = 0
        return np.array([[val],[val]])
    
    def elemental_strain(ue,Le,Zta,r):
        B = configure_B_matrix(Le,Zta,r)
        strain = np.dot(B,ue)
        return strain

    def elemental_internal_force(ue,Le,w,Zta,r_zta,rn,strain_int,sigma_ov_int,E,v,dt,Q,T):

        r = r_zta(Zta,Le,rn)
        B = configure_B_matrix(Le,Zta,r)
        #N = shape_fn(Zta)

        strain_next= elemental_strain(ue,Le,Zta,r)
        sigma,sigma_ov = compute_sigma(strain_next,strain_int,sigma_ov_int,E,v,dt,T,Q)
        Fe_int = w*(np.dot(B.T,sigma)*r*(Le/2))
        return Fe_int,sigma_ov
    
    def gauss_quadrature_Ke_t(w,Le,r_zta,E,v,Q,dt,T,Zta,rn):

        r = r_zta(Zta,Le,rn)
        B = configure_B_matrix(Le,Zta,r)
        C_t = compute_Ct(E,v,Q,dt,T)
        temp_ket_val_1 = np.dot(B.T,C_t)
        temp_ket_val_2 = r*(Le/2)*w

        Ke_t = np.dot(temp_ket_val_1,B)*temp_ket_val_2
        return Ke_t
    

    return gauss_quadrature_Ke_t,element_external_force,elemental_internal_force,configure_B_matrix

def global_routine(num_elements):

    gauss_quadrature_Ke_t,element_external_force,elemental_internal_force,configure_B_matrix = element_routine()
    def assignment_matrix(element_i,number_nodes):
        A = np.zeros((2,number_nodes))
        start_index = element_i - 1
        A[0][start_index] = 1
        A[1][element_i] = 1
        return A
    
    def assemble_Kt(num_elements,element_lengths,E,v,Q,T,dt,r_nodes):

        w =2
        Zta = 0
        node = num_elements+1
        Kt = np.zeros((node,node))

        def r_zta(Zta,Le,rn):
            return (Le/2)*(1+Zta) + rn

        for i,Le in zip(range(1, num_elements+1),element_lengths):
            rn = r_nodes[i-1]
            Ae = assignment_matrix(i,node)
            Ket = gauss_quadrature_Ke_t(w,Le,r_zta,E,v,Q,dt,T,Zta,rn)
            tempK1 = np.dot(Ae.T,Ket)
            tempK2 = np.dot(tempK1,Ae)
            Kt = np.add(Kt,tempK2)
            
        return Kt

    def assemble_Fext(num_elements,element_lengths,p,a):

        node = num_elements+1
        F_ext = np.zeros((node,1))
        for i in range(1, num_elements+1):
            Fe = element_external_force()
            Ae = assignment_matrix(i,node)
            tempK1 = np.dot(Ae.T,Fe)
            F_ext = np.add(F_ext,tempK1)
        F_ext[0][0] = p*a
        return F_ext
    
    def assemble_Fint(U,strain,sigma_ov_int,E,v,dt,Q,T,r_nodes,element_lengths):
        
        w =2
        Zta = 0
        node = num_elements+1
        F_int = np.zeros((node,1))
        sigma_ov = np.array([[],[]])
        def r_zta(Zta,Le,rn):
            return (Le/2)*(1+Zta) + rn
        
        for i,Le in zip(range(1, num_elements+1),element_lengths):
            ue = np.array([U[i-1],U[i]])
            strain_int = np.array([[strain[:,i-1][0]],[strain[:,i-1][1]]])
            sigma_ov_int_n = np.array([[sigma_ov_int[:,i-1][0]],[sigma_ov_int[:,i-1][1]]])
            rn = r_nodes[i-1]
            Ae = assignment_matrix(i,node)
            Fi,sigma_ov_next = elemental_internal_force(ue,Le,w,Zta,r_zta,rn,strain_int,sigma_ov_int_n,E,v,dt,Q,T)
            temp_val = np.dot(Ae.T,Fi)
            F_int = np.add(F_int,temp_val)
            sigma_ov = np.append(sigma_ov,sigma_ov_next,axis=1)

        return F_int,sigma_ov
    
    def Newton_Raphson_method(K_max,tolerence,num_elements,element_lengths,E,v,Q,T,dt,p,a,strain_int,sigma_ov_int,r_nodes,U_int):

        Kt = assemble_Kt(num_elements,element_lengths,E,v,Q,T,dt,r_nodes)
        F_ext = assemble_Fext(num_elements,element_lengths,p,a)

        k =1
        def Euclidean_norm(y):
            result = np.linalg.norm(y,ord=np.inf)
            return result.item()

        while True:
            F_int,sigma_ov = assemble_Fint(U_int,strain_int,sigma_ov_int,E,v,dt,Q,T,r_nodes,element_lengths)
            detKt = np.linalg.det(Kt)
            if( detKt != 0):
                Kt_inv = np.linalg.inv(Kt)
            else:
                print('Singular Matrix error')
            R = (F_int - F_ext)
            U_next = U_int - np.dot(Kt_inv,R)
            norm_R = Euclidean_norm(R)
            norm_delta_U = Euclidean_norm(U_next-U_int)
            norm_Fint = Euclidean_norm(F_int)
            norm_U = Euclidean_norm(U_int)
            if (not(norm_R > tolerence*norm_Fint or norm_delta_U > tolerence*norm_U) and (not k < K_max)):
                return U_next,sigma_ov
            #if k == K_max-1:
                #print('Not converging')
                #return U_next,sigma_ov,False
            
            U_int = U_next
            k+=1

    def assemble_strain(U,number_elements,element_lengths,r_nodes):

        Zta = 0
        Strain = np.array([[],[]])

        def r_zta(Zta,Le,rn):
            return (Le/2)*(1+Zta) + rn

        for i,Le in zip(range(1, num_elements+1),element_lengths):
            ue = np.array([U[i-1],U[i]])
            rn = r_nodes[i-1]
            r = r_zta(Zta,Le,rn)
            B = configure_B_matrix(Le,Zta,r)
            strain = np.dot(B,ue)
            Strain = np.append(Strain,strain,axis =1)
        return  Strain

        
    
    def finite_element_method(num_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f):

        time = np.arange(0,t_f,dt)
        U = np.array([])
        U_int = np.zeros((len(r_nodes),1))
        strain = np.zeros((2,num_elements))
        stress_ov = np.zeros((2,num_elements))
        p =0

        for t in time[1:]:

            if( t <= t_l):
                p+= 25*dt

            K_max = 20
            tolerence = 0.005
            Um,sigma_ov_updated = Newton_Raphson_method(K_max,tolerence,num_elements,element_lengths,E,v,Q,T,dt,p,a,strain,stress_ov,r_nodes,U_int)
            U = Um
            stress_ov = sigma_ov_updated
            strain = assemble_strain(Um,num_elements,element_lengths,r_nodes)
            U_int = Um
        
        return U,strain
    
    return finite_element_method

def analytical_solution(p,E,v,r,a,b):

    temp_val_1 = ((1+v)*(p/E)*(a**2/(b**2 - a**2)))
    temp_val_2 = (((1-2*v)*r) + (b**2/r))

    return temp_val_1*temp_val_2

def post_processing(u,p,E,v,r,a,b):

    u_analytical = analytical_solution(p,E,v,r,a,b)
    u_numerical = u[:,0]

    print(u_analytical)
    print(u_numerical)

    fig,ax = plt.subplots()
    ax.plot(r,u_analytical,label='Analytical')
    ax.plot(r,u_numerical,'x',label='Numerical')
    ax.set(xlabel='r',ylabel='Displacement')
    plt.title('Displacment along radial direction for elastic case (Q=0)')
    plt.legend()
    plt.show()





def main_program():

    r_inner = 40
    r_outer = 80
    meshrefinementfactor = 2
    num_elements = 30
    E = 70e3
    v =0.25
    #Q = 35e3
    Q =0
    T =1 
    P_max =50
    a= 40
    dt = 0.1
    t_l =2
    t_f = 10
    r_nodes,element_lengths = meshing(r_inner,r_outer,meshrefinementfactor,num_elements)
    finite_element_method = global_routine(num_elements)

    U,strain = finite_element_method(num_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f)

    print(f'Strain : {strain}')
    #print(f'Displacement : {U}')

    post_processing(U,P_max,E,v,r_nodes,r_inner,r_outer)

main_program()
    
