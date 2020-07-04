import Global_routine
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


def analytical_solution(p,E,v,a,b,r):

    temp_val_1 = ((1+v)*(p/E)*(a**2/(b**2 - a**2)))
    temp_val_2 = (((1-2*v)*r) + (b**2/r))
    return temp_val_1*temp_val_2

def elastic_convergence_study():

    r_inner = 40
    r_outer = 80
    meshrefinementfactor = 2
    #num_elements = 30
    E = 70e3
    v =0.25
    Q =0
    T =1 
    P_max =50
    a= 40
    dt = 1
    t_l =2
    t_f = 10
    P = []
    time = np.arange(0,t_f,dt)
    p =0

    for t in time:

        if(t <= t_l):
            p+= 25*dt
            P.append(p)
        else:
            P.append(P_max)
    
    print(P)

    def call_FEM_solver(number_elements):

        r_nodes,element_lengths = meshing(r_inner,r_outer,meshrefinementfactor,number_elements)
        U,strain = Global_routine.non_linear_fem_solver(number_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f)
        return U[0],r_nodes
    
    
    u_numerical_10_elements,r_10 = call_FEM_solver(10)
    u_numerical_15_elements,r_15 = call_FEM_solver(15)
    u_numerical_20_elements,r_20 = call_FEM_solver(20)
    u_numerical_25_elements,r_25 = call_FEM_solver(25)
    u_numerical_90_elements,r_90 = call_FEM_solver(30)
    u_analytical = analytical_solution(P[0],E,v,r_inner,r_outer,np.array(r_90))
    print(f'Analytical u : {u_analytical}')
    print(f'numerical u : {u_numerical_10_elements}')
 
    fig,ax = plt.subplots()
    ax.plot(r_90,u_analytical,color ='black',label='Analytical')
    ax.plot(r_10,u_numerical_10_elements,'+',label='Numerical 10 elements')
    ax.plot(r_15,u_numerical_15_elements,'*',label='Numerical 15 elements')
    ax.plot(r_20,u_numerical_20_elements,'--',label='Numerical 20 elements')
    ax.plot(r_25,u_numerical_25_elements,'--',label='Numerical 25 elements')
    ax.plot(r_90,u_numerical_90_elements,'+',label='Numerical 90 elements')
    ax.set(xlabel='r',ylabel='Displacement')
    plt.title('Displacment along radial direction for elastic case (Q=0)')
    plt.legend()
    plt.show()

elastic_convergence_study()
