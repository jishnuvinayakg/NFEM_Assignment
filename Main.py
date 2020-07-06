import Global_routine
import numpy as np
import matplotlib.pyplot as plt 

def read_input_data(file_name):

    values_read= []

    with open(file_name,'r') as f:
        data = f.readlines()
    
    for line in data:
        key_value = line.rstrip().split(':')
        values_read.append(float(key_value[1]))
        
    return values_read
        

    


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

    # r_inner = 40
    # r_outer = 80
    # meshrefinementfactor = 2
    # E = 70e3
    # v =0.25
    # Q =0
    # T =1 
    # P_max =50
    # a= 40
    # dt = 1
    # t_l =2
    # t_f = 10
    r_inner,r_outer,meshrefinementfactor,E,v,_,T,P_max,t_l,t_f,_,_ = read_input_data('input_data.txt')
    Q =0
    dt = 1
    a = r_inner

    P = []
    time = np.arange(0,t_f+1,dt)
    p =0

    for t in time:

        if(t <= t_l):
            p+= 25*dt
            P.append(p)
        else:
            P.append(P_max)

    def call_FEM_solver(number_elements):

        r_nodes,element_lengths = meshing(r_inner,r_outer,meshrefinementfactor,number_elements)
        U,_ = Global_routine.non_linear_fem_solver(number_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f)
        #print(U)
        return U[0],r_nodes
    
    
    u_numerical_3_elements,r_10 = call_FEM_solver(3)
    u_numerical_5_elements,r_15 = call_FEM_solver(5)
    u_numerical_10_elements,r_20 = call_FEM_solver(10)
    u_numerical_15_elements,r_25 = call_FEM_solver(15)
    u_analytical = analytical_solution(P[0],E,v,r_inner,r_outer,np.array(r_25))
 
    fig,ax = plt.subplots()
    ax.plot(r_25,u_analytical,color ='black',label='Analytical')
    ax.plot(r_10,u_numerical_3_elements,'+--',label='Numerical 3 elements')
    ax.plot(r_15,u_numerical_5_elements,'*--',label='Numerical 5 elements')
    ax.plot(r_20,u_numerical_10_elements,'o--',label='Numerical 10 elements')
    ax.plot(r_25,u_numerical_15_elements,'1--',label='Numerical 15 elements')
    ax.set(xlabel='r',ylabel='Displacement')
    plt.title('Displacment along radial direction for elastic case (Q=0)')
    fig.savefig('Elastic convergence study')
    plt.legend()
    plt.show()

def visco_elastic_convergence_study():
    # r_inner = 40
    # r_outer = 80
    # meshrefinementfactor = 2
    # E = 70e3
    # v =0.25
    # Q = 35e3
    # T =1 
    # P_max =50
    # a= 40
    # t_l =2
    # t_f = 10

    r_inner,r_outer,meshrefinementfactor,E,v,Q,T,P_max,t_l,t_f,_,_ = read_input_data('input_data.txt')
    a = r_inner
    r,_ = meshing(r_inner,r_outer,meshrefinementfactor,25)
    u_analytical = analytical_solution(50,E,v,r_inner,r_outer,r)

    def call_FEM_solver(number_elements,dt):

        r_nodes,element_lengths = meshing(r_inner,r_outer,meshrefinementfactor,number_elements)
        U,_ = Global_routine.non_linear_fem_solver(number_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f)

        return U[-1],r_nodes

    #Keeping number of elements constant and changing dt

    #10 elements
    u_numerical_10_elements_1dt,r_10_1dt = call_FEM_solver(3,1.5)
    u_numerical_10_elements_2dt,r_10_2dt = call_FEM_solver(3,1.3)
    u_numerical_10_elements_3dt,r_10_3dt = call_FEM_solver(3,1.2)
    u_numerical_10_elements_4dt,r_10_4dt = call_FEM_solver(3,0.1)

    #30 elements
    u_numerical_30_elements_1dt,r_30_1dt = call_FEM_solver(5,1.5)
    u_numerical_30_elements_2dt,r_30_2dt = call_FEM_solver(5,1.3)
    u_numerical_30_elements_3dt,r_30_3dt = call_FEM_solver(5,1.2)
    u_numerical_30_elements_4dt,r_30_4dt = call_FEM_solver(5,0.1)

    #60 elements
    u_numerical_60_elements_1dt,r_60_1dt = call_FEM_solver(10,1.5)
    u_numerical_60_elements_2dt,r_60_2dt = call_FEM_solver(10,1.3)
    u_numerical_60_elements_3dt,r_60_3dt = call_FEM_solver(10,1.2)
    u_numerical_60_elements_4dt,r_60_4dt = call_FEM_solver(10,0.1)

    #90 elements
    u_numerical_90_elements_1dt,r_90_1dt= call_FEM_solver(15,1.5)
    u_numerical_90_elements_2dt,r_90_2dt = call_FEM_solver(15,1.3)
    u_numerical_90_elements_3dt,r_90_3dt = call_FEM_solver(15,1.2)
    u_numerical_90_elements_4dt,r_90_4dt = call_FEM_solver(15,0.1)

    fig,ax = plt.subplots(2,2)
    #plt.title('Conergence study of visco elastic case')
    ax[0,0].plot(r,u_analytical,color ='black',label='Analytical')
    ax[0,0].plot(r_10_1dt,u_numerical_10_elements_1dt,'--r*',label='Δt-1.5')
    ax[0,0].plot(r_10_2dt,u_numerical_10_elements_2dt,'--y+',label='Δt-1.3')
    ax[0,0].plot(r_10_3dt,u_numerical_10_elements_3dt,'--m1',label='Δt-1.2')
    ax[0,0].plot(r_10_4dt,u_numerical_10_elements_4dt,'--g.',label='Δt- 0.1')
    ax[0,0].set_title('3 Elements')
    ax[0,0].legend()

    ax[0,1].plot(r,u_analytical,color ='black',label='Analytical')
    ax[0,1].plot(r_30_1dt,u_numerical_30_elements_1dt,'--r*',label='Δt-1.5')
    ax[0,1].plot(r_30_2dt,u_numerical_30_elements_2dt,'--y+',label='Δt-1.3')
    ax[0,1].plot(r_30_3dt,u_numerical_30_elements_3dt,'--m1',label='Δt-1.2')
    ax[0,1].plot(r_30_4dt,u_numerical_30_elements_4dt,'--g.',label='Δt-0.1')
    ax[0,1].set_title('5 Elements')
    ax[0,1].legend()

    ax[1,0].plot(r,u_analytical,color ='black',label='Analytical')
    ax[1,0].plot(r_60_1dt,u_numerical_60_elements_1dt,'--r*',label='Δt-1.5')
    ax[1,0].plot(r_60_2dt,u_numerical_60_elements_2dt,'--y+',label='Δt-1.3')
    ax[1,0].plot(r_60_3dt,u_numerical_60_elements_3dt,'--m1',label='Δt-1.2')
    ax[1,0].plot(r_60_4dt,u_numerical_60_elements_4dt,'--g.',label='Δt-0.1')
    ax[1,0].set_title('10 Elements')
    ax[1,0].legend()

    ax[1,1].plot(r,u_analytical,color ='black',label='Analytical')
    ax[1,1].plot(r_90_1dt,u_numerical_90_elements_1dt,'--r*',label='Δt-1.5')
    ax[1,1].plot(r_90_2dt,u_numerical_90_elements_2dt,'--y+',label='Δt-1.3')
    ax[1,1].plot(r_90_3dt,u_numerical_90_elements_3dt,'--m1',label='Δt-1.2')
    ax[1,1].plot(r_90_4dt,u_numerical_90_elements_4dt,'--g.',label='Δt-0.1')
    ax[1,1].set_title('15 Elements')
    ax[1,1].legend()
    fig.set_size_inches(12,10)
    fig.savefig('Viscoelastic convergence study')
    plt.show()

def extract_required_results():
    # r_inner = 40
    # r_outer = 80
    # meshrefinementfactor = 2
    # number_elements = 10
    # E = 70e3
    # v =0.25
    # Q =35e3
    # T =1 
    # P_max =50
    # a= 40
    # dt = 0.1
    # t_l =2
    # t_f = 10
    r_inner,r_outer,meshrefinementfactor,E,v,Q,T,P_max,t_l,t_f,number_elements,dt = read_input_data('input_data.txt')
    number_elements = int(number_elements)
    a = r_inner
    t_f =int(t_f)
    t_l = int(t_l)
    time = np.arange(0,t_f+1,dt)
    time = time[time<=10.0]

    r_nodes,element_lengths = meshing(r_inner,r_outer,meshrefinementfactor,number_elements)
    U,Stress = Global_routine.non_linear_fem_solver(number_elements,element_lengths,E,v,Q,T,dt,P_max,a,r_nodes,t_l,t_f)

    u_r_outer_tF =[]
    for u in U:
        u_r_outer_tF.append(u[-1])

    fig,ax = plt.subplots()
    ax.plot(time[1:],u_r_outer_tF,'g')
    ax.set(xlabel = 'time (s)',ylabel ='Displacement(r mm)')
    plt.title("History of widening of pipe Ur(r=b,t)")
    fig.savefig('History of widening')
    plt.show()

    print(Stress)
    with open('result.txt','w') as f:
        f.write('The displacement at final time step is \n ' + str(U[-1]) +'\n\n\n')
        f.write('The stress at final time step is \n\n Stress σ rr \n' + str(Stress[0]) +'\n\n\n' +'Stress σ φφ \n'+str(Stress[1]))




#elastic_convergence_study()
#visco_elastic_convergence_study()
extract_required_results()




