# SOLVING ALGORITHMS

#IMPORTS
import numpy as np 
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from utils import * 






###############################
###############################
######## SOLVING ALGOS ########
###############################
###############################



#ADMs



def ADM(y,a_init,x_init,lmbda,max_iter,tol,prnt='T'):
    """
    The ADM (Alternating Descent Method) algorithm to solve the Sas problem.

    Inputs : 
        y = (n,) arrray, the observed signal
        a_init = (n0,) arrray, the initialization of the convolution kernel
        x_init = (n,) arrray, the initialization of the convolution signal
        lmbda = scalar, the regularization (sparsity) parameter
        max_iter = int, the number of iteration maximal
        tol = scalar, the minimum difference between two consecutives iterates
        print = sequence, 'T' -> print the error of each iterations

    Outputs : 
        error = (max_iter,) array, the precision error at each iteration (norm(conv(a,x)-y))
        sparsity =(max_iter,) array, the % of sparsity of x at each iteration
        a = (n0,) array, the optimal convolution kernel
        x =  (n,) array, the optimal convolution signal

    """
    a=a_init
    x=x_init

    #Record of the solution path 
    Psi_val = []
    error = []
    sparsity = []

    #Initialization of the stepsize
    t=1

    for iter in range(max_iter):
        
        #Fix a and take a descent step on x
        f_x = psy_val(y,a,x)
        grad_f_x = compute_gradient(a,x,y,'x')

        x_old=x
        t,x=backtracking_rule(t,x,y,a,f_x,grad_f_x,lmbda)

        #Fix x and take a descent step on a

        f_a = psy_val(y,a,x)
        grad_f_a = compute_gradient(a,x,y,'a')

        a_old=a 
        tau,a = riemanian_linesearch(y, a, x, f_a, grad_f_a)


        if (np.linalg.norm(a_old -a) <= tol) and (np.linalg.norm(x_old -x) <= tol) and (iter>3):
            break

        Psi_val.append(Psy_val(y,a,x,lmbda)) 
        err=np.linalg.norm(y-cconv(a,x))
        error.append(err)
        sparsity.append(np.sum(x<1e-4)/len(x))

        if prnt=='T':
            print(f'error of the {iter}th iteration : {err}')
    
    
    return error,sparsity,a,x



def ADM_lamb_adaptatif(y,a_init,x_init,max_iter,tol,lmbda_init=1,prnt='T'):
    """
    The ADM (Alternating Descent Method) algorithm with adptative lambda to solve the Sas problem.

    Inputs : 
        y = (n,) arrray, the observed signal
        a_init = (n0,) arrray, the initialization of the convolution kernel
        x_init = (n,) arrray, the initialization of the convolution signal
        max_iter = int, the number of iteration maximal
        tol = scalar, the minimum difference between two consecutives iterates
        lmbda_init = scalar, the initialization for the regularization (sparsity) parameter
        print = sequence, 'T' -> print the error of each iterations

    Outputs : 
        error = (max_iter,) array, the precision error at each iteration (norm(conv(a,x)-y))
        sparsity =(max_iter,) array, the % of sparsity of x at each iteration
        a = (n0,) array, the optimal convolution kernel
        x =  (n,) array, the optimal convolution signal

    """
    a=a_init
    x=x_init

    #Record of the solution path 
    Psi_val = []
    error = []
    sparsity = []

    #Initialization of the stepsize
    t=1
    lmbda=lmbda_init

    for iter in range(max_iter):
        
        #Fix a and take a descent step on x
        f_x = psy_val(y,a,x)
        grad_f_x = compute_gradient(a,x,y,'x')

        x_old=x
        t,x=backtracking_rule(t,x,y,a,f_x,grad_f_x,lmbda)

        #Fix x and take a descent step on a

        f_a = psy_val(y,a,x)
        grad_f_a = compute_gradient(a,x,y,'a')

        a_old=a 
        tau,a = riemanian_linesearch(y, a, x, f_a, grad_f_a)


        if (np.linalg.norm(a_old -a) <= tol) and (np.linalg.norm(x_old -x) <= tol) and (iter>3):
            break

        Psi_val.append(Psy_val(y,a,x,lmbda)) 
        err=np.linalg.norm(y-cconv(a,x))
        error.append(err)
        sparsity.append(np.sum(x<1e-4)/len(x))

        if prnt=='T':
            print(f'error of the {iter}th iteration : {err}')

        #aptative lambda
        lmbda=lmbda_init/(iter+1)
    

    return error,sparsity,a,x



def IADM(y,a_init,x_init,lmbda,beta,max_iter,tol,prnt='T'):
    """
    The iADM (Inertial Alternating Descent Method) algorithm to solve the Sas problem.

    Inputs : 
        y = (n,) arrray, the observed signal
        a_init = (n0,) arrray, the initialization of the convolution kernel
        x_init = (n,) arrray, the initialization of the convolution signal
        beta = scalar in [0,1], the momentum parameter
        lmbda = scalar, the regularization (sparsity) parameter
        max_iter = int, the number of iteration maximal
        tol = scalar, the minimum difference between two consecutives iterates
        print = sequence, 'T' -> print the error of each iterations

    Outputs : 
        error = (max_iter,) array, the precision error at each iteration (norm(conv(a,x)-y))
        sparsity =(max_iter,) array, the % of sparsity of x at each iteration
        a = (n0,) array, the optimal convolution kernel
        x =  (n,) array, the optimal convolution signal

    """

    a=a_init
    a_old=a_init
    x=x_init+0.01*np.random.normal(0,1,len(x_init))
    x_old=x_init

    #Record of the solution path 
    Psi_val = []
    error = []
    sparsity = []

    n=len(y)

    #Initialization of the stepsize
    t=1

    for iter in range(max_iter):

        #Fix a and take a descent step on x

        w = x+beta*(x-x_old)

        f_x = psy_val(y,a,w)

        grad_f_x = compute_gradient(a,w,y,'x')
 

        x_old=x
        t,x=backtracking_rule(t,w,y,a,f_x,grad_f_x,lmbda)

        #Fix x and take a descent step on a

        #z = retraction_operator(a,beta*inverserse_retraction_operator(a_old,a))
        D=a-a_old
        z = retraction_operator(a,beta*D)

        f_a = psy_val(y,z,x)

        grad_f_a = compute_gradient(z,x,y,'a')

        a_old=a 
        tau,a = riemanian_linesearch(y, z, x, f_a, grad_f_a)


        if (np.linalg.norm(a_old -a) <= tol) and (np.linalg.norm(x_old -x) <= tol) and (iter>3):
            break

        Psi_val.append(Psy_val(y,a,x,lmbda)) 
        err=np.linalg.norm(y-cconv(a,x))
        error.append(err)
        sparsity.append(np.sum(x<1e-4)/len(x))

        if prnt=='T':
            print(f'error of the {iter}th iteration : {err}')
    
    
    return error,sparsity,a,x





def IADM_lamb_adaptatif(y,a_init,x_init,beta,max_iter,tol,lmbda_init=1,prnt='T'):
    """
    The iADM (Inertial Alternating Descent Method) algorithm to solve the Sas problem.

    Inputs : 
        y = (n,) arrray, the observed signal
        a_init = (n0,) arrray, the initialization of the convolution kernel
        x_init = (n,) arrray, the initialization of the convolution signal
        beta = scalar in [0,1], the momentum parameter
        max_iter = int, the number of iteration maximal
        tol = scalar, the minimum difference between two consecutives iterates
        lmbda_init = scalar, the initialization for the regularization (sparsity) parameter
        print = sequence, 'T' -> print the error of each iterations

    Outputs : 
        error = (max_iter,) array, the precision error at each iteration (norm(conv(a,x)-y))
        sparsity =(max_iter,) array, the % of sparsity of x at each iteration
        a = (n0,) array, the optimal convolution kernel
        x =  (n,) array, the optimal convolution signal

    """

    a=a_init
    a_old=a_init
    x=x_init+0.01*np.random.normal(0,1,len(x_init))
    x_old=x_init

    #Record of the solution path 
    Psi_val = []
    error = []
    sparsity = []

    n=len(y)

    #Initialization of the stepsize
    t=1
    lmbda=lmbda_init

    for iter in range(max_iter):

        #Fix a and take a descent step on x

        w = x+beta*(x-x_old)

        f_x = psy_val(y,a,w)

        grad_f_x = compute_gradient(a,w,y,'x')
 

        x_old=x
        t,x=backtracking_rule(t,w,y,a,f_x,grad_f_x,lmbda)

        #Fix x and take a descent step on a

        #z = retraction_operator(a,beta*inverserse_retraction_operator(a_old,a))
        D=a-a_old
        z = retraction_operator(a,beta*D)

        f_a = psy_val(y,z,x)

        grad_f_a = compute_gradient(z,x,y,'a')

        a_old=a 
        tau,a = riemanian_linesearch(y, z, x, f_a, grad_f_a)


        if (np.linalg.norm(a_old -a) <= tol) and (np.linalg.norm(x_old -x) <= tol) and (iter>3):
            break

        Psi_val.append(Psy_val(y,a,x,lmbda)) 
        err=np.linalg.norm(y-cconv(a,x))
        error.append(err)
        sparsity.append(np.sum(x<1e-4)/len(x))

        if prnt=='T':
            print(f'error of the {iter}th iteration : {err}')

        #adaptative lambda
        lmbda=lmbda_init/(iter+1)

    
    return error,sparsity,a,x





#HOMOTOPY 





def homotopy_continuation(y,a_init,x_init,lmbda_init,lmbda_final,eta,delta,eps_tol,case,max_iter_in=100,beta_iadm=None,prnt='F'):
    """
    The Homotopy Continuation algorithm to solve the Sas problem.

    Inputs : 
        y = (n,) arrray, the observed signal
        a_init = (n0,) arrray, the initialization of the convolution kernel
        x_init = (n,) arrray, the initialization of the convolution signal
        lmbda_init = scalar, the intial regularization (sparsity) parameter
        lmbda_final = scalar, the final regularization (sparsity) parameter
        eta = scalar in [0,1], decay penalty parameter
        delta = scalar in [0,1], precision factor
        eps_tol = scalar, the minimum difference between two consecutives iterates
        case = sequence, 'IADM' -> use of the iADM algo or 'ADM' -> use of the ADM algo (also 'IADMadapt' and 'ADMadapt' for adptative version)
        max_iter_in = int, the number of iteration maximal of the IADM or ADM 
        beta_iadm = scalar in [0,1], momentum parameter for the IADM

    Outputs : 
        errors = (max_iter,) array, the precision error at each iteration (norm(conv(a,x)-y))
        sparsitys =(max_iter,) array, the % of sparsity of x at each iteration
        a = (n0,) array, the optimal convolution kernel
        x =  (n,) array, the optime convolution signal

    """

    lmbda=lmbda_init
    eps=delta*lmbda

    a=a_init
    x=x_init

    errors=np.array([])
    sparsitys=np.array([])

    if beta_iadm==None:
        beta_iadm=0.9

    K=int(np.log(lmbda_final/lmbda_init)/np.log(eta))+15

    print('number of iteration :',K)

    for _ in range(K):
        if case=='IADM':
            error,sparsity,a,x = IADM(y=y,a_init=a,x_init=x,lmbda=lmbda,beta=beta_iadm,max_iter=max_iter_in,tol=eps,prnt=prnt)
        elif case=='ADM':
            error,sparsity,a,x = ADM(y=y,a_init=a,x_init=x,lmbda=lmbda,max_iter=max_iter_in,tol=eps,prnt=prnt)
        elif case=='AMDadapt':
            error,sparsity,a,x = ADM_lamb_adaptatif(y=y,a_init=a,x_init=x,max_iter=max_iter_in,tol=eps,lmbda_init=lmbda,prnt=prnt)
        elif case=='IAMDadapt':
            error,sparsity,a,x = IADM_lamb_adaptatif(y=y,a_init=a,x_init=x,beta=beta_iadm,max_iter=max_iter_in,tol=eps,lmbda_init=lmbda,prnt=prnt)

        errors=np.concatenate((errors,np.array(error)))
        sparsitys=np.concatenate((sparsitys,np.array(sparsity)))
        lmbda=eta*lmbda
        eps=delta*lmbda

    #Final round
    if case=='IADM':
        error,sparsity,a,x = IADM(y=y,a_init=a,x_init=x,lmbda=lmbda,beta=beta_iadm,max_iter=max_iter_in,tol=eps_tol,prnt=prnt)
        print(f'The error for the last iteration is {error[-1]}')
    elif case=='ADM':
        error,sparsity,a,x = ADM(y=y,a_init=a,x_init=x,lmbda=lmbda,max_iter=max_iter_in,tol=eps_tol,prnt=prnt)
        print(f'The error for the last iteration is {error[-1]}')
    elif case=='AMDadapt':
        error,sparsity,a,x = ADM_lamb_adaptatif(y=y,a_init=a,x_init=x,max_iter=max_iter_in,tol=eps,lmbda_init=lmbda,prnt=prnt)
        print(f'The error for the last iteration is {error[-1]}')
    elif case=='IAMDadapt':
        error,sparsity,a,x = IADM_lamb_adaptatif(y=y,a_init=a,x_init=x,beta=beta_iadm,max_iter=max_iter_in,tol=eps,lmbda_init=lmbda,prnt=prnt)
        print(f'The error for the last iteration is {error[-1]}')

    errors=np.concatenate((errors,np.array(error)))
    sparsitys=np.concatenate((sparsitys,np.array(sparsity)))

    return errors,sparsitys,a,x
