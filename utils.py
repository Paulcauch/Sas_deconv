# UTILS 

#IMPORT 
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt




###############################
###############################
###### GENERAL FUNCTIONS ######
###############################
###############################


#PADDING

def pad_end(v,n):
    """
    Pad the vector with n more zeros.

    Inputs :
        v = (n0,) array, vector
        n = int, size 
    Outputs : 
        padded = (n+n0,) array, the padded vector
    """
    
    padded = np.pad(v,n)[n:]

    return padded



#CONVOLUTION OPERATORS

def cconv(a,x,N=None):
    """
    Circular convolution between the kernel a and the signal x.
    
    Inputs :
        a = (n0,) array, convolution kernel
        x = (n,) array, signal convoluted
        N = int, size of the result of the convolution (=n)
    
    Outputs :
        y = (N,) array, convolution product
    """

    if N==None:
        N=len(x)

    # Compute FFTs
    signal_fft = np.fft.fft(x, N)
    kernel_fft = np.fft.fft(a, N)
    
    # Compute the product in the frequential domain (circ conv) 
    result_fft = signal_fft * kernel_fft
    
    # Back to the temporal domain
    result = np.fft.ifft(result_fft)
    
    conv = np.real(result)

    return conv


def cconv_loop(a,x):
        """
        Circular Convolution with loops
        
        Inputs : 
            a = (n0,) array, convolution kernel
            x = (n,) array, signal convoluted
        Outputs : 
            y = (n,) array, convolution product
        """

        n=len(x)
        d=len(a)

        y=np.zeros(n)

        for k in range(n):
            sum=0
            for s in range(d):
                sum+=a[s]*x[k-s]
            y[k]=sum
        return y 


def circulant_matrix(v):
    """
    Compute the Circulant matrix of v.

    Inputs : 
        v = (n,) array, vector
    Outputs : 
        C = (n,n) array, the circulant matrix
    """
    C=np.zeros((len(v),len(v)))

    for i in range(len(v)):

        C[:,i]=np.roll(v,i)

    return C



def reduce_coor(z,n):
    """
    Reduce a vector to the n firt elements.
    """
    return z[:n]





#INITIALISATION


def init_sol(n,d,k,seed=None):
    """
    Initialization of the solution for the SaS problem :
    a_sol is a realization of a standard normal distribution N(0,Id).
    A subset S ⊂ {1,...,n} is chosen uniformly at random among all subsets with cardinality k. 
    For each s ∈ S, x_sol,s is chosen according to a normal law N(0,1) and, for each s ∈/ S, x_sol,s = 0

    Inputs : 
        n = int, size of x
        k = int, <= n with the number of non zero coordinate of x
        d = int, <= size of a
        seed = int, set a seed to compare results
    
    Outputs : 
        a_sol = (n,) array, convolution kernel solution to the SaS problem
        x_sol = (d,) array, signal convoluted to the SaS problem
        y = (n,) array, observed solution to the Sas problem 
    
    """
    if seed!=None :
        np.random.seed(seed)

    x = np.zeros(n)

    a = np.random.uniform(low=0,high=1,size=d)

    S = np.random.choice(np.arange(n),size=k,replace=False)
    x[S] = np.random.uniform(low=0,high=1,size=k)

    a_sol = a/np.linalg.norm(a)
    x_sol = x*np.linalg.norm(a)
    y = cconv(a,x)

    return a_sol,x_sol,y


def init_a(y,d):

    i=random.randint(0,len(y)-d-1)
    a_init=np.pad(y[i:d+i],d-1)

    return a_init

def init_a_wo_pad(y,d):

    i=random.randint(0,len(y)-d-1)
    a_init=y[i:d+i] 

    return a_init



#COHERENCE

def shift_coherence_kernel(a):
    """
    shift-coherence of the kernel a, which measures the “similarity” between a and its cyclic-shifts;
    smaller shift_coherence ->>> "easier" problem.

    Inputs : 
        a = (n0,) array, convolution kernel 

    Outputs : 
        coherence = scalar in [0,1], coherence of the kernel 
    """
    
    list = []
    n=len(a)

    for i in range(1,n):
        list.append(np.abs((a/np.linalg.norm(a))@(np.roll(a,i)/np.linalg.norm(a))))
    
    coherence = max(list)

    return coherence





#OBJECTIVE FUNCTIONS

def Psy_val(y,a,x,lmbda) : 
    """
    Compute the Psy value function (the objective of the SaS problem)

    Inputs : 
        y = (n,) array, the observed signal
        a = (d,) arrray, the convolution kernel that we want to recover
        x = (n,) array, the convolution signal that we want to recover
        lmbda = scalar, regularization (sparse) parameter of the problem

    Outputs : 
        val = scalar, the value  of the objective function

    """
    val = (1/2) * np.linalg.norm(y-cconv(a,x))**2 + lmbda*np.sum(np.abs(x))

    return val 

def psy_val(y,a,x) : 
    """
    Compute the Psy value function (the objective of the SaS problem) without the g(x)

    Inputs : 
        y = (n,) array, the observed signal
        a = (d,) arrray, the convolution kernel that we want to recover
        x = (n,) array, the convolution signal that we want to recover

    Outputs : 
        val = scalar, the value  of the objective function

    """
    val = (1/2) * np.linalg.norm(y-cconv(a,x))**2 

    return val 




#FIND THE SOLUTION

def shift_correction(a_opt,a_real):
    """
    Produces the shift to recover a using the real solution : 
    We can optimize with a in R^3*d-2 with possibly a shift, as a belongs to R^d we need to find 
    the right shift to recover a.

    Inputs :
        a_opt = (3*d-2,) array, the optimal kernel found by an optimization algorithm
        a_real = (d,) array, the ground truth kernel

    Outputs : 
        a_recov = (d,) array, the optimal kernel with the good shape and sign

    """
    d=len(a_real)

    #Compute the cross correlation between the ground truth and the optimal signal
    corr = np.correlate(a_real, a_opt, mode='full')

    ind = np.argmax(np.abs(corr))

    max_corr = corr[ind]
    sign_max_corr = np.sign(max_corr)

    shift_amount = ind - (len(a_real) - 1)

    a_recov = sign_max_corr*np.roll(a_opt, -shift_amount)[:d]

    return a_recov

def find_sol(y,a_opt,x_opt,d):
    """
    Produces the shift to recover a not using the real solution : 
    We can optimize with a in R^3*d-2 with possibly a shift, as a belongs to R^d we need to find 
    the right shift to recover a.

    Inputs :
        y = (n,) array, the observed signal
        a_opt = (3*d-2,) array, the optimal kernel found by an optimization algorithm
        x_opt = (n,) array, the optima convolution signal found by an optimization algorithm
        d = int, the dimension of the solution a

    Outputs : 
        a_recov = (d,) array, the optimal kernel with the good shape

    """
    error=[]

    for i in range(len(a_opt)):

        a_hat=np.roll(a_opt,-i+1)[:d]
        x_hat=np.roll(x_opt,i-1)
        y_hat=cconv(a_hat,x_hat)

        error.append(np.linalg.norm(y-y_hat))

    j=np.argmin(error)
    a_recov=np.roll(a_opt,-j+1)[:d]
    x_recov=np.roll(x_opt,j-1)

    return a_recov ,x_recov


def success(a_opt,x_opt,a_real,x_real):
    """
    Success of the algortihm :
    We say that the algorithm “succeeds” if it returns (a,x) such that 
    min(||(a_opt, x_opt)-(a_real, x_real)||_2, ||(a_opt, x_opt)+(a_real, x_real)||_2) ≤ 0.01||(a_real, x_real)||_2.
    """

    success=0

    diff = np.linalg.norm(np.concatenate((a_opt,x_opt))-np.concatenate((a_real,x_real)))
    sum = np.linalg.norm(np.concatenate((a_opt,x_opt))+np.concatenate((a_real,x_real)))

    opt = min(diff,sum)
    truth = np.linalg.norm(np.concatenate((a_real,x_real)))*0.01

    if opt<=truth:
        success=1

    return success, opt, truth








###############################
###############################
####### ALGOS FUNCTIONS #######
###############################
###############################




#PROJECTION

def soft_thres(z, lmbda):
    """
    Compute the soft threshold to compute the proximal operator of g

    Inputs : 
        z = (n,) array , the vector that we want to apply the proximal operator
        lmbda = scalar 
    
    Output : 
        prox = (n,) array, the updated vector
    """

    prox = np.sign(z) * np.maximum(np.abs(z) - lmbda, 0)

    return prox

def proj_a(a, z):
    """
    Project z onto the orthogonal complement of a.
    
    Inputs :
        a = A 1D numpy array representing the vector w.
        z = A 1D numpy array representing the vector z.
    
    Returns:
        projection = The projection of z onto the space orthogonal to a
    """
    d = len(a)

    p = np.identity(d)-(a[:,None]@a[:,None].T)/np.linalg.norm(a)

    projection = p @ z

    return projection





#RETRACTION

def retraction_operator(a,delta):
    """
    Retract back to oblique manifold 
    """

    norm_delta = np.linalg.norm(delta)
    
    #To avoid numericals issues we use an approximation
    #when the norm is too small
    if norm_delta>1e-3:
        return a*np.cos(norm_delta)+(delta/norm_delta)*np.sin(norm_delta)
    else :
        return (a+delta)/np.linalg.norm(a+delta)
    

def inverserse_retraction_operator(a,delta):
    """
    The inverse of the Retraction operator
    """
    d=len(a)
    alpha = np.arccos(np.clip(a.T@delta,a_min=-1+1e-9,a_max=1-1e-9))
    
    p = np.identity(d)-(a[:,None]@a[:,None].T)/np.linalg.norm(a)

    inv_retract = (alpha/np.sin(alpha))*p@delta

    return inv_retract





#LINESEARSH

def riemanian_linesearch(y ,a , x, f_a, grad_f_a ,tau_init=1,beta=1/2 ,stop=100):
    """
    Riemannian linesearch for stepsize tau and take steps on the a variable

    Inputs : 
        y = (n,) array, the observed signal
        a = (d,) array, the current iterate for the kernel
        x = (n,) array, the fixed convolution signal
        f_a = scalar, the Psy function at the current estimates
        grad_f_a = (d,) array, the grad of the Psy function at the current estimates
        beta = scalar in ]0,1[, the shriking factor for the stepsize
        stop = int, number of loop 

    Outputs : 
        tau = scalar, the updated stepsize 
        a = (d,) array, the updated iterate of the kernel 
    """

    #Initialization of the parameters 
    tau = tau_init
    eta = 0.8 

    a1=retraction_operator(a,-tau*grad_f_a)

    cmpt=0
    while psy_val(y,a1,x) > f_a-eta*tau*np.linalg.norm(grad_f_a)**2 :

        tau=tau*beta
        a1=retraction_operator(a,-tau*grad_f_a)

        cmpt+=1

        if cmpt>=stop:
            break
    
    return tau,a1


def backtracking_rule(t0,x,y,a,f_x,grad_f_x,lmbda,beta=1/2,stop=100):
    """
    Backtracking rule for stepsize t and take steps on the x variable

    Inputs : 
        t0 = scalar, the current stepsize
        x = (n,) array, the current iterate for the convolution signal
        y = (n,) array, the observed signal
        a = (d,) array, the fixed kernel
        f_x = scalar, the Psy function at the current estimates
        grad_f_x = (n,) array, the grad of the Psy function at the current estimates
        lmbda = scalar, the regularization parameter of the problem
        beta = scalar in ]0,1[, the shriking factor for the stepsize
        stop = int, number of loop 
    
    Outputs : 
        t = scalar, the updated stepsize 
        P = (n,) array, the updated iterate of the convolution signal (x)
    """

    #Define Q a quadratic approximation of the objective function 
    Q = lambda h, t : f_x + grad_f_x @ (h-x) + (1/(2*t))*np.linalg.norm(h-x)**2 + lmbda*np.sum(np.abs(h))


    #compute P_t(x) and t<-t0
    t = 8*t0 
    P = soft_thres(x-t*grad_f_x,lmbda*t0)


    #loop count
    cmpt=0

    while Psy_val(y,a,P,lmbda) >= Q(P,t):

        #t<-beta*t and update P_t
        t = beta*t
        P = soft_thres(x-t*grad_f_x,lmbda*t)

        cmpt+=1

        if cmpt>=stop : 
            break

    return t, P




# GRADIENTS


def compute_gradient(a,x,y,case):
    """
    Compute the gradient of psy w.r.t a or x. (Matricial version)

    Inputs : 
        x = (n,) array, the convolution signal
        y = (n,) array, the observed signal
        a = (d,) array, the kernel
        case = sequence 'x' -> gradient w.r.t x or 'a' -> gradient w.r.t to a
    Outputs : 
        gradient
    """ 
    n=len(x)
    d=len(a)

    if case == 'x':
        grad = circulant_matrix(pad_end(a,len(y)-len(a))).T@(cconv(a,x)-y)
        return grad
    
    if case == 'a':
        grad= reduce_coor(circulant_matrix(x).T@(cconv(a,x)-y),d)
        riemanian_grad = proj_a(a,grad)
        return riemanian_grad



def compute_gradient2(a,x,y,case):
    """
    Compute the gradient of psy w.r.t a or x.
    Inputs : 
        x = (n,) array, the convolution signal
        y = (n,) array, the observed signal
        a = (d,) array, the kernel
        case = sequence 'x' -> gradient w.r.t x or 'a' -> gradient w.r.t to a
    Outputs : 
        gradient
    """ 

    def reversal(a):
        a_rev= np.concatenate(([a[0]], a[:0:-1]))
        return a_rev
    
    n=len(x)
    d=len(a)

    if case == 'x':
        grad = cconv(reversal(a),cconv(a,x)-y) 
        return grad
    if case == 'a':
        grad= reduce_coor(cconv(reversal(x),cconv(a,x)-y,n),d)
        riemanian_grad = proj_a(a,grad)
        return riemanian_grad
