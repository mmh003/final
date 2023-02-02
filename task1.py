
import numpy as np
import matplotlib.pyplot as plt

# import compass dynamics
import dynamic_leg as dyn

# import cost functions
import stagecost as cst
import cost_funct as cst_
import affineLQR as aff
import solver_LQR_matrix as solver
import referencecurve as REF
# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

######################################
# Algorithm parameters
######################################
max_iters = int(2e2)
beta = 0.5
armijo_maxiters = 200 # number of Armijo iterations
Stop_too_short_Jupgrade = 1e-19
visu_armijo = True

tf = 10     # final time in seconds
Nmax = 100
dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni
TT = int(tf/dt) # discrete-time samples

#######################################
# Trajectory parameters
#######################################

theta_sw_start = 0
theta_sw_end = 20

xx_des,uu_des = dyn.desired_step(theta_sw_start, theta_sw_end )

x_init = xx_des[0,:,:]
u_init = uu_des[0,:,:]
print(x_init)
print(u_init)

# we define the first guess vector state on the basis of our desired configuration
xx_guess = dyn.dynamic_leg(x_init, u_init)[0]
uu_guess = uu_des

# Extract system dimensions
ns = x_init.shape[0]
ni = u_init.shape[0]

# Flag signals
descent = 0
JJ_cost = np.zeros((Nmax,1,1)) # collect cost
descent = np.zeros((Nmax,1,1)) # collect descent
uu_temp = np.zeros((TT,ni,1))

last_iter = 0

# Define empty structures
xxx = np.zeros((Nmax,TT+1,ns,1))
uuu = np.zeros((Nmax,TT+1,ni,1))
#------------------------------------------
Duu = np.zeros((TT,ni))
sigma = np.zeros((TT,ni,1))
xx_opt = np.zeros((TT+1,ns,1))
uu_opt = np.zeros((TT,ni,1))
lambda_k = np.zeros((TT+1,ns,1))
nabla_J = np.zeros((TT,ni,1))  
#-----------------------------------------
AA = np.zeros((TT,ns,ns))
BB = np.zeros((ns,ni,TT))
QQ = np.zeros((TT,ns,ns))
RR = np.zeros((TT,ni,ni))
SS = np.zeros((TT,ni,ns))
qq = np.zeros((TT,ns,1))
rr = np.zeros((TT,ni,1))

AA2 = np.zeros((ns,ns,TT))
BB2 = np.zeros((ns,ni,TT))
QQ2 = np.zeros((ns,ns,TT))
RR2 = np.zeros((ni,ni,TT))
SS2 = np.zeros((ni,ns,TT))
qq2 = np.zeros((ns,TT))
rr2 = np.zeros((ni,TT))
  
# Initialization: the same starting point for all sequences
xxx[0,:,:,:] = xx_guess
uuu[0,:,:,:] = uu_guess

for k in range(0,Nmax):
    xxx[k,0,:,:] = x_init

####################################################################################
# Newton's algorithm

for k in range(0, Nmax-1):
  
    print("Iteration k =",k)
    JJ_cost[k,0,0] = cst_.fulltime_cost(x_init, uuu[k,:,:,:], xx_des, uu_des)
    print("J_cost(",k,") =",JJ_cost[k,0,0])
    

    # DESCEND DIRECTION CALCULATION ---------------------------------------------------

    # Solve backward the adjoint equation
    lambda_k[TT,:,:] = cst.QQT@(xxx[k,TT,:,:]- xx_des[TT,:,:])

    for t in reversed(range(TT)):
      dx_F,du_F = dyn.dynamic_leg(xxx[k,t,:,:], uuu[k,t,:,:])[1:]
     
      dx_l =cst.QQt@(xxx[k,t,:,:] - xx_des[t,:,:])
      du_l  = cst.RRt@( uuu[k,t,:,:] - uu_des[t,:,:])

      lambda_k[t,:,:] = dx_F.T@ lambda_k[t+1,:,:] + dx_l
      nabla_J[t,:,:]  = du_F@ lambda_k[t+1,:,:] + du_l
    # end of for cycle

    # ONLY IN NEWTON -----------------------------------------
    # Version 1
    
    for t in range(0,TT):
        dx_F,du_F = dyn.dynamic_leg(xxx[k,t,:,:], uuu[k,t,:,:])[1:]
        dx_l =cst.QQt@(xxx[k,t,:,:]- xx_des[t,:,:])
        du_l  = cst.RRt@( uuu[k,t,:,:]- uu_des[t,:,:])
        AA[t,:,:] = dx_F.T
        BB[:,:,t] = du_F.T
        QQ[t,:,:] = cst.QQt
        RR[t,:,:] = cst.RRt
        qq[t,:,:] = dx_l
        rr[t,:,:] = du_l
    QT = cst.QQT
    qT = cst.QQT@(xxx[k,TT,:,:] - xx_des[TT,:,:])

    KK, Duu, Dxx = aff.affineLQR(AA, BB, QQ, RR, SS, QT, x_init, qq, rr, qT,TT)

    # ONLY IN NEWTON END -----------------------------------------
    # Duu = K Dxx + sigma  -->  sigma = Duu - K Dxx
    for t in range(0,TT):
      sigma[t,:,:] = Duu[t,:,:] - KK[t,:,:]@Dxx[t,:,:]
    
    # Gradient method substitution
    # Duu = -nabla_J
    for t in range(0,TT):
      descent[k,0,0] += nabla_J[t,:,:].T @Duu[t,:,:]
      

    # STEPSIZE SELECTION ----------------------------------------------------------
    stepsize = 0.7
    cc = 0.5
    beta = 0.7

    # Armijo selection
    for i in range(0,armijo_maxiters):

      for t in range(0,TT):
        uu_temp[t,:,:] = uuu[k,t,:,:] + stepsize*Duu[t,:,:]

      JJ_temp = cst_.fulltime_cost(x_init, uu_temp, xx_des, uu_des)

      if(JJ_temp > (JJ_cost[k,0,0]+cc*stepsize*descent[k,0,0])):
        stepsize = beta*stepsize
      else:
        print('Armijo stepsize = {}'.format(stepsize))
        break
    print("J_down of =", JJ_cost[k,0,0]-JJ_temp)
  
    # Plot delle rette di Armijo
    print("stepsize =", stepsize)

    if visu_armijo:
      steps = np.linspace(0,stepsize,int(1e1)) #appartiene a [0;1]
      costs = np.zeros(len(steps))
      for(j) in range(len(steps)):
        for t in range(0,TT):
          uu_temp[t,:,:] = uuu[k,t,:,:] + Duu[t,:,:]*steps[j]
        costs[j] = cst_.fulltime_cost(x_init, uu_temp, xx_des, uu_des)
      # end for cycle

      plt.figure(1)
      plt.clf()
      plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k +stepsize*d^k)$')
      plt.plot(steps, JJ_cost[k,0,0]+descent[k,0,0]*steps, color='r', label='$J(\\mathbf{u}^k) + stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
      plt.plot(steps, JJ_cost[k,0,0]+cc*descent[k,0,0]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) + stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
      plt.scatter(stepsize, JJ_temp, marker='*') # plot the tested stepsize
      plt.grid()
      plt.xlabel('stepsize')
      plt.legend()
      plt.draw()
      plt.show()

     # TRAJECTORY UPDATE ------------------------------------------------------
    close_loop = True
    if close_loop:
      for t in range(0,TT):
        Duu[t,:,:] = KK[t,:,:]@(xxx[k+1,t,:,:]-xxx[k,t,:,:]) + sigma[t,:,:]
        uuu[k+1,t,:,:] = uuu[k,t,:,:] + stepsize*Duu[t,:,:]
        xxx[k+1,t+1,:,:] = cst_.full_state_dyn(xxx[k+1,t,:,:], uuu[k+1,t,:,:])
    else:
      for t in range(0,TT):
        uuu[k+1,t,:,:] = uuu[k,t,:,:] + stepsize*Duu[t,:,:]
    
    #xxx[k+1,:,:,:] = cst_.fulltime_dyn(x_init, uuu[k+1,:,:,:])
    #end of trajectory update--------------------------------------------------
  
    last_iter = k
    # Stopping condition 
    if (k>10)and(abs(JJ_cost[k,0,0]-JJ_cost[k-1,0,0])<Stop_too_short_Jupgrade):
      break
    

    # End of Newton's algorithm
    #####################################################################################
    # Package and deliver
    xx_opt = xxx[last_iter,:,:,:]
    uu_opt = uuu[last_iter,:,:,:]

#####################################################################################
# Here we test the system

x_init = xx_des[0,:,:]
u_init = uu_des[0,:,:]

# Extract system dimensions
ns = x_init.shape[0]
ni = u_init.shape[0]

# We want to compute gradient J di uu_opt
# nabla_J[t,:,:] = du_F@ lambda_k[t+1,:,:] +du_l

lambda_k1 = np.zeros((TT+1,ns,1))
nabla_J1 = np.zeros((TT,ni,1))
norm_nabla_ut_J1 = np.zeros((TT,1,1))

# Solve backward the adjoint equation
lambda_k1[TT,:,:] = cst.QQT@(xx_opt[TT,:,:]-xx_des[TT,:,:])
    
for t in reversed(range(TT)):
    
    dx_F,du_F = dyn.dynamic_leg(xx_opt[t,:,:], uu_opt[t,:,:])[1:]
    dx_l =cst.QQt@(xx_opt[t,:,:] - xx_des[t,:,:])
    du_l  = cst.RRt@( uu_opt[t,:,:] - uu_des[t,:,:])

    lambda_k1[t,:,:] = dx_F@ lambda_k1[t+1,:,:] +dx_l
    nabla_J1[t,:,:] = du_F@ lambda_k1[t+1,:,:] +du_l
    # end of for cycle
        
# Norm of gradient J with respect to ut
norm_all_J = 0
for t in range(0,TT):
    norm_nabla_ut_J1[t,:,:] = nabla_J1[t,:,:].T @ nabla_J1[t,:,:]
    norm_all_J = norm_all_J + norm_nabla_ut_J1[t,:,:]

print("Norm of nabla_J(uu_opt) =", norm_all_J)
#####################################################################################
plt.figure('Cost J(k) evolution')
plt.semilogy(np.arange(JJ_cost.shape[0]), JJ_cost[:,0,0])
plt.xlabel('k')
plt.ylabel('logJ(k)')
plt.grid()
plt.show(block=True)

plt.figure('Descend norm evolution')
plt.semilogy(np.arange(descent.shape[0]), -descent[:,0,0])
plt.xlabel('k')
plt.ylabel('log||d_k||^2')
plt.grid()
plt.show(block=True)

plt.figure('Optimal theta_st(t) angle')
plt.legend('theta_sw_opt','theta_sw_des')
plt.plot(np.arange(TT+1), xx_opt[:,0,0])
plt.plot(np.arange(TT+1), xx_des[:,0,0],'--')
plt.xlabel('t [ms]')
plt.ylabel('theta_sw [rad]')
plt.grid()
plt.show(block=True)

plt.figure('Optimal theta_sw(t) angle')
plt.legend('theta_st_opt','theta_st_des')
plt.plot(np.arange(TT+1), xx_opt[:,1,0])
plt.plot(np.arange(TT+1), xx_des[:,1,0],'--')
plt.xlabel('t [ms]')
plt.ylabel('theta_st[rad]')
plt.grid()
plt.show(block=True)

plt.figure(' Optimal theta_st speed')
plt.legend('theta_swdot_opt','theta_swdot_des')
plt.plot(np.arange(TT+1), xx_opt[:,2,0])
plt.plot(np.arange(TT+1), xx_des[:,2,0],'--')
plt.xlabel('t [ms]')
plt.ylabel('dot theta_sw[rad/s]')
plt.grid()
plt.show(block=True)

plt.figure('Optimal theta_sw speed')
plt.legend('theta_stdot_opt','theta_stdot_des')
plt.plot(np.arange(TT+1), xx_opt[:,3,0])
plt.plot(np.arange(TT+1), xx_des[:,3,0],'--')
plt.xlabel('t [ms]')
plt.ylabel('dot theta_st[rad/s/s]')
plt.grid()

########################
plt.show(block=True)
plt.figure('Optimal tau(t) torque')
plt.plot(np.arange(TT+1), uu_opt[:,0,0])
plt.plot(np.arange(TT+1), uu_des[:,0,0],'--')
plt.xlabel('t [ms]')
plt.ylabel('tau_t [Nm]')
plt.grid()
plt.show(block=True)


  
    
   
   


  