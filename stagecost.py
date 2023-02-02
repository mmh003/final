#
# Gradient method for Optimal Control
# Cost functions
# 
#
#

import numpy as np

import dynamic_leg as dyn

tf = 10 # final time in seconds
dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni
TT = int(tf/dt) 

QQt = np.array([[1e2, 0, 0, 0],
                [0, 1e2, 0, 0],
                [0, 0, 1e0, 0], 
                [0, 0, 0, 1e0]])
RRt = 1e0*np.eye(ni)

QQT = 2*QQt
# # State stage cost weights
# w_x1 = 0.00001
# w_x2 = 1
# w_x3 = 0.01
# w_x4 = 0.01
# w_x5 = 0.01
# w_x6 = 0.001
# # Input stage cost weights
# w_u1 = 0.001
# w_u2 = 0.001
# # State terminal cost weights
# w_T1 = 0.00001
# w_T2 = 1
# w_T3 = 0.01
# w_T4 = 0.01
# w_T5 = 0.01
# w_T6 = 0.001
# # Diagonal cost matrices


# Q = np.diag([w_x1, w_x2, w_x3, w_x4, w_x5, w_x6])
# R = np.diag([w_u1, w_u2])
# QT = np.diag([w_T1, w_T2, w_T3, w_T4, w_T5, w_T6])
# ######################################
# # Reference curve
# ######################################

# 
# 
# xx_ref_T = np.zeros((ns,))
# uu_ref_T = np.zeros((ni,))


# fx,fu = dyn.dynamics(xx_ref_T,uu_ref_T)[1:]

# AA = fx.T; BB = fu.T




def stagcost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  # Setup empty structures
  x = np.zeros((6,1))
  x_des = np.zeros((6,1))
  u = np.zeros((2,1))
  u_des = np.zeros((2,1))
  J_temp = np.zeros((TT,1,1))
  cost_J = 0


  # Computation of stage cost
  for t in range(TT-1):
    x = xx[t,:,:]
    x_des = xx_ref[t,:,:]
    u = uu[t,:,:]
    u_des = uu_ref[t,:,:]

    J_temp = (0.5)*(x-x_des).T@QQt@(x-x_des) + (0.5)*(u-u_des).T@RRt@(u-u_des)
    cost_J = cost_J + J_temp
  # end of for cycle
  print("TT",TT)
  # Computation of terminal cost
  J_temp = (0.5)*(xx[-1,:,:]-xx_ref[-1,:,:]).T@QQT@(xx[-1,:,:]-xx_ref[-1,:,:])
  cost_J = cost_J + J_temp
  #derivative 
  # lx = QQt@(xx - xx_ref)
  # lu = RRt@(uu - uu_ref)
  # lTx = QQT@(xx - xx_ref)
  # Package and deliver

  return cost_J



def fulltime_cost(xx,uu, xx_ref, uu_ref,):
    """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref) + 1/2 (x - x_ref)^T QT (x - x_ref) 

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu,xxT
  
  """

    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]
    ll= 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref) +  0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)
    return ll

    