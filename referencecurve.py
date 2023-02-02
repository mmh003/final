
import numpy as np
# import compass dynamics
import dynamic_leg as dyn
dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni


tf = 20
T = int(tf/dt)
def desired_step(theta_st,theta_sw):
    """
    ARGS
       
    RETURNS
        - xx_des in desired state behaviour time sequence [from t=0 to t=T-1]
        - uu_des  desired input time sequence [from t=0 to t=T-1]
    """
    # Define empty structure

    xx_des = np.zeros((T+1,ns,1))
    uu_des = np.zeros((T+1,ni,1))
    thets_sto= np.deg2rad(theta_st )
    thets_swo= np.deg2rad(theta_sw )

    # RAMP keeping angle of attack fixed
    for t in range(0,T+1):
        xx_des[t,0,0] = thets_swo
        xx_des[t,1,0] = thets_sto 
        xx_des[t,2,0] = 0
        xx_des[t,3,0] = 0
        uu_des[t,0,0] =  -(-(s2*s3*np.sin(thets_swo))+s1*s4*np.cos(thets_sto -
                        thets_swo)*np.sin(thets_sto ))/(-s2+ s1*np.cos(thets_sto -thets_swo))

  # Package and deliver
    return xx_des, uu_des[:T,:,:]

# def f_desired_jump(z0, z1, type):
#   """
#     ARGS
#       - z0 in R scalar fixed low altitude (z0<z1)
#       - z1 in R scalar fixed high altitude (z0<z1)
#       - type integer design choice parameter
#     RETURNS
#       - xx_des in (T+1)x(R^6x1) desired state behaviour time sequence [from t=0 to t=T]
#       - uu_des Tx(R^2x1) desired input time sequence [from t=0 to t=T-1]
#   """
#   # Define empty structure
#   xx_des = np.zeros((T+1,6,1))
#   uu_des = np.zeros((T+1,2,1))

#   # STEPWISE
#   if (type==1):
#     for t in range(0,T+1):
#       xx_des[t,0,0] = dt*V_eq*t
#       if(t<int(T/4)):
#         xx_des[t,1,0] = z0
#       else:
#         xx_des[t,1,0] = z1
#       xx_des[t,2,0] = alpha_eq
#       xx_des[t,3,0] = V_eq
#       xx_des[t,4,0] = 0
#       xx_des[t,5,0] = 0
#       uu_des[t,0,0] = Th_eq
#       uu_des[t,1,0] = 0
     
#   # SMOOTH TRY
#   if(type==2):
#     # We compute the Sigmoid in z and its derivatives and we give that to the desired angles
#     zz_sig, dz_sig = ff3.f_Sigmoid(z0, z1)[:2]
  
#     for t in range(0,T):
#       xx_des[t,0,0] = dt*V_eq*t
#       xx_des[t,1,0] = zz_sig[t,0,0]
#       xx_des[t,2,0] = 0.1*dz_sig[t,0,0] +alpha_eq
#       xx_des[t,3,0] = V_eq*(1-0.1*dz_sig[t,0,0])
#       xx_des[t,4,0] = 0.1*dz_sig[t,0,0]
#       xx_des[t,5,0] = 0
#       uu_des[t,0,0] = Th_eq
#       uu_des[t,1,0] = 0
#     # end of for cycle
#     xx_des[T,:,:] = xx_des[T-1,:,:]


#   if(type==21):
#     # We compute the Sigmoid in z, then we compute some angles trying to achieve that
#     # We consider a simpler model keeping thrust and speed fixed
#     zz_sig, dz_sig = ff3.f_Sigmoid(z0, z1)[:2]
#     V = V_eq
#     gaga = np.zeros(T+1)
#     for t in range(0,T):
#       gaga[t] = 0.01*(dz_sig[t,0,0]/(dt*V_eq))
#     gaga[T] = gaga[T-1]
#     thth = np.zeros(T+1)
#     for t in range(0,T+1):
#       thth[t] = gaga[t]+alpha_eq

#     theta_feas, q_feas, M_feas = f_pitch_solver(thth)
#     for t in range(0,T):
#       xx_des[t,0,0] = dt*(V_eq*cos(gaga[t]))*t
#       xx_des[t,1,0] = zz_sig[t]
#       xx_des[t,2,0] = gaga[t]+alpha_eq
#       xx_des[t,3,0] = V_eq
#       xx_des[t,4,0] = gaga[t]
#       xx_des[t,5,0] = q_feas[t]
#       uu_des[t,0,0] = Th_eq
#       uu_des[t,1,0] = M_feas[t]
#     # end of for cycle
#     xx_des[T,:,:] = xx_des[T-1,:,:]


#   if(type==22):
#     # We compute the Sigmoid in z, then we compute a guess trajectory achieving that
#     # We consider a simpler model keeping thrust and speed fixed
#     # zz_sig, dz_sig, ddz_sig = ff3.f_Sigmoid(z0, z1)[:3]
#     zz_feas, theta_feas, ga_feas, q_feas, M_feas = f_guess_traj(z0, z1)[1:]
  
#     for t in range(0,T):
#       xx_des[t,0,0] = dt*(V_eq*cos(gaga[t]))*t
#       xx_des[t,1,0] = zz_feas[t]
#       xx_des[t,2,0] = theta_feas[t]
#       xx_des[t,3,0] = V_eq
#       xx_des[t,4,0] = ga_feas[t]
#       xx_des[t,5,0] = q_feas[t]
#       uu_des[t,0,0] = Th_eq
#       uu_des[t,1,0] = M_feas[t]
#     # end of for cycle
#     xx_des[T,:,:] = xx_des[T-1,:,:]

#   # Package and deliver
#   return xx_des, uu_des[:T,:,:]


# # ref_theta_sw_0 = 0 # in gradi
# # ref_theta_st_0 = 0

# # ref_theta_sw_T = 45 # in gradi
# # ref_theta_st_T = 45

# # T_eq = dyn.T_eq

# # x_ref = np.zeros((ns, DT))
# # u_ref = np.zeros((ni, DT))
# # x_ref[0]=np.deg2rad(ref_theta_sw_T )
# # x_ref[1]=np.deg2rad(0)
# # x_ref[2]=np.deg2rad(ref_theta_st_T )
# # x_ref[3]=np.deg2rad(0)

# # #x_ref[0,:int(DT/2)] = np.ones((1,int(DT/2)))*np.deg2rad(ref_theta_st_T)
# # u_ref[0,:] = T_eq*np.sin(x_ref[0,:])
# # print("x_ref[:,t]",x_ref)
# # print("u_ref[:,t]",u_ref)
# # print(" xref", x_ref.shape)
# # print(" xrefff", x_ref[0,int(DT/2):].shape)
# # print(" uref", u_ref.shape)

# # x0 = x_ref[:,0]
# # print(" xo", x0)
# # print(" xo", x0.shape)