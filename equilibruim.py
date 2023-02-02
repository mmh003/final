
import numpy as np
from scipy.optimize import fsolve

g = 9.8    # gravity 
dt = 1e-3
a = 0.7    # distance from center of mass to hip in meter
b = 0.5   # distance center mass of mass to ground in meter
l = a + b  # the total leg lenght
m_h = 2    # the hip mass in kg
m = 0.2    # leg mass in center mass

s = m*b**2
s1 = m*l*b
s2 = (m_h+m)*l**2 + m*a*a
s3 = m*g*b
s4= (m*(a+l) + m_h*l)*g

"""
        
    xx_plus= np.zeros((4,1))

    xx_plus[0] = x1 + dt * x3
    xx_plus[1] = x2 + dt * x4
    xx_plus[2] = x3+ dt * (((s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x3**2 +s2*s1*np.sin(x2-x1)*x4**2)-
            (-(s2*s3*np.sin(x1))+s1*s4*np.cos(x2-x1)*np.sin(x2))+u*(-s2+s1*np.cos(x2-x1)))/den
            
    xx_plus[3] = x4 + dt * ((s*s1*np.sin(x2-x1)*x3**2+(s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x4**2) -
            (-(s1*s3*np.cos(x2-x1)*np.sin(x1)) + s*s4*np.sin(x2)) +u*(s - s1*np.cos(x2-x1)))/den

    x3 = 0
    x4 = 0
    u1 = -(((s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x3**2 +s2*s1*np.sin(x2-x1)*x4**2)-
        (-(s2*s3*np.sin(x1))+s1*s4*np.cos(x2-x1)*np.sin(x2)))/(-s2+s1*np.cos(x2-x1)) 

    u2 = -((s*s1*np.sin(x2-x1)*x3**2+(s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x4**2) -       
        (-(s1*s3*np.cos(x2-x1)*np.sin(x1)) + s*s4*np.sin(x2))) / (s - s1*np.cos(x2-x1))

    
    0 = -(-(s2*s3*np.sin(x1))+s1*s4*np.cos(x2-x1)*np.sin(x2))+u*(-s2+s1*np.cos(x2-x1)))/den
    0 = -(-(s1*s3*np.cos(x2-x1)*np.sin(x1)) + s*s4*np.sin(x2)) +u*(s - s1*np.cos(x2-x1)))/den

    print('Iter {}\tu1 = {}\tu2 = {}'.format(u1, u2))

"""
"""
    w1 = -(-(s2*s3*np.sin(x1))+s1*s4*np.cos(x2-x1)*np.sin(x2))+ u*(-s2+s1*np.cos(x2-x1))
    w2 = -(-(s1*s3*np.cos(x2-x1)*np.sin(x1)) + s*s4*np.sin(x2)) + u*(s - s1*np.cos(x2-x1))

    we have two eqation in three unknow varaibles
    we can fix one and try the two orthers
"""

# Here we fixe the two theta position
def equations_0(x1 ,x2):
   
    u1 = ((s2*s3*np.sin(x1)) - s1*s4*np.cos(x2-x1)*np.sin(x2)) / (-s2 + s1*np.cos(x2-x1)) 
    u2 = ((s1*s3*np.cos(x2-x1)*np.sin(x1)) - s*s4*np.sin(x2)) / (s - s1*np.cos(x2-x1))
    return u1, u2

# here we fixe only one angola position
def equations_1(variables,x1) :
    (x2, u) = variables
    eqn_1 = (s2*s3*np.sin(x1)) - s1*s4*np.cos(x2-x1)*np.sin(x2) + u*(-s2 + s1*np.cos(x2-x1)) 
    eqn_2 = (s1*s3*np.cos(x2-x1)*np.sin(x1)) - s*s4*np.sin(x2) + u*(s - s1*np.cos(x2-x1))
    
    return[eqn_1, eqn_2]

# we find the coordinates of first equilibruim point 
# for this we fixe theta_st = 0 

def equilibruim(theta):
    """
        Args
        - theta one position of the leg

        Return 
        - the total vector position
        - input u
  
    """

    x , u = fsolve(equations_1,(0, 0),args=(theta,))
    
    E = np.array([theta,x,0,0])
    return E.T, u

tf = 10
T = int(tf/dt)
ns = 4
ni = 1

def desired_step(theta_sw_start,theta_sw_end):
    """
    ARGS
       one of the two angles at the begining and end of trajectory
    RETURNS
        - xx_des in desired state behaviour time sequence [from t=0 to t=T-1]
        - uu_des  desired input time sequence [from t=0 to t=T-1]
    """
    # Define empty structure

    xx_des = np.zeros((T+1,ns,1))
    uu_des = np.zeros((T+1,ni,1))
    
    theta_sw_0 = np.deg2rad(theta_sw_start)
    theta_sw_1 = np.deg2rad(theta_sw_end)

    for t in range(0,T+1):

      if(t<int(T/2)):

        xx_des[t,:,0], uu_des[t,0,0] = equilibruim(theta_sw_0)
      
      else:
        xx_des[t,:,0], uu_des[t,0,0] = equilibruim(theta_sw_1)

    return xx_des, uu_des




print("first equilibruim point\n")
theta_st = 0
x2_1, u_1 = fsolve(equations_1,(0, 0),args=(theta_st,))
print('theta_sw 1 = {}\tinput u1 = {}'.format(x2_1, u_1))

u1_1, u2_1 = equations_0(theta_st,x2_1)
print('u1 = {}\tu2 = {}'.format(u1_1, u2_1))

E1 = np.array([theta_st,x2_1,0,0])
print("\n the first eql point is:\n ",E1.T)
print(" the coresponding input is : ",u_1)

# now we find the second equilibrum point
# we fixe theta_st = 20 degreed
print("second equilibruim point\n")
theta_st = np.deg2rad(0.7)
x2_2, u_2 = fsolve(equations_1,(0, 0),args=(theta_st,))
print('theta_sw 2 = {}\tinput u2 = {}\t theta_sw_deg = {}'.format(x2_2, u_2,np.rad2deg(x2_2)))

u1_2, u2_2 = equations_0(theta_st,x2_2)
print('u1 = {}\tu2 = {}'.format(u1_2, u2_2))

E2 = np.array([theta_st,x2_2,0,0])
print("\n the first eql point is:\n ",E2.T)
print(" the coresponding input is : ",u_2)

a , b = equilibruim(np.deg2rad(20))
print('position = {}\tu = {}'.format(a, b))