import numpy as np
from scipy.optimize import fsolve

#Initialization
ns = 4 # numbers of state variables
ni = 1 # number of input
T_eq = 10

#PARAMETERS
g = 9.8    # gravity 
dt = 1e-3
b = 0.7    # distance from center of mass to hip in meter
a = 0.5   # distance center mass of mass to ground in meter
l = a + b  # the total leg lenght
m_h = 2    # the hip mass in kg
m = 0.2    # leg mass in center mass
    
s = m*b**2
s1 = m*l*b
s2 = (m_h+m)*l**2 + m*a*a
s3 = m*g*b
s4= (m*(a+l) + m_h*l)*g



def dynamic_leg(q, uu):
    
    # Dynamics of a discrete-time Compass gait
    """
    function dynamic_leg
    Args
      - x \in \R^4 state at time t
      - u \in \R^1 input at time t

    Return 
      - next state x_i_{t+1}
      - gradient of f wrt x,
      - gradient of f wrt u, 
  
  """

    # inputs data 
    theta_sw = q[0]   
    theta_st = q[1]
    theta_sw_dot = q[2]   
    theta_st_dot = q[3]
    u=uu[0]

    
    ################## discretization step ###################################
    x1 = theta_sw
    x2= theta_st
    x3 = theta_sw_dot
    x4 = theta_st_dot

    den = (s*s2 - s1**2*np.cos(x1-x2)**2)


    xx_plus= np.zeros((4,1))

    xx_plus[0] = x1 + dt * x3

    xx_plus[1] = x2 + dt * x4

    xx_plus[2] = x3+ dt * (-((s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x3**2 + s2*s1*np.sin(x2-x1)*x4**2) +
            (-(s2*s3*np.sin(x1) - s1*s4*np.cos(x2-x1)*np.sin(x2))) + u*(s2 - s1*np.cos(x2-x1)))/den

    xx_plus[3] = x4 + dt * (-(s*s1*np.sin(x2-x1)*x3**2 + (s1**2)*np.cos(x2-x1)*np.sin(x2-x1)*x4**2) +
            (-(s1*s3*np.cos(x2-x1)*np.sin(x1) - s*s4*np.sin(x2))) + u*(-s + s1*np.cos(x2-x1)))/den

    df1_dx1= 1
    df1_dx2= 0
    df1_dx3= dt
    df1_dx4= 0

    df2_dx1= 0
    df2_dx2= 1
    df2_dx3= 0
    df2_dx4= dt

    df3_dx1= -2*b**2*dt*l**2*m**2*(b**2*l**2*m**2*x3**2*np.sin(x1 - x2)*np.cos(x1 - x2) +
        b*g*l*m*(l*m_h + m*(a + l))*np.sin(x2)*np.cos(x1 - x2) - b*g*m*(a**2*m + l**2*(m + m_h))*np.sin(x1) + 
        b*l*m*x4**2*(a**2*m + l**2*(m + m_h))*np.sin(x1 - x2) + u*(a**2*m - b*l*m*np.cos(x1 - x2) + 
        l**2*(m + m_h)))*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h))
        )**2 + dt*(-b**2*l**2*m**2*x3**2*np.sin(x1 - x2)**2 + b**2*l**2*m**2*x3**2*np.cos(x1 - x2)**2 - 
        b*g*l*m*(l*m_h + m*(a + l))*np.sin(x2)*np.sin(x1 - x2) - b*g*m*(a**2*m + l**2*(m + m_h))*np.cos(x1) + b*l*m*u*np.sin(x1 - x2) + 
        b*l*m*x4**2*(a**2*m + l**2*(m + m_h))*np.cos(x1 - x2))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))

    df3_dx2= 2*b**2*dt*l**2*m**2*(b**2*l**2*m**2*x3**2*np.sin(x1 - x2)*np.cos(x1 - x2) + 
        b*g*l*m*(l*m_h + m*(a + l))*np.sin(x2)*np.cos(x1 - x2) - b*g*m*(a**2*m + l**2*(m + m_h))*np.sin(x1) + 
        b*l*m*x4**2*(a**2*m + l**2*(m + m_h))*np.sin(x1 - x2) + u*(a**2*m - b*l*m*np.cos(x1 - x2) + l**2*(m + m_h))
        )*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))**2 + dt*(
        b**2*l**2*m**2*x3**2*np.sin(x1 - x2)**2 - b**2*l**2*m**2*x3**2*np.cos(x1 - x2)**2 + b*g*l*m*(l*m_h + m*(a + l)
        )*np.sin(x2)*np.sin(x1 - x2) + b*g*l*m*(l*m_h + m*(a + l))*np.cos(x2)*np.cos(x1 - x2) - b*l*m*u*np.sin(x1 - x2) - 
        b*l*m*x4**2*(a**2*m + l**2*(m + m_h))*np.cos(x1 - x2))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))

    df3_dx3= 2*b**2*dt*l**2*m**2*x3*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + 
        b**2*m*(a**2*m + l**2*(m + m_h))) + 1

    df3_dx4= 2*b*dt*l*m*x4*(a**2*m + l**2*(m + m_h))*np.sin(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + 
        b**2*m*(a**2*m + l**2*(m + m_h)))

    df4_dx1= -2*b**2*dt*l**2*m**2*(b**3*l*m**2*x3**2*np.sin(x1 - x2) - b**2*g*l*m**2*np.sin(x1)*np.cos(x1 - x2) + 
        b**2*g*m*(l*m_h + m*(a + l))*np.sin(x2) + b**2*l**2*m**2*x4**2*np.sin(x1 - x2)*np.cos(x1 - x2) + 
        u*(-b**2*m + b*l*m*np.cos(x1 - x2)))*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + 
        b**2*m*(a**2*m + l**2*(m + m_h)))**2 + dt*(b**3*l*m**2*x3**2*np.cos(x1 - x2) + b**2*g*l*m**2*np.sin(x1)*np.sin(x1 - x2) - 
        b**2*g*l*m**2*np.cos(x1)*np.cos(x1 - x2) - b**2*l**2*m**2*x4**2*np.sin(x1 - x2)**2 + b**2*l**2*m**2*x4**2*np.cos(x1 - x2)**2 - 
        b*l*m*u*np.sin(x1 - x2))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))

    df4_dx2= 2*b**2*dt*l**2*m**2*(b**3*l*m**2*x3**2*np.sin(x1 - x2) - b**2*g*l*m**2*np.sin(x1)*np.cos(x1 - x2) + 
        b**2*g*m*(l*m_h + m*(a + l))*np.sin(x2) + b**2*l**2*m**2*x4**2*np.sin(x1 - x2)*np.cos(x1 - x2) + 
        u*(-b**2*m + b*l*m*np.cos(x1 - x2)))*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 +
        b**2*m*(a**2*m + l**2*(m + m_h)))**2 + dt*(-b**3*l*m**2*x3**2*np.cos(x1 - x2) - b**2*g*l*m**2*np.sin(x1)*np.sin(x1 - x2) +
        b**2*g*m*(l*m_h + m*(a + l))*np.cos(x2) + b**2*l**2*m**2*x4**2*np.sin(x1 - x2)**2 - b**2*l**2*m**2*x4**2*np.cos(x1 - x2)**2 +
        b*l*m*u*np.sin(x1 - x2))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))

    df4_dx3= 2*b**3*dt*l*m**2*x3*np.sin(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + 
        b**2*m*(a**2*m + l**2*(m + m_h)))

    df4_dx4= 2*b**2*dt*l**2*m**2*x4*np.sin(x1 - x2)*np.cos(x1 - x2)/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + 
        b**2*m*(a**2*m + l**2*(m + m_h))) + 1


    df1_du= 0
    df2_du= 0

    df3_du= dt*(a**2*m - b*l*m*np.cos(x1 - x2) + l**2*(m + m_h))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 +
        b**2*m*(a**2*m + l**2*(m + m_h)))
 
    df4_du= dt*(-b**2*m + b*l*m*np.cos(x1 - x2))/(-b**2*l**2*m**2*np.cos(x1 - x2)**2 + b**2*m*(a**2*m + l**2*(m + m_h)))

    A=np.array([[df1_dx1,df2_dx1,df3_dx1,df4_dx1],
                [df1_dx2,df2_dx2,df3_dx2,df4_dx2],
                [df1_dx3,df2_dx3,df3_dx3,df4_dx3],
                [df1_dx4,df2_dx4,df3_dx4,df4_dx4]],dtype=np.float32)

    # gradient of f wrt u == matrix B
    
    B =np.array([[df1_du,df2_du,df3_du,df4_du]],dtype=np.float32)
    # the dynamics is given by x_plus 
   

    return xx_plus,A,B

# xplus, A, B = dynamic_leg(q, uu)
# print("xplu",xplus)
# print("A",A)
# print("B",B)
tf = 10
T = int(tf/dt)

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
def equilibruim(theta_st):
    """
        Args
        - theta one position of the leg

        Return 
        - the total vector position
        - input u
  
    """

    theta_sw, u = fsolve(equations_1,(0, 0),args=(theta_st,))
    
    E = np.array([theta_sw,theta_st,0,0])
    return E.T, u

def desired_step(theta_sw_start,theta_sw_end):
    """
    ARGS
       
    RETURNS
        - xx_des in desired state behaviour time sequence [from t=0 to t=T-1]
        - uu_des  desired input time sequence [from t=0 to t=T-1]
    """
    # Define empty structure

    xx_des = np.zeros((T+1,ns,1))
    uu_des = np.zeros((T+1,ni,1))
    
    theta_sw_0 = np.deg2rad(theta_sw_start)
    theta_sw_1 = np.deg2rad(theta_sw_end)

    # RAMP keeping angle of attack fixed
    for t in range(0,T+1):

      if(t<int(T/2)):

        xx_des[t,:,0], uu_des[t,0,0] = equilibruim(theta_sw_0)
      
      else:
        xx_des[t,:,0], uu_des[t,0,0] = equilibruim(theta_sw_1)

    return xx_des, uu_des
