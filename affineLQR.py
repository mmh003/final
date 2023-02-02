import numpy as np

def affineLQR(AA, BB, QQ, RR, SS, QT, x0, qq, rr, qT,T):
  """
  Our version of prof's solver
  ARGS:
    - AA in Tx(R^ns x ns) matrix time sequence [from t=0 to t=T-1]
    - BB in (R^nsxni)xT matrix time sequence [from t=0 to t=T-1] questa la vuole così 
    - QQ Tx(R^ns x ns) matrix time sequence [from t=0 to t=T-1] 
    - RR Tx(R^ni x ni) matrix time sequence [from t=0 to t=T-1]
    - SS Tx(R^ni x ns) matrix time sequence [from t=0 to t=T-1]
    - QT (R^ns x ns) matrix
    - qq Tx(R^nsx1) affine term time sequence [from t=0 to t=T-1]
    - rr Tx(R^nix1) affine term time sequence [from t=0 to t=T-1]
    - qT Tx(R^nsx1) affine term 
  RETURNS:
    - KK Tx(R^nixns) optimal gain time sequence [from t=0 to t=T-1]
    - delta_u Tx(R^nix1) optimal descend direction time sequence [from t=0 to t=T-1]
  """
	
  # Extract number of states and inputs
  ns = AA.shape[1]
  ni = BB.shape[1]
  
  # System empty solution
  xx = np.zeros((T,ns+1,1))
  uu = np.zeros((T,ni,1))
  # Augmented state
  xx[:,0,:].fill(1)
  xx[0,1:,:] = x0

  # Augmented empty structures
  KK_a = np.zeros((T,ni,ns+1))
  PP_a = np.zeros((T,ns+1,ns+1))
  QQ_a = np.zeros((T,ns+1,ns+1))
  QT_a = np.zeros((ns+1,ns+1))
  SS_a = np.zeros((T,ni,ns+1))
  RR_a = np.zeros((T,ni,ni))  # Must be positive definite
  AA_a = np.zeros((T,ns+1,ns+1))
  BB_a = np.zeros((T,ns+1,ni))

  # Fill augmented terms
  for t in range(T):
    QQ_a[t,1:,0] = 0.5*qq[t,:,0]
    QQ_a[t,0,1:] = 0.5*qq[t,:,0].T
    QQ_a[t,1:,1:] = QQ[t,:,:]
    RR_a[t,:,:] = RR[t,:,:]
    SS_a[t,:,:] = 0.5*rr[t,:,:]
    SS_a[t,:,1:] = SS[t,:,:]
    AA_a[t,0,0] = 1
    AA_a[t,1:,1:] = AA[t,:,:]
    BB_a[t,1:,:] = BB[:,:,t]

  QT_a[1:,:] = 0.5*qT
  QT_a[:,1:] = 0.5*qT.T
  QT_a[1:,1:] = QT

  # Solve Riccati equation backward in time
  PP_a[-1,:,:] = QT_a
  for t in reversed(range(T-1)):
    QQt = QQ_a[t,:,:]
    RRt = RR_a[t,:,:]
    AAt = AA_a[t,:,:]
    BBt = BB_a[t,:,:]
    SSt = SS_a[t,:,:]
    PPtp = PP_a[t+1,:,:]
    PP_a[t,:,:] = QQt + AAt.T@PPtp@AAt - \
        + (BBt.T@PPtp@AAt + SSt).T@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt + SSt)
  
  # print("PP_a =",PP_a[100,:,:])

  # Forward computation of gain KK sequence
  for t in range(T-1):
    QQt = QQ_a[t,:,:]
    RRt = RR_a[t,:,:]
    AAt = AA_a[t,:,:]
    BBt = BB_a[t,:,:]
    SSt = SS_a[t,:,:]
    PPtp = PP_a[t+1,:,:]
    # Check positive definiteness
    MM = RRt + BBt.T@PPtp@BBt
    if not np.all(np.linalg.eigvals(MM) > 0):
      MM += 0.5*np.eye(ni)
    KK_a[t,:,:] = -np.linalg.inv(MM)@(BBt.T@PPtp@AAt + SSt)
  

  # Compute the optimal trajectory
  for t in range(T-1):
    # Trajectory
    uu[t,:,:] = KK_a[t,:,:]@xx[t,:,:]
    xx[t+1,:,:] = AA_a[t,:,:]@xx[t,:,:] + BB_a[t,:,:]@uu[t,:,:]


  # Package and deliver
  KK = KK_a[:,:,1:]
  return KK, uu, xx[:,1:,:]