import numpy as np
from scipy.linalg import logm

def logmap(COV, type):
    """
    Logarithmic mapping on centralized signal covariance matrices
    
    Input:
        COV: K*K*N, centralized signal covariance matrices
        type: str, type of features to extract ('ERP' or 'MI')
    
    Output:
        Fea: tangent space features, d*N

    """
    NTrial = COV.shape[2]
    N_elec = COV.shape[0]
    
    if type == 'ERP':
        # Select upper right elements related to temporal information
        N = int(N_elec/2)
        Fea = np.zeros((N*N, NTrial))
        for i in range(NTrial):
            Tn = np.real(np.logm(COV[:,:,i]))
            Fea[:,i] = np.reshape(Tn[0:N, N:N_elec], -1, order='F')
    
    elif type == 'MI':
        
        # Select upper triangular elements related to spatial information
        Fea = np.zeros((int(N_elec*(N_elec+1)/2), NTrial))
        index = np.ravel(np.triu(np.ones((N_elec, N_elec)), k=1)) == 0
        for i in range(NTrial):
            Tn = logm(COV[:, :, i])
            tmp = np.reshape(np.sqrt(2)*np.triu(Tn, k=1) + np.diag(np.diag(Tn)), -1, order='F')
            Fea[:,i] = tmp[index]
    
    return Fea
