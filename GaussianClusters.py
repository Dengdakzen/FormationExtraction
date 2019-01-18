import numpy as np
import json
from scipy.optimize import linear_sum_assignment

def GaussianCluster(input_pos,iteration = 50,threshold = 10):
    '''
    input_pos: numpy array of shape (player number,dimension,time) 
    '''
    #Initialize the GMM model
    init_avg_pos_frame = np.average(input_pos,axis = 0)
    normalized_pos = input_pos - init_avg_pos_frame[np.newaxis,:,:]
    distribution = np.zeros([normalized_pos.shape[0],normalized_pos.shape[1] + 3])
    distribution[:,0:2] = normalized_pos[:,:,0]
    distribution[:,2] = 1
    distribution[:,3] = 1
    distribution[:,4] = 0
    last_indices = np.zeros([normalized_pos.shape[0],normalized_pos.shape[2]])
    new_pos = np.zeros_like(normalized_pos)
    new_pos[:,:,0] = normalized_pos[:,:,0].copy()
    indices = np.zeros([normalized_pos.shape[0],normalized_pos.shape[2]])
    indices[:,0] = np.array(range(normalized_pos.shape[0]))
    for i in range(iteration):
        print(i)
        if i == 0:
            start = 1
        else:
            start = 0
        for k in range(start,normalized_pos.shape[2]):
            x = normalized_pos[:,0,k].copy()[np.newaxis,:]
            y = normalized_pos[:,1,k].copy()[np.newaxis,:]
            miu1 = distribution[:,0].copy()[:,np.newaxis]
            miu2 = distribution[:,1].copy()[:,np.newaxis]
            rho = distribution[:,4].copy()[:,np.newaxis]
            c1 = 1/(1 - distribution[:,4]*distribution[:,4])[:,np.newaxis] #1/(1-rho**2)
            c2 = 1/(distribution[:,2] * distribution[:,3])[:,np.newaxis]
            c3 = 1/(distribution[:,2] * distribution[:,2])[:,np.newaxis]
            c4 = 1/(distribution[:,3] * distribution[:,3])[:,np.newaxis]

            p = np.sqrt(c1)*c2/(2*np.pi)*np.exp(-0.5*c1*(c3*(x - miu1)**2 - 2*c2*rho*(x - miu1)*(y - miu2) + c4*(y - miu2)**2))
            loss = -np.log(p + 0.1)
            _, col_ind = linear_sum_assignment(loss)
            indices[:,k] = col_ind
            new_pos[:,:,k] = normalized_pos[:,:,k][col_ind].copy()
        delta = np.sum(last_indices != indices)
        print(delta)
        distribution[:,0:2] = np.average(new_pos,axis = 2)
        distribution[:,2] = np.sqrt(np.var(new_pos[:,0,:],axis = 1,ddof = 1))
        distribution[:,3] = np.sqrt(np.var(new_pos[:,1,:],axis = 1,ddof = 1))
        distribution[:,4] = np.sum((new_pos[:,0,:] - np.average(new_pos[:,0,:],axis = 1)[:,np.newaxis])\
                            *(new_pos[:,1,:] - np.average(new_pos[:,1,:],axis = 1)[:,np.newaxis]),axis = 1)\
                            /(new_pos.shape[2] - 1)/(distribution[:,2]*distribution[:,3])
        if delta <= threshold:
            break
        last_indices = indices.copy()
    # print(indices)
    indices = indices.astype(int)
    un_normalized_pos = np.zeros_like(input_pos)
    for j in range(input_pos.shape[2]):
        un_normalized_pos[:,:,j] = input_pos[indices[:,j],:,j]
    # print(un_normalized_pos.shape)
    init_avg_pos = np.average(un_normalized_pos,axis = 2)
    distribution[:,0:2] += init_avg_pos
    return distribution,indices,delta,i
        




if __name__ == "__main__":
    with open("position.json",'r') as w:
        original_data = json.load(w)

    points = np.zeros([10,2,500])
    for j in range(500):
        for i in range(10):
            points[i,0,j] = original_data[1300 +j]['Argentina'][i]['x']
            points[i,1,j] = original_data[1300 +j]['Argentina'][i]['y']

    distribution,indices,error,j = GaussianCluster(points)
    print(distribution)
    print(indices)
    print(error)
    print(j)
    # print(np.average(new_pos,axis = 2))
    # # print(np.cov(new_pos[:,0,:],new_pos[:,0,:]).shape)
    # # cov = np.cov()
    # print(np.var(new_pos[:,0,:],axis = 1,ddof = 1))
    # print(np.var(new_pos[:,1,:],axis = 1,ddof = 1))
    # print(np.corrcoef(new_pos[0,0,:],new_pos[0,1,:]))
    # print(np.sum((new_pos[:,0,:] - np.average(new_pos[:,0,:],axis = 1)[:,np.newaxis])*(new_pos[:,1,:] - np.average(new_pos[:,1,:],axis = 1)[:,np.newaxis]),axis = 1)/(new_pos.shape[2] - 1))
