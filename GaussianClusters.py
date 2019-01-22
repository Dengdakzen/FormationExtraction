import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches

def eigsorted(cov):
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def GaussianCluster(input_pos,iteration = 50):
    '''
    input_pos: numpy array of shape (player number,dimension,time) 
    '''
    #Initialize the GMM model
    init_avg_pos_frame = np.average(input_pos,axis = 0)
    # print(init_avg_pos_frame.shape)
    normalized_pos = input_pos - init_avg_pos_frame[np.newaxis,:,:]
    distribution = np.zeros([normalized_pos.shape[0],normalized_pos.shape[1] + 3])
    distribution[:,0:2] = np.average(normalized_pos,axis = 2)
    distribution[:,2] = np.sqrt(np.var(normalized_pos[:,0,:],axis = 1,ddof = 1))
    distribution[:,3] = np.sqrt(np.var(normalized_pos[:,1,:],axis = 1,ddof = 1))
    distribution[:,4] = np.sum((normalized_pos[:,0,:] - np.average(normalized_pos[:,0,:],axis = 1)[:,np.newaxis])\
                        *(normalized_pos[:,1,:] - np.average(normalized_pos[:,1,:],axis = 1)[:,np.newaxis]),axis = 1)\
                        /(normalized_pos.shape[2] - 1)/(distribution[:,2]*distribution[:,3])
    # print(distribution)
    # print(np.cov(normalized_pos[1,0,:],normalized_pos[1,1,:]))
    # print(np.corrcoef(normalized_pos[1,0,:],normalized_pos[1,1,:]))
    last_indices = np.zeros([normalized_pos.shape[0],normalized_pos.shape[2]])
    new_pos = np.zeros_like(normalized_pos)
    new_pos[:,:,0] = normalized_pos[:,:,0].copy()
    indices = np.zeros([normalized_pos.shape[0],normalized_pos.shape[2]])
    indices[:,0] = np.array(range(normalized_pos.shape[0]))
    for i in range(iteration):
        print(i)
        for k in range(normalized_pos.shape[2]):
            x = normalized_pos[:,0,k].copy()[np.newaxis,:]
            y = normalized_pos[:,1,k].copy()[np.newaxis,:]
            miu1 = distribution[:,0].copy()[:,np.newaxis]
            miu2 = distribution[:,1].copy()[:,np.newaxis]
            rho = distribution[:,4].copy()[:,np.newaxis]
            c1 = 1/(1 - distribution[:,4]*distribution[:,4])[:,np.newaxis] #1/(1-rho**2)
            c2 = 1/(distribution[:,2] * distribution[:,3])[:,np.newaxis]
            c3 = 1/(distribution[:,2] * distribution[:,2])[:,np.newaxis]
            c4 = 1/(distribution[:,3] * distribution[:,3])[:,np.newaxis]
            # print((x - miu1).shape)

            p = np.sqrt(c1)*c2/(2*np.pi)*np.exp(-0.5*c1*(c3*(x - miu1)**2 - 2*c2*rho*(x - miu1)*(y - miu2) + c4*(y - miu2)**2))
            loss = -np.log(p + 0.1)
            _, col_ind = linear_sum_assignment(loss)
            indices[:,k] = col_ind
            new_pos[:,:,k] = normalized_pos[col_ind,:,k].copy()
        delta = np.sum(last_indices != indices)
        if i == 0:
            threshold = 0.002 * delta
            print(threshold)
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
    init_avg_pos = np.average(un_normalized_pos,axis = (0,2))
    distribution[:,0:2] += init_avg_pos
    new_pos += init_avg_pos[np.newaxis,:,np.newaxis]
    return distribution,un_normalized_pos,new_pos,indices,delta,i
        
def Vis_Distributions(distribution,new_pos = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    
    plt.xlim(0,1050)
    plt.ylim(0,680)
    fig.gca().invert_yaxis()
    ax.set_aspect('equal')
    for i in range(10):
        if new_pos is not None:
            x = new_pos[i,0,:]
            y = new_pos[i,1,:]
        cov = np.array([[distribution[i][2]**2,distribution[i][2]*distribution[i][3]*distribution[i][4]],[distribution[i][2]*distribution[i][3]*distribution[i][4],distribution[i][3]**2]])
        m,n = eigsorted(cov)
        theta = np.degrees(np.arctan2(*n[:,0][::-1]))
        e1 = patches.Ellipse((distribution[i,0],distribution[i,1]), width=2*np.sqrt(5.991*m[0]),height=2*np.sqrt(5.991*m[1]),angle=theta, linewidth=2, fill=False, zorder=2)
        if new_pos is not None:
            plt.scatter(x,y)
        ax.add_patch(e1)
    plt.show()


if __name__ == "__main__":
    with open("position.json",'r') as w:
        original_data = json.load(w)

    points = np.zeros([10,2,25])
    for j in range(25):
        for i in range(10):
            points[i,0,j] = original_data[13000+j]['Argentina'][i]['x']
            points[i,1,j] = original_data[13000+j]['Argentina'][i]['y']

    distribution,un_normalized_pos,new_pos, indices,error,j = GaussianCluster(points)
    Vis_Distributions(distribution,new_pos)
