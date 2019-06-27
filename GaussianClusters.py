import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import matplotlib
# matplotlib.use('TkAgg')
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
    collection = []
    for i in range(10):
        collection.append(np.sum(indices == i,axis = 1))
    collection = np.transpose(np.array(collection))
    collection = collection/input_pos.shape[2]
    return distribution,un_normalized_pos,new_pos,collection,delta,i
        
def Vis_Distributions(distribution,new_pos = None,names = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    
    plt.xlim(-200,200)
    plt.ylim(-100,100)
    fig.gca().invert_yaxis()
    ax.set_aspect('equal')
    for i in range(10):
        if new_pos is not None:
            x = new_pos[i,0,:]
            y = new_pos[i,1,:]
        cov = np.array([[distribution[i][2]**2,distribution[i][2]*distribution[i][3]*distribution[i][4]],[distribution[i][2]*distribution[i][3]*distribution[i][4],distribution[i][3]**2]])
        m,n = eigsorted(cov)
        theta = np.degrees(np.arctan2(*n[:,0][::-1]))
        e1 = patches.Ellipse((distribution[i,0],distribution[i,1]), width=1*np.sqrt(5.991*m[0]),height=1*np.sqrt(5.991*m[1]),angle=theta, linewidth=2, fill=False, zorder=2)
        if names is not None:
            plt.text(distribution[i,0],distribution[i,1],names[i])
        if new_pos is not None:
            plt.scatter(x,y)
        ax.add_patch(e1)
    plt.show()


if __name__ == "__main__":
    # with open("position.json",'r') as w:
    #     original_data = json.load(w)

    # points = np.zeros([10,2,25])
    # for j in range(25):
    #     for i in range(10):
    #         points[i,0,j] = original_data[13000+j]['Argentina'][i]['x']
    #         points[i,1,j] = original_data[13000+j]['Argentina'][i]['y']

    # with open('/home/2TB_disk/data/ChelseaVsArsenal.json','r') as w:
    #     original_data = json.load(w)['Trajectory']
    # names = []
    # # print(original_data)
    # timelen = 1000
    # points = np.zeros([10,2,timelen])
    # for j in range(timelen):
    #     print(original_data[j]['time'])
    #     for i in range(10):
    #         if j == 0:
    #             names.append(original_data[j]['Players'][i+19]['name'])
    #         points[i,0,j] = original_data[j]['Players'][i+19]['x']
    #         points[i,1,j] = original_data[j]['Players'][i+19]['y']
    # distribution,un_normalized_pos,new_pos, collection,error,j = GaussianCluster(points)
    # print(collection)
    # Vis_Distributions(distribution,new_pos,names)

    current_time = 0
    with open('/home/2TB_disk/data/ManUnitedVsChelsea.json','r') as w:
        event_data = json.load(w)['Events']
    print(event_data)

    point_x_1st = []
    point_y_1st = []

    first_half_idx = 0
    for i in event_data:
        if i['Time'] >= current_time:
            current_time = i['Time']
        else:
            break
        first_half_idx += 1

        point_x_1st.append(i['Time'])
        if i['Description'] == 'Pass' or i['Description'] == 'Side' or i['Description'] == 'Shot' or i['Description'] == 'ShotOT':
            if i['Team'] == 1:
                point_y_1st.append(1)
            else:
                point_y_1st.append(0)
        else:
            point_y_1st.append(0.5)
    # print(first_half_idx)

    # plt.plot(point_x_1st,point_y_1st)
    # plt.show()
    team_label = 0
    intevals = []
    if point_y_1st[0] == team_label:
        start  = True
        inteval = [point_x_1st[0]]
    else:
        start = False
    for i in range(1,len(point_x_1st)):
        if point_y_1st[i] == team_label and i < len(point_x_1st) - 1:
            if start == True:
                continue
            else:
                start  = True
                inteval = [point_x_1st[i]]
        elif point_y_1st[i] == team_label and i == len(point_x_1st) - 1:
            if start == True:
                inteval.append(point_x_1st[i])
                intevals.append(inteval)
        else:
            if start == True:
                if point_x_1st[i-1] > inteval[0]:
                    inteval.append(point_x_1st[i-1])
                    intevals.append(inteval)
                inteval = []
                start = False

    # for i in intevals:
    #     plt.plot(i,[team_label,team_label],color = 'red')
    # plt.show()


    with open('/home/2TB_disk/data/ManUnitedVsChelsea.json','r') as w:
        original_data = json.load(w)['Trajectory']
    names = []
    # print(original_data)
    team_offset = 19
    timelen = 5000
    points = np.zeros([10,2,timelen])
    inteval_idx = 0
    inteval = intevals[inteval_idx]
    time_count = 0
    for j in range(timelen):
        print(original_data[j]['time'])
        if original_data[j]['time'] < inteval[0]:
            continue
        elif original_data[j]['time'] > inteval[1]:
            if inteval_idx == len(intevals) - 1:
                break
            else:
                inteval_idx += 1
                inteval = intevals[inteval_idx]
                continue
        for i in range(10):
            if len(names) < 10:
                names.append(original_data[j]['Players'][i+team_offset]['name'])
            points[i,0,time_count] = original_data[j]['Players'][i+team_offset]['x']
            points[i,1,time_count] = original_data[j]['Players'][i+team_offset]['y']
        time_count += 1
    print("finally:",time_count)
    points = points[:,:,:time_count]
    distribution,un_normalized_pos,new_pos, collection,error,j = GaussianCluster(points)
    print(collection)
    Vis_Distributions(distribution,new_pos,names)

                

        