import json
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import datetime

def AffineMatrixEstimate(input_vec,output_vec):
    '''
    input_vec: shape (m,2)
    output_vec: shape (m,2)
    return matrix: T:input_vec->output_vec
    '''
    assert input_vec.shape == output_vec.shape
    # print(input_vec.shape)
    num = np.shape(input_vec)[0]
    ones_col = np.ones([num,1])
    ones_zeros = np.zeros([num,3])
    A_l = np.column_stack((input_vec,ones_col,ones_zeros))
    A_r = np.column_stack((ones_zeros,input_vec,ones_col))
    A = np.column_stack((A_l.reshape(num*2,-1),A_r.reshape(num*2,-1)))
    B = output_vec.reshape([-1,1])
    # print(np.round(np.dot(A.T,A)))
    res = np.dot(inv(np.dot(A.T,A)),np.dot(A.T,B)).reshape([2,3])
    # print(np.round(np.dot(res,np.column_stack((input_vec,ones_col)).T)))
    # print(output_vec.T)
    return np.row_stack((res,(0,0,1)))

def Project(input_vec,affine_matrix):
    num = np.shape(input_vec)[0]
    ones_col = np.ones([num,1])
    A = np.column_stack((input_vec,ones_col)).T
    res = np.dot(affine_matrix,A)
    return res[0:2,:].T

def compute_Error(vec1,vec2):
    num = np.shape(vec1)[0]
    element_2 = (vec1 - vec2)**2
    element_sqrt = np.sqrt(np.sum(element_2,axis = 1))
    return np.sum(element_sqrt)/num



def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape
    loss_matrix = cdist(src,dst)
    # print(loss_matrix)
    row_ind, col_ind = linear_sum_assignment(loss_matrix)
    distances = loss_matrix[row_ind, col_ind]
    return distances,col_ind

def EvaluateTemplateError(field_vec,template_vec,affine_matrix):
    '''
    affine_matrix: The homogeneous transformation matrix from field_vec to template_vec
    '''
    a_inv = inv(affine_matrix)
    tem_to_field = Project(template_vec,a_inv)
    distance,_ = nearest_neighbor(field_vec,tem_to_field)
    return np.sum(distance)

def icp(A,B,min_local_optimazation = 20,max_iterations = 2000,tolerance=0.00001):
    assert A.shape == B.shape

    src = np.copy(A)
    dst = np.copy(B)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src, dst)
        # print(indices)

        # compute the transformation between the current source and nearest destination points
        T = AffineMatrixEstimate(src, dst[indices])

        # update the current source
        src = Project(src,T)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance and i > min_local_optimazation:
            break
        prev_error = mean_error
    T = AffineMatrixEstimate(A, src)
    return T, distances, i, indices

def ImportFromTemplate(filepath):
    with open(filepath,'r') as w:
        template = json.load(w)
    data = []
    name = []
    for i in template:
        name.append(i['name'])
        merged = i['backward'] + i['midfielder'] + i['forward']
        data.append(merged)
    return np.array(data),name

def time_to_str():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    

if __name__ == "__main__":
    # output = 'result.json'
    # www = open(output,'w+')
    data = []
    hist = np.zeros(16)
    count = 0
    with open("on-off.json",'r') as w:
        original_label = json.load(w)["Events"]
    with open("position.json",'r') as w:
            original_data = json.load(w)
    templates, names = ImportFromTemplate('templates.json')
    on_off_idx = 0
    while on_off_idx < len(original_label):
        start_idx = original_label[on_off_idx]["frame"]
        end_idx = original_label[on_off_idx + 1]["frame"]
        on_off_idx += 2

        for k in range(start_idx,end_idx):
            points = np.zeros([10,2])
            for i in range(10):
                points[i,0] = original_data[k]['Brazil'][i]['x']
                points[i,1] = original_data[k]['Brazil'][i]['y']
            least_err = 1000000
            least_idx = 0
            err_array = np.zeros(np.shape(templates)[0])
            ProjectedToTemplate = []
            for j in range(np.shape(templates)[0]):
                C = points.copy()
                D = templates[j].copy()
                if k < 64767:
                    D[:,0] = 1050 - D[:,0]
                T,err,i,indices = icp(C,D)
                E = Project(C,T)
                ProjectedToTemplate.append(E.tolist())
                error = EvaluateTemplateError(C,D,T)
                err_array[j] = error
                if error < least_err:
                    least_err = error
                    least_idx = j
            sort_error_index = np.argsort(err_array)
            # print(sort_error_index[0:5])
            # print(err_array[sort_error_index[0:5]])
            # print(names[sort_error_index[0:5]])
            print("number, 5 smallest errors & names, variance:\n",k,err_array[sort_error_index[0:5]],[names[sort_error_index[i]] for i in range(5)],np.var(err_array[sort_error_index[0:5]]))
            this_element = {'frame':k,'errors':err_array[sort_error_index[0:16]].tolist(),"names":[names[sort_error_index[i]] for i in range(16)],"indices":indices.tolist(),"Projected_To_Template":ProjectedToTemplate}
            data.append(this_element)
            print(names[least_idx])
            hist[least_idx] += 1
            if count > 0 and count % 1000 == 0:
            #     y_pos = np.arange(len(names))
            #     plt.barh(y_pos, hist, align='center', alpha=0.5)
            #     plt.yticks(y_pos, names)
            #     plt.xlabel('Usage')
            #     fig_name = 'result/' + 'count_' + str(count) + '_' + time_to_str() + '.png'
            #     plt.title('formation detection result' + ' count: ' + str(count) +'_' + time_to_str())
            #     plt.savefig(fig_name)
            #     plt.close()
                save_name = 'result/json_perK/Brazil/' + 'count_' + str(count) + '_' + time_to_str() + '.json'
                with open(save_name,'w+') as w:
                    json.dump(data,w,indent = 4)
                data = []
            count += 1
        #     if count >5:
        #         break
        # break
    save_name = 'result/json_perK/Brazil/' + 'count_' + str(count) + '_' + time_to_str() + '.json'
    with open(save_name,'w+') as w:
        json.dump(data,w)

        
    y_pos = np.arange(len(names))
    
    # plt.barh(y_pos, hist, align='center', alpha=0.5)
    # plt.yticks(y_pos, names)
    # plt.xlabel('Usage')
    # plt.title('formation detection result')
    # plt.savefig('plt_pics/result2.png')
    # plt.show()
    
