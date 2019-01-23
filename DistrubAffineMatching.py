import numpy as np
import json
from GaussianClusters import GaussianCluster,Vis_Distributions
import AffineTransformation as AT
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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

def EvaluateTemplateEntropy(distribution,template_vec,affine_matrix):
    '''
    affine_matrix: The homogeneous transformation matrix from field_vec to template_vec
    '''
    a_inv = inv(affine_matrix)
    tem_to_field = Project(template_vec,a_inv)
    x = tem_to_field[:,0].copy()[np.newaxis,:]
    y = tem_to_field[:,1].copy()[np.newaxis,:]
    miu1 = distribution[:,0].copy()[:,np.newaxis]
    miu2 = distribution[:,1].copy()[:,np.newaxis]
    rho = distribution[:,4].copy()[:,np.newaxis]
    c1 = 1/(1 - distribution[:,4]*distribution[:,4])[:,np.newaxis] #1/(1-rho**2)
    c2 = 1/(distribution[:,2] * distribution[:,3])[:,np.newaxis]
    c3 = 1/(distribution[:,2] * distribution[:,2])[:,np.newaxis]
    c4 = 1/(distribution[:,3] * distribution[:,3])[:,np.newaxis]
    p = np.sqrt(c1)*c2/(2*np.pi)*np.exp(-0.5*c1*(c3*(x - miu1)**2 - 2*c2*rho*(x - miu1)*(y - miu2) + c4*(y - miu2)**2))
    loss_matrix = np.exp(p)
    row_ind, col_ind = linear_sum_assignment(loss_matrix)
    ps = p[row_ind, col_ind]
    plogp = -ps*np.log2(ps + 0.1)
    return np.sum(plogp)

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

if __name__ == "__main__":
    with open("position.json",'r') as w:
        original_data = json.load(w)
    templates, names = AT.ImportFromTemplate('templates.json')
    k = 0
    points = np.zeros([10,2,25])
    for j in range(25):
        for i in range(10):
            points[i,0,j] = original_data[k+j]['Argentina'][i]['x']
            points[i,1,j] = original_data[k+j]['Argentina'][i]['y']

    distribution,un_normalized_pos,new_pos, indices,error,j = GaussianCluster(points)
    Vis_Distributions(distribution,new_pos)
    fig, axs = plt.subplots(4, 4,figsize=(25,25))
    least_err = 1000000
    least_idx = 0
    err_array = np.zeros(np.shape(templates)[0])
    ProjectedToTemplate = []
    for j in range(np.shape(templates)[0]):
        C = distribution[:,0:2].copy()
        D = templates[j].copy()
        if k < 64767:
            D[:,0] = 1050 - D[:,0]
        T,err,i,indices = AT.icp(C,D)
        E = AT.Project(C,T)
        ProjectedToTemplate.append(E.tolist())
        error = AT.EvaluateTemplateError(C,D,T)
        # error = EvaluateTemplateEntropy(distribution,D,T)
        err_array[j] = error
        if error < least_err:
            least_err = error
            least_idx = j
    sort_error_index = np.argsort(err_array)
    print(err_array[sort_error_index])
    for index,j in enumerate(sort_error_index):
        D = templates[j].copy()
        if k < 64767:
            D[:,0] = 1050 - D[:,0]
        ax = axs[int(index/4),index%4]
        ax.scatter(np.array(ProjectedToTemplate[j])[:,0],np.array(ProjectedToTemplate[j])[:,1],c='r')
        ax.scatter(D[:,0],D[:,1],c='b')
        ax.set_title(names[j])
    plt.show()