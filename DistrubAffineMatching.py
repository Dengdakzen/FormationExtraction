import numpy as np
import json
from GaussianClusters import GaussianCluster,Vis_Distributions
import AffineTransformation as AT
from matplotlib import pyplot as plt


if __name__ == "__main__":
    with open("position.json",'r') as w:
        original_data = json.load(w)
    templates, names = AT.ImportFromTemplate('templates.json')
    k = 13000
    points = np.zeros([10,2,25])
    for j in range(25):
        for i in range(10):
            points[i,0,j] = original_data[k+j]['Argentina'][i]['x']
            points[i,1,j] = original_data[k+j]['Argentina'][i]['y']

    distribution,un_normalized_pos,new_pos, indices,error,j = GaussianCluster(points)
    # Vis_Distributions(distribution,new_pos)
    fig, axs = plt.subplots(4, 4)
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
        err_array[j] = error
        if error < least_err:
            least_err = error
            least_idx = j
        ax = axs[int(j/4),j%4]
        plt.scatter(E[:,0],E[:,1],c='r')
        plt.scatter(D[:,0],D[:,1],c='b')
        # ax.add_patch(e1)
        # ax.add_patch(e2)
    plt.show()
    sort_error_index = np.argsort(err_array)
    # print(sort_error_index[0:5])
    # print(err_array[sort_error_index[0:5]])
    # print(names[sort_error_index[0:5]])
    print("number, 5 smallest errors & names, variance:\n",k,err_array[sort_error_index[0:5]],[names[sort_error_index[i]] for i in range(5)],np.var(err_array[sort_error_index[0:5]]))
    this_element = {'frame':k,'errors':err_array[sort_error_index[0:16]].tolist(),"names":[names[sort_error_index[i]] for i in range(16)],"indices":indices.tolist(),"Projected_To_Template":ProjectedToTemplate}
    # data.append(this_element)
    print(names[least_idx])
    # hist[least_idx] += 1
    # if count > 0 and count % 1000 == 0:
    # y_pos = np.arange(len(names))
    # plt.barh(y_pos, hist, align='center', alpha=0.5)
    # plt.yticks(y_pos, names)
    # plt.xlabel('Usage')
    # fig_name = 'result/' + 'count_' + str(count) + '_' + time_to_str() + '.png'
    # plt.title('formation detection result' + ' count: ' + str(count) +'_' + time_to_str())
    # plt.savefig(fig_name)
    # plt.close()
    #     save_name = 'result/json_perK/Brazil/' + 'count_' + str(count) + '_' + time_to_str() + '.json'
    #     with open(save_name,'w+') as w:
    #         json.dump(data,w,indent = 4)
    #     data = []
    # count += 1
