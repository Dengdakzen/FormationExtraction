from matplotlib import pyplot as plt
import os
import numpy as np
import json
from AffineTransformation import time_to_str, ImportFromTemplate


if __name__ == "__main__":
    '''
    Merge all the json files
    '''
    # path = "result/json_perK"
    # files= os.listdir(path)
    # counts = []
    # for i in files:
    #     res = i.split('_')
    #     counts.append(res[1])
    # arg_sort_index = np.argsort(counts)
    # print(arg_sort_index)
    # data = []
    # for i in arg_sort_index:
    #     file_name = path + '/' + files[i]
    #     with open(file_name,'r') as w:
    #         this_file = json.load(w)
    #         data = data + this_file
    # print(len(data))
    # with open(final_file_name,'w+') as w:
    #     json.dump(data,w,indent = 4)

    _, names = ImportFromTemplate('templates.json')
    path = "result/json_perK/Brazil/count_1000_19-01-09-16-22.json"
    with open(path,'r') as w:
        data = json.load(w)
    points_first = []
    value = {}
    for index,i in enumerate(names):
        value[i] = 15 - index
    y_val = []
    for i in data:
        y_val.append(value[i["names"][0]])
    x_val = range(1001)
    plt.plot(x_val,y_val)
    plt.show()