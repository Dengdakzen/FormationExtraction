import numpy as np
import json
from AffineTransformation import ImportFromTemplate
from scipy.spatial.qhull import Delaunay

def template_dict(templates):
    this_dict = {}
    for i in templates:
        a = len(i['backward'])
        b = len(i['midfielder'])
        c = len(i['forward'])
        arr = [a,b,c]
        this_dict[i['name']] = arr
    return this_dict

def template_to_index(template_array,indices):
    count = 0
    Res = []
    for i in range(3):
        this = []
        print(template_array)
        for k in range(template_array[i]):
            this.append(indices[count] + 1)
            count += 1
        Res.append(this)
    return Res

def Area(Tri, points):
    point_rep = points[Tri.simplices]
    print(point_rep.shape)
    a = point_rep[:,1,:] - point_rep[:,0,:]
    b = point_rep[:,2,:] - point_rep[:,0,:]
    ab = np.cross(a,b)
    return np.sum(ab)/2

# templates, names = ImportFromTemplate('templates.json')
# print(templates)
path = "result/json_perK/Brazil/count_1000_19-01-11-20-17.json"
with open(path,'r') as w:
    data = json.load(w)
with open("position.json",'r') as w:
    positional_data = json.load(w)
with open("templates.json",'r') as w:
    templates = template_dict(json.load(w))

a = data[0]['names'][0]
indices = data[0]['indices']
print(template_to_index(templates[a],indices))


final = []
start_frame = data[0]['frame']
end_frame = data[-1]['frame']
count = 0
area_count = 0
for i in range(start_frame,end_frame + 1):
    if data[count]['frame'] == i and i != end_frame:
        count += 1
        area_count += 1
        flag = 0
        if area_count % 50 == 0 or i == 0:
            positions = positional_data[i]['Brazil']
            points = np.zeros([10,2])
            for index in range(10):
                points[index][0] = positions[index]['x']
                points[index][1] = positions[index]['y']
            tri = Delaunay(points)
            area = Area(tri,points)
            print(i)
            node = {"frame":i,"formation":template_to_index(templates[data[count]['names'][0]],data[count]['indices']),"area":area,"next":1 }
            final.append(node)
            area_count = 0
            flag = 1
    elif data[count - 1]['frame'] == i - 1 and flag == 0:
        positions = positional_data[i - 1]['Brazil']
        points = np.zeros([10,2])
        for index in range(10):
            points[index][0] = positions[index]['x']
            points[index][1] = positions[index]['y']
        tri = Delaunay(points)
        area = Area(tri,points)
        node = {"frame":i,"formation":template_to_index(templates[data[count - 1]['names'][0]],data[count - 1]['indices']),"area":area,"next":0}
        final.append(node)
        area_count = 0
        flag = 1
    elif i == end_frame and data[count]['frame'] == i:
        positions = positional_data[i]['Brazil']
        points = np.zeros([10,2])
        for index in range(10):
            points[index][0] = positions[index]['x']
            points[index][1] = positions[index]['y']
        tri = Delaunay(points)
        area = Area(tri,points)
        print(i)
        node = {"frame":i,"formation":template_to_index(templates[data[count]['names'][0]],data[count]['indices']),"area":area,"next":0 }
        final.append(node)
        area_count = 0
        flag = 1
with open("sankey_Brazil.json","w+") as w:
    json.dump(final,w,indent = 4)