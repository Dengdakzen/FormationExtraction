import os
import json
import cv2
import copy
import numpy as np

# def draw_rectangle(event,x,y,flags,param):
#     global ix, iy
#     if event==cv2.EVENT_LBUTTONDOWN:
#         ix, iy = x, y
#         print("point1:=", x, y)
#     elif event==cv2.EVENT_LBUTTONUP:
#         print("point2:=", x, y)
#         print("width=",x-ix)
#         print("height=", y - iy)
#         cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

def get_Point_Position(event,x,y,flags,param):
    global ix, iy,flag
    if event==cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        print([ix,iy])
        flag = 1

ix = 0
iy = 0
flag = 0
cv2.namedWindow('image')
cv2.setMouseCallback('image', get_Point_Position)
index = 0
line_names = ["backward","midfielder","forward"]
line = []
path = "template"
files= os.listdir(path)
sep = '.'
formations = []
for i in files:
    filename = i.split(sep, 1)[0]
    index = 0
    print(filename)
    formation = {}
    formation['name'] = filename
    img = cv2.imread(path+'/'+i)
    shp = np.shape(img)
    width = shp[1]
    height = shp[0]
    print(shp)
    while(1):
        cv2.imshow('image', img)
        if flag == 1:
            line.append([int(copy.deepcopy(iy)/height*1050),int(copy.deepcopy(ix)/width*680)])
            flag = 0
        if cv2.waitKey(20) & 0xFF == 27:
            break
        elif cv2.waitKey(20) & 0xFF == 32:
            formation[line_names[index]] = line.copy()
            print(formation)
            line = []
            index += 1
            if index == 3:
                break
    formations.append(formation)
    print(formations)
with open('templates.json','w+') as w:
    json.dump(formations,w,indent = 4)

