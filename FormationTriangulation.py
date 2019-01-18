import numpy as np
import json
from scipy.spatial.qhull import Delaunay

with open("position.json",'r') as w:
    original_data = json.load(w)[0]['Argentina']

points = np.zeros([10,2])
for i in range(10):
    points[i,0] = original_data[i]['x']
    points[i,1] = original_data[i]['y']
tri = Delaunay(points)
import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
