import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import yaml
import pandas as pd
from kitti_util import *
from matplotlib.lines import Line2D
import cv2



###instance point cloud###
v_path = '6397_Car_0.bin'

num_features = 4 

points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

points_v_ = points_v[:,0:3]

#### 5.2*5.2*2.4
####  center 2.6 2.6 1.2

a = np.arange(0.2, 5.2, 0.4)
b = np.arange(0.2, 5.2, 0.4)
c = np.arange(0.4, 2.4, 0.8)

all = a.shape[0] * b.shape[0] * c.shape[0]

pos = np.zeros(points_v_.shape, dtype = np.int)

for i in range(points_v_.shape[0]):
    x = points_v_[i,0]
    y = points_v_[i,1]
    z = points_v_[i,2]
    diff_x = np.abs(x + 2.6 - a)  
    x_pos = np.where(diff_x==np.min(diff_x))[0]

    diff_y = np.abs(y + 2.6 - b)  
    y_pos = np.where(diff_y==np.min(diff_y))[0]

    diff_z = np.abs(z - c)  
    z_pos = np.where(diff_z==np.min(diff_z))[0]  
    
    pos[i,0] = x_pos
    pos[i,1] = y_pos
    pos[i,2] = z_pos
    
  
    
pos = np.unique(pos, axis=0)    
    
save_ = np.zeros((7,0),dtype=float)
for i in range(pos.shape[0]):
       
    x_pos = a[pos[i,0]]
    y_pos = b[pos[i,1]]
    z_pos = c[pos[i,2]]
    
    x1 = x_pos - 0.2 - 2.6
    x2 = x_pos + 0.2 - 2.6
    x3 = x_pos + 0.2 - 2.6
    x4 = x_pos - 0.2 - 2.6
    
    y1 = y_pos - 0.2 - 2.6
    y2 = y_pos - 0.2 - 2.6
    y3 = y_pos + 0.2 - 2.6
    y4 = y_pos + 0.2 - 2.6
    
    z1 = z_pos + 0.4 
    z2 = z_pos + 0.4 
    z3 = z_pos + 0.4 
    z4 = z_pos + 0.4
    
    z11 = z_pos - 0.4 
    z22 = z_pos - 0.4 
    z33 = z_pos - 0.4 
    z44 = z_pos - 0.4 
        
    
    save_points = draw_3D_box(x1,x2,x3,x4, y1,y2,y3,y4, z1,z2,z3,z4, x1,x2,x3,x4, y1,y2,y3,y4, z11,z22,z33,z44)
    
    save_ = np.concatenate((save_,save_points), axis = 1)


## Delete duplicate data
save_ = np.unique(save_, axis=1)

pad = np.zeros((points_v.shape[0],3), dtype = np.float) 
points = np.concatenate((points_v, pad), axis = 1)



''' 
## draw circle 
# circle center 
center = [0,0,0.78]
r = 1.6 
save_circle1 = draw_circle(center, r, axis=2, color=[0,0,255], res=0.01)
save_circle2 = draw_circle(center, r, axis=1, color=[0,0,255], res=0.01)
save_circle3 = draw_circle(center, r, axis=0, color=[0,0,255], res=0.01)
points = np.concatenate((points,save_.T, save_circle1.T, save_circle2.T, save_circle3.T), axis = 0)
np.savetxt(v_path + '_circle_voxel_Lidar_1.6.txt', points)
'''

#np.savetxt(v_path + '_Lidar_ori.txt', points)

points = np.concatenate((points,save_.T), axis = 0)

np.savetxt(v_path + '_voxel_Lidar.txt', points)
