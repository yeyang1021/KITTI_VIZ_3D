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




def read_detection(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'width', 'height', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
#     df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
#     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
#    df = df[df['type']=='Car']
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df.reset_index(drop=True, inplace=True)
    return df

img_id = 1470


path = '/home/yeyang/data/kitti/testing/velodyne/%06d.bin'%img_id

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

df = read_detection('/home/yeyang/pp_test/%06d.txt'%img_id)


print (df)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    #y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h,h,h,h,0,0,0,0]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print(corners_3d)
    #print(x,y,z)
    #corners_3d += np.vstack([x, y, z])
    corners_3d += np.vstack([x, y, z])
    return corners_3d


def compute_3d_box_cam3(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    #a = np.vstack([z, x, y])#, [x-1, y, z],[x+1, y, z])
    #print(h, w, l, x, y, z, yaw)
    a = np.vstack([x, y, z])
    return a
    
    
save_ = np.zeros((7,0),dtype=float)
for o in range(len(df)):

    corners_3d = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
    
    x1,x2,x3,x4 = corners_3d[0,0:4]
    y1,y2,y3,y4 = corners_3d[2,0:4]
    z1,z2,z3,z4 = corners_3d[1,0:4]

    x11,x22,x33,x44 = corners_3d[0,4:8]
    y11,y22,y33,y44 = corners_3d[2,4:8]
    z11,z22,z33,z44 = corners_3d[1,4:8]
    
    save_points1 = compute_3D_line(x1,x2, y1,y2, z1,z2)
    save_points2 = compute_3D_line(x2,x3, y2,y3, z2,z3)
    save_points3 = compute_3D_line(x3,x4, y3,y4, z3,z4)
    save_points4 = compute_3D_line(x4,x1, y4,y1, z4,z1)
    
    save_points = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)
    
    
    save_points11 = compute_3D_line(x11,x22, y11,y22, z11,z22)
    save_points22 = compute_3D_line(x22,x33, y22,y33, z22,z33)
    save_points33 = compute_3D_line(x33,x44, y33,y44, z33,z44)
    save_points44 = compute_3D_line(x44,x11, y44,y11, z44,z11)
    
    save_points_ = np.concatenate((save_points11,save_points22,save_points33,save_points44), axis = 1)
    
    
    
    save_points111 = compute_3D_line(x1,x11, y1,y11, z1,z11)
    save_points222 = compute_3D_line(x2,x22, y2,y22, z2,z22)
    save_points333 = compute_3D_line(x3,x33, y3,y33, z3,z33)
    save_points444 = compute_3D_line(x4,x44, y4,y44, z4,z44)    
    save_points__ = np.concatenate((save_points, save_points111,save_points222,save_points333,save_points444), axis = 1)    

    #n1,n2, m1,m2, l1,l2 = (x11+x22)/2,(x33+x44)/2, (y11+y22)/2, (y33+y44)/2, (z11+z22)/2, (z33+z44)/2
    n1,n2, m1,m2, l1,l2 = (x11+x22)/2,(x33+x44)/2, (y1+y2)/2, (y3+y4)/2, (z11+z22)/2, (z33+z44)/2
    #################
    save_points_center = compute_3D_line_(n1,n2, m1,m2, l1,l2)
    #################
    
    
    save_points_center_ = save_points_center[:,0:150]

    save_ = np.concatenate((save_,save_points,save_points_, save_points__,save_points_center_), axis = 1)
    

def remove_points(points):
    label1 = points[:,1] > 40 
    label2 = points[:,1] < -40


    a = np.where(points[:,1] > 40)
    points = np.delete(points, a, axis=0)   
    
    a = np.where(points[:,1] < -40)
    points = np.delete(points, a, axis=0)       
    
    return points            

  
pad = np.zeros((points.shape[0],3), dtype = np.float)   
save_[0:3,:] = save_[[0,2,1],:]

points = np.concatenate((points,pad), axis = 1)
points = remove_points(points)
points = np.concatenate((points,save_.T), axis = 0)

np.savetxt(str(img_id) + '_test_file_Lidar.txt', points)




