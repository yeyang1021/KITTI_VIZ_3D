# KITTI_VIZ_3D
Visualization 3D object detection results using meshlab.

This code is used for visualization by adding 3D bounding boxes to the LiDAR point cloud and storing it in a txt file.

The demo file is meshlab_file.py

###############Some information need changed when you use this code.###############


img_id = 5147

calib = Calibration('/home1/yang_ye/data/Kitti/testing/calib/%06d.txt'%img_id)

path = '/home1/yang_ye/data/Kitti/testing/velodyne/%06d.bin'%img_id

path_img = '/home1/yang_ye/data/Kitti/testing/image_2/%06d.png'%img_id

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

df = read_detection('/home2/yang_ye/results_kitti/%06d.txt'%img_id)

###############Some information need changed when you use this code.###############

#####################################origin image##################################
![Ori](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/005147.png)

#################################3D Visualization##################################

![3D](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/snapshot_514700.png)


###############################3D projected to image##############################
![3D_to_2D](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/5147_img.png)

