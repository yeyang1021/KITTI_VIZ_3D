# KITTI_VIZ_3D
Visualization 3D object detection results using meshlab.

This code is used for visualization by adding 3D bounding boxes to the LiDAR point cloud and storing it in a txt file.

## Requrements

numba

opencv

matplotlib

pandas


## python code

kitti_util.py

meshlab_file.py    #using this to create meslab file, the detected *.txt using Camera coordinate  

draw3Dbox2img.py

draw_voxel_circle.py  

527_test_file_Lidar.txt is a test file 

## Some information need changed when you use this code.

```
img_id = 5147

calib = Calibration('/home1/yang_ye/data/Kitti/testing/calib/%06d.txt'%img_id)

path = '/home1/yang_ye/data/Kitti/testing/velodyne/%06d.bin'%img_id

path_img = '/home1/yang_ye/data/Kitti/testing/image_2/%06d.png'%img_id

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

df = read_detection('/home2/yang_ye/results_kitti/%06d.txt'%img_id)
```


```
An example for '/home2/yang_ye/results_kitti/%06d.txt'%img_id
Car 0.0000 0.0000 1.3057 32.7170 174.0567 237.6250 296.2407 1.6493 1.7758 4.1813 -7.9406 1.6728 11.8281 0.7246 0.9034
Car 0.0000 0.0000 0.4979 153.2972 177.7395 352.8660 244.3964 1.5039 1.6312 4.0061 -8.5172 1.6277 17.3175 0.0466 0.7844
Car 0.0000 0.0000 1.1924 0.0000 180.3295 86.5943 254.3225 1.4746 1.6550 4.1006 -13.9844 1.6628 16.4232 0.4950 0.3663
```
## how to use meshlab 
![meshlab](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/config.png)

## origin image
![Ori](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/005147.png)

## 3D Visualization

![3D](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/snapshot_514700.png)


## 3D projected to image
![3D_to_2D](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/5147_img.png)


## Draw voxels and circles
![voxels](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/voxel.png)
![circles](https://github.com/yeyang1021/KITTI_VIZ_3D/blob/master/circle.png)
