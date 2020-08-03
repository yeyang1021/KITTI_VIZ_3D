# KITTI_VIZ_3D
Visualization 3D object detection results using meshlab.


Some information need changed when you use this code.

img_id = 5147

calib = Calibration('/home1/yang_ye/data/Kitti/testing/calib/%06d.txt'%img_id)

path = '/home1/yang_ye/data/Kitti/testing/velodyne/%06d.bin'%img_id

path_img = '/home1/yang_ye/data/Kitti/testing/image_2/%06d.png'%img_id

points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

df = read_detection('/home2/yang_ye/results_kitti/%06d.txt'%img_id)
