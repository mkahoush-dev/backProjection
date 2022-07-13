import os
import numpy
from skimage import io
import skimage.transform
import skimage.io
import open3d as o3d
import xml.etree.ElementTree as ET
from class_util import class_names, class_colors
import timeit
import argparse
import cv2



def contains_duplicates(X):
    return len(numpy.unique(X)) != len(X)
#Format: xy: [color, count, xyz]


label_id_map = {j:i for i,j in enumerate(class_names)}
label_color_map = {i:j for i,j in enumerate(class_colors)}
label_color_id_map = {str(j):i for i,j in enumerate(class_colors)}
print('label_id_map', label_id_map)
print('label_color_map', label_color_map)
print('label_color_id_map', label_color_id_map)
#base_dir = FLAGS.base_dir
base_dir = './I-675-location1/7-22-2021/60degrees/'

# Load point cloud data
pcd_data = []
load_from_npy = True
pcd_file = base_dir + '/original.ply'
pcd = o3d.io.read_point_cloud(pcd_file)
xyz = numpy.asarray(pcd.points)
pcd_data.append(xyz)
print (len(xyz))
print (xyz)
pcd_labels = [0 for x in range(len(pcd_data[0]))]

# Assume exported from Pix4D
f = open(base_dir + 'params/60 degrees wo reference_calibrated_camera_parameters.txt', 'r')
for i in range(8):
    f.readline()
intrinsics = []
extrinsics = []
image_names = []
original_width = 0
original_height = 0
while True:
    l = f.readline()
    if not l:
        break
    image_names.append(l.split()[0])
    w, h = [int(t) for t in l.split()[1:]]
    focal, _, cx = [float(t) for t in f.readline().split()]
    cy = float(f.readline().split()[2])
    intrinsics.append([focal, 0, 0])
    for i in range(3):
        f.readline()
    R = numpy.zeros((3, 3))
    t = numpy.array([float(j) for j in f.readline().split()])
    R[0, :] = [float(j) for j in f.readline().split()]
    R[1, :] = [float(j) for j in f.readline().split()]
    R[2, :] = [float(j) for j in f.readline().split()]
    extrinsics.append(numpy.eye(4))
    extrinsics[-1][:3, :3] = R
    extrinsics[-1][:3, 3] = -R.dot(t)
intrinsics = numpy.array(intrinsics)[numpy.argsort(image_names)]
extrinsics = numpy.array(extrinsics)[numpy.argsort(image_names)]
f.close()
cx_offset = cx - (w - 1) * 0.5
cy_offset = cy - (h - 1) * 0.5
flip_u = True

confidences = []
confidences
confidence_dir = base_dir + "/conf/"
for i in sorted(os.listdir(confidence_dir)):
    c = numpy.load(os.path.join(confidence_dir, i))
    confidences.append(c)

original_height = 3648
original_width = 5472
images_folder = base_dir + '/results'
image_height = None
image_width = None
image_downsample = 8
camera_id = 0
c = 0
total_color = []
total_points = []
points_set = {}
total_confidence = []
for i in sorted(os.listdir(images_folder)):
    xy_map = {}
    start = timeit.default_timer()
    color = []
    points = []
    I = skimage.io.imread(images_folder + '/' + i)
    I = skimage.transform.resize(I, (original_height, original_width), order=0, preserve_range=True, anti_aliasing=False).astype(numpy.uint8)
    camera_id = i.split("_")[1].split('.')[0]
    C = confidences[int(c)]
    C = skimage.transform.resize(C, (original_height, original_width), order=0, preserve_range=True, anti_aliasing=False).astype(numpy.uint8)
    if (int(c) >= extrinsics.shape[0]):
        continue
    image_height = I.shape[0]
    image_width = I.shape[1]
    cx = (image_width-1)*0.5 + cx_offset
    cy = (image_height-1)*0.5 + cy_offset
    print('Image dimensions', image_width, image_height, cx, cy)
    R = extrinsics[int(c), :3, :3]
    t = extrinsics[int(c), :3, 3]
    f, k1, k2 = intrinsics[int(c), :]
    k1 = k2 = 0
    xyz = pcd_data[0]
    #total_points.append(xyz)
    print('Lenght xyz',len(xyz))
    xyz = xyz.dot(R.T) + t
    xy = xyz[:, :2] / xyz[:, 2:3]
    rp = 1.0 + k1 * (xy**2).sum(axis=1) + k2 * (xy**2).sum(axis=1)**2
    xy = rp.reshape(-1, 1) * xy
    xy = f * xy
    if flip_u:
        xy[:, 0] = -xy[:, 0]
    xy += [cx, cy]
    xy = numpy.round(xy).astype(numpy.int32)
    valid = xy[:, 0] >= 0
    valid = numpy.logical_and(valid, xy[:, 0] < image_width)
    valid = numpy.logical_and(valid, xy[:, 1] >= 0)
    valid = numpy.logical_and(valid, xy[:, 1] < image_height)
    xy = xy[valid, :]
    print('Length xy',len(xy))
    xyz = pcd_data[0][valid, :]
    for j in range (len(xy)):
        x = xy[j][0]
        y = xy[j][1]
        key = []
        label = I[int(round(y))][int(round(x))]
        co = C[int(round(y))][int(round(x))]
        key.append(xy[j][0])
        key.append(xy[j][1])
        if (len(xy_map.keys()) == 0 or not tuple(key) in xy_map):
            colors = [label]
            count = [0]
            a = numpy.around(xyz[j], 1)
            xy_map[tuple(key)] = [colors, count, a, co]
        else:
            value = xy_map[tuple(key)]
            colors = value[0]
            count = value[1]
            if ((label == colors).all(axis = 1).any()):
                index = 0
                for index in range(len(colors)):
                    if (colors[index] == label).all():
                        break;
                count[index] = count[index] + 1
                a = numpy.around(xyz[j], 1)
                value = [colors, count, a, co]
                xy_map[tuple(key)] = value
            else:
                colors.append(label)
                count.append(0)
                a = numpy.around(xyz[j], 1)
                value = [colors, count, a, co]
                xy_map[tuple(key)] = value
        comparison = label == [0,0,0]
        if (not comparison.all()):
            pcd_labels[j] = label_color_id_map[str(label)]
    print("Len xy: ", len(xy_map.keys()))
    values = xy_map.values()
    # print(xy_map.keys())
    # print(contains_duplicates(total_points))
    for value in values:
        colors = value[0]
        count = value[1]
        max_count_index = count.index(max(count))
        conf = value[3]
        # color.append((colors[max_count_index]))
        # points.append(value[2])
        # numpy.any(numpy.all(numpy.isin(total_points,values[2],True),axis=1))

        #filter points that are close to each other by rounding to first decimal place

        if len(total_points) == 0 or not (tuple(value[2]) in points_set.keys()):
            # points.append(value[2])
            # color.append(colors[max_count_index])
            total_points.append(value[2])
            total_color.append(colors[max_count_index])
            points_set[tuple(value[2])] = 0
            total_confidence.append(conf[max_count_index])
    print("Fin")
    end = timeit.default_timer()
    print(end-start)
    # color = numpy.asarray(color)
    # points = numpy.asarray(points)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(color.astype(numpy.float) / 255.0)
    # o3d.io.write_point_cloud("./data" + str(c) + ".ply", pcd)
    c = c + 1
    #camera_id += 1
final_arr = [total_points, total_color, total_confidence]
final_arr = numpy.array(final_arr)
numpy.save(base_dir + "/conf.npy", final_arr)
total_color = numpy.asarray(total_color)
total_points = numpy.asarray(total_points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(total_points)
pcd.colors = o3d.utility.Vector3dVector(total_color.astype(numpy.float) / 255.0)
o3d.io.write_point_cloud("./data.ply", pcd)
