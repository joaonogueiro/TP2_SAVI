#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# José Nuno Cunha, DEM, UA


from copy import deepcopy
from functools import partial
from random import randint
from matplotlib import cm
from more_itertools import locate

import cv2
import numpy as np
import open3d as o3d
import math
import os



# view = {"class_name": "ViewTrajectory",
#         "interval": 29,
#         "is_loop": False,
#         "trajectory":
#         [
#             {
#                 "boundingbox_max": [6.5291471481323242, 34.024543762207031, 11.225864410400391],
#                 "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
#                 "field_of_view": 60.0,
#                 "front": [0.48005911651460004, -0.71212541184952816, 0.51227008740444901],
#                 "lookat": [-10.601035566791843, -2.1468729890773046, 0.097372916445466612],
#                 "up": [-0.28743522255406545, 0.4240317338845464, 0.85882366146617084],
#                 "zoom": 0.3412
#             }
#         ],
#         "version_major": 1,
#         "version_minor": 0
#         }


# vista da nuvem de pontos escolhida por mim 
view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.7932640408991194, 2.4011368432071523, 0.77045684766887101 ],
			"boundingbox_min" : [ -2.4695077294195116, -2.2598605371423122, -1.0083387351096422 ],
			"field_of_view" : 60.0,
			"front" : [ 0.64061242236824634, 0.56496236990434212, 0.52003196526709472 ],
			"lookat" : [ 0.16542321574590738, 0.60646546102710608, 0.12107869807647861 ],
			"up" : [ -0.29410717573734757, -0.44506713620246319, 0.84582280263205201 ],
			"zoom" : 0.42120000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}



# Function to get each objects color
def get_object_colors(pcd_objects, labels):
    object_colors = []

    for group_idx in set(labels):
        if group_idx == -1:
            continue  # Skip unassigned points

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))
        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)
        colors = np.asarray(pcd_separate_object.colors)
        object_color = np.mean(colors, axis=0)  # Calculate mean color of the object
        object_colors.append(object_color)

    return object_colors



# Function to convert ply to pcd
def convert_ply_to_pcd(ply_file, pcd_file):
    # Read the .ply file
    point_cloud = o3d.io.read_point_cloud(ply_file)

    # Write the point cloud to a .pcd file
    o3d.io.write_point_cloud(pcd_file, point_cloud)



def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])



def main():


    # --------------------------------------
    # Initialization
    # --------------------------------------

    filename = '../rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/03.ply'
    # the clustering in scenario 5, 6, 7 and 8 is not working properly

    point_cloud_original = o3d.io.read_point_cloud(filename)


    # --------------------------------------
    # Convert ply to pcd
    # --------------------------------------

    os.system('pcl_ply2pcd ' +filename+ ' pcd_point_cloud.pcd')
    point_cloud_original = o3d.io.read_point_cloud('pcd_point_cloud.pcd')


    point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.02)
    print(point_cloud_downsampled)



    # --------------------------------------
    # Estimate normals
    # --------------------------------------

    point_cloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Create transformation
    T1 = np.zeros((4,4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    R = point_cloud_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R

    # # Add null rotation, put a entity matrix
    # T[0:3, 0] = [1, 0, 0] # add n vector
    # T[0:3, 1] = [0, 1, 0] # add s vector
    # T[0:3, 2] = [0, 0, 1] # add a vector
    
    # Add a translation
    T1[0:3, 3] = [0, 0, 0]
    print('T1=\n' + str(T1))

    # Create transformation
    T2 = np.zeros((4,4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation, put a entity matrix
    T2[0:3, 0] = [1, 0, 0] # add n vector
    T2[0:3, 1] = [0, 1, 0] # add s vector
    T2[0:3, 2] = [0, 0, 1] # add a vector
    
    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.4]
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))



    # --------------------------------------
    # Execution
    # --------------------------------------

    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
    frame_table = frame_table.transform(T)

    point_cloud_downsampled = point_cloud_downsampled.transform(np.linalg.inv(T))

    # create the vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float) # array 8x3

    # s from scale -> size of the box
    sx = sy = 0.6
    sz_top = 0.4
    sz_bottom = -0.1
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    print('np_vertices =\n' + str(np_vertices))
    vertices = o3d.utility.Vector3dVector(np_vertices)

    # create the bbox
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    print(bbox)

    pcd_cropped = point_cloud_downsampled.crop(bbox)



    # --------------------------------------
    # Plane segmentation
    # --------------------------------------

    plane_model, inlier_idx = pcd_cropped.segment_plane(distance_threshold=0.02,
                                         ransac_n=3,
                                         num_iterations=100)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idx, invert=False)
    pcd_table.paint_uniform_color([1,0,0]) # paint the inliers, points that belong to the ground

    pcd_objects = pcd_cropped.select_by_index(inlier_idx, invert=True)



    # --------------------------------------
    # Clustering
    # --------------------------------------

    labels = pcd_objects.cluster_dbscan(eps=0.056, min_points=50, print_progress=True)
    # eps is the distance between objects, currently 5cm
    # 0,0059 is the value with most accurate results
    # for scenario 1, 0,056
    # for scenario 2, 0,056
    # for scenario 3, 0,056
    # for scenario 4, 0,056


    print("Max label:", max(labels))

    group_idxs = list(set(labels))
    # group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    colormap = cm.Set1(range(0, num_groups))

    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        color = colormap[group_idx, 0:3]
        # pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)




    # --------------------------------------
    # ICP for object classification
    # -------------------------------------
    
    pcd_cap = o3d.io.read_point_cloud('../cap_1/rgbd-dataset/cap/cap_1/cap_1_1_1.pcd')
    pcd_cap_downsampled = pcd_cap.voxel_down_sample(voxel_size=0.005)
    pcd_cap_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_cap_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # objects_data = []
    # for idx, pcd_separate_object in enumerate(pcd_separate_objects):

    #     Tinit = np.eye(4, dtype=float)  # null transformation
    #     reg_p2p = o3d.pipelines.registration.registration_icp(pcd_cap_downsampled, pcd_separate_object, 0.9, Tinit,
    #                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                                                           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    #     print('object idx ' + str(idx))
    #     print('reg_p2p = ' + str(reg_p2p))

    #     print("Transformation is:")
    #     print(reg_p2p.transformation)

    #     # draw_registration_result(pcd_separate_object, pcd_cap_downsampled, np.linalg.inv(reg_p2p.transformation))
    #     objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})

    # # Select which of the objects in the table is a cereal box by getting the minimum rmse
    # min_rmse = None
    # min_rmse_idx = None

    # for idx, object_data in enumerate(objects_data):

    #     if min_rmse is None:  # first object, use as minimum
    #         min_rmse = object_data['rmse']
    #         min_rmse_idx = idx

    #     if object_data['rmse'] < min_rmse:
    #         min_rmse = object_data['rmse']
    #         min_rmse_idx = idx

    # print('Object idx ' + str(min_rmse_idx) + ' is the cap')
    # draw_registration_result(pcd_separate_objects[min_rmse_idx], pcd_cap_downsampled,
    #                          np.linalg.inv(objects_data[min_rmse_idx]['transformation']))

    # print(objects_data)


    
    # --------------------------------------
    # Visualization
    # --------------------------------------
        
    # point_cloud_downsampled.paint_uniform_color([0.4, 0.3, 0.3])
    # pcd_cropped.paint_uniform_color([0.0, 0.9, 0.0])
    # pcd_table.paint_uniform_color([0.0, 0.0, 0.9])

    # print tests
    print('group_idxs = ', group_idxs)
    print('pcd_separate_objects = ', pcd_separate_objects)

    
    # pcds_to_draw = [point_cloud_downsampled, pcd_cropped, pcd_table]
    # pcds_to_draw = [pcd_cropped]
    # pcds_to_draw = [pcd_table]
    # pcds_to_draw = [pcd_table_original_colors]
    # pcds_to_draw = [pcd_cap_downsampled]
    # pcds_to_draw = [pcd_separate_objects[0], pcd_separate_objects[1], pcd_separate_objects[2],pcd_separate_objects[3], pcd_separate_objects[4]]
    # pcds_to_draw.extend(pcd_separate_objects)
    pcds_to_draw = []
    entities = [] 
    # pcds_to_draw.extend(pcd_separate_objects)
    
    # Iterate through pcd_separate_objects, excluding the last object
    for idx in range(len(pcd_separate_objects) - 1):
        pcds_to_draw.append(pcd_separate_objects[idx])

        # Compute the oriented bounding box for the current object
        bbox = pcd_separate_objects[idx].get_oriented_bounding_box()

        # Append the bounding box to entities and draw the bbox
        entities.append(bbox)

        # Extract centroid
        centroid = bbox.get_center()

        # Extract dimensions (length, width, height)
        lengths = bbox.get_max_bound() - bbox.get_min_bound()
        length, width, height = lengths[0], lengths[1], lengths[2]

        # Calculate area and volume of each object
        area = 2 * (length * width + width * height + height * length)
        volume = length * width * height

        # Print properties
        print(f"Object {idx + 1}:")
        print(f"Centroid: {centroid}")
        print(f"Dimensions (LxWxH): {length} x {width} x {height}")
        print(f"Area: {area}")
        print(f"Volume: {volume}")

        # get each objects color
        object_colors = get_object_colors(pcd_objects, labels)
        for idx, color in enumerate(object_colors):
            print(f"Color of object {idx + 1}: {color}")


        

    


    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    
    # entities = [] 
    entities.append(frame_world)
    # entities.append(frame_table)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()

