#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# José Nuno Cunha, DEM, UA


from copy import deepcopy
from functools import partial
from random import randint

import cv2
import numpy as np
import open3d as o3d
import math



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
			"boundingbox_max" : [ 2.7116048336029053, 1.2182252407073975, 3.8866173028945923 ],
			"boundingbox_min" : [ -2.4257750511169434, -1.6397310495376587, -1.3339539766311646 ],
			"field_of_view" : 60.0,
			"front" : [ 0.90806389038388724, -0.36136608280300542, -0.21174164725082129 ],
			"lookat" : [ 0.062821566363608611, -0.21461364585355228, 1.3956862887565959 ],
			"up" : [ -0.41235505009531731, -0.85992086516689492, -0.30083121232928484 ],
			"zoom" : 0.48120000000000007
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    filename = '../rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/01.ply'
    point_cloud_original = o3d.io.read_point_cloud(filename)

    point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.01)
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

    pcds_to_draw = [point_cloud_downsampled]

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))


    
    
    entities = [] 
    # entities.append(frame_world)
    entities.append(frame_table)
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

