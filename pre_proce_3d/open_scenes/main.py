#!/usr/bin/env python3

from copy import deepcopy
import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.7116048336029053, 1.2182252407073975, 3.8905272483825684 ],
			"boundingbox_min" : [ -2.4257750511169434, -1.6397310495376587, -1.3339539766311646 ],
			"field_of_view" : 60.0,
			"front" : [ -0.42435722763098271, -0.32421248457999857, -0.84546271839733456 ],
			"lookat" : [ -2.9988803724941846, -3.4141542364877826, -5.8212961795348663 ],
			"up" : [ 0.15190921113526218, -0.94595945552288074, 0.28650357777716767 ],
			"zoom" : 0.02
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    #/home/joao/Documents/SAVI/Repositório/Aula1_SAVI_mine/TP2_ideas/rgbd-scenes-v2_pc/rgbd-scenes-v2
    #/home/joao/Documents/SAVI/Repositório/Aula1_SAVI_mine/TP2_ideas/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/01.ply
    filename = '../rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/01.ply'
    print('Loading file ' + filename)
    pcd_original = o3d.io.read_point_cloud(filename)
    print(pcd_original)

    # --------------------------------------
    # Execution
    # --------------------------------------
    
    #
    # Aply Downsampling
    #    
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    #pcd_downsampled.paint_uniform_color([1,0,0])#pintar a malha com uma cor
    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    #
    # find the table plan
    #

    

    #
    # Sistem cordinates on the table center
    #

    #criar o referecial em [0 0 0]
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    

    #------------------------------------------------------------------
    # Visualization 
    #------------------------------------------------------------------
    
    pcds_to_draw = [pcd_downsampled]
    
   
    entities = []
    entities.append(frame)
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
