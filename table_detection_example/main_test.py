#!/usr/bin/env python3

from copy import deepcopy

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
			"boundingbox_max" : [ 0.5, 0.76599996999999997, 0.5 ],
			"boundingbox_min" : [ -0.041739031891304339, -0.029999999999999999, -0.054244022437499997 ],
			"field_of_view" : 60.0,
			"front" : [ 0.27107632577314983, -0.94928574351125505, 0.15929282084140803 ],
			"lookat" : [ 0.22913048405434783, 0.36799998499999997, 0.22287798878125001 ],
			"up" : [ 0.021657299327180426, 0.17146224227027537, 0.98495261858705796 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    pcd_original = o3d.io.read_point_cloud('../rgbd-dataset_pcd/coffee_mug/coffee_mug_4/coffee_mug_4_1_110.pcd')

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    # pcd_downsampled.paint_uniform_color([1,0,0])

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Visualization ----------------------
    pcds_to_draw = [pcd_downsampled]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = []
    entities.append(frame)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
