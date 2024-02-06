#!/usr/bin/env python3

# objectdetector

from copy import deepcopy
import math
import open3d as o3d
import numpy as np
import sys
from matplotlib import cm
from more_itertools import locate
import os



view ={
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.70914378762245178, 0.22959384077520512, 2.0597424507141113 ],
			"boundingbox_min" : [ -0.73871105909347534, -0.30997958779335022, 0.73439687490463257 ],
			"field_of_view" : 60.0,
			"front" : [ 0.14095174342777114, -0.41341896209763335, -0.89956509925785111 ],
			"lookat" : [ 0.26243634826393109, -1.2711610996847007, -1.584954423393417 ],
			"up" : [ -0.10355452711760713, -0.90980883554453118, 0.40190091151744506 ],
			"zoom" : 0.02
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}



class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane 

    def segment(self, distance_threshold=0.04, ransac_n=3, num_iterations=100):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        
        [self.a, self.b, self.c, self.d] = plane_model
        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)
        inlier_cloud  = self.point_cloud.select_by_index(inlier_idxs, invert=False)
        
        return (outlier_cloud,inlier_cloud)

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    
    filename = '../rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/01.ply'
   
    #05 06 07 08 gives me problems,table legs separate from the table
    # 13 14 gives me probles, 
    print('Loading file ' + filename)
    pcd_original = o3d.io.read_point_cloud(filename)
    print(pcd_original)

    #
    # Convert ply to pcd
    #    
    os.system('pcl_ply2pcd ' +filename+ ' pcd_point_cloud.pcd')
    pcd_original = o3d.io.read_point_cloud('pcd_point_cloud.pcd')

    # --------------------------------------
    # Execution
    # --------------------------------------
    
    #
    # Aply Downsampling
    #    
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    #pcd_downsampled.paint_uniform_color([0,0,1])#pintar a malha com uma cor
    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    #
    # find the ground plane
    #
    pcd_point_cloud = deepcopy(pcd_downsampled)
    planes=[]
    max_planes=1
    plane_min_points=1000 #25 as a good value
    while True:
        plane=PlaneDetection(pcd_point_cloud)
        [pcd_point_cloud,pcd_inlier_cloud] = plane.segment(distance_threshold=0.035, ransac_n=3, num_iterations=400) #num off iterations 200
        #pcd_inlier_cloud.paint_uniform_color((1, 0, 0))
        print(plane)
        planes.append(plane)
        if len(planes) >= max_planes: # stop detection planes
            print('Detected planes >=' + str(max_planes))
            break
        elif len(pcd_point_cloud.points) < plane_min_points:
            print('number of remaining points <' + str(plane_min_points))
            break
    

    table_plane_mean_xy = 10000
    centers=[]
    for plane_idx,plane in enumerate(planes):
        center=plane.inlier_cloud.get_center()
        centers.append(center)
        print('Cloud ' + str(plane_idx) + ' has center '+str(center))
        mean_x = center[0]
        mean_y = center[1]
        mean_z = center[2]

        mean_xy = abs(mean_x) + abs(mean_y)
        if mean_xy < table_plane_mean_xy:
            table_plane = plane
            table_plane_mean_xy = mean_xy

    #
    # create the reference of the found plans
    #
            
    frame_1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=2, origin=np.array([centers[0][0],centers[0][1],centers[0][2]]))
    #frame_2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([centers[1][0],centers[1][1],centers[1][2]]))
    frame_main = o3d.geometry.TriangleMesh().create_coordinate_frame(size=2, origin=np.array([0,0,0]))
    
    #
    # Clustering aply to the big point clouds
    # 
    labels = pcd_point_cloud.cluster_dbscan(eps=0.08, min_points=15, print_progress=True)
    print("Max label:", max(labels))
    group_idxs = list(set(labels))
    print('groud_idx:',group_idxs)
    num_groups = len(group_idxs)
    colormap = cm.Pastel1(range(0, num_groups))

    center_clouds=[]
    norms_vetor_center=[]
    pcd_sep_objects = []
    a=1000
    for group_idx in group_idxs:  # Cycle all groups, i.e.,
        group_points_idxs = list(locate(labels, lambda x: x == group_idx))
        pcd_separate_object = pcd_point_cloud.select_by_index(group_points_idxs, invert=False)
        color = colormap[group_idx, 0:3]
        #pcd_separate_object.paint_uniform_color(color)
        pcd_sep_objects.append(pcd_separate_object)
        
        #print(pcd_separate_object)
        print('The point cloud nº ',group_idx,' is centered at',center)
        center=pcd_separate_object.get_center()
        center_clouds.extend(center)
        quadratic= ((center[0]-mean_x)**2)+((center[1]-mean_y)**2)+((center[2]-mean_z)**2) 
        norm=math.sqrt(quadratic)
        print('the norm value',norm)
        
        frame_2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=2, origin=np.array([center[0],center[1],center[2]]))
        if norm<a and group_idx!=-1:
            id_table=group_idx
            a=norm
    print('The point cloud closest to the center is:',id_table)

    #
    # plane segmentation, for remove the table surface
    #
    pcd_table_id=pcd_sep_objects[id_table]
    plane_2=PlaneDetection(pcd_table_id)
    [pcd_point_cloud_2,pcd_inlier_cloud_2]=plane_2.segment(distance_threshold=0.02, ransac_n=3, num_iterations=400)
    #pcd_point_cloud_2.paint_uniform_color((0, 0, 1)) # point cloud on the surface
    pcd_inlier_cloud_2.paint_uniform_color((1, 0, 0)) #table surface
    
    #
    # Clustering applied to the point cloud on the table surface
    #
    labels_2 = pcd_point_cloud_2.cluster_dbscan(eps=0.04, min_points=30, print_progress=True)
    print("Max label:", max(labels_2))
    group_idxs_2 = list(set(labels_2))
    print('groud_idx:',group_idxs_2)
    num_groups_2 = len(group_idxs_2)
    
    pcd_sep_objects_2=[]
    for group_idx_2 in group_idxs_2:
        group_points_idxs_2 = list(locate(labels_2, lambda x: x == group_idx_2))
        pcd_sep_object_2=pcd_point_cloud_2.select_by_index(group_points_idxs_2,invert=False)
        #pcd_sep_object_2.paint_uniform_color(color)
        pcd_sep_objects_2.append(pcd_sep_object_2)

    #------------------------------------------------------------------
    # Visualization 
    #------------------------------------------------------------------
    
    pcds_to_draw=[]
    max_index=len(pcd_sep_objects_2)
    print('max_index',max_index)
    for i in range(max_index-1):
        pcds_to_draw.append(pcd_sep_objects_2[i])
        print(i)
    
    #pcds_to_draw = [pcd_inlier_cloud_2,pcd_sep_objects_2[:group_idx_2]]
     
    
    

    entities = []
    # entities.append(frame_main)
    #entities.append(frame_1)
    #entities.append(pcd_inlier_cloud)
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
