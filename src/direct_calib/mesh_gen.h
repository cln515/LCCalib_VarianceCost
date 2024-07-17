//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once
#define NOMINMAX
#define _HAS_STD_BYTE 0
#include <ply_object.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>


pcl::PointCloud<pcl::Normal>::Ptr surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
pcl::PolygonMesh Generate_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void genMesh(cvl_toolkit::plyObject& po);
