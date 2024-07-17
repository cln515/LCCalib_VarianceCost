//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <direct_calib/mesh_gen.h>


pcl::PointCloud<pcl::Normal>::Ptr surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

	ne.setRadiusSearch(0.05);
	ne.compute(*cloud_normals);

	return cloud_normals;
}

pcl::PolygonMesh Generate_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	cloud_normals = surface_normals(cloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *cloud_normals, *cloud_with_normals);

	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	gp3.setSearchRadius(0.2);

	gp3.setMu(50.0);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);
	gp3.setMinimumAngle(M_PI / 10);
	gp3.setMaximumAngle(2 * M_PI / 3);
	gp3.setNormalConsistency(true);

	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);

	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	return triangles;
}

void genMesh(cvl_toolkit::plyObject& po){
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for (int i = 0; i < po.getVertexNumber(); i++) {
		double x = po.getVertecesPointer()[i * 3];
		double y = po.getVertecesPointer()[i * 3 + 1];
		double z = po.getVertecesPointer()[i * 3 + 2];
		cloud->points.push_back(pcl::PointXYZ(x, y, z));
	}

	pcl::PolygonMesh mesh = Generate_mesh(cloud);
	//std::vector<unsigned int> faces;
	unsigned int* faces = (unsigned int*)malloc(sizeof(unsigned int)*mesh.polygons.size()*3);
	for (int i = 0; i < mesh.polygons.size(); i++) {
		faces[i*3] = (mesh.polygons.at(i).vertices[0]);
		faces[i * 3 + 1] = (mesh.polygons.at(i).vertices[1]);
		faces[i * 3 + 2] = (mesh.polygons.at(i).vertices[2]);
	}

	po.setFacePointer(faces, mesh.polygons.size());

}