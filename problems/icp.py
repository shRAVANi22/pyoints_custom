import numpy as np

from pyoints import (
	storage,
	Extent,
	transformation,
	filters,
	registration,
	normals,
	IndexKD
)

A = storage.loadPly('../data/scan02_overlap.ply')
B = storage.loadPly('../data/scan04_overlap.ply')
C = storage.loadPly('bun090_binary.ply')
# coords_dict = {}
# normals_dict = {}


def roughly_align():
	T_A = transformation.r_matrix([90*np.pi/180, 0, 0])
	A.transform(T_A)
	T_B = transformation.r_matrix([86*np.pi/180, 0, 45*np.pi/180])
	B.transform(T_B)
	T_C = transformation.r_matrix([95*np.pi/180, 0, 90*np.pi/180])
	C.transform(T_C)
	coords_dict = {
		'A': A.coords,
		'B': B.coords,
		'C': C.coords
	}
	return coords_dict
	pass


def coarse_registration(coords_dict):
	d_th = 0.04
	radii = [d_th, d_th, d_th]
	icp = registration.ICP(
		radii,
		max_iter=60,
		max_change_ratio=0.000001,
		k=1
	)
	T_dict, pairs_dict, report = icp(coords_dict)
	return T_dict, pairs_dict, report
	pass


def compute_normals(coords_dict):
	normals_dict = {
		key: normals.fit_normals(coords_dict[key], k=5, preferred=[0, -1, 0])
		for key in coords_dict
	}
	return normals_dict
	pass


def fine_registration(coords_dict, normals_dict):
	d_th = 0.04
	n_th = np.sin(15 * np.pi / 180)
	radii = [d_th, d_th, d_th, n_th, n_th, n_th]
	nicp = registration.ICP(
		radii,
		max_iter=60,
		max_change_ratio=0.000001,
		update_normals=True,
		k=1
	)
	T_dict, pairs_dict, report = nicp(coords_dict, normals_dict)
	return T_dict, pairs_dict, report


def run():
	point_clouds_dict = roughly_align()
	T_dict_coarse, pairs_dict_coarse, report_coarse = coarse_registration(point_clouds_dict)
	normals_dict = compute_normals(point_clouds_dict)
	T_dict, pairs_dict, report = fine_registration(point_clouds_dict, normals_dict)


run()