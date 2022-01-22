#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import open3d as o3d
from pyoints import (
    storage,
    Extent,
    transformation,
    filters,
    registration,
    normals,
)
pcd = o3d.geometry.PointCloud()
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# get_ipython().run_line_magic('matplotlib', 'inline')
folder = r'D:\PycharmProjects\pyoints-v0.2.0\pyoints_custom\data'

if not os.path.exists(folder + '\\Merge'):
    os.mkdir(folder + '\\Merge')
if not os.path.exists(folder + '\\T_Matrix'):
    os.mkdir(folder + '\\T_Matrix')
if not os.path.exists(folder + '\\Iterations'):
    os.mkdir(folder + '\\Iterations')
# output_ply = folder + '\\Merge\\V_Block_dth_30.ply'

A = storage.loadPly(folder + '\\Side_1__aligned.ply')
# print(A.shape)
# print(A.dtype.descr)
# print("1\n")
# print(A)

B = storage.loadPly(folder + '\\Side_2__aligned.ply')
# print(B.shape)
# print(B.dtype.descr)

##########################################   LOAD POINTS   #############################################################

# r = 2.5
# A = A[list(filters.ball(A.indexKD(), r))]
# # print("2\n")
# # print(A)
# B = B[list(filters.ball(B.indexKD(), r))]

axes_lims = Extent([
    A.extent().center - 0.5 * A.extent().ranges.max(),
    A.extent().center + 0.5 * A.extent().ranges.max()
])
colors = {'A': 'green', 'B': 'blue'}

fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*A.coords.T, color=colors['A'], alpha=0.5, label='UP__A')
ax.scatter(*B.coords.T, color=colors['B'], alpha=0.5, label='DOWN__B')

plt.title("Loaded Sparse points")
plt.show()

##########################################    ALIGNMENT     #############################################################
#------------17thJan
T_A = transformation.r_matrix([-15 * np.pi/180, 0, 0])
T_A1 = transformation.t_matrix([45, 0, 0])
A.transform(T_A)
A.transform(T_A1)

T_B = transformation.r_matrix([20 * np.pi/180, 0, 0])
T_B1 = transformation.t_matrix([45, -50, 0])
B.transform(T_B)
B.transform(T_B1)


axes_lims = Extent([
    A.extent().center - 0.5 * A.extent().ranges.max(),
    A.extent().center + 0.5 * A.extent().ranges.max()
])

fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(*A.coords.T, color=colors['A'], label='UP__A', alpha=0.4)
ax.scatter(*B.coords.T, color=colors['B'], label='DOWN__B', alpha=0.4)

ax.legend()
plt.title("Roughly Aligned Points")
plt.show()

##########################################    COARSE     #############################################################
def registration_gocator():

    # print(A.coords)

    coords_dict = {
        'A': A[list(filters.ball(A.indexKD(), 2.5))].coords,
        'B': B[list(filters.ball(B.indexKD(), 2.5))].coords,
    }
    coords_dict_fine = {
        'A': A.coords,
        'B': B.coords,
    }

    d_th = 30
    radii = [d_th, d_th, d_th]
    icp = registration.ICP(
        radii,
        max_iter=300,
        max_change_ratio=0.000001,
        k=5
    )

    T_dict, pairs_dict, report = icp(coords_dict)

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_xlim(axes_lims[0], axes_lims[3])
    ax.set_ylim(axes_lims[1], axes_lims[4])
    ax.set_zlim(axes_lims[2], axes_lims[5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    file_TC = open(folder + "\\T_Matrix\\T_matrix_coarse_"+str(d_th)+".txt", 'w')
    combine_coords = []

    for key in coords_dict_fine:
        file_TC.write(str(T_dict[key]) + '\n')
        coords = transformation.transform(coords_dict_fine[key], T_dict[key])
        ax.scatter(*coords.T, color=colors[key], label=key)

    file_TC.close()
    ax.legend()
    plt.title("Coarse Registration")
    plt.show()


    fig = plt.figure(figsize=(25, 12))
    plt.xlim(0, len(report['RMSE']) + 1)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.bar(np.arange(len(report['RMSE'])) + 1, report['RMSE'], color='gray')
    for x, y in enumerate(report['RMSE']):
        plt.text(x + 0.9, y + .02, str(round(y,3)), color='blue', fontsize=5, rotation='vertical')
    plt.title("Coarse Iteration")
    # plt.show()
    plt.savefig(folder + "\\Iterations\\Coares_dth_"+str(d_th)+".png",bbox_inches='tight')


    # ##########################################     NORMALS      #########################################################
    normals_dict = {
        key: normals.fit_normals(coords_dict[key], k=5, preferred=[0, -1, 0])
        for key in coords_dict
    }

    # fig = plt.figure(figsize=(15, 15))
    # ax = plt.axes(projection='3d')
    # ax.set_xlim(axes_lims[0], axes_lims[3])
    # ax.set_ylim(axes_lims[1], axes_lims[4])
    # ax.set_zlim(axes_lims[2], axes_lims[5])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # ax.scatter(*A.coords.T, c=normals_dict['A'][:, 2], cmap='coolwarm')
    # for coord, normal in zip(coords_dict_fine['A'], normals_dict['A']):
    #     ax.plot(*np.vstack([coord, coord + normal * 0.1]).T, color='black')
    #
    # plt.title("Normals")
    # plt.show()

    ##########################################     FINE     #############################################################

    n_th = np.sin(15 * np.pi / 180)
    radii = [d_th, d_th, d_th, n_th, n_th, n_th]
    nicp = registration.ICP(
        radii,
        max_iter=500,
        max_change_ratio=0.000001,
        update_normals=True,
        k=5
    )

    T_dict, pairs_dict, report = nicp(coords_dict, normals_dict)

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_xlim(axes_lims[0], axes_lims[3])
    ax.set_ylim(axes_lims[1], axes_lims[4])
    ax.set_zlim(axes_lims[2], axes_lims[5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    file_TF = open(folder + "\\T_Matrix\\T_matrix_fine_"+str(d_th)+".txt", 'w')

    for key in coords_dict:
        file_TF.write(str(T_dict[key]) + '\n')
        coords = transformation.transform(coords_dict_fine[key], T_dict[key])
        ax.scatter(*coords.T, color=colors[key], label=key, alpha=0.5)
        # file_name = 'E:\\' + str(key) + '.txt'
        # file = open(file_name, 'w')
        # for point in coords:
        #     file.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')
        # file.close()
        for point in coords:
            combine_coords.append(point)

    file_TF.close()
    ax.legend()
    plt.title("Fine Registration")
    # plt.show()

    fig = plt.figure(figsize=(25, 12))
    plt.xlim(0, len(report['RMSE']) + 1)
    # for x, y in enumerate(len(report['RMSE']) + 1):
    #     plt.text(x + 0.9, y + .02, str(round(y,3)), color='blue', fontsize=5, rotation='vertical')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')

    plt.bar(np.arange(len(report['RMSE']))+1, report['RMSE'], color='gray')
    plt.title("Fine Iteration")
    # plt.show()
    plt.savefig(folder + "\\Iterations\\Fine_dth_" + str(d_th) + ".png",bbox_inches='tight')


    ###########################################     SAVE .pny FILE     #############################################################
    output_ply = folder + '\\Merge\\V_Block_dth_'+str(d_th)+'.ply'
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(combine_coords)
    o3d.write_point_cloud(output_ply, pcd)

#################################################################################################################################################################################################################

registration_gocator()