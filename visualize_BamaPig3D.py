import numpy as np 
import cv2 
import json 
from tqdm import tqdm 
import pickle 
from utils import *
from matplotlib.patches import Patch 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import os 
from visualize_BamaPig2D import COLORS, draw_keypoints, draw_mask

# this function is used to transfer 2d labeled json files to unified pickle file for easier usage. 
# if you load BamaPig3D_pure_pickle, it already contains these two files. 
def read_2dlabel_to_pickle(BamaPig3D_folder):
    camids = [0,1,2,5,6,7,8,9,10,11]
    all_2D_points = np.zeros([70,10,4,19,3]) # store all 2d keypoints  
    mask_all_frames = [] # store all silhouettes 
    for i in tqdm(range(70)): 
        mask_all_views = {}
        for index, camid in enumerate(camids):
            frameid = 25 * i 
            outfilename = BamaPig3D_folder + "/label_images/cam{}/{:06d}.json".format(camid, frameid) 
            mask_4 = [[], [], [], []]
            with open(outfilename, 'r') as f: 
                data = json.load(f) 
                for part in data['shapes']: 
                    if part['shape_type'] == 'point':             
                        point = np.asarray(part['points'][0], dtype=np.int32)         
                        group_id = part["group_id"]
                        label = int(part['label'])
                        if label == 18: 
                            label = 17 
                        elif label == 20: 
                            label = 18 
                        elif label > 18: 
                            print("error") 
                            return 
                        try: 
                            all_2D_points[i, index, group_id, label, 0:2] = point 
                            all_2D_points[i, index, group_id, label, 2] = 1 
                        except: 
                            from IPython import embed; embed() # debug
                            exit()
                    elif part['shape_type'] == 'polygon': 
                        group_id = part["group_id"]
                        if len(part["points"]) > 0: 
                            mask_4[group_id].append(part["points"])
            mask_all_views.update({camid:mask_4})
        mask_all_frames.append(mask_all_views)
    with open(BamaPig3D_folder + "label_keypoints2d.pkl", 'wb') as f: 
        pickle.dump(all_2D_points, f) 
    with open(BamaPig3D_folder + "label_silhouettes2d.pkl", 'wb') as f: 
        pickle.dump(mask_all_frames, f) 


## This function load all 2D keypoint annotations and combine them into `output/points2d.pkl` file for 
def count_visibility(BamaPig3D_folder):
    camids = [0,1,2,5,6,7,8,9,10,11]
    all_2D_points = np.zeros([70,10,4,19,3])
    if os.path.exists(BamaPig3D_folder + "/label_keypoints2d.pkl"): 
        with open(BamaPig3D_folder + "/label_keypoints2d.pkl", 'rb') as f: 
            all_2D_points = pickle.load(f) 
    else: 
        print("please run read_2dlabel_to_pickle() function first!")
        return 
    cam_level1 = [3]
    cam_level2 = [0, 5, 6]
    cam_level3 = [1,2,7,8]
    cam_level4 = [4, 9]
    total_visible_point_num = all_2D_points[:,:,:,:,2].sum() 
    total_point_num = 70 * 10 * 4 * 19 
    print("visible_ratio of all keypoints   : ", (float(total_visible_point_num) / float(total_point_num) ) ) 
    print("visible_ratio for each part      : ")
    for k in range(19): 
        print("{:10s}".format(g_jointnames[g_all_parts[k]]), all_2D_points[:,:,:,k,2].sum() / (70 * 10 * 4) )

    part_levels = [ 
        [0,17,18], 
        [1,2,3,4],
        [5,6,7,8,9,10,11,12,13,14,15,16]
    ]
    part_level_names = ["Trunk", "Head", "Limb"]
    for index, part_level in enumerate(part_levels): 
        N = len(part_level)
        print(part_level_names[index])
        print("  cam  level1 : ", all_2D_points[:, cam_level1, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 1) )
        print("  cam  level2 : ", all_2D_points[:, cam_level2, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 3) )
        print("  cam  level3 : ", all_2D_points[:, cam_level3, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 4) )
        print("  cam  level4 : ", all_2D_points[:, cam_level4, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 2) )

    print("ratio of visible to more than 1 views: ") 
    for k in range(19): 
        part_sum = all_2D_points[:,:,:,k,2].sum(axis=1)
        print("{:10s}".format(g_jointnames[g_all_parts[k]]), (part_sum > 1).sum() / (70 * 4) )

# This function draws Supplementary Fig. 8c in the paper. 
def draw_visibility_level(BamaPig3D_folder):
    if not os.path.exists(BamaPig3D_folder + "/label_keypoints2d.pkl"): 
        print("Please run read_2dlabel_to_pickle() function to generate label_keypoints2d.pkl")
        return 
    with open(BamaPig3D_folder + "/label_keypoints2d.pkl", 'rb') as f: 
        all_2D_points = pickle.load(f) 

    part_levels = [ 
        [0,17,18], # trunk
        [1,2,3,4], # head 
        [5,6,7,8,9,10,11,12,13,14,15,16] # limbs
    ]

    cam_level1 = [3]
    cam_level2 = [0, 5, 6]
    cam_level3 = [1,2,7,8]
    cam_level4 = [4, 9]

    data = np.zeros([4,3])
    for index, part_level in enumerate(part_levels): 
        N = len(part_level)
        data[0,index] = all_2D_points[:, cam_level1, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 1)
        data[1,index] = all_2D_points[:, cam_level2, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 3)
        data[2,index] = all_2D_points[:, cam_level3, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 4)
        data[3,index] = all_2D_points[:, cam_level4, :, :, 2][:,:,:,part_level].sum() / (70 * 4 * N * 2)
    
    mpl.rc('font', family='Arial')
    fig = plt.figure(figsize=(1.8, 1.4)) 
    colormaps = np.loadtxt("colormaps/tab.txt") / 255
    part_level_names = ["Trunk", "Head", "Limbs"]
    xs = np.asarray([0,1,2])
    for x_index in range(3): 
        plt.bar(xs-0.3, data[0,:], width=0.2, color=colormaps[0], linewidth=0.5)
        plt.bar(xs-0.1, data[1,:], width=0.2, color=colormaps[1], linewidth=0.5)
        plt.bar(xs+0.1, data[2,:], width=0.2, color=colormaps[2], linewidth=0.5)
        plt.bar(xs+0.3, data[3,:], width=0.2, color=colormaps[3], linewidth=0.5)
    plt.legend(["Top view", "Corner views", "Middle views", "Side views"], fontsize=6, frameon=False, ncol=1)
    plt.ylim(0,1) 
    plt.xticks(xs, part_level_names, fontsize=7)
    plt.yticks([0,0.2,0.4,0.6,0.8,1], [0,20,40,60,80,100], fontsize=7)
    plt.xlabel("Body Parts", fontsize=7)
    plt.ylabel("Percentage of Visible\n Keypoints (%)", fontsize=7)
    ax = fig.get_axes()[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for line in ["bottom", "left", "right"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.savefig("output/supp_fig_8c.png", dpi=1000, bbox_inches='tight', pad_inches=0)
    # plt.savefig("output/supp_fig_8c.svg", dpi=1000, bbox_inches='tight', pad_inches=0)

# This function draws Supplementary Fig. 8d in the paper. 
def draw_keypoint_visibility_hist(BamaPig3D_folder): 
    jointnames = [g_jointnames[k] for k in g_all_parts]
    if not os.path.exists(BamaPig3D_folder + "/label_keypoints2d.pkl"): 
        print("Please run read_2dlabel_to_pickle() function first!")
        return 
    with open(BamaPig3D_folder + "/label_keypoints2d.pkl", 'rb') as f: 
        all_2D_points = pickle.load(f) 
    data = np.zeros([19,4])
    for k in range(19): 
        part_sum = all_2D_points[:,:,:,k,2].sum(axis=1)
        data[k,2] = (part_sum > 1).sum() / (70 * 4) 
        data[k,3] = (part_sum > 4).sum() / (70 * 4)
        data[k,1] = (part_sum == 1).sum() / (70 * 4)
        data[k,0] = 1  
    mpl.rc('font', family='Arial') 

    fig = plt.figure(figsize=(4,1.4)) 

    colormaps = np.loadtxt("colormaps/tab20c.txt") / 255 

    xs = np.arange(0,19,1)
    plt.bar(xs, data[:,0], color=colormaps[3], edgecolor=(0,0,0), lw=0.5)
    plt.bar(xs, data[:,1] + data[:,2], color=colormaps[2], edgecolor=(0,0,0), lw=0.5)
    plt.bar(xs, data[:,2], color=colormaps[1], edgecolor=(0,0,0), lw=0.5)
    plt.bar(xs, data[:,3], color=colormaps[0], edgecolor=(0,0,0), lw=0.5)


    plt.xticks(xs,jointnames, rotation=45, ha='right', fontsize=7)
    plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,20,40,60,80,100], fontsize=7)
    legend_elements = [
        Patch(facecolor=colormaps[3], edgecolor='black', label="Visible to 0 view", linewidth=0.5), 
        Patch(facecolor=colormaps[2], edgecolor='black', label="Visible to 1 view", linewidth=0.5),
        Patch(facecolor=colormaps[1], edgecolor='black', label="Visible to 2~4 views", linewidth=0.5),
        Patch(facecolor=colormaps[0], edgecolor='black', label="Visible to 5~10 views", linewidth=0.5),
    ]
    plt.legend(handles=legend_elements, fontsize=6, ncol=2, loc='upper left', bbox_to_anchor=(0.0, 1.3), frameon=False)
    ax = fig.get_axes()[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    plt.xlim(-1, 19)
    
    plt.xlabel("", fontsize=7)
    plt.ylabel("Percentage of Visible Keypoints (%)", fontsize=7)
    plt.ylim(0,1)

    plt.savefig("output/supp_fig_8d.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig("output/supp_fig_8d.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)

# demo for how to load and draw silhouettes used in BamaPig3D dataset. 
def demo_draw_mask(BamaPig3D_folder):
    camid = 0
    frameid = 0 
    imgfile = BamaPig3D_folder + "/label_images/cam{}/{:06d}.jpg".format(camid, frameid) 
    img = cv2.imread(imgfile) 

    camids = [0,1,2,5,6,7,8,9,10,11]
    with open(BamaPig3D_folder + "/label_silhouettes2d.pkl", 'rb') as f: 
        mask_label = pickle.load(f) 
    
    mask_label_current = mask_label[frameid][camid]

    for pid in range(4): 
        draw_mask(img, np.asarray(mask_label_current[pid]), COLORS[pid])
    if not os.path.exists("output"):
        os.makedirs("output")
    cv2.imwrite("output/demo_sil_BamaPig3D_frame0_cam0.png", img) 

# 
def demo_how_to_project_points_with_extrinsic_params(BamaPig3D_folder):
    frameid = 0 
    camids = [0,1,2,5,6,7,8,9,10,11]

    with open(BamaPig3D_folder + "/intrinsic_camera_params/distortion_info.pkl", 'rb') as f: 
        intrinsic_params = pickle.load(f) 
    K = intrinsic_params["newcameramtx"]

    for camid in camids: 
        undist_image = cv2.imread(BamaPig3D_folder + "/label_images/cam{}/{:06d}.jpg".format(camid, frameid))

        extrinsic_params = np.loadtxt(BamaPig3D_folder + "/extrinsic_camera_params/{:02d}.txt".format(camid)).squeeze() 
        # extrinsic_params = np.loadtxt("H:/MAMMAL_core/data/calibdata/adjust/{:02d}.txt".format(camid)).squeeze() 
        R = cv2.Rodrigues(extrinsic_params[0:3])[0]
        T = extrinsic_params[3:]

        # if you use BamaPig3D_pure_pickle, just load pickle files for all 3D keypoints GT
        for pid in range(4):
            points3d = np.loadtxt(BamaPig3D_folder + "/label_mix/pig_{}_frame_{:06d}.txt".format(pid, frameid))
            points3d = points3d[g_all_parts] # remove invalid ones to get 19 keypoints only 
            ## KEY process 
            points2d = (points3d @ R.T + T) @ K.T 
            points2d =  points2d[:,0:2] / points2d[:,2:]
            draw_keypoints(undist_image, points2d, pid, g_bones_19)

        cv2.imwrite("output/demo_BamaPig3D_proj_3d_keypoints_{}.png".format(camid), undist_image)

if __name__ == "__main__": 
    # To run this file, you should change this folder to your own BamaPig3D dataset path  
    BamaPig3D_folder = "H:/examples/BamaPig3D/"
    # count_visibility(BamaPig3D_folder) 

    # output supp_fig_8c.png
    draw_visibility_level(BamaPig3D_folder)

    # output supp_fig_8d.png
    draw_keypoint_visibility_hist(BamaPig3D_folder)

    # demo for how to load and draw sihouettes 
    demo_draw_mask(BamaPig3D_folder)

    # this function shows how to project 3D keypoints to 2D 
    demo_how_to_project_points_with_extrinsic_params(BamaPig3D_folder)