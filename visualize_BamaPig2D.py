

import numpy as  np 
import json 
import cv2 
import os 
from utils import g_bones
import pickle 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

COLORS = np.loadtxt("colormaps/anliang_render.txt")
COLORS = COLORS[:,(2,1,0)].astype(np.int32).tolist()


def draw_mask(img, mask, color):
    # cv2.fillPoly(img, np.int32([mask + 0.5]), color)
    cv2.polylines(img,  np.int32([mask + 0.5]), isClosed=True, color=color, thickness=5)

## points : [N, 3]
## if N == 23: set bone=g_bones
## if N == 19: set bone=g_bones_19
def draw_keypoints(img, points, id, bone=None):
    base_color = np.asarray(COLORS[id])
    for i in range(points.shape[0]):
        if points.shape[1] == 3 and points[i,2] == 2: # visible 
            color = base_color
            color_tuple = (int(color[0]), int(color[1]), int(color[2])) 
            cv2.circle(img, tuple(points[i,0:2].astype(np.int32)), 9, color_tuple, thickness=-1)
        elif points.shape[1] == 2: 
            color = base_color
            color_tuple = (int(color[0]), int(color[1]), int(color[2])) 
            cv2.circle(img, tuple(points[i,0:2].astype(np.int32)), 9, color_tuple, thickness=-1)
    if bone is None:
        return

    for pair in bone: 
        i, dad_id = pair
        sun = points[i,0:2].astype(np.int32)
        if points.shape[1] == 3: 
            v_sun = points[i,2]
        else: 
            v_sun = 1 
        if dad_id < 0: 
            continue 
        dad = points[dad_id,0:2].astype(np.int32)
        if points.shape[1] == 3: 
            v_dad = points[dad_id,2].astype(np.int32)
        else: 
            v_dad = 1 
        if v_sun == 0 or v_dad == 0: 
            continue 
        color = base_color
        color_tuple = (int(color[0]), int(color[1]), int(color[2])) 
        cv2.line(img, tuple(sun), tuple(dad), color_tuple, thickness=4)

def draw_box(img, box, colorid):
    x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3] 
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2),int(y2)), tuple(COLORS[colorid]), thickness=3)


# This function is used to reproduce BamaPig2D visualization results in the 
# Supplementary Fig. 4b of the paper. 
def visualize_BamaPig2D(datafolder, output_folder,image_ids_to_rend=[0]): 
    with open(datafolder + "/annotations/eval_pig_cocostyle.json", 'r') as f: 
        annos_eval = json.load(f) 
    with open(datafolder + "/annotations/train_pig_cocostyle.json", 'r') as f: 
        annos_train = json.load(f)
    annotations = annos_eval["annotations"] + annos_train["annotations"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for imgid in image_ids_to_rend:
        img = cv2.imread(datafolder + "/images/{:06d}.png".format(imgid))
        colorid = 0 
        for A in annotations: 
            if A['image_id'] != imgid: 
                continue 
            box = np.asarray(A['bbox'])
            keypoints = np.asarray(A['keypoints'])
            masks = A['segmentation']
            draw_box(img, box, colorid)
            for m in masks: 
                m = np.asarray(m)
                m = m.reshape([-1,2])
                draw_mask(img, m, COLORS[colorid])
            draw_keypoints(img, keypoints.reshape([-1,3]), colorid, g_bones)
            colorid += 1
        cv2.imwrite(output_folder + "/{:06d}.png".format(imgid), img)
        print("write image ", imgid)


def draw_vis_2D(BamaPig2D_folder):
    with open(BamaPig2D_folder + "/annotations/eval_pig_cocostyle.json", 'r') as f: 
        annos_eval = json.load(f)
    with open(BamaPig2D_folder + "/annotations/train_pig_cocostyle.json", 'r') as f: 
        annos_train = json.load(f)  
    Total = 11504 * 19 
    part_levels = [ 
        [0,17,18], # trunk
        [1,2,3,4], # head 
        [5,6,7,8,9,10,11,12,13,14,15,16] # limbs
    ]

    Vis = np.zeros(3)
    for A in (annos_train['annotations'] + annos_eval['annotations']): 
        keypoints = np.asarray(A['keypoints'])
        keypoints = keypoints.reshape([-1,3])
        for k in range(3): 
            visible = (keypoints[part_levels[k],2] > 1).sum() 
            Vis[k] += visible 
    Vis[0] /= (11504 * 3)
    Vis[1] /= (11504 * 4)
    Vis[2] /= (11504 * 12)

    mpl.rc('font', family='Arial')
    fig = plt.figure(figsize=(1.2, 1.4)) 
    colormaps = np.loadtxt("colormaps/tab20c.txt") / 255
    part_level_names = ["Trunk", "Head", "Limbs"]
    xs = np.asarray([0,1,2])
    plt.bar(xs, Vis, color=colormaps[0], width=0.6, edgecolor=(0,0,0), lw=0.5)
    plt.ylim(0,0.5) 
    plt.xticks(xs, part_level_names, fontsize=7)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5], [0,10,20,30,40,50], fontsize=7)
    plt.ylabel("Percentage of Visible\n Keypoints (%)", fontsize=7)
    ax = fig.get_axes()[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for line in ["bottom", "left", "right"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.savefig("output/supp_fig_4d.png", dpi=1000, bbox_inches='tight', pad_inches=0)
    # plt.savefig("output/supp_fig_4d.svg", dpi=1000, bbox_inches='tight', pad_inches=0) # uncomment this line to generate vector image file.


if __name__ == "__main__": 
    # You may change the BamaPig2D_folder to your own path of BamaPig2D
    BamaPig2D_folder = "H:/examples/BamaPig2D/"

    # [0,100,1000,2000,3000,3009,3100,3300] These images are used for generating Supplementary Fig. 4b. 
    visualize_BamaPig2D(BamaPig2D_folder, "output", [0,100,1000,2000,3000,3009,3100,3300])

    draw_vis_2D(BamaPig2D_folder)

