import numpy as np 
import os 

# 23 keypoints names
# only 19 have not `none` name 
# This is an issue rooted in history
# in fact, one can slightly modify the dataset to remove these 4 non-used keypoints. 
g_jointnames = [
    "nose", 
    "l_eye", 
    "r_eye", 
    "l_ear", 
    "r_ear", 
    "l_shoulder", 
    "r_shoulder", 
    "l_elbow", 
    "r_elbow", 
    "l_paw", 
    "r_paw", 
    "l_hip", 
    "r_hip", 
    "l_knee", 
    "r_knee", 
    "l_foot", 
    "r_foot", 
    "none", # not used. 
    "tail", 
    "none", # not used
    "center", 
    "none", # not used
    "none"  # not used
]

g_bones = [ # bone structure for 23 keypoints 
    [0,1],
    [0,2],
    [1,2],
    [1,3], 
    [2,4],
    [0,20], 
    [20,18],
    [20,5],
    [5,7],
    [7,9],
    [20,6],
    [6,8],
    [8,10],
    [18,11],
    [11,13], 
    [13,15],
    [18,12],
    [12,14],
    [14,16]
]

g_bones_19 = [ # bone structure for the final 19 valid keypoints  
    [0,1],
    [0,2],
    [1,2],
    [1,3], 
    [2,4],
    [0,18], 
    [18,17],
    [18,5],
    [5,7],
    [7,9],
    [18,6],
    [6,8],
    [8,10],
    [17,11],
    [11,13], 
    [13,15],
    [17,12],
    [12,14],
    [14,16]
]

# group name of each keypoint
# e.g. nose, eyes, ears are all Head part. 
g_groupnames = [
    "Head", 
    "Head", 
    "Head", 
    "Head", 
    "Head", 
    "L_arm", 
    "R_arm", 
    "L_arm", 
    "R_arm", 
    "L_arm", 
    "R_arm", 
    "L_leg", 
    "R_leg", 
    "L_leg", 
    "R_leg", 
    "L_leg", 
    "R_leg", 
    "none", 
    "Tail", 
    "none", 
    "Center", 
    "none", 
    "none"
]

# indices of not `none` keypoints 
g_all_parts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20]

# some other part divide for flexible usage. 
g_head = [0,1,2,3,4]
g_left_front_leg = [5,7,9]
g_right_front_leg = [6,8,10]
g_left_hind_leg = [11,13,15]
g_right_hind_leg = [12,14,16]
g_legs = g_left_front_leg + g_left_hind_leg + g_right_front_leg + g_right_hind_leg
g_leg_level1 = [5,6,11,12]
g_leg_level2 = [7,8,13,14]
g_leg_level3 = [9,10,15,16]
g_trunk = [20,18]
g_pig_ids_for_eval = [0,1,2,3]

# This function is used to load all 3D labeled data. 
def load_joint23(folder, start = 0, step = 25, num = 70, order=[0,1,2,3]):
    all_data = [] 
    for i in range(num):
        frameid = start + step * i 
        single_frame = [0,1,2,3] 
        for pid in range(4):
            filename = folder + "/pig_{}_frame_{:06d}.txt".format(pid, frameid)
            data = np.loadtxt(filename) 
            index = order[pid]
            single_frame[index] = data 
        all_data.append(single_frame) 
    all_data = np.asarray(all_data) 

    return all_data 