import numpy as np 
import os 
import shutil 
import json 
import pickle 
from tqdm import tqdm 

g_all_parts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20]

## This function build a slim version of BamaPig3D dataset with most data encoded as .pkl binary files. 
def clean_label3d():
    subfolders = ["label_3d","label_mix", "label_mesh"]
    ## src_folder is the path of your own BamaPig3D folder. 
    src_folder = "H:/examples/BamaPig3D/"
    ## tgt_folder is the path of your output. 
    tgt_folder = "H:/examples/BamaPig3D_pure_pickle/" 
    for sub in subfolders:
        all_data = [] 
        for k in range(70):
            framedata = [] 
            frameid = k * 25 
            for pid in range(4): 
                srcfile = src_folder + sub + "/pig_{}_frame_{:06d}.txt".format(pid, frameid)
                mat = np.loadtxt(srcfile) 
                framedata.append(mat) 
            all_data.append(framedata)

        all_data_np = np.asarray(all_data)
        all_data_np = all_data_np[:,:,g_all_parts,:]
        print(all_data_np.shape) 
        with open(tgt_folder + sub + ".pkl", 'wb') as f: 
            pickle.dump(np.asarray(all_data), f) 

def clean_pose_params():
    src_folder = "H:/examples/BamaPig3D/label_pose_params/"
    tgt_folder = "H:/examples/BamaPig3D_pure_pickle/" 
    g_r = np.zeros([70,4,3])
    g_t = np.zeros([70,4,3])
    g_scale = np.zeros([70,4])
    g_aa = np.zeros([70,4,62,3])
    for k in range(70):
        frameid = k * 25 
        for pid in range(4): 
            srcfile = src_folder + "/pig_{}_frame_{:06d}.txt".format(pid, frameid)
            data = np.loadtxt(srcfile) 
            scale = data[-1]
            trans = data[0:3]
            poseparams = data[3:-1].reshape([-1,3])
            rotation = poseparams[0].copy() 
            poseparams[0] = 0
            g_r[k,pid] = rotation 
            g_t[k,pid] = trans 
            g_scale[k,pid] = scale 
            g_aa[k,pid] = poseparams 

    param_dict = { 
        "global_rotations": g_r, 
        "global_translations": g_t, 
        "global_scales": g_scale, 
        "joint_rotations": g_aa,
        "description" : "global_rotations is [70,4,3] dim contains euler angle of 4 pigs in 70 frames which represent global orientation. \
            global_translations is [70,4,3]. global_scale is [70,4]. joint_rotations is [70,4,62,3] contains axis-angle format joint rotation for 62 joints of each pig. "
    }
    with open(tgt_folder + "label_pose_params.pkl", 'wb') as f: 
        pickle.dump(param_dict, f) 

def clean_label_images():
    BamaPig3D_folder = "H:/examples/BamaPig3D/"
    output_folder = "H:/examples/BamaPig3D_pure_pickle/"
    camids = [0,1,2,5,6,7,8,9,10,11]
    all_2D_points = np.zeros([70,10,4,19,3])
    mask_all_frames = []
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
                # for k in range(4):
                #     print(len(mask_4[k]))
            mask_all_views.update({camid:mask_4})
        mask_all_frames.append(mask_all_views)
    with open(output_folder + "label_keypoints2d.pkl", 'wb') as f: 
        pickle.dump(all_2D_points, f) 
    with open(output_folder + "label_silhouettes2d.pkl", 'wb') as f: 
        pickle.dump(mask_all_frames, f) 



if __name__ == "__main__":
    clean_label3d() 
    # clean_pose_params()
    clean_label_images()