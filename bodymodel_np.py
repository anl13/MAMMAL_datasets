import numpy as np 
import os 
from scipy.spatial.transform import Rotation

'''
To use this model, please first download the model files from (TODO link) and 
put them as the folder `./PIG_Model/` (you can rename it as you like and change the 
`model_folder` variable below. ) 
'''
class BodyModelNumpy(object):
    def __init__(self, model_folder = "./PIG_Model/"):
        self.model_folder = model_folder
        self.vertices, self.parents, self.joints, self.weights, self.faces = \
            self.readmodel(self.model_folder)
        self.joint_num = self.joints.shape[0] 

        # [target joint, type, source index, weight]
        self.optimize_pair = [
            [ 0, 1, 10895, 1 ], # nose
            [ 1, 1, 938, 1 ], # left eye
            [ 2, 1, 6053, 1 ], # right eye
            [ 3, 1, 1368, 1 ], # left ear
            [ 4, 1, 6600, 1 ], # right ear
            [ 5, 0, 15, 1 ], # left shouder
            [ 6, 0, 7, 1 ], # right shoulder
            [ 7, 0, 16, 1 ], # left elbow
            [ 8, 0, 8, 1 ], # right elbow
            [ 9, 0, 17, 1 ], # left paw
            [ 10, 0, 9, 1 ], # right paw
            [ 11, 0, 56, 1 ], # left hip
            [ 12, 0, 40, 1 ], # right hip
            [ 13, 0, 57, 1 ], # left knee
            [ 14, 0, 41, 1 ], # right knee
            [ 15, 0, 58, 1 ], # left foot
            [ 16, 0, 42, 1 ], # right foot
            [ 17, -1, 0, 0], # neck(not use)
            [ 18, 1, 7903, 1 ], # tail 
            [ 19, -1, 0,0], #wither (not use) 
            [ 20, 0, 2, 1 ], # center
            [ 21, -1, 0,0], # tail middle (not use) 
            [ 22, -1, 0,0] # tail end (not use)
        ]

        self.translation = np.zeros(3, dtype=np.float32) 
        self.poseparam = np.zeros([self.joint_num, 3], dtype=np.float32)
        self.scale = 1
        self.posed_joints = self.joints.copy() 
        self.posed_vertices = self.vertices.copy() 

    # function to parse the pose param txt file. 
    def readstate(self,filename):
        states = np.loadtxt(filename)
        translation = states[0:3] # the first 3 dim is global translation 
        scale = states[-1] # the last param is global scale 
        poseparam = states[3:-1].reshape([-1,3]) # others are pose params, in which the first 3-dim rotation is global rotation. 
        
        self.translation = translation
        self.poseparam = poseparam
        self.scale = scale 
        return translation, poseparam, scale


    def readmodel(self, model_folder):
        vertices_np = np.loadtxt(os.path.join(model_folder, "vertices.txt"))
        parents_np = np.loadtxt(os.path.join(model_folder, "parents.txt")).squeeze()
        joints_np = np.loadtxt(os.path.join(model_folder, "t_pose_joints.txt"))
        weights = np.zeros((vertices_np.shape[0], parents_np.shape[0]))
        _weights = np.loadtxt(os.path.join(model_folder, "skinning_weights.txt"))
        for i in range(_weights.shape[0]):
            jointid = int(_weights[i,0])
            vertexid = int(_weights[i,1])
            value = _weights[i,2]
            weights[vertexid, jointid] = value 
        faces_vert = np.loadtxt(os.path.join(model_folder, "faces_vert.txt"))
        
        return vertices_np, parents_np.astype(np.int), joints_np, weights, faces_vert.astype(np.int)


    def poseparam2Rot(self, poseparam):
        Rot = np.zeros((poseparam.shape[0], 3, 3), dtype=np.float32)
        r_tmp1 = Rotation.from_euler('ZYX', poseparam[0], degrees=False)
        Rot[0] = r_tmp1.as_matrix()
        r_tmp2 = Rotation.from_rotvec(poseparam[1:])
        Rot[1:] = r_tmp2.as_matrix()
        
        return Rot


    def write_obj(self, filename):
        with open(filename, 'w') as fp:
            for v in self.posed_vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


    def joint_Rot(self, Rot):
        skinmat = np.repeat(np.eye(4, dtype=np.float32).reshape(1, 4, 4), repeats=self.joints.shape[0], axis=0)
        skinmat[0, :3, :3] = Rot[0]
        skinmat[0, :3, 3] = self.joints[0]
        for jind in range(1, self.joints.shape[0]):
            skinmat[jind, :3, :3] = Rot[jind]
            skinmat[jind, :3, 3] = self.joints[jind] - self.joints[self.parents[jind]]
            skinmat[jind] = np.matmul(skinmat[self.parents[jind]], skinmat[jind])

        joints_final = skinmat[:, :3, 3].copy()
        joints_deformed = np.zeros((self.joints.shape[0], 4), dtype=np.float32)
        for jind in range(self.joints.shape[0]):
            joints_deformed[jind, :3] = np.matmul(skinmat[jind, :3, :3], self.joints[jind])
        skinmat[:, :, 3] = skinmat[:, :, 3] - joints_deformed
        return skinmat[:, :3, :], joints_final

    def regress_verts(self, skinmat):
        vertsmat = np.tensordot(self.weights, skinmat, axes=([1], [0]))
        verts_final = np.zeros((self.vertices.shape[0], 3), dtype=np.float32)
        for vind in range(self.vertices.shape[0]):
            verts_final[vind] = np.matmul(vertsmat[vind, :, :3], self.vertices[vind]) + vertsmat[vind, :, 3]
        
        return verts_final

    def forward(self, pose, trans=np.zeros(3, dtype=np.float32), scale=1):
        rot = self.poseparam2Rot(pose)
        skinmat, joints_final = self.joint_Rot(rot)
        self.posed_joints = joints_final * scale + trans 
        verts = self.regress_verts(skinmat) 
        self.posed_vertices = verts * scale + trans 
        return self.posed_vertices, self.posed_joints

    # This function regress all the 23 keypoints with non-used ones set as zero.
    # return: np.ndarray (float32), [23,3]
    def regress_keypoints(self):
        keynum = len(self.optimize_pair)
        keypoints = np.zeros((keynum, 3), dtype=np.float32)
        for i in range(keynum):
            if self.optimize_pair[i][1] == 0:
                keypoints[i] = self.posed_joints[self.optimize_pair[i][2]]
            elif self.optimize_pair[i][1] == 1:
                keypoints[i] = self.posed_vertices[self.optimize_pair[i][2]]
        return keypoints

    # This function regress 19 used keypoints from the model. 
    # return: np.ndarray (float32), [19,3]
    def regress_keypoints_pack(self):
        keynum = 19 
        keypoints = np.zeros((keynum, 3), dtype=np.float32)
        non_zero = 0
        for i in range(len(self.optimize_pair)): 
            if self.optimize_pair[i][1] == 0:
                keypoints[non_zero] = self.posed_joints[self.optimize_pair[i][2]]
                non_zero += 1 
            elif self.optimize_pair[i][1] == 1:
                keypoints[non_zero] = self.posed_vertices[self.optimize_pair[i][2]]
                non_zero += 1
        return keypoints 

if __name__ == "__main__":
    bm = BodyModelNumpy()
    BamaPig3D_path = "H:/examples/BamaPig3D/"
    for k in range(25):
        frameid = k * 25 
        for pid in range(4):
            filename = BamaPig3D_path + "label_pose_params/pig_{}_frame_{:06d}.txt".format(pid, frameid)
            trans, poseparam, scale = bm.readstate(filename) 
            # V: [11239, 3]; 
            # J: [62, 3]
            V, J = bm.forward(poseparam, trans=trans, scale = scale) 
            # keypoints: [23, 3]
            keypoints = bm.regress_keypoints() 
            
            # you can do something here 

            
