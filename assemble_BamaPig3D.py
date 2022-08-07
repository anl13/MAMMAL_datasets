import numpy as np 

# This function shows how to use "label_mesh" and "label_3d" annotations to get the 
# final label_mix gt annotation. 
# This function will help you to understand Supplementary Fig. 8 in the paper. 
# 
BamaPig3D_path = "H:/examples/BamaPig3D/" 
def assemble():
    folder1 = BamaPig3D_path + "label_mesh" 
    order1 = [0,1,2,3]
    folder2 = BamaPig3D_path + "label_3d" 
    order2 = [0,1,2,3]
    folder3 = BamaPig3D_path + "label_mix"

    for i in range(65,66):
        frameid = 25 * i
        all_data1 = np.zeros((4, 23, 3))  
        all_data2 = np.zeros((4, 23, 3))
        all_data3 = np.zeros((4, 23, 3))
        for pid in range(4):
            filename1 = folder1 + "/pig_{}_frame_{:06d}.txt".format(pid, frameid) 
            data = np.loadtxt(filename1) 
            all_data1[order1[pid]] = data 
            filename2 = folder2 + "/pig_{}_frame_{:06d}.txt".format(pid, frameid) 
            data = np.loadtxt(filename2) 
            all_data2[pid] = data 
        all_data3 = all_data2.copy() 
        for pid in range(4): 
            for jid in range(23): 
                if np.linalg.norm(all_data3[pid, jid]) == 0 and np.linalg.norm(all_data1[pid,jid]) > 0: 
                    all_data3[pid, jid] = all_data1[pid, jid] 
        
        for pid in range(4): 
            filename = folder3 +"/pig_{}_frame_{:06d}.txt".format(pid, frameid) 
            np.savetxt(filename, all_data3[pid])

if __name__ == "__main__":
    assemble() 