import numpy as np 
from bodymodel_np import BodyModelNumpy

# This function shows how to use "label_mesh" and "label_3d" annotations to get the 
# final label_mix gt annotation. 
# This function will help you to understand Supplementary Fig. 6 in the paper.  
def assemble():
    folder1 = "label_mesh" 
    order1 = [0,1,2,3]
    folder2 = "label3d" 
    order2 = [0,1,2,3]
    folder3 = "label_mix"

    for i in range(70):
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