import numpy as np
import yaml
import os
import random
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import threading

class LESS():
    def __init__(self,config,config_LESS,args,split='train', label_directory='scribbles',target_directory='LESS') -> None:
        self.config = config
        self.split = split
        self.label_directory = label_directory
        self.true_directory = 'labels'
        self.target_directory = config_LESS['target_directory']
        self.t = config_LESS['RANSAC']['scans_per_subsequence']
        self.l_grid = config_LESS['RANSAC']['l_grid']
        self.dist = config_LESS['RANSAC']['dist']
        self.iter_max = config_LESS['RANSAC']['iter_max']
        self.percent = config_LESS['RANSAC']['percent']
        self.d = config_LESS['cluster']['d']
        self.tic = 0
        self.Save = config_LESS['save']
        self.OOM_bar = config_LESS['RANSAC']['OOM_bar']
        self.save_file_number = config_LESS['save_file_number']
        self.seq_to_process = int(args.solve_seq)
        self.start_number_in_one_seq = config_LESS['start_number_in_one_seq']
        self.print_detail = config_LESS['print_detail']
        self.add_list = config_LESS['add_list']
        self.z_sort_bar = config_LESS['RANSAC']['z_sort_bar']
        self.count_classify_error = np.zeros([1,20])
        self.class_name_map = config['labels']

    def map_label(self,label, map_dict):
        maxkey = 0
        for key, data in map_dict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in map_dict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        return lut[label]

    def run_LESS(self):
        root_dir = self.config['root_dir']
        cnt_seq = 0
        for seq in self.config['split'][self.split]:
            if self.seq_to_process >= 0:
                if seq != self.seq_to_process:
                    continue
            seq = '{0:02d}'.format(int(seq))
            label_class_dir = os.path.join(root_dir, seq,self.target_directory, 'group')
            labels_dir = os.path.join(root_dir, seq,self.target_directory, 'labels')
            if not os.path.exists(label_class_dir):
                    os.makedirs(label_class_dir)
            if not os.path.exists(labels_dir):
                    os.makedirs(labels_dir)   
            
            Calculate_Result = np.zeros([19,5],dtype=np.int64)      # row represents class_name, column is [Weak_Wrong, Propogated_Wrong, Weak_Num, Propogated_Num, All_Point]


            lidar_dir = os.path.join(root_dir, seq, 'velodyne')
            lidar_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(lidar_dir)) for f in fn if f.endswith('.bin')]
            self.seq_len = len(lidar_path)
            self.scans_cnt = 0
            label_dir = os.path.join(root_dir, seq, self.label_directory)
            true_label_dir = os.path.join(root_dir,seq,self.true_directory)
            label_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_dir)) for f in fn if f.endswith('.label')]
            true_label_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(true_label_dir)) for f in fn if f.endswith('.label')]
            # assert (len(lidar_path) == len(label_path))
            # label_paths.extend(label_path)
            LESS_label_dir = os.path.join(root_dir, seq, self.target_directory,'labels')
            LESS_labels_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(LESS_label_dir)) for f in fn if f.endswith('.label')]
            # assert (len(lidar_paths) == len(LESS_labels_paths))
            label_group_dir = os.path.join(root_dir, seq, self.target_directory,'group')
            label_group_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_group_dir)) for f in fn if f.endswith('.label')]
            pose_path = os.path.join(root_dir,seq,'poses.txt')
            self.fusion(lidar_path=lidar_path,pose_path=pose_path,label_path=label_path,t=self.t,l_grid=self.l_grid,dist=self.dist,iter_max=self.iter_max,percent=self.percent,d=self.d,seq=seq,true_label_path=true_label_path,LESS_labels_path=LESS_labels_path,label_group_paths=label_group_paths,Calculate_Result=Calculate_Result)
        
        return 1
    

    # t represent t scans to be combined with
    # l_grid means we use l*l grid to run RANSAC
    def fusion(self,lidar_path,pose_path,label_path,t,l_grid,dist,iter_max,percent,d,seq,true_label_path,LESS_labels_path,label_group_paths,Calculate_Result):     
        lidar_path.sort()
        label_path.sort()
        true_label_path.sort()
        LESS_labels_path.sort()
        label_group_paths.sort()
        cnt = 0
        for lidar_file,label_file,true_label_file,LESS_label_file,label_group_file in zip(lidar_path,label_path,true_label_path,LESS_labels_path,label_group_paths):
            # start a new subsequence 
            lidar_xyz_unified = np.array([[0,0,0]],dtype=np.float32)
            label_unified = np.array([[0]])
            true_label_unified = np.array([[0]])

            # read the pose and aggregate scans
            # pose_SE3 = np.array([[r11,r12,r13,t14],[r21,r22,r23,t24],[r31,r32,r33,t34],[0,0,0,1]])
            # pose_SE3_inv = np.linalg.inv(pose_SE3)
            # (r11,r12,r13,t14),(r21,r22,r23,t24),(r31,r32,r33,t34),(_,_,_,_)=pose_SE3_inv
            lidar_data = np.fromfile(lidar_file,dtype=np.float32)
            lidar_data = lidar_data.reshape((-1,4))
            lidar_x = lidar_data[:,0]
            lidar_y = lidar_data[:,1]
            lidar_z = lidar_data[:,2]
            label = np.fromfile(label_file, dtype=np.int32)
            label = label.reshape((-1)) & 0xFFFF
            label = self.map_label(label, self.config['learning_map'])
            true_label = np.fromfile(true_label_file, dtype=np.int32)
            true_label = true_label.reshape((-1)) & 0xFFFF
            true_label = self.map_label(true_label, self.config['learning_map'])
            # Matrix_RT = np.array([r11,r12,r13,t11],[r21,r22,r23,t12],[r31,r32,r33,t13],[0,0,0,1])
            LESS_label = np.fromfile(LESS_label_file, dtype=bool)
            LESS_label = LESS_label.reshape((-1, 20))
            label_group = np.fromfile(label_group_file,dtype=np.int8)
            # lidar_x = lidar_x*r11 +r12*lidar_y+r13*lidar_z + t14
            # lidar_y = lidar_x*r21 +r22*lidar_y+r23*lidar_z + t24
            # lidar_z = lidar_x*r31 +r32*lidar_y+r33*lidar_z + t34

            lidar_x = np.expand_dims(lidar_x,axis=1)
            lidar_y = np.expand_dims(lidar_y,axis=1)
            lidar_z = np.expand_dims(lidar_z,axis=1)
            label   = np.expand_dims(label,  axis=1)
            lidar_xyz = np.concatenate((lidar_x,lidar_y,lidar_z),axis=1)
            lidar_xyz_unified = np.concatenate((lidar_xyz_unified,lidar_xyz),axis=0)        # OK
            label_unified = np.concatenate((label_unified,label),axis=0)        # OK
            true_label_unified = np.concatenate((true_label_unified,np.expand_dims(true_label,axis=1)),axis=0)        # OK
            lidar_xyz_unified = lidar_xyz_unified[1:,:]     # get all points in one axis
            label_unified = label_unified[1:,:]
            true_label_unified = true_label_unified[1:,:]


            # row represents class_name, column is [Weak_Wrong, Propogated_Wrong, Weak_Num, Propogated_Num]
            for i in range(1,20):
                index = np.where(true_label_unified==i)[0]
                true_label_in_index = true_label_unified[index]
                LESS_label_in_index = LESS_label[index]
                label_group_in_index = label_group[index]
                propogated_index = np.where(label_group_in_index==2)[0]
                weak_inedx = np.where(label_group_in_index==3)[0]
                weak_label = LESS_label_in_index[weak_inedx]
                weak_label[:,0] = False
                true_label_in_weak_index = true_label_in_index[weak_inedx]
                weak_label_right = weak_label[np.arange(len(true_label_in_weak_index)), true_label_in_weak_index.ravel()].reshape(-1, 1)
                Calculate_Result[i-1,0] += weak_label_right.shape[0] - weak_label_right.sum()
                Calculate_Result[i-1,2] += weak_label_right.shape[0]

                propogated_label = LESS_label_in_index[propogated_index]
                propogated_label = np.where(propogated_label==1)[1]
                propogated_label = propogated_label[propogated_label!=0]
                true_label_in_propogated_index = true_label_in_index[propogated_index]
                True_propogated_point = np.count_nonzero(np.expand_dims(propogated_label,axis=1) == true_label_in_propogated_index)
                Calculate_Result[i-1,1] += true_label_in_propogated_index.shape[0] - True_propogated_point
                Calculate_Result[i-1,3] += true_label_in_propogated_index.shape[0]        
                Calculate_Result[i-1,4] += index.shape[0]
            cnt += 1
            print('seq ',seq,'scan ',cnt,' finish')
        Result_list = []
        for i in range(1,20):
            class_name = self.class_name_map[i]
            one_line_list = [class_name,Calculate_Result[i-1,0], Calculate_Result[i-1,1], Calculate_Result[i-1,2], Calculate_Result[i-1,3], Calculate_Result[i-1,4],Calculate_Result[i-1,0]/Calculate_Result[i-1,2],Calculate_Result[i-1,1]/Calculate_Result[i-1,3]]
            Result_list.append(one_line_list)
        columns = ["Class Name", "Weak_Wrong", "Propogated_Wrong", "Weak_Num", "Propogated_Num", "Points_Num",'Weak_Wrong_Percent','Propogated_Wrong_Percent']
        df = pd.DataFrame(Result_list, columns=columns)
        excel_filename = "seq "+seq+".xlsx"
        df.to_excel(excel_filename, index=False)


if __name__ == '__main__':
    # config_path = 'config/LESS_dist3.yaml'
    config_path = 'config/LESS.yaml'
    dataset_config_path = 'config/dataset/semantickitti.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))
    parser = argparse.ArgumentParser()
    parser.add_argument('--solve_seq', default=-1,help='-1 means all,0-10 should be input')
    args = parser.parse_args()
    less = LESS(config['dataset'],config['LESS'],args=args)
    less.run_LESS()

    
    