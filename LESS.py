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
import open3d as p3d
    
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
            cluster_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'cluster')
            if not os.path.exists(label_class_dir):
                    os.makedirs(label_class_dir)
            if not os.path.exists(labels_dir):
                    os.makedirs(labels_dir)   
                        
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
            pose_path = os.path.join(root_dir,seq,'poses.txt')
            self.fusion(lidar_path=lidar_path,pose_path=pose_path,label_path=label_path,t=self.t,l_grid=self.l_grid,dist=self.dist,iter_max=self.iter_max,percent=self.percent,d=self.d,seq=seq,true_label_path=true_label_path)
        
        return 1
    

    # t represent t scans to be combined with
    # l_grid means we use l*l grid to run RANSAC
    def fusion(self,lidar_path,pose_path,label_path,t,l_grid,dist,iter_max,percent,d,seq,true_label_path):     
        pose_file = open(pose_path) 
        lidar_path.sort()
        label_path.sort()
        true_label_path.sort()
        cnt = 0
        # label_file_exist = os.listdir(label_path[0].replace('scribbles',self.target_directory+'/group').split('/000000.label')[0])
        label_file_exist = os.listdir(label_path[0].replace('scribbles',self.target_directory+'/group').split('/000000.label')[0])
        label_file_exist.sort()
        if not self.Save:
            if self.start_number_in_one_seq != 0:
                len_file_exist = self.start_number_in_one_seq
            else:
                len_file_exist = len(label_file_exist)
                pass
        else:
            len_file_exist = self.save_file_number
        # len_file_exist = 4000

        for lidar_file,label_file,true_label_file in zip(lidar_path,label_path,true_label_path):
            
            # pass the existed files
            if self.scans_cnt < len_file_exist:
                self.scans_cnt += 1
                pose = pose_file.readline()
                continue
            
            # start a new subsequence 
            if cnt == 0: 
                file_list = []
                file_len_list = []
                lidar_xyz_unified = np.array([[0,0,0]],dtype=np.float32)
                label_unified = np.array([[0]])
                true_label_unified = np.array([[0]])

            # read the pose and aggregate scans
            pose = pose_file.readline()
            # pose_ = self.read_specific_line_from_file(pose_file,self.scans_cnt)
            r11, r12, r13, t14, r21, r22, r23, t24, r31, r32, r33, t34 = map(float, pose.split(' '))
            # pose_SE3 = np.array([[r11,r12,r13,t14],[r21,r22,r23,t24],[r31,r32,r33,t34],[0,0,0,1]])
            # pose_SE3_inv = np.linalg.inv(pose_SE3)
            # R_inv = np.linalg.inv(R)
            # T_inv = -np.dot(R_inv, T)
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
            original_lidar_x = lidar_x.copy()
            original_lidar_y = lidar_y.copy()
            original_lidar_z = lidar_z.copy()

            # lidar_x = original_lidar_x*r11 +r12*original_lidar_y+r13*original_lidar_z + t14
            # lidar_y = original_lidar_x*r21 +r22*original_lidar_y+r23*original_lidar_z + t24
            # lidar_z = original_lidar_x*r31 +r32*original_lidar_y+r33*original_lidar_z + t34

            # lidar_x = original_lidar_x*r11 +r21*original_lidar_y+r31*original_lidar_z - t14
            # lidar_y = original_lidar_x*r12 +r22*original_lidar_y+r32*original_lidar_z - t24
            # lidar_z = original_lidar_x*r13 +r23*original_lidar_y+r33*original_lidar_z - t34

            lidar_x -= t14
            lidar_y -= t24
            lidar_z -= t34

            lidar_x = np.expand_dims(lidar_x,axis=1)
            lidar_y = np.expand_dims(lidar_y,axis=1)
            lidar_z = np.expand_dims(lidar_z,axis=1)
            label   = np.expand_dims(label,  axis=1)
            lidar_xyz = np.concatenate((lidar_x,lidar_y,lidar_z),axis=1)
            lidar_xyz_unified = np.concatenate((lidar_xyz_unified,lidar_xyz),axis=0)        # OK
            label_unified = np.concatenate((label_unified,label),axis=0)        # OK
            true_label_unified = np.concatenate((true_label_unified,np.expand_dims(true_label,axis=1)),axis=0)        # OK

            file_list.append(label_file.split(self.label_directory+'/')[1])
            file_len_list.append(len(lidar_xyz_unified)-1)

            cnt += 1
            if cnt == t:
                label_save = np.zeros((lidar_xyz_unified.shape[0]-1,20),dtype=bool)
                group_save = np.zeros((lidar_xyz_unified.shape[0]-1,1),dtype=np.int32)
                label_class = np.expand_dims(np.zeros(lidar_xyz_unified.shape[0]-1,dtype=np.int8),axis=1)
                self.tic = time.time()
                lidar_xyz_unified = lidar_xyz_unified[1:,:]     # get all points in one axis
                label_unified = label_unified[1:,:]
                true_label_unified = true_label_unified[1:,:]
                # For visualization, save Origin points
                if self.Save:
                    # lidar_xyz_unified.tofile(str(lidar_xyz_unified.shape[0])+'_Origin.bin')
                    self.save_matrix_to_txt(lidar_xyz_unified,'points_with_ground.txt')
                y_max = np.max(lidar_xyz_unified[:,1])
                y_min = np.min(lidar_xyz_unified[:,1])
                x_max = np.max(lidar_xyz_unified[:,0])
                x_min = np.min(lidar_xyz_unified[:,0])
                flag_points_less_than_OOM = True
                cnt_OOM = 0
                lidar_xyz_unified_copy = lidar_xyz_unified
                dist = self.dist
                # to save all points with color
                # Points_with_rgb = np.zeros([lidar_xyz_unified.shape[0],6])
                # for i in range(0,20):
                #     Points_with_rgb[np.where(label==i)[0],3:6] = self.config['color_map'][i]
                # Points_with_rgb[:,0:3] = lidar_xyz_unified
                # self.save_matrix_to_txt(Points_with_rgb,'all_points_with_ground_truth.txt')
                while flag_points_less_than_OOM:
                    lidar_xyz_unified = lidar_xyz_unified_copy.copy()
                    self.cnt_all_ground_block = 0
                    self.punish_error = 0
                    self.punish_error = 0

                    self.propogated_error = 0 
                    self.punish_error_cluster = 0
                    self.propogated_error_cluster = 0 
                    self.cnt_all_group_clutser = 0
                    # run RANSAC for each block
                    for i in range(int(np.floor(x_min/l_grid)),int(np.ceil(x_max/l_grid))):
                        for j in range(int(np.floor(y_min/l_grid)),int(np.ceil(y_max/l_grid))):
                            self.cnt_all_ground_block += 1
                            grid_x_min = i*l_grid
                            grid_x_max = (i+1)*l_grid
                            grid_y_min = j*l_grid
                            grid_y_max = (j+1)*l_grid
                            indice = np.logical_and(np.logical_and(lidar_xyz_unified[:,1]>grid_y_min ,lidar_xyz_unified[:,1]<grid_y_max),np.logical_and(lidar_xyz_unified[:,0]>grid_x_min ,lidar_xyz_unified[:,0]<grid_x_max ) ) 
                            if np.where(indice==True)[0].shape[0] >= 3:
                                lidar_xyz_unified[indice],label_save[indice],label_class[indice] = self.RANSAC(lidar_xyz_unified=lidar_xyz_unified[indice],label_save=label_save[indice],label_class=label_class[indice],label_unified=label_unified[indice],true_label_unified=true_label_unified[indice],dist=dist,iter_max=iter_max,percent=percent,group_save=group_save[indice])
                            group_save[indice] = - self.cnt_all_ground_block                   
                    # avoid OOM
                    if len(np.unique(np.where(lidar_xyz_unified!=np.array([0,0,0])[0]))) < self.OOM_bar:
                        flag_points_less_than_OOM = False
                    else:
                        cnt_OOM += 1
                        if cnt_OOM >= 5:
                            cnt_OOM = 0
                            dist += 0.05 
                        print('points shape is (',len(np.unique(np.where(lidar_xyz_unified!=np.array([0,0,0])[0]))),', 3), May cause OOM',flush=True)
                
                cnt = 0
                self.cluster(lidar_xyz_unified=lidar_xyz_unified,labels=label_unified,true_label_unified=true_label_unified,label_save=label_save,label_class=label_class,file_list=file_list,file_len_list=file_len_list,d=d,seq=seq,group_save=group_save)
                print('Ground: propogated error is',self.propogated_error,'/',self.cnt_all_ground_block,'; weak error is',self.punish_error,'/',self.cnt_all_ground_block)
                print('Cluster: propogated error is',self.propogated_error_cluster,'/',self.cnt_all_group_clutser,'; weak error is',self.punish_error_cluster,'/',self.cnt_all_group_clutser)                
                # if only save points, don't need to process next subsequence 
                if self.Save:
                    exit()

            # for a squence, but not enough for x scans, which means the end of a sequence
            # run the step of thw whole subsequence anyway
            # just a copy of code above 
            elif label_file == label_path[len(label_path)-1]:
                label_save = np.zeros((lidar_xyz_unified.shape[0]-1,20),dtype=bool)
                label_class = np.expand_dims(np.zeros(lidar_xyz_unified.shape[0]-1,dtype=np.int8),axis=1)
                self.tic = time.time()
                lidar_xyz_unified = lidar_xyz_unified[1:,:]     # get all points in one axis
                label_unified = label_unified[1:,:]

                # For visualization, save Origin points
                if self.Save:
                    # lidar_xyz_unified.tofile(str(lidar_xyz_unified.shape[0])+'_Origin.bin')
                    self.save_matrix_to_txt(lidar_xyz_unified,'points_with_ground.txt')
                y_max = np.max(lidar_xyz_unified[:,1])
                y_min = np.min(lidar_xyz_unified[:,1])
                x_max = np.max(lidar_xyz_unified[:,0])
                x_min = np.min(lidar_xyz[:,0])
                flag_points_less_than_OOM = True
                cnt_OOM = 0
                lidar_xyz_unified_copy = lidar_xyz_unified
                dist = self.dist
                while flag_points_less_than_OOM:
                    lidar_xyz_unified = lidar_xyz_unified_copy.copy()
                    self.cnt_all_ground_block = 0
                    self.punish_error = 0
                    self.propogated_error = 0 
                    self.punish_error_cluster = 0
                    self.propogated_error_cluster = 0 
                    self.cnt_all_group_clutser = 0
                    # run RANSAC for each block
                    for i in range(int(np.round(x_min/l_grid)),int(np.round(x_max/l_grid))):
                        for j in range(int(np.round(y_min/l_grid)),int(np.round(y_max/l_grid))):
                            self.cnt_all_ground_block += 1
                            grid_x_min = i*l_grid
                            grid_x_max = (i+1)*l_grid
                            grid_y_min = j*l_grid
                            grid_y_max = (j+1)*l_grid
                            indice = np.logical_and(np.logical_and(lidar_xyz_unified[:,1]>grid_y_min ,lidar_xyz_unified[:,1]<grid_y_max),np.logical_and(lidar_xyz_unified[:,0]>grid_x_min ,lidar_xyz_unified[:,0]<grid_x_max ) ) 
                            if np.where(indice==True)[0].shape[0] >= 3:
                                lidar_xyz_unified[indice],label_save[indice],label_class[indice] = self.RANSAC(lidar_xyz_unified=lidar_xyz_unified[indice],label_save=label_save[indice],label_class=label_class[indice],label_unified=label_unified[indice],true_label_unified=true_label_unified[indice],dist=dist,iter_max=iter_max,percent=percent,group_save=group_save[indice])
                            group_save[indice] = -self.cnt_all_ground_block                   
                    # avoid OOM
                    if len(np.unique(np.where(lidar_xyz_unified!=np.array([0,0,0])[0]))) < self.OOM_bar:
                        flag_points_less_than_OOM = False
                    else:
                        cnt_OOM += 1
                        if cnt_OOM >= 5:
                            cnt_OOM = 0
                            dist += 0.2 
                        print('points shape is (',len(np.unique(np.where(lidar_xyz_unified!=np.array([0,0,0])[0]))),', 3), May cause OOM',flush=True)
                
                cnt = 0
                self.cluster(lidar_xyz_unified=lidar_xyz_unified,labels=label_unified,true_label_unified=true_label_unified,label_save=label_save,label_class=label_class,file_list=file_list,file_len_list=file_len_list,d=d,seq=seq,group_save=group_save)
                print('Ground: propogated error is',self.propogated_error,'/',self.cnt_all_ground_block,'; weak error is',self.punish_error,'/',self.cnt_all_ground_block,flush=True)
                print('Cluster: propogated error is',self.propogated_error_cluster,'/',self.cnt_all_group_clutser,'; weak error is',self.punish_error_cluster,'/',self.cnt_all_group_clutser,flush=True)
                # if only save points, don't need to process next subsequence 
                if self.Save:
                    exit()


    def RANSAC(self,lidar_xyz_unified,label_save,dist,label_class,label_unified,true_label_unified,iter_max,percent,group_save):
        iter = 0
        lidar_ground_max = [np.array([0])]
        z_sort = np.sort(lidar_xyz_unified[:,2])[round(lidar_xyz_unified.shape[0]*self.z_sort_bar)]
        lidar_xyz_to_be_chosen = lidar_xyz_unified[lidar_xyz_unified[:,2]<=z_sort]
        if lidar_xyz_to_be_chosen.shape[0]<=3:
            lidar_xyz_to_be_chosen = lidar_xyz_unified
        if lidar_xyz_unified.shape[0]<=10000:
            iter_max = 100
        cnt = 0
        while 1:
            p1 = lidar_xyz_to_be_chosen[random.randint(0,lidar_xyz_to_be_chosen.shape[0]-1),:]
            p2 = lidar_xyz_to_be_chosen[random.randint(0,lidar_xyz_to_be_chosen.shape[0]-1),:]
            p3 = lidar_xyz_to_be_chosen[random.randint(0,lidar_xyz_to_be_chosen.shape[0]-1),:]
            # 加一个分位数限制
            # 加一个分类别的error



            ## ax+by+cz+d = 0
            a =  (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) 
            b =  (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) 
            c =  (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) 
            d =  0-(a*p1[0]+b*p1[1]+c*p1[2]) 
            if a*a+b*b+c*c == 0:
                cnt += 1
                if cnt == 50000 and lidar_xyz_to_be_chosen.shape[0] == 3:
                    return lidar_xyz_unified,label_save,label_class
                continue
            # get distance of each point to the flat
            distance = abs(lidar_xyz_unified[:,0]*a + lidar_xyz_unified[:,1]*b + lidar_xyz_unified[:,2]*c + d) / np.sqrt(a*a+b*b+c*c)
            
            # get ground point, which means it is ground, only index
            # get remain point, which means it is not ground, also only index
            lidar_ground = np.where(distance<=dist)
            # update the lidar_return_max, to get the min reamin point(not ground point)
            if lidar_ground[0].shape[0] > lidar_ground_max[0].shape[0]:
                lidar_ground_max = lidar_ground
            # update the iter
            iter += 1
            # exit if iter equals to iter_max or ground point is bigger than ?% of whole points.
            if lidar_ground_max[0].shape[0] >= percent*lidar_xyz_unified.shape[0]:
                # get the ground points' labels
                labels_scribble = label_unified[lidar_ground_max[0],:]
                labels_true = true_label_unified[lidar_ground_max[0],:]
                # get their unique labels
                label_true_unique = np.unique(labels_true)
                labels_unique = np.unique(labels_scribble)
                if not labels_unique.__len__() == 1 & (labels_unique==0)[0]:
                    labels_unique = np.append(labels_unique,self.add_list)
                    labels_unique = np.unique(labels_unique)
                    pass
                # create one hot encode, and let the unique labels be the 1
                label_add = np.zeros((1,20))
                label_add[0,labels_unique]=1
                # let this row be the label_add
                label_save[lidar_ground_max[0],:] = label_add

                # get this group of groud points should be what kind label: weak propogated or scribble.
                if len(labels_unique) == 1:
                        if labels_scribble[0] == 0:
                            # this ground is nothing, not be labeled
                            label_class[lidar_ground_max[0]] = 0
                        else:
                            # this ground is propogated
                            # Hard to reach, but possible
                            label_class[lidar_ground_max[0]] = 2
                            max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                            if labels_unique[0] != max_label_in_ground_truth:
                                self.propogated_error += 1
                # then, it can be propogated and weak, which propogated is just appear when len is 2 and label is something and 0(no label).
                # So if len is 2 and one is 0, it must be propogated
                elif (len(labels_unique) == 2) & (labels_unique[0] == 0):
                    # this group is propogated
                    label_class[lidar_ground_max[0]] = 2
                    max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                    if labels_unique[1] != max_label_in_ground_truth:
                        self.propogated_error += 1
                        if self.print_detail:
                            print('Ground: propogated result is',labels_unique,'while the max of ground truth is',max_label_in_ground_truth,flush=True)
                else:
                    # this group is weak
                    if not (np.unique(np.isin(label_true_unique,labels_unique)).shape[0]==1):
                        self.punish_error +=1 
                        if self.print_detail:
                            print('Ground: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique,flush=True)
                    # if (np.unique(labels_unique[labels_unique!=0]==label_true_unique).shape[0] == 1):
                    #         if (np.unique(labels_unique[labels_unique!=0]==label_true_unique) == False):       # weak与ground truth有不一样的
                    #             self.punish_error +=1 
                    #             if self.print_detail:
                    #                 print('Ground: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique)
                    label_class[lidar_ground_max[0]] = 3
                # let the ground point equal to (0,0,0)
                lidar_xyz_unified[lidar_ground_max[0],:] = 0
                break
            elif iter == iter_max:
                # get the ground points' labels
                labels_scribble = label_unified[lidar_ground_max[0],:]
                # get their unique labels
                labels_true = true_label_unified[lidar_ground_max[0],:]
                label_true_unique = np.unique(labels_true)
                labels_unique = np.unique(labels_scribble)
                # create one hot encode, and let the unique labels be the 1
                if not labels_unique.__len__() == 1 & (labels_unique==0)[0]:
                    labels_unique = np.append(labels_unique,self.add_list)
                    labels_unique = np.unique(labels_unique)
                    pass
                label_add = np.zeros((1,20))
                label_add[0,labels_unique]=1
                # let this row be the label_add
                label_save[lidar_ground_max[0],:] = label_add

                # get this group of groud points should be what kind label: weak propogated or scribble.
                if len(labels_unique) == 1:
                        if labels_scribble[0] == 0:
                            # this ground is nothing, not be labeled
                            label_class[lidar_ground_max[0]] = 0
                        else:
                            # this ground is propogated
                            # Hard to reach, but possible
                            max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                            if labels_unique[0] != max_label_in_ground_truth:
                                self.propogated_error += 1
                                if self.print_detail:
                                    print('Ground: propogated result is',labels_unique,'while the max of ground truth is',max_label_in_ground_truth)
                            label_class[lidar_ground_max[0]] = 2
                # then, it can be propogated and weak, which propogated is just appear when len is 2 and label is something and 0(no label).
                # So if len is 2 and one is 0, it must be propogated
                elif (len(labels_unique) == 2) & (labels_unique[0] == 0):
                    # this group is propogated
                    max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                    if labels_unique[1] != max_label_in_ground_truth:
                        self.propogated_error += 1
                        if self.print_detail:
                            print('Ground: propogated result is',labels_unique,'while the max of ground truth is',max_label_in_ground_truth,flush=True)
                    label_class[lidar_ground_max[0]] = 2
                else:
                    # this group is weak
                    # this group is weak
                    if not (np.unique(np.isin(label_true_unique,labels_unique)).shape[0]==1):
                        self.punish_error +=1
                        if self.print_detail: 
                            print('Ground: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique,flush=True)
                    # if (np.unique(labels_unique[labels_unique!=0]==label_true_unique).shape[0] == 1):
                    #         if (np.unique(labels_unique[labels_unique!=0]==label_true_unique) == False):       # weak与ground truth有不一样的
                    #             self.punish_error +=1
                    #             if self.print_detail: 
                    #                 print('Ground: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique)
                    label_class[lidar_ground_max[0]] = 3
                # let the ground point equal to (0,0,0)
                lidar_xyz_unified[lidar_ground_max[0],:] = 0
                break
        return lidar_xyz_unified,label_save,label_class


    def cluster(self,lidar_xyz_unified,labels,true_label_unified,label_save,label_class,file_list,file_len_list,d,seq,group_save):
        # varible which restores this label is nothing, weak, propogated or scribble
        # tips: nothing and ground(which is not reduce by last step) will not be used in 
        # noting, scribbles, propogated, weak
        #   0         1          2         3 
        points_index = np.where(lidar_xyz_unified!=np.array([0,0,0]))
        points = lidar_xyz_unified[np.unique(points_index[0]),:].astype(np.float32)
        points_index = np.expand_dims(np.unique(points_index[0]),axis=1)
        # get N*N distance, N represents the number of remained points
        # if t=10, cannot pass here
        print('points shape is',points.shape,flush=True)
        distances = np.zeros([points.shape[0],points.shape[0]],dtype=np.float64)
        cdist(points, points, out=distances)  # Assuming 'points' is N*3 array
        # get thresholds
        sensor_centers = np.array([0,0,0])
        ru = np.linalg.norm(points - sensor_centers, axis=1)
        rv = np.linalg.norm(points - sensor_centers, axis=1)
        thresholds = np.maximum(ru, rv) * d
        # cluster 
        adjacency_matrix = distances <= thresholds
        # Convert the adjacency matrix to a sparse matrix for efficient computation
        adjacency_sparse = csr_matrix(adjacency_matrix)
        # Compute the connected components using the sparse matrix
        n_components, groups = connected_components(adjacency_sparse)
        self.cnt_all_group_clutser = groups.max() + 1 
        # save results
        if self.Save:
            self.save_matrix_to_txt(points,'points_without_ground.txt')
            points_with_rgb = self.get_points_with_rgb(points,groups)
            self.save_matrix_to_txt(points_with_rgb,'rgb_points_without_ground.txt')
            # points.tofile(str(points.shape[0])+'_withoutground.bin')
            # groups.tofile(str(points.shape[0])+'_groups.bin')
        for i in range(n_components):
            # get each label of points in the group i
            index_in_origin_point_cloud = points_index[np.where(groups==i)]
            labels_scribble = labels[index_in_origin_point_cloud]
            group_save[index_in_origin_point_cloud] = i+1
            labels_unique = np.unique(labels_scribble)
            labels_true = true_label_unified[index_in_origin_point_cloud]
            label_true_unique = np.unique(labels_true)
            label_add = np.zeros((1,20))
            label_add[0,labels_unique]=1
            # divide this group into weak,propogated or nothing(when no label in this group)
            # first, get the number of unique label. If only 1, it can just be the propogated or nothing
            if len(labels_unique) == 1:
                if labels_scribble[0] == 0:
                    # this group is nothing
                    label_class[index_in_origin_point_cloud] = 0
                else:
                    # this group is propogated
                    # Hard to reach, but possible
                    max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                    if labels_unique[0] != max_label_in_ground_truth:
                        self.propogated_error_cluster += 1
                        if self.print_detail:
                            print('Cluster: propogated result is',labels_unique,'while the max of ground truth is',max_label_in_ground_truth,flush=True)
                    label_class[index_in_origin_point_cloud] = 2
            # then, it can be propogated and weak, which propogated is just appear when len is 2 and label is something and 0(no label).
            # So if len is 2 and one is 0, it must be propogated
            elif (len(labels_unique) == 2) & (labels_unique[0] == 0):
                # this group is propogated
                max_label_in_ground_truth = np.argmax(np.bincount(labels_true[labels_true!=0]))
                if labels_unique[1] != max_label_in_ground_truth:
                    self.propogated_error_cluster += 1
                    if self.print_detail:
                        print('Cluster: propogated result is',labels_unique,'while the max of ground truth is',max_label_in_ground_truth,flush=True)
                label_class[index_in_origin_point_cloud] = 2
            else:
                # # this group is weak
                # if not ((np.unique(labels_unique[labels_unique!=0]==label_true_unique).shape[0] == 1) & np.unique(labels_unique[labels_unique!=0]==label_true_unique) ==True):       # weak与ground truth有不一样的
                #     self.punish_error_cluster +=1 
                #     if self.print_detail:
                #         print('Cluster: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique)
                if not np.unique(np.isin(label_true_unique,labels_unique)).shape[0]==1:       # weak与ground truth有不一样的
                    self.punish_error_cluster +=1 
                    if self.print_detail:
                        print('Cluster: weak labels is',labels_unique[labels_unique!=0],'while ground truth is',label_true_unique,flush=True)
                label_class[index_in_origin_point_cloud] = 3
            # save the label by one hot encode.
            label_save[index_in_origin_point_cloud,:] = label_add
        # Scribble Labels should be 1
        label_class[np.where(labels!=0)] = 1
        # Save the label in LESS
        label_class_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'group')
        labels_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'labels')
        cluster_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'cluster')

        # save group array, size N*1, value can be 0,1,2,3, which represents the nothing, scribbles, propogated and weak.
        # save the label array, size N*20, value is one hot encoded
        file_len_old = 0
        # nothing = label_class[label_class==0]
        # scribble = label_class[label_class==1]
        # propogated = label_class[label_class==2]
        # weak = label_class[label_class==3]
        # print(nothing.shape,'\n',scribble.shape,'\n',propogated.shape,'\n',weak.shape,'\n',)
        # weak_label = label_save[np.where(label_class==3)[0],:]
        # propogated_label = label_save[np.where(label_class==2)[0],:]
        propogated_label_true = true_label_unified[label_class==2]
        propogated_label = np.where(label_save[np.where(label_class == 2)[0],:] == True)[1]
        propogated_label = propogated_label[propogated_label!=0]
        True_propogated_point = np.count_nonzero(propogated_label == propogated_label_true)
        if propogated_label.shape[0] > 0:
            print('propogated points right:',True_propogated_point,'/',propogated_label.shape[0],'. Wrong is',(1 - True_propogated_point/propogated_label.shape[0])*100,'%',flush=True)
        weak_label_true = true_label_unified[label_class==3]
        weak_label_true = np.expand_dims(weak_label_true,axis=1)
        weak_label = label_save[np.where(label_class == 3)[0],:] == True
        weak_label_right = weak_label[np.arange(len(weak_label_true)), weak_label_true.ravel()].reshape(-1, 1)
        print('Weak points right:',weak_label_right.sum(),'/',weak_label.shape[0],'. Wrong is',(1 - weak_label_right.sum()/weak_label.shape[0])*100,'%',flush=True)
        if not self.Save:
            for file_len,file_name in zip(file_len_list,file_list):
                label_class_file = os.path.join(label_class_dir,file_name)
                labels_file = os.path.join(labels_dir,file_name)
                cluster_file = os.path.join(cluster_dir,file_name)
                label_class[file_len_old:file_len].tofile(label_class_file)
                label_save[file_len_old:file_len,:].tofile(labels_file)
                group_save[file_len_old:file_len,:].tofile(cluster_file)
                file_len_old = file_len
                self.scans_cnt += 1
                toc = time.time()
                take_time = toc - self.tic
                print('processing seq:',seq,',complete',self.scans_cnt,'/',self.seq_len,',RANSAC+cluster takes',take_time,'s',flush=True)

    def save_matrix_to_txt(self,matrix, filename):
        # Define the format string to separate elements with spaces
        format_str = ' '.join(['%s'] * matrix.shape[1])
        # Save the matrix to the text file
        np.savetxt(filename, matrix, fmt=format_str)


    def get_points_with_rgb(self,points, groups):
        # Get the unique group labels
        unique_groups = np.unique(groups)

        # Generate a color map based on the number of unique groups
        num_colors = max(len(unique_groups), 20)
        if num_colors <= 20:
            cmap = plt.get_cmap('tab20b')
        else:
            cmap = plt.get_cmap('rainbow')

        colors = cmap(np.linspace(0, 1, num_colors))

        # Create an RGB matrix
        rgb_matrix = np.zeros((groups.shape[0], 3), dtype=np.uint8)

        # Assign colors to each group label
        for i, group in enumerate(unique_groups):
            indices = np.where(groups == group)[0]
            rgb_matrix[indices] = (colors[i-1][:3] * 255).astype(np.uint8)

        points_with_rgb = np.concatenate((points,rgb_matrix),axis=1)

        return points_with_rgb
    
    def read_specific_line_from_file(self,file, line_number):
        try:
            current_line_number = 0
            for line in file:
                current_line_number += 1
                if current_line_number == line_number:
                    return line
            # If the line number is out of range, return an error message
            return f"Line {line_number} not found in the file."
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
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
    parser.add_argument('--solve_seq', default=0,help='-1 means all,0-10 should be input')
    args = parser.parse_args()
    less = LESS(config['dataset'],config['LESS'],args=args)
    less.run_LESS()

    
    