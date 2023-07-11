import numpy as np
import yaml
import os
import random
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time

    
class LESS():
    def __init__(self,config,config_LESS,split='train', label_directory='scribbles',target_directory='LESS') -> None:
        self.config = config
        self.split = split
        self.label_directory = label_directory
        self.target_directory = target_directory
        self.t = config_LESS['RANSAC']['scans_per_subsequence']
        self.l_grid = config_LESS['RANSAC']['l_grid']
        self.dist = config_LESS['RANSAC']['dist']
        self.iter_max = config_LESS['RANSAC']['iter_max']
        self.percent = config_LESS['RANSAC']['percent']
        self.d = config_LESS['cluster']['d']
        self.tic = 0

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
        for seq in self.config['split'][self.split]:
            seq = '{0:02d}'.format(int(seq))
            label_class_dir = os.path.join(root_dir, seq,self.target_directory, 'group')
            labels_dir = os.path.join(root_dir, seq,self.target_directory, 'labels')
            if not os.path.exists(label_class_dir):
                    os.makedirs(label_class_dir)
            if not os.path.exists(labels_dir):
                    os.makedirs(labels_dir)           
            lidar_dir = os.path.join(root_dir, seq, 'velodyne')
            lidar_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(lidar_dir)) for f in fn if f.endswith('.bin')]
            self.seq_len = len(lidar_path)
            self.scans_cnt = 0
            label_dir = os.path.join(root_dir, seq, self.label_directory)
            label_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_dir)) for f in fn if f.endswith('.label')]
            # assert (len(lidar_path) == len(label_path))
            # label_paths.extend(label_path)
            pose_path = os.path.join(root_dir,seq,'poses.txt')
            self.fusion(lidar_path=lidar_path,pose_path=pose_path,label_path=label_path,t=self.t,l_grid=self.l_grid,dist=self.dist,iter_max=self.iter_max,percent=self.percent,d=self.d,seq=seq)
        
        return 1
    

    # t represent t scans to be combined with
    # l_grid means we use l*l grid to run RANSAC
    def fusion(self,lidar_path,pose_path,label_path,t,l_grid,dist,iter_max,percent,d,seq):     
        pose_file = open(pose_path) 
        lidar_path.sort()
        label_path.sort()
        cnt = 0
        label_file_exist = os.listdir(label_path[0].replace('scribbles','LESS/group').split('/000000.label')[0])
        label_file_exist.sort()
        len_file_exist = len(label_file_exist)

        for lidar_file,label_file in zip(lidar_path,label_path):
            # label_file_exist = os.path.join(seq_path,label_file.split('/scribbles/')[1])
            # label_file_exist = label_file.replace('scribbles','LESS/group')
            if self.scans_cnt < len_file_exist:
                self.scans_cnt += 1
                continue
            # if os.path.exists(label_file_exist):
            #     self.scans_cnt += 1
            #     continue

            if cnt == 0: 
                file_list = []
                file_len_list = []
                lidar_xyz_unified = np.array([[0,0,0]],dtype=np.float32)
                label_unified = np.array([[0]])
            pose = pose_file.readline()
            r11,r12,r13,t14,r21,r22,r23,t24,r31,r32,r33,t34 = list(map(float, pose.split(' ')))
            lidar_data = np.fromfile(lidar_file,dtype=np.float32)
            lidar_data = lidar_data.reshape((-1,4))
            lidar_x = lidar_data[:,0]
            lidar_y = lidar_data[:,1]
            lidar_z = lidar_data[:,2]
            label = np.fromfile(label_file, dtype=np.int32)
            label = label.reshape((-1)) & 0xFFFF
            label = self.map_label(label, self.config['learning_map'])
            # Matrix_RT = np.array([r11,r12,r13,t11],[r21,r22,r23,t12],[r31,r32,r33,t13],[0,0,0,1])
            lidar_x = lidar_x*r11 +r12*lidar_y+r13*lidar_z + t14
            lidar_y = lidar_x*r21 +r22*lidar_y+r23*lidar_z + t24
            lidar_z = lidar_x*r31 +r32*lidar_y+r33*lidar_z + t34
            lidar_x = np.expand_dims(lidar_x,axis=1)
            lidar_y = np.expand_dims(lidar_y,axis=1)
            lidar_z = np.expand_dims(lidar_z,axis=1)
            label   = np.expand_dims(label,  axis=1)
            lidar_xyz = np.concatenate((lidar_x,lidar_y,lidar_z),axis=1)
            lidar_xyz_unified = np.concatenate((lidar_xyz_unified,lidar_xyz),axis=0)        # OK
            label_unified = np.concatenate((label_unified,label),axis=0)        # OK
            file_list.append(label_file.split(self.label_directory+'/')[1])
            file_len_list.append(len(lidar_xyz_unified)-1)
            label_save = np.zeros((lidar_xyz_unified.shape[0]-1,20),dtype=bool)
            label_class = np.expand_dims(np.zeros(lidar_xyz_unified.shape[0]-1,dtype=np.int8),axis=1)

            cnt += 1
            if cnt == t:
                self.tic = time.time()
                lidar_xyz_unified = lidar_xyz_unified[1:,:]     # get all points in one axis
                label_unified = label_unified[1:,:]
                y_max = np.max(lidar_xyz[:,1])
                y_min = np.min(lidar_xyz[:,1])
                x_max = np.max(lidar_xyz[:,0])
                x_min = np.min(lidar_xyz[:,0])
                for i in range(int(np.round(x_min/l_grid)),int(np.round(x_max/l_grid))):
                    for j in range(int(np.round(y_min/l_grid)),int(np.round(y_max/l_grid))):
                        grid_x_min = i*l_grid
                        grid_x_max = (i+1)*l_grid
                        grid_y_min = j*l_grid
                        grid_y_max = (j+1)*l_grid
                        indice = np.logical_and(np.logical_and(lidar_xyz_unified[:,1]>grid_y_min ,lidar_xyz_unified[:,1]<grid_y_max),np.logical_and(lidar_xyz_unified[:,0]>grid_x_min ,lidar_xyz_unified[:,0]<grid_x_max ) )       
                        if np.where(indice==True)[0].shape[0] >= 3:
                            lidar_xyz_unified[indice],label_save[indice],label_class[indice] = self.RANSAC(lidar_xyz_unified=lidar_xyz_unified[indice],label_save=label_save[indice],label_class=label_class[indice],label_unified=label_unified[indice],dist=dist,iter_max=iter_max,percent=percent)
                cnt = 0
                self.cluster(lidar_xyz_unified=lidar_xyz_unified,labels=label_unified,label_save=label_save,label_class=label_class,file_list=file_list,file_len_list=file_len_list,d=d,seq=seq)
            # lidar_xyz_SE3 = np.concatenate(lidar_xyz)
            pass
        pass

    def RANSAC(self,lidar_xyz_unified,label_save,dist,label_class,label_unified,iter_max,percent):
        iter = 0
        lidar_ground_max = [np.array([0])]
        while 1:
            p1 = lidar_xyz_unified[random.randint(0,lidar_xyz_unified.shape[0]-1),:]
            p2 = lidar_xyz_unified[random.randint(0,lidar_xyz_unified.shape[0]-1),:]
            p3 = lidar_xyz_unified[random.randint(0,lidar_xyz_unified.shape[0]-1),:]
            
            ## ax+by+cz+d = 0
            a =  (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) 
            b =  (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) 
            c =  (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) 
            d =  0-(a*p1[0]+b*p1[1]+c*p1[2]) 

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
                break
            elif iter == iter_max:
                break
        
        # get the ground points' labels
        labels_scribble = label_unified[lidar_ground_max[0],:]
        # get their unique labels
        labels_unique = np.unique(labels_scribble)
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
        # then, it can be propogated and weak, which propogated is just appear when len is 2 and label is something and 0(no label).
        # So if len is 2 and one is 0, it must be propogated
        elif (len(labels_unique) == 2) & (labels_unique[0] == 0):
            # this group is propogated
            label_class[lidar_ground_max[0]] = 2
        else:
            # this group is weak
            label_class[lidar_ground_max[0]] = 3
        # let the ground point equal to (0,0,0)
        lidar_xyz_unified[lidar_ground_max[0],:] = 0
        return lidar_xyz_unified,label_save,label_class


    def cluster(self,lidar_xyz_unified,labels,label_save,label_class,file_list,file_len_list,d,seq):
        # varible which restores this label is nothing, weak, propogated or scribble
        # tips: nothing and ground(which is not reduce by last step) will not be used in 
        # noting, scribbles, propogated, weak
        #   0         1          2         3 
        points_index = np.where(lidar_xyz_unified!=np.array([0,0,0]))
        points = lidar_xyz_unified[points_index].reshape((-1,3)).astype(np.float32)
        points_index = np.expand_dims(np.unique(points_index[0]),axis=1)
        # get N*N distance, N represents the number of remained points
        # if t=10, cannot pass here
        distances = cdist(points, points)  # Assuming 'points' is N*3 array
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
        # 
        for i in range(n_components):
            # get each label of points in the group i
            index_in_origin_point_cloud = points_index[np.where(groups==i)]
            labels_scribble = labels[index_in_origin_point_cloud]
            labels_unique = np.unique(labels_scribble)
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
                    label_class[index_in_origin_point_cloud] = 2
            # then, it can be propogated and weak, which propogated is just appear when len is 2 and label is something and 0(no label).
            # So if len is 2 and one is 0, it must be propogated
            elif (len(labels_unique) == 2) & (labels_unique[0] == 0):
                # this group is propogated
                label_class[index_in_origin_point_cloud] = 2
            else:
                # this group is weak
                label_class[index_in_origin_point_cloud] = 3
            # save the label by one hot encode.
            label_save[index_in_origin_point_cloud,:] = label_add
        # Scribble Labels should be 1
        label_class[np.where(labels!=0)] = 1
        # label_class = np.concatenate((label_class,label_save),axis=1)
        # Save the label in LESS
        # the array should be N*21, the 21 represent 1+20, which 1 is the group, and the 20 is (one) hot encode
        label_class_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'group')
        labels_dir = os.path.join(self.config['root_dir'], seq, self.target_directory,'labels')
        # label_class_dir = os.path.join('./data', seq, self.target_directory,'group')
        # labels_dir = os.path.join('./data', seq, self.target_directory,'labels')
        # label_class_dir = os.path.join('./data', seq, self.target_directory,'group')
        # labels_dir = os.path.join(self.config['root_dir'], seq, self.target_directory)
        # labels_dir = os.path.join('./data', seq, self.target_dir#ectory)

        # save group array, size N*1, value can be 0,1,2,3, which represents the nothing, scribbles, propogated and weak.
        # save the label array, size N*20, value is one hot encoded
        file_len_old = 0
        for file_len,file_name in zip(file_len_list,file_list):
            label_class_file = os.path.join(label_class_dir,file_name)
            labels_file = os.path.join(labels_dir,file_name)
            label_class[file_len_old:file_len].tofile(label_class_file)
            label_save[file_len_old:file_len].tofile(labels_file)
            file_len_old = file_len
            self.scans_cnt += 1
            toc = time.time()
            take_time = toc - self.tic
            print('processing seq:',seq,',complete',self.scans_cnt,'/',self.seq_len,',RANSAC+cluster takes',take_time,'s',flush=True)

if __name__ == '__main__':
    # result = np.array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    config_path = 'config/LESS.yaml'
    dataset_config_path = 'config/dataset/semantickitti.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))

    less = LESS(config['dataset'],config['LESS'])
    less.run_LESS()

    
    