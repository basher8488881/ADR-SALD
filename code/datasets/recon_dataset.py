import torch.utils.data as data
from utils.general import *
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import utils.general as utils
import numpy as np 
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
import torch 
import point_cloud_utils as pcu 

import os, sys



class ReconDataSet(data.Dataset):

    def __init__(self,split,dataset_path,dist_file_name, num_of_points, num_of_tr_ts_points):

        self.num_of_tr_ts_points = num_of_tr_ts_points
        #self.num_of_points = num_of_points

        #model = trimesh.load(dataset_path)

        self.device = torch.device('cuda')
        #self.conf = conf

        self.data_dir = dataset_path #.split('.')[0] #conf.get_string('data_dir')
        dataname = dataset_path.split('.') 
        #print(dataname)
        if len(dataname) == 3:
            self.np_data_name = '.' + dataname[1] + '.npz' # if the path is relative with one dot (./), if absolute path is provided then [0]  
        else:
            self.np_data_name = dataname[1] + '.npz'
        #print(self.data_dir, dataname, self.np_data_name)

        if os.path.exists(self.np_data_name):
            print('Data existing. Loading data...')
            load_data = np.load(self.np_data_name)
            self.dists = np.asarray(load_data['dist']) 
            self.mnf_point = np.asarray(load_data['point']) 
            self.scale = np.asarray(load_data['scale'])
            self.center = np.asarray(load_data['center'])  
            self.npyfiles_mnfld = [dataset_path]           
        else:
            print('Data not found. Processing data...')
            model = trimesh.load(dataset_path)
            if not type(model) == trimesh.PointCloud:
                model = utils.as_mesh(trimesh.load(dataset_path))
                if (model.faces.shape[0] > 0):
                    #self.points = sample_surface(model,num_of_points)[0]
                    total_size = (model.bounds[1] - model.bounds[0]).max()
                    centers = (model.bounds[1] + model.bounds[0]) / 2

                    model.apply_translation(-centers)
                    model.apply_scale(1 / total_size)                    
                    sample = sample_surface(model, int(num_of_points*2))
                    
                    center = 0 * np.mean(sample[0], axis=0)
                    scale = 1.0
                    face_nrmls = model.face_normals[sample[1]]



                    surface_sample_scales = [0.005,  0.03,  0.05, 0.5]
                    surface_sample_ratios = [0.20,  0.20,  0.30, 0.2]  # sum: 0.9

                    bbox_sample_scale, bbox_sample_ratio, bbox_padding = 0.07, 0.08, 0.15
                    space_sample_scale, space_sample_ratio, space_size = 0.09, 0.02, 0.65
                    surface_pairs = []

                    mesh = model 
                    sample_num = 100000


                    surface_pairs = []
                    # sample near surface
                    for sample_ratio, sample_scale in zip(surface_sample_ratios, surface_sample_scales):
                        sample_points_num = int(sample_num * sample_ratio)
                        surface_points = mesh.sample(sample_points_num)
                        random_pairs = np.tile(surface_points, (1, 2))
                        assert random_pairs.shape[1] == 6  # shape: N x 6
                        random_pairs = random_pairs + np.random.normal(0.0, 1.0, size=random_pairs.shape) * sample_scale
                        surface_pairs.append(random_pairs)
                    surface_pairs = np.concatenate(surface_pairs, axis=0)

                    # sample in bbox
                    bbox_points_num = int(sample_num * bbox_sample_ratio)
                    extents, transform = trimesh.bounds.to_extents(mesh.bounds)
                    padding_extents = extents + bbox_padding
                    bbox_points = trimesh.sample.volume_rectangular(padding_extents, bbox_points_num*6, transform=transform)
                    bbox_pairs = np.tile(bbox_points, (1, 2))
                    bbox_pairs = bbox_pairs + np.random.normal(0.0, 1.0, size=bbox_pairs.shape) * bbox_sample_scale

                    # sample in space
                    space_points_num = int(sample_num * space_sample_ratio)
                    space_points = (np.random.normal(0.0, 1.0, size=(int(space_points_num*2 ), 3)) * 2 - 1) * space_size
                    space_pairs0 = np.tile(space_points, (1, 2))
                    space_pairs0 = space_pairs0 + np.random.normal(0.0, 1.0, size=space_pairs0.shape) * space_sample_scale

                    # sample points in bbox and space
                    extents, transform = trimesh.bounds.to_extents(mesh.bounds)
                    bbox_points = trimesh.sample.volume_rectangular(extents, space_points_num*2, transform=transform)
                    space_points = (np.random.normal(0.0, 1.0, size=(int(space_points_num*2 ), 3))) * space_size
                    space_pairs1 = np.concatenate([bbox_points, space_points], axis=1)
                    space_pairs = np.concatenate([space_pairs0, space_pairs1], axis=0)

                    sample_pairs = np.concatenate([surface_pairs, bbox_pairs, space_pairs], axis=0)
                    cpnts = np.concatenate([sample_pairs[:,:3],sample_pairs[:,3:] ], axis=0)


                    n250kpnts_sample = sample_surface(model, num_of_points)
                    pnts = n250kpnts_sample[0]

                    triangles = []
                    for tri in model.triangles:
                        a = Point_3((tri[0][0] - center[0]) / scale,
                                (tri[0][1] - center[1]) / scale,
                                (tri[0][2] - center[2]) / scale)
                        b = Point_3((tri[1][0] - center[0]) / scale,
                                    (tri[1][1] - center[1]) / scale,
                                    (tri[1][2] - center[2]) / scale)
                        c = Point_3((tri[2][0] - center[0]) / scale,
                                    (tri[2][1] - center[1]) / scale,
                                    (tri[2][2] - center[2]) / scale)
                        triangles.append(Triangle_3(a, b, c))
                    tree = AABB_tree_Triangle_3_soup(triangles)
                    sigmas = []
                    ptree = cKDTree(pnts)
                    i = 0
                    for p in np.array_split(pnts, 100, axis=0):
                        d = ptree.query(p, 11)
                        sigmas.append(d[0][:, -1])

                        i = i + 1

                    sigmas = np.concatenate(sigmas)
                    sigmas_big = 1.0 * np.ones_like(sigmas)
                    pnts = np.tile(pnts, [num_of_points // pnts.shape[0], 1])
                    sample2 = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                            pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0,size=pnts.shape)],
                                            axis=0)

                    join_sample =  np.concatenate([sample2, cpnts],axis=0)

                    dists = []
                    normals = []
                    
                    for np_query in tqdm(join_sample):
                        cgal_query = Point_3(np_query[0].astype(np.double),
                                        np_query[1].astype(np.double),
                                        np_query[2].astype(np.double))

                        cp = tree.closest_point(cgal_query)
                        cp = np.array([cp.x(), cp.y(), cp.z()])
                        dist = np.sqrt(((cp - np_query) ** 2).sum(axis=0))
                        n = (np_query - cp) / dist
                        normals.append(np.expand_dims(n.squeeze(), axis=0))
                        dists.append(dist)

                    dists = np.array(dists)
                    normals = np.concatenate(normals, axis=0)
                    self.dists = np.concatenate([join_sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                    self.mnf_point = np.concatenate([sample[0], face_nrmls], axis=-1)
                    self.npyfiles_mnfld = [dataset_path]
                    self.scale = scale 
                    self.center = center 
                    np.savez(self.np_data_name, dist = self.dists, point = self.mnf_point, scale= self.scale, center= self.center)
                else:
                    if num_of_points < max((model.vertices).shape):
                        random_idx = np.random.choice(model.vertices.shape[0], num_of_points, replace = False)
                        self.points = model.vertices[random_idx,:]
                    else:
                        self.points = model.vertices
                        num_of_points = max((model.vertices).shape)

                    self.points = self.points - self.points.mean(0,keepdims=True)
                    scale = np.abs(self.points).max()
                    center = center = 0 * np.mean(self.points, axis=0)
                    self.points = self.points / scale

                    sigmas = []

                    ptree = cKDTree(self.points)

                    for p in tqdm(np.array_split(self.points, 10, axis=0)):
                        d = ptree.query(p, np.int(11.0))
                        sigmas.append(d[0][:, -1])

                    sigmas = np.concatenate(sigmas)
                    sigmas = np.tile(sigmas, [num_of_points // sigmas.shape[0]])
                    sigmas_big = 0.3 * np.ones_like(sigmas)
                    sigmas_big2 = 0.5 * np.ones_like(sigmas) #
                    pnts = np.tile(self.points, [num_of_points // self.points.shape[0], 1])
                    # 
                    sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                            pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape), pnts + np.expand_dims(sigmas_big2, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                            axis=0)

                    dists = []
                    normals = []
                    for np_query in tqdm(sample):
                        dist, idx = ptree.query(np_query)
                        n = (np_query - self.points[idx]) / dist
                        dists.append(dist)
                        normals.append(np.expand_dims(n.squeeze(), axis=0))

                    dists = np.array(dists)
                    normals = np.concatenate(normals, axis=0)
                    self.dists = np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                    self.mnf_point = np.concatenate([self.points, normals[:num_of_points,:]], axis=-1)
                    self.npyfiles_mnfld = [dataset_path]
                    self.scale = scale 
                    self.center = center
                    np.savez(self.np_data_name, dist = self.dists, point = self.mnf_point, scale= self.scale, center= self.center) 
            else:
                if num_of_points < max((model.vertices).shape):
                    random_idx = np.random.choice(model.vertices.shape[0], num_of_points, replace = False) #[np.random.randint(0, num_of_points) for p in range(0, num_of_points)]
                    self.points = model.vertices[random_idx,:]
                else:
                    self.points = model.vertices
                    #pointclouds = self.points
                    num_of_points = max((model.vertices).shape)


                # ********************* If you are using watertight point clouds from ShapeNet,  then use following lines of code ***************************
                
                self.points = self.points - self.points.mean(0,keepdims=True)
                scale = 1.0 #np.abs(self.points).max()
                center = 0 * np.mean(self.points, axis=0)
                self.points = self.points / scale
                pnts = np.tile(self.points, [num_of_points // self.points.shape[0], 1])

                sigmas = []

                ptree = cKDTree(pnts)

                for p in tqdm(np.array_split(pnts, 10, axis=0)):
                    d = ptree.query(p, np.int(11.0))
                    sigmas.append(d[0][:, -1])

                sigmas = np.concatenate(sigmas)
                sigmas = np.tile(sigmas, [num_of_points // sigmas.shape[0]])
                sigmas_big = 0.3 * np.ones_like(sigmas)
                
                sigmas_big2 = 0.5 * np.ones_like(sigmas) #
                sigmas_big3 = 0.05 * np.ones_like(sigmas) #
                

                sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                        pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape), 
                                        pnts + np.expand_dims(sigmas_big2, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                        pnts + np.expand_dims(sigmas_big3, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                        axis=0)

                dists = []
                normals = []
                for np_query in tqdm(sample):
                    dist, idx = ptree.query(np_query)
                    n = (np_query - pnts[idx]) / dist
                    dists.append(dist)
                    normals.append(np.expand_dims(n.squeeze(), axis=0))

                dists = np.array(dists)
                normals = np.concatenate(normals, axis=0)
                self.dists = np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                self.mnf_point = np.concatenate([pnts, normals[:num_of_points,:]], axis=-1)
                self.npyfiles_mnfld = [dataset_path]
                self.scale = scale 
                self.center = center 
                
                # *********** If not watertight point clouds from ShapeNet,  then use following lines of code and comment the previous block of code *********************

                # self.points = self.points - self.points.mean(0,keepdims=True)
                # scale = np.abs(self.points).max()
                # center = center = 0 * np.mean(self.points, axis=0)
                # self.points = self.points / scale

                # sigmas = []

                # ptree = cKDTree(self.points)

                # for p in tqdm(np.array_split(self.points, 10, axis=0)):
                #     d = ptree.query(p, np.int(51.0))
                #     sigmas.append(d[0][:, -1])

                # sigmas = np.concatenate(sigmas)
                # sigmas = np.tile(sigmas, [num_of_points // sigmas.shape[0]])
                # sigmas_big = 0.3 * np.ones_like(sigmas)
                # pnts = np.tile(self.points, [num_of_points // self.points.shape[0], 1])
                # sigmas_big2 = 0.5 * np.ones_like(sigmas) #

                # sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                #                         pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape), pnts + np.expand_dims(sigmas_big2, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                #                         axis=0)

                # dists = []
                # normals = []
                # for np_query in tqdm(sample):
                #     dist, idx = ptree.query(np_query)
                #     n = (np_query - self.points[idx]) / dist
                #     dists.append(dist)
                #     normals.append(np.expand_dims(n.squeeze(), axis=0))

                # dists = np.array(dists)
                # normals = np.concatenate(normals, axis=0)
                # self.dists = np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                # self.mnf_point = np.concatenate([self.points, normals[:num_of_points,:]], axis=-1)
                # self.npyfiles_mnfld = [dataset_path]
                # self.scale = scale 
                # self.center = center
                np.savez(self.np_data_name, dist = self.dists, point = self.mnf_point, scale= self.scale, center= self.center) 

    def __getitem__(self, index):
        
        point_set_mnlfld = torch.from_numpy(self.mnf_point).float()
        sample_non_mnfld = torch.from_numpy(self.dists).float()
        random_idx = (torch.rand(self.num_of_tr_ts_points**2) * point_set_mnlfld.shape[0]).long()
        #random_idx = (torch.rand(100**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld,0,random_idx)
        normal_set_mnfld = point_set_mnlfld[:,3:] 
        point_set_mnlfld = point_set_mnlfld[:,:3]# + self.normalization_params[index].float()

        random_idx = (torch.rand(self.num_of_tr_ts_points** 2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld, 0, random_idx)
        normalization_param1 = torch.from_numpy(np.asarray(self.center))
        normalization_param2 = torch.from_numpy(np.asarray(self.scale))
        normalization_param = {'center': normalization_param1, 'scale': normalization_param2}

                                         
        return point_set_mnlfld,normal_set_mnfld,sample_non_mnfld,normalization_param, index
    def __len__(self):
        return 1


