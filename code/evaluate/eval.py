import argparse
import sys
sys.path.append('../code')
import utils.general as utils
import os
import json
import trimesh
import utils.general as utils
import point_cloud_utils as pcu
import logging
from trimesh.sample import sample_surface
import torch
from pyhocon import ConfigFactory
import utils.plots as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import os
import GPUtil
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from utils.plots import plot_cuts,get_scatter_trace,plot_surface,plot_threed_scatter

def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def optimize_latent(conf, latent, ds, itemindex, network,lat_vecs):
    latent.detach_()
    latent.requires_grad_()
    lr = 1.0e-3
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_func = utils.get_class(conf.get_string('network.loss.loss_type'))(
        recon_loss_weight=1,grad_on_surface_weight=0,grad_loss_weight=0.1,z_weight=0.001,latent_reg_weight=0)

    num_iterations = 800

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    network.with_sample = False
    network.adaptive_with_sample = False
    idx_latent = utils.get_cuda_ifavailable(torch.arange(lat_vecs.num_embeddings))
    for e in range(num_iterations):
        #network.with_sample = e > 100
        pnts_mnfld,normals_mnfld,sample_nonmnfld,indices = ds[itemindex]
        
        pnts_mnfld = utils.get_cuda_ifavailable(pnts_mnfld).unsqueeze(0)
        normals_mnfld = utils.get_cuda_ifavailable(normals_mnfld).unsqueeze(0)
        sample_nonmnfld = utils.get_cuda_ifavailable(sample_nonmnfld).unsqueeze(0)
        indices = utils.get_cuda_ifavailable(indices).unsqueeze(0)
        outside_latent = lat_vecs(idx_latent[np.random.choice(np.arange(lat_vecs.num_embeddings),4,False)])

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        outputs = network(pnts_mnfld, None, sample_nonmnfld[:,:,:3], latent, False, only_decoder_forward=False)
        loss_res = loss_func(network_outputs=outputs, normals_gt=normals_mnfld, normals_nonmnfld_gt = sample_nonmnfld[:,:,3:6], pnts_mnfld=pnts_mnfld, gt_nonmnfld=sample_nonmnfld[:,:,-1],epoch=-1)
        loss = loss_res["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info("iteration : {0} , loss {1}".format(e, loss.item()))
        logging.info("mean {0} , std {1}".format(latent.mean().item(), latent.std().item()))

    
    #network.with_sample = True
    return latent



def evaluate_with_load(gpu, conf, exps_folder_name, override, timestamp, checkpoint ,parallel, resolution, recon_only=False, with_gt=True, plot_cmpr=False,with_opt=False, recon_chamfer=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info("running")
    conf = ConfigFactory.parse_file(conf)

    if override != '':
        expname = override    
    else:
        expname = conf.get_string('train.expname') 

    if timestamp == 'latest':
        timestamps = os.listdir(os.path.join(conf.get_string('train.base_path'),exps_folder_name, expname))
        timestamp = sorted(timestamps)[-1]
    elif timestamp == 'find':
        timestamps = [x for x in os.listdir(os.path.join('../',exps_folder_name,expname))
                      if not os.path.isfile(os.path.join('../',exps_folder_name,expname,x))]
        for t in timestamps:
            cpts = os.listdir(os.path.join('../',exps_folder_name,expname,t,'checkpoints/ModelParameters'))

            for c in cpts:
                if args.epoch + '.pth' == c:
                    timestamp = t
    else:
        timestamp = timestamp
    
    base_dir = os.path.join(conf.get_string('train.base_path'),exps_folder_name, expname, timestamp)
    if (gpu == 'cpu'):
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"),map_location=torch.device('cpu'))
    else:
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"))
    logging.info('loaded model')
    saved_model_epoch = saved_model_state["epoch"]

    network = utils.get_class(conf.get_string('train.network_class'))(conf=conf.get_config('network'),latent_size=conf.get_int('train.latent_size'),auto_decoder=conf.get_int('train.auto_decoder'))

    if (parallel):
        network.load_state_dict(
            {'.'.join(k.split('.')[1:]): v for k, v in saved_model_state["model_state_dict"].items()})
    else:
        network.load_state_dict(saved_model_state["model_state_dict"])

    if conf.get_bool('train.auto_decoder') :
        split_filename = './confs/splits/{0}'.format(conf.get_string('train.data_split'))
        with open(split_filename, "r") as f:
            split = json.load(f)

        ds = utils.get_class(conf.get_string('train.dataset.class'))(split=split, with_gt=True,
                                                                     **conf.get_config('train.dataset.properties'))
        total_files = len(ds)

        lat_vecs = torch.nn.Embedding(total_files, conf.get_int('train.latent_size'), max_norm=1.0)
        if os.path.isfile(os.path.join(base_dir,'checkpoints', "LatentCodes", checkpoint + '.pth')):
            data = torch.load(os.path.join(base_dir,'checkpoints', "LatentCodes",  checkpoint + ".pth"))
            lat_vecs.load_state_dict(data['latent_codes'])
            lat_vecs = utils.get_cuda_ifavailable(lat_vecs)
        else:
            logging.info("NO LATENT FILE")
            lat_vecs = None
    else:
        lat_vecs = None

    evaluate(
        network=utils.get_cuda_ifavailable(network),
        exps_folder_name=exps_folder_name,
        experiment_name=expname,
        timestamp=timestamp,
        epoch=saved_model_epoch,
        resolution=resolution,
        conf=conf,
        index=-1,
        recon_only=recon_only,
        lat_vecs=lat_vecs,
        plot_cmpr=plot_cmpr,
        with_gt=True,
        with_opt=with_opt, recon_chamfer=recon_chamfer
    )


def evaluate(network,exps_folder_name, experiment_name, timestamp, epoch, resolution, conf, index, recon_only,lat_vecs,plot_cmpr=False,with_gt=False, with_opt=False, recon_chamfer=False):

    if type(network) == torch.nn.parallel.DataParallel:
        network = network.module
        
    
    chamfer_results = dict(files=[],reg_to_gen_chamfer=[],gen_to_reg_chamfer=[],scan_to_gen_chamfer=[],gen_to_scan_chamfer=[])

    if (conf.get_string('train.test_split') == 'None'):
        ds = utils.get_class(conf.get_string('train.dataset.class'))(split=None,
                                                                            **conf.get_config('train.dataset.properties'))
        if recon_chamfer:
            recon_raw_data_path = conf.get_config('train.dataset.properties')['dataset_path']
            #print(recon_raw_data_path)

    else:
        split_filename = './confs/splits/{0}'.format(conf.get_string('train.test_split' ))
        with open(split_filename, "r") as f:
            split = json.load(f)
        
        ds = utils.get_class(conf.get_string('train.dataset.class'))(split=split,with_gt=with_gt,with_scans=True,scans_file_type='obj',
                                                                        **conf.get_config('train.dataset.properties'))
    total_files = len(ds)
    logging.info ("total files : {0}".format(total_files))
    prop = conf.get_config('train.dataset.properties')
    #print(prop)
    #prop['num_of_points'] = int(30000)
    prop['num_of_tr_ts_points'] = int(100)
    
    if with_gt:
        recon_raw_data_path = conf.get_config('train.dataset.properties')['dataset_path']
        ds_eval_scan = utils.get_class(conf.get_string('train.dataset.class'))(split=None,
                                                                **prop)
    # else:
    #     ds_eval_scan = utils.get_class(conf.get_string('train.dataset.class'))(split=None,
    #                                                                         **conf.get_config('train.dataset.properties'))
    utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation'))
    
    if with_gt:
        # utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation', split_filename.split('/')[-1].split('.json')[0]))
        # path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation', split_filename.split('/')[-1].split('.json')[0], str(epoch))
        path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation', str(epoch))
    else:
        path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation', str(epoch))
    utils.mkdir_ifnotexists(path)

    counter = 0
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)

    #names = ['_'.join([ds.npyfiles_dist[i].split('/')[-3:][0],ds.npyfiles_dist[i].split('/')[-3:][2]]).split('_dist_triangle.npy')[0] for i in range(len(ds.npyfiles_dist))]
    
    eval_same_dataloader = torch.utils.data.DataLoader(ds_eval_scan,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)
    i = 1
    # index = index + 1
    for data in tqdm(dataloader):
        if ((index == -1 or index == i )):
            logging.info (counter)
            #logging.info (ds.npyfiles_mnfld[data[-1].item()].split('/'))
            counter = counter + 1

            [logging.debug("evaluating " + ds.npyfiles_mnfld[data[-1][i]]) for i in range(len(data[-1]))]

            input_pc = utils.get_cuda_ifavailable(data[0])
            input_normal = utils.get_cuda_ifavailable(data[1])
            if with_gt:
                #filename = ['{0}/nonuniform_iteration_{1}_{2}_id.ply'.format(path, epoch, ds.npyfiles_mnfld[data[-1][i].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1][i].item()].split('/')[-1].split('.npy')[0]) for i in range(len(data[-1]))][0]
                filename = ['{0}/nonuniform_iteration_{1}_id.ply'.format(path, epoch)][0]
            else:
                filename = ['{0}/nonuniform_iteration_{1}_id.ply'.format(path, epoch)][0]

            if conf.get_bool('train.auto_decoder'):
                if not os.path.isfile(filename):
                    if with_opt:
                        latent = utils.get_cuda_ifavailable(torch.zeros([1, conf.get_int('train.latent_size')]))
                        latent = latent + 1e-5*torch.randn_like(latent)
                        latent = optimize_latent(conf,
                                                latent,
                                                ds,
                                                data[-1],
                                                network,
                                                lat_vecs)
                    else:
                        latent = lat_vecs(utils.get_cuda_ifavailable(data[-1]))
                        
                else:
                    latent = None
            else:
                _,latent,_ = network(manifold_points=input_pc,
                                manifold_normals=input_normal,
                                sample_nonmnfld=None,
                                latent=None,
                                only_encoder_forward=True,
                                only_decoder_forward=False)
            pnts_to_plot = input_pc

            

            filename1 = '{0}/nonuniform_iteration_{1}_{2}'.format(path, epoch, ds.npyfiles_mnfld[data[-1].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1].item()].split('/')[-1].split('.npy')[0])
            if (os.path.isfile(filename)):
                reconstruction = trimesh.load(filename)
                logging.info ('loaded : {0}'.format(filename))
            else:
                    if not latent is None:
                        #reconstruction = extract_fields(resolution=100,  query_func=network, points=pnts_to_plot, latent=latent, out_dir=filename1, iter_step=epoch) 
                        reconstruction,_ = plt.plot_surface(with_points=False,
                                        points=pnts_to_plot.detach()[0],
                                        decoder=network,
                                        latent=latent,
                                        path=path,
                                        epoch=epoch,
                                        in_epoch=ds.npyfiles_mnfld[data[-1].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1].item()].split('/')[-1].split('.npy')[0],
                                        shapefile=ds.npyfiles_mnfld[data[-1].item()],
                                        resolution=resolution,
                                        mc_value=0,
                                        is_uniform_grid=False,
                                        verbose=True,
                                        save_html=False,
                                        save_ply=True,
                                        overwrite=True,
                                        is_3d=True,
                                        z_func={'id':lambda x:x})
            if reconstruction is None and not latent is None:
                i = i + 1
                continue
            recon_chamfer_results = dict(files=[],reg_to_gen_chamfer=[],gen_to_reg_chamfer=[])
            if recon_chamfer:
                
                ground_truth_recon_model = trimesh.load(recon_raw_data_path)
                if not type(ground_truth_recon_model) == trimesh.PointCloud:
                    ground_truth_recon_model = utils.as_mesh(trimesh.load(recon_raw_data_path))
                    if (ground_truth_recon_model.faces.shape[0] > 0):
                        gt_recon_points = sample_surface(ground_truth_recon_model, 30000)[0]
                    else:
                        random_idx = [np.random.randint(0, 30000) for p in range(0, 30000)]
                        gt_recon_points = ground_truth_recon_model.vertices[random_idx,:] 

                else:
                    random_idx = [np.random.randint(0, 990000) for p in range(0, 990000)]
                    gt_recon_points = ground_truth_recon_model.vertices[random_idx,:] 
                dists_to_reg = utils.compute_trimesh_chamfer(
                    gt_points=trimesh.Trimesh(gt_recon_points),
                    gen_mesh=reconstruction,
                    offset=data[3]['center'].detach().cpu().numpy(),
                    scale=1./data[3]['scale'].detach().cpu().numpy()
                    )

                # dists_to_reg = utils.compute_dists(
                #     gt_points=gt_recon_points,
                #     recon_points=sample_surface(reconstruction, 30000)[0]
                #     ) 

                recon_chamfer_results['files'].append(recon_raw_data_path)
                recon_chamfer_results['reg_to_gen_chamfer'].append(dists_to_reg['gt_to_gen_chamfer'])
                recon_chamfer_results['gen_to_reg_chamfer'].append(dists_to_reg['gen_to_gt_chamfer'])
                pd.DataFrame(recon_chamfer_results).to_csv(os.path.join(path,"eval_results.csv"))
            if with_gt:
                for eval_data in tqdm(eval_same_dataloader):
                    eval_gt_pnts = eval_data[0] #.cpu().numpy() #utils.get_cuda_ifavailable(eval_data[0])
                    #print(eval_gt_pnts.size())
                    dists_to_reg = utils.compute_trimesh_chamfer(
                        gt_points=eval_gt_pnts.cpu().numpy(),
                        gen_mesh=reconstruction,
                        offset=-eval_data[3]['center'].detach().cpu().numpy(),
                        scale=1.0 #/eval_data[3]['scale'].detach().cpu().numpy()
                        )
                    # dists_to_reg = utils.compute_dists(
                    #     gt_points=np.reshape(eval_gt_pnts, (max(eval_gt_pnts.shape), 3)),
                    #     recon_points=sample_surface(reconstruction, 30000)[0]
                    #     )
                                    
                recon_chamfer_results['files'].append(recon_raw_data_path)
                recon_chamfer_results['reg_to_gen_chamfer'].append(dists_to_reg['gt_to_gen_chamfer'])
                recon_chamfer_results['gen_to_reg_chamfer'].append(dists_to_reg['gen_to_gt_chamfer'])
                pd.DataFrame(recon_chamfer_results).to_csv(os.path.join(path,"eval_results.csv"))
    #         if with_gt:
                
    #             normalization_params_filename = ds.normalization_files[data[-1]]
    #             logging.debug("normalization params are " + normalization_params_filename)
                    
    #             normalization_params = np.load(normalization_params_filename,allow_pickle=True)
    #             scale = normalization_params.item()['scale']
    #             center = normalization_params.item()['center']

    #             if with_gt:
    #                 gt_mesh_filename = ds.gt_files[data[-1]]
    #                 ground_truth_points = trimesh.Trimesh(trimesh.sample.sample_surface(utils.as_mesh(trimesh.load(gt_mesh_filename)), 30000)[0])
    #                 dists_to_reg = utils.compute_trimesh_chamfer(
    #                     gt_points=ground_truth_points,
    #                     gen_mesh=reconstruction,
    #                     offset=-center,
    #                     scale=1./scale,
    #                 )


    #             dists_to_scan = utils.compute_trimesh_chamfer(
    #                 gt_points=trimesh.Trimesh(ds_eval_scan[data[-1]][0].cpu().numpy()),
    #                 gen_mesh=reconstruction,
    #                 offset=0,
    #                 scale=1.,
    #             )

    #             if with_gt:
    #                 chamfer_results['files'].append(ds.gt_files[data[-1]])
    #                 chamfer_results['reg_to_gen_chamfer'].append(dists_to_reg['gt_to_gen_chamfer'])
    #                 chamfer_results['gen_to_reg_chamfer'].append(dists_to_reg['gen_to_gt_chamfer'])
                    

    #             chamfer_results['scan_to_gen_chamfer'].append(dists_to_scan['gt_to_gen_chamfer'])
    #             chamfer_results['gen_to_scan_chamfer'].append(dists_to_scan['gen_to_gt_chamfer'])
                
                    

    #             if (plot_cmpr):
                                        
    #                 fig = make_subplots(rows=1, cols=2 + int(with_gt), specs=[[{"type": "scene"}] * (2 + int(with_gt))],
    #                                     subplot_titles=("input scan", "Ours","Registration") if with_gt else ("input pc", "Ours"))

    #                 fig.layout.scene.update(dict(camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),xaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             yaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             zaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             aspectratio=dict(x=1, y=1, z=1)))
    #                 fig.layout.scene2.update(dict(camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),xaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             yaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             zaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                             aspectratio=dict(x=1, y=1, z=1)))
    #                 if with_gt:
    #                     fig.layout.scene3.update(dict(camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),xaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                                 yaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                                 zaxis=dict(range=[-1.5, 1.5], autorange=False),
    #                                                 aspectratio=dict(x=1, y=1, z=1)))

    #                 scan_mesh = utils.as_mesh(trimesh.load(ds.scans_files[data[-1]]))

    #                 scan_mesh.vertices = (scan_mesh.vertices - center)/scale

    #                 def tri_indices(simplices):
    #                     return ([triplet[c] for triplet in simplices] for c in range(3))

    #                 I, J, K = tri_indices(scan_mesh.faces)
    #                 color = '#ffffff'
    #                 trace = go.Mesh3d(x=scan_mesh.vertices[:, 0], y=scan_mesh.vertices[:, 1],
    #                                 z=scan_mesh.vertices[:, 2],
    #                                 i=I, j=J, k=K, name='scan',
    #                                 color=color, opacity=1.0, flatshading=False,
    #                                 lighting=dict(diffuse=1, ambient=0, specular=0), lightposition=dict(x=0, y=0, z=-1))
    #                 fig.add_trace(trace, row=1, col=1)

    #                 I, J, K = tri_indices(reconstruction.faces)
    #                 color = '#ffffff'
    #                 trace = go.Mesh3d(x=reconstruction.vertices[:, 0], y=reconstruction.vertices[:, 1], z=reconstruction.vertices[:, 2],
    #                                     i=I, j=J, k=K, name='our',
    #                                     color=color, opacity=1.0,flatshading=False,lighting=dict(diffuse=1,ambient=0,specular=0),lightposition=dict(x=0,y=0,z=-1))
                    
    #                 fig.add_trace(trace,row=1,col=2)

    #                 if with_gt:
    #                     gtmesh = utils.as_mesh(trimesh.load(gt_mesh_filename))
    #                     gtmesh.vertices = (gtmesh.vertices - center)/scale
    #                     I, J, K = tri_indices(gtmesh.faces)
    #                     trace = go.Mesh3d(x=gtmesh.vertices[:, 0], y=gtmesh.vertices[:, 1],
    #                                     z=gtmesh.vertices[:, 2],
    #                                     i=I, j=J, k=K, name='gt',
    #                                     color=color, opacity=1.0, flatshading=False,
    #                                     lighting=dict(diffuse=1, ambient=0, specular=0),
    #                                     lightposition=dict(x=0,y=0,z=-1))
                        
    #                     fig.add_trace(trace, row=1, col=3)


    #                 div = offline.plot(fig, include_plotlyjs=False, output_type='div', auto_open=False)
    #                 div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
                    
    #                 js = '''
    #                                 <script>
    #                                 var gd = document.getElementById('{div_id}');
    #                                 var isUnderRelayout = false
        
    #                                 gd.on('plotly_relayout', () => {{
    #                                 console.log('relayout', isUnderRelayout)
    #                                 if (!isUnderRelayout) {{
    #                                         Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
    #                                         .then(() => {{ isUnderRelayout = false }}  )
    #                                         Plotly.relayout(gd, 'scene3.camera', gd.layout.scene.camera)
    #                                         .then(() => {{ isUnderRelayout = false }}  )
    #                                     }}
        
    #                                 isUnderRelayout = true;
    #                                 }})
    #                                 </script>'''.format(div_id=div_id)

    #                 # merge everything
    #                 div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
    #                 with open(os.path.join(path, "compare_{0}.html".format(ds.npyfiles_mnfld[data[-1][0].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1][0].item()].split('/')[-1].split('.npy')[0])),
    #                         "w") as text_file:
    #                     text_file.write(div)
    #     i = i + 1
    #     logging.info (i)
    # if (index == -1) and with_gt:
    #     pd.DataFrame(chamfer_results).to_csv(os.path.join(path,"eval_results.csv"))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--expname", required=False, help='The experiment name to be evaluated.',default='')
    arg_parser.add_argument("--override", required=False, help='Override exp name.',default='')
    arg_parser.add_argument("--exps_folder_name", default="exps", help='The experiments directory.')
    arg_parser.add_argument("--timestamp", required=False, default='latest')
    arg_parser.add_argument("--conf", required=False , default='./confs/recon_vae.conf')
    arg_parser.add_argument("--checkpoint", help="The checkpoint to test.", default='latest')
    arg_parser.add_argument("--split", required=False,help="The split to evaluate.",default='')
    arg_parser.add_argument("--parallel", default=False, action="store_true", help="Should be set to True if the loaded model was trained in parallel mode")
    arg_parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto].')
    arg_parser.add_argument('--with_opt', default=False, action="store_true", help='If set, optimizing latent with reconstruction Loss versus input scan')
    arg_parser.add_argument('--resolution', default=256, type=int, help='Grid resolution')
    arg_parser.add_argument('--index', default=-1, type=int, help='')
    arg_parser.add_argument('--recon_only', default=False,action="store_true")
    arg_parser.add_argument('--with_gt', default=True,action="store_true")
    arg_parser.add_argument('--plot_cmpr', default=False,action="store_true")
    


    
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logging.info ("running")

    args = arg_parser.parse_args()
    
    if args.gpu != 'ignore':
        if args.gpu == "auto":
            deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                        excludeUUID=[])
            gpu = deviceIDs[0]
        else:
            gpu = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate_with_load(gpu=args.gpu,
                       parallel=args.parallel,
                       conf=args.conf,
                       exps_folder_name=args.exps_folder_name,
                       timestamp=args.timestamp,
                       checkpoint=args.checkpoint,
                       resolution=args.resolution,
                       override=args.override,
                       with_opt=args.with_opt,
                       recon_chamfer=args.recon_only,
                       with_gt = args.with_gt
                       )

