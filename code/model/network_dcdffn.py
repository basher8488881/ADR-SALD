import torch
from torch import nn
import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch.autograd import grad
import utils.plots as plt
from torch import distributions as dist
import logging


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class Conv1dGrad(nn.Conv1d):
    def forward(self, input, input_grad, compute_grad=False, is_first=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None

        if len(self.weight.size())>2:
            #print('self.weight:',self.weight.size())
            weights = self.weight.squeeze(2)
            #print('after squeeze--> self.weight:', weights.size())
            if input_grad != None:
                #print('input_grad:',input_grad.size())
                input_grad = input_grad.permute(0, 2, 1, 3)
            
        #output_grad = self.weight[:,:3] if is_first else self.weight.matmul(input_grad)
        output_grad = weights[:,:3] if is_first else weights.matmul(input_grad)
        #print('self.weight, output_grad:', self.weight.size(), output_grad.size())
        # if is_first:
        #     output_grad = output_grad.squeeze()
        #output_grad = output_grad.view(output_grad.shape[2]*output_grad.shape[0], output_grad.shape[1])
        #     output_grad = output_grad.permute(2,0,1)
        # else:
        #     output_grad = output_grad.permute(0,2,1,3)
        # print('output, output_grad:', output.size(), output_grad.size())
        return output , output_grad

class LinearGrad(nn.Linear):
    def forward(self, input, input_grad, compute_grad=False, is_first=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None

        #tt = self.weight
        #print('self.weight:',tt.size)
        #output_grad = self.weight[:,:3] if is_first else self.weight.matmul(input_grad)
        # if is_first:
        #     #output_grad = output_grad.view(output_grad.shape[2]*output_grad.shape[0], output_grad.shape[1])
        #     output_grad = output_grad.permute(2,0,1)
        # # else:
        #     output_grad= output_grad.permute(0,2,1,3)
        #print('output, output_grad:', output.size(), output_grad.size())
        return output #, output_grad


class TanHGrad(nn.Tanh):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = (1 - torch.tanh(input).pow(2)).unsqueeze(-1) * input_grad
        return output, output_grad


class SoftplusGrad(nn.Softplus):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        #print('softplus:',output.size(), input.size(), input_grad.size())
        #input = input.permute(0, 2, 1)
        #print('input squeeze:',torch.sigmoid(self.beta * input).unsqueeze(-1).size(), input_grad.size())
        output_grad = torch.sigmoid(self.beta * input).unsqueeze(-1).permute(0,2,1,3) * input_grad #
        return output , output_grad


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=256, dim=3, hidden_dim=256):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, kernel_size=1, padding=0)
        self.fc_0 = nn.Conv1d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_1 = nn.Conv1d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_2 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_3 = nn.Conv1d(4*hidden_dim, hidden_dim,kernel_size=1, padding=0)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        #self.pool = maxpool
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        
        self.pool2 = maxpool

    def forward(self, p):

        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        net1 = net 
        
        fc1_net = net.permute(0,2,1)
        f_c1_net = self.pool2(fc1_net, dim=1)
        f_c1_mean = self.fc_mean(self.actvn(f_c1_net))
        f_c1_std = self.fc_std(self.actvn(f_c1_net))

        #pooled = self.pool(net)
        pooled1 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled1], dim=1)
        net = self.fc_1(self.actvn(net))
        net2 = net 

        fc2_net = net.permute(0,2,1)
        f_c2_net = self.pool2(fc2_net, dim=1)
        f_c2_mean = self.fc_mean(self.actvn(f_c2_net))
        f_c2_std = self.fc_std(self.actvn(f_c2_net))

        #pooled = self.pool(net)
        pooled2 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net1, net2, pooled2], dim=1)
        net = self.fc_2(self.actvn(net))
        net3 = net 
        
        fc3_net = net.permute(0,2,1)
        f_c3_net = self.pool2(fc3_net, dim=1)
        f_c3_mean = self.fc_mean(self.actvn(f_c3_net))
        f_c3_std = self.fc_std(self.actvn(f_c3_net))

        #pooled = self.pool(net)
        pooled3 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net1, net2, net3, pooled3], dim=1)

        net = self.fc_3(self.actvn(net))
        net = net.permute(0,2,1)

        f_c4_net = self.pool2(net, dim=1)
        f_c4_mean = self.fc_mean(self.actvn(f_c4_net))
        f_c4_std = self.fc_std(self.actvn(f_c4_net))

        # c_mean = self.fc_mean(self.actvn(net))
        # c_std = self.fc_std(self.actvn(net))
        # c_mean = f_c1_mean + f_c2_mean + f_c3_mean + f_c4_mean
        # c_std = f_c1_std + f_c2_std + f_c3_std + f_c4_std
        
        #print('Encoder:', c_mean.size(), c_std.size())
        
        #return c_mean,c_std
        #return f_c1_mean , f_c2_mean , f_c3_mean , f_c4_mean, f_c1_std , f_c2_std , f_c3_std , f_c4_std 
        return  f_c4_mean,  f_c4_std 


class ImplicitMap(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            activation=None,
            latent_dropout=False,
            xyz_dim=3,
            geometric_init=True,
            beta=100
    ):
        super().__init__()

        bias = 1.0
        self.latent_size = latent_size
        last_out_dim = 1
        dims = [latent_size + xyz_dim] + dims + [last_out_dim]
        self.d_in = latent_size + xyz_dim
        self.latent_in = latent_in
        self.num_layers = len(dims)
        self.in_dim = 0
        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            #print('decoder filter size:',dims[l], out_dim)
            #lin = LinearGrad(dims[l], out_dim)
            if l == 0:
            	Conv1d = Conv1dGrad(dims[l], out_dim, kernel_size=1, padding=0)
            	self.in_dim = self.in_dim + out_dim
            elif l!=0 and l < self.num_layers - 2:
                Conv1d = Conv1dGrad(self.in_dim, out_dim, kernel_size=1, padding=0)
                self.in_dim = self.in_dim + out_dim
            else:
                Conv1d = Conv1dGrad(self.in_dim, out_dim, kernel_size=1, padding=0)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(Conv1d.weight, mean=np.sqrt(np.pi) / np.sqrt(self.in_dim), std=0.0001)
                    torch.nn.init.constant_(Conv1d.bias, -bias)
                else:
                    torch.nn.init.constant_(Conv1d.bias, 0.0)
                    torch.nn.init.normal_(Conv1d.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                Conv1d = nn.utils.weight_norm(Conv1d)

            setattr(self, "Conv1d" + str(l), Conv1d)

        self.softplus = SoftplusGrad(beta=beta)
        

    def forward(self, input, latent, compute_grad=False, cat_latent=True):
        '''
        :param input: [shape: (N x d_in)]
        :param compute_grad: True for computing the input gradient. default=False
        :return: x: [shape: (N x d_out)]
                 x_grad: input gradient if compute_grad=True [shape: (N x d_in x d_out)]
                         None if compute_grad=False
        '''

        if len(input.size())!= 3:
            input = input.unsqueeze(0)
        x = input
        #xyz = input[:,:, -3:]
        input_con = latent.unsqueeze(1).repeat(1, input.shape[1], 1) if self.latent_size > 0 else input
        #print('decoder x:', x.size(), 'latent:', input_con.size())
        if self.latent_size > 0 and cat_latent:
            x = torch.cat([x, input_con], dim=-1) if len(x.shape) == 3 else torch.cat(
                [x, latent.repeat(input.shape[0], 1)], dim=-1)
            #print('x:',x.size())
        input_con = x
        to_cat = x
        x_grad = None
        dfeatures = None
        ax_grad = None
        for l in range(0, self.num_layers - 1):
            Conv1d = getattr(self, "Conv1d" + str(l))
            #print('x:', x.size())
            # if l in self.latent_in:
            #     #print('self.latent_in:',x.size(), to_cat.size())
            #     x = torch.cat([x, to_cat], -1) / np.sqrt(2)
            #     #print('x:', x.size())
            #     if compute_grad:
            #         skip_grad = torch.eye(self.d_in, device=x.device)[:, :3].unsqueeze(0).repeat(input.shape[0],input.shape[1], 1, 1)
            #         #print('compute_grad:',skip_grad.size(), x_grad.size())
            #         x_grad = torch.cat([x_grad, skip_grad], 2) / np.sqrt(2)


            #print('x:', x.size())
            if l == 0:
                if len(x.size())==3:
                    x = x.permute(0, 2, 1)
                x, x_grad = Conv1d(x, x_grad, compute_grad, l == 0)
                if l==0:
                    dfeatures = x
                if l==0 and x_grad is not None and compute_grad:
                    ax_grad = x_grad
            if l!=0 and l < self.num_layers-2:
                dfeatures = dfeatures / np.sqrt(2)
                #print('dfeatures:', dfeatures.shape)
                if compute_grad:
                    ax_grad = ax_grad / np.sqrt(2) 
                x, x_grad = Conv1d(dfeatures, ax_grad, compute_grad, l == 0)

            if l == self.num_layers-2:
                dfeatures = dfeatures / np.sqrt(2)
                #print('dfeatures:', dfeatures.shape)
                if compute_grad:
                    ax_grad = ax_grad / np.sqrt(2) 
                x, x_grad = Conv1d(dfeatures, ax_grad, compute_grad, l==0)

            if l < self.num_layers - 2:
                #x_grad = x_grad.permute(0, 2, 1, 3)
                x, x_grad = self.softplus(x, x_grad, compute_grad)
                if l==0:
                    dfeatures = x
                if l==0 and x_grad is not None and compute_grad:
                    ax_grad = x_grad.permute(0, 2, 1, 3) 
                if l > 0:
                    if len(x.shape) == 3:
                        
                        dfeatures = torch.cat([dfeatures, x], axis=1) #/ np.sqrt(2)
                    if len(x.shape) == 2:
                        
                        dfeatures = torch.cat([dfeatures, x], axis=0) #/ np.sqrt(2)
           
                if l > 0 and x_grad is not None and compute_grad:
                    x_grad = x_grad.permute(0, 2, 1, 3)
                    ax_grad = torch.cat([ax_grad, x_grad], axis=1) #/ np.sqrt(2)
                #print('soft plus--> dfeatures, ax_grad:', dfeatures.shape, ax_grad.shape)

                x = x.permute(0, 2, 1)
        return x, x_grad, input_con


class Network(nn.Module):
    def __init__(self, conf, latent_size, auto_decoder):
        super().__init__()
        
        self.latent_size = latent_size
        self.with_normals = conf.get_bool('encoder.with_normals')
        encoder_input_size = 6 if self.with_normals else 3
        #hidden_dim= latent_size, c_dim=latent_size,
        self.encoder = SimplePointnet(dim=encoder_input_size) if not auto_decoder and latent_size > 0 else None

        self.implicit_map = ImplicitMap(latent_size=latent_size, **conf.get_config('decoder_implicit'))
        #print(self.implicit_map)

        self.predict_normals_on_surfce = conf.get_bool('predict_normals_on_surfce')
        
        logging.debug("""self.latent_size = {0},
                      self.with_normals = {1}
                      self.predict_normals_on_surfce = {2}
                      """.format(self.latent_size,
                                                            self.with_normals,
                                                            self.predict_normals_on_surfce))

    def forward(self, manifold_points, manifold_normals, sample_nonmnfld, latent,
                only_encoder_forward, only_decoder_forward,epoch=-1):
        output = {}

        if self.encoder is not None and not only_decoder_forward:
            encoder_input = torch.cat([manifold_points, manifold_normals],
                                      axis=-1) if self.with_normals else manifold_points
            encoder_input = encoder_input.permute(0, 2, 1)
            #print(encoder_input.size())
            #q_latent_mean, q_latent_std = self.encoder(encoder_input)
            #qlm1,qlm2,qlm3,qlm4, qls1, qls2,qls3,qls4 = self.encoder(encoder_input)
            qlm4, qls4 = self.encoder(encoder_input)
            #q_z1 = dist.Normal(qlm1, torch.exp(qls1))
            #l1 = q_z1.rsample()
            
            #q_z2 = dist.Normal(qlm2, torch.exp(qls2))
            #l2 = q_z2.rsample()
            
            #q_z3 = dist.Normal(qlm3, torch.exp(qls3))
            #l3 = q_z3.rsample()
            
            q_z4 = dist.Normal(qlm4, torch.exp(qls4))
            l4 = q_z4.rsample()
            #latent = torch.cat([l1, l2, l3, l4], axis=1)
            latent = l4
           
            #q_latent_mean = torch.cat([qlm1, qlm2, qlm3, qlm4], axis=1) 
            #q_latent_std = torch.cat([qls1, qls2, qls3 , qls4], axis=1)
            q_latent_mean =  qlm4 
            q_latent_std =  qls4
            #print('encoder output:',q_latent_mean.size(), q_latent_std.size())
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
            output['latent_reg'] = latent_reg

            if only_encoder_forward:
                return latent, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None, None

        if only_decoder_forward:
            #manifold_points = manifold_points.permute(0, 2, 1)
            #print('manifold_points:', manifold_points.size())
            return self.implicit_map(manifold_points, latent, False)[0]
        else:
            #sample_nonmnfld = sample_nonmnfld.permute(0,2,1)  
            #print('networ--> else block:',sample_nonmnfld.size(), latent.size())
            non_mnfld_pred, non_mnfld_pred_grad, _ = self.implicit_map(sample_nonmnfld, latent, True)
            

            output['non_mnfld_pred_grad'] = non_mnfld_pred_grad
            output['non_mnfld_pred'] = non_mnfld_pred

            if not latent is None:
                output['norm_square_latent'] = (latent**2).mean(-1)

            if self.predict_normals_on_surfce:
                #manifold_points = manifold_points.permute(0, 2, 1)  
                _, grad_on_surface, _ = self.implicit_map(manifold_points, latent, True)
                output['grad_on_surface'] = grad_on_surface

            return output

    

