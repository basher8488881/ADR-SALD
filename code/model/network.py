from turtle import forward
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
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def minpool(x, dim=-1, keepdim=False):
    out, _ = x.min(dim=dim, keepdim=keepdim)
    return out
class Conv1dGrad(nn.Conv1d):
    def forward(self, input, input_grad, compute_grad=False, is_first=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None

        if len(self.weight.size())>2:
            #print('self.weight:',self.weight.size())
            weights = self.weight.squeeze(2)
        output_grad = weights[:,:3] if is_first else weights.matmul(input_grad)
        return output , output_grad



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
        output_grad = torch.sigmoid(self.beta * input).unsqueeze(-1).permute(0,2,1,3) * input_grad #
        return output , output_grad




from einops import rearrange, repeat
import math

def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

class N2PAttention(nn.Module):
    def __init__(self, in_channels, K=32):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = K
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        #self.ff = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(in_channels, in_channels, 1, bias=False))
        self.bn1 = nn.GroupNorm(1, in_channels) #nn.BatchNorm1d(in_channels)
        #self.bn2 = nn.GroupNorm(1, in_channels)  # nn.BatchNorm1d(in_channels)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        #tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        #x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


class PointAttentionNetwork(nn.Module):
    def __init__(self,C, ratio = 8):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.GroupNorm(1, C//ratio) #nn.BatchNorm1d(C//ratio)
        self.bn2 = nn.GroupNorm(1, C//ratio)  #nn.BatchNorm1d(C//ratio)
        self.bn3 = nn.GroupNorm(1, C)  #nn.BatchNorm1d(C)
        #self.gn_out = nn.GroupNorm(1, C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                nn.ReLU(),
                                self.bn1
                                )
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                nn.ReLU(), 
                                self.bn2
                                )
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                nn.ReLU(),
                                self.bn3
                                )

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        #b,c,n = x.shape

        a = self.conv1(x).permute(0,2,1) # b, n, c/ratio

        b = self.conv2(x) # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b)) # b,n,n

        d = self.conv3(x) # b,c,n
        out = x + torch.bmm(d, s.permute(0, 2, 1))
        
        return out


class ResnetPoolBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None, shortcut=True, atten=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.shortcut = shortcut
        self.atten = atten
        self.gn_0 = nn.GroupNorm(1, size_in)
        self.gn_1 = nn.GroupNorm(1, size_h)
        self.pool = maxpool

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(2*size_h, size_out, 1)
        self.p2p_attention_global = PointAttentionNetwork(size_out)
        self.n2p_attention_local = N2PAttention(size_out)

        self.fc_out = nn.Conv1d(2*size_h, 2*size_out, 1)
        self.gn_out = nn.GroupNorm(1, 2*size_h)
        self.actvn = nn.ReLU()

    def forward(self, x, pf):
        net = self.gn_0(self.actvn(self.fc_0(x)))
        pooled2 = self.pool(net, keepdim=True).clone().expand(net.size())
        comb_fea = torch.cat([net, pooled2],1)
        dx = self.gn_1(self.actvn(self.fc_1(comb_fea)))
        
        if self.atten:
            global_atten = self.p2p_attention_global(dx)

        if self.shortcut is not False:

            feature = dx + pf  
        else:
            p2pAtt = self.n2p_attention_local(dx)
            feature = torch.cat([p2pAtt + pf, global_atten + pf], 1)

            feature = self.gn_out(self.fc_out(self.actvn(feature))) 
            feature = feature.permute(0,2,1)
            feature = self.pool(feature, dim=1)
        

        return feature



class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=256, dim=3, hidden_dim=256, normal_channel=False):
        super().__init__()
        self.c_dim = c_dim
        self.fc_pos = nn.Conv1d(dim, hidden_dim, 1) #hidden_dim+
        #self.fc_pos = nn.Conv1d(dim, hidden_dim, 1) #hidden_dim+
        self.block_0 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=True, atten=False)
        self.block_1 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=True, atten=False)
        self.block_2 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=True, atten=False)
        self.block_3 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=True, atten=False)
        self.block_4 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=True, atten=False)
        self.block_5 = ResnetPoolBlockConv1d(hidden_dim, hidden_dim,hidden_dim, shortcut=False, atten=True)
        self.fc_mean1 = nn.Linear(2*hidden_dim, c_dim)
        self.fc_std = nn.Linear(2*hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.gn_pos = nn.GroupNorm(1, hidden_dim)

        torch.nn.init.constant_(self.fc_mean1.weight,0.0)
        torch.nn.init.constant_(self.fc_mean1.bias, 0.0)

        torch.nn.init.constant_(self.fc_std.weight, 0.0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

    def forward(self, p):
        #net = self.fc_pos(p)
        net = self.gn_pos(self.actvn(self.fc_pos(p)))
        net = self.block_0(net, net)
        net = self.block_1(net, net)
        net = self.block_2(net, net)
        net = self.block_3(net, net)
        net = self.block_4(net, net)
        net = self.block_5(net, net)

        f_mean = self.fc_mean1(self.actvn(net))
        f_std = self.fc_std(self.actvn(net))
        return f_mean, f_std 





class ResNext_fExtractor(nn.Module):
    def __init__(self, in_channel, filter_list, beta=100):
        super(ResNext_fExtractor, self).__init__()

        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        self.softplus = SoftplusGrad(beta=beta)
        for u, out_channel in enumerate(filter_list):
            Conv1d = nn.utils.weight_norm(Conv1dGrad(last_channel, out_channel, 1))
            self.mlp_convs.append(Conv1d)
            setattr(self, "Conv1d" + str(u), Conv1d)
            torch.nn.init.normal_(Conv1d.weight, 0.0, np.sqrt(2) / np.sqrt(last_channel))
            torch.nn.init.constant_(Conv1d.bias, 0.0)

    def forward(self, x_feature, grade_feature, compute_grad=False):
        x_c_feature = None 
        x_grad_c_feature = None 
        for i, conv in enumerate(self.mlp_convs):
            #print(i)
            x_feature1, grade_feature1 =  conv(x_feature, grade_feature, compute_grad, False)
            x_feature1, grade_feature1 = self.softplus(x_feature1, grade_feature1, compute_grad)
            if i== 0:
                x_c_feature, x_grad_c_feature = x_feature1, grade_feature1
            elif i!= 0 and compute_grad == True:
                x_c_feature = torch.cat([x_c_feature, x_feature1], 1)
                x_grad_c_feature = torch.cat([x_grad_c_feature, grade_feature1], 2)
            elif i!=0 and compute_grad==False:
                x_c_feature = torch.cat([x_c_feature, x_feature1], 1)
                x_grad_c_feature = None 

        return x_c_feature, x_grad_c_feature





class ImplicitMap_fExtrator(nn.Module):
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
        self.base_nf = 384
        self.d_conv1d_1 = Conv1dGrad(259, 515, kernel_size=1)

        self.d_conv1d_10 = Conv1dGrad(896, 1, kernel_size=1)

        self.softplus = SoftplusGrad(beta=beta)

        
        torch.nn.init.normal_(self.d_conv1d_1.weight, 0.0, np.sqrt(2) / np.sqrt(515))
        torch.nn.init.constant_(self.d_conv1d_1.bias, 0.0)


        torch.nn.init.normal_(self.d_conv1d_10.weight, mean=np.sqrt(np.pi) / np.sqrt(896), std=0.0001)
        torch.nn.init.constant_(self.d_conv1d_10.bias, -bias)


        nn.utils.weight_norm(self.d_conv1d_1)

        nn.utils.weight_norm(self.d_conv1d_10)

        self.ResNextFExtractor1 = ResNext_fExtractor(in_channel=515, filter_list=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])
        self.ResNextFExtractor1_1 = ResNext_fExtractor(in_channel= 512, filter_list=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])
        self.ResNextFExtractor2 = ResNext_fExtractor(in_channel=512, filter_list=[48,48,48,48,48,48,48,48,48,48,48,48])
        self.ResNextFExtractor2_2 = ResNext_fExtractor(in_channel=576, filter_list=[48,48,48,48,48,48,48,48,48,48,48,48])
        self.ResNextFExtractor3 = ResNext_fExtractor(in_channel=576, filter_list=[64,64,64,64,64,64,64,64,64,64,64,64])
        self.ResNextFExtractor3_3 = ResNext_fExtractor(in_channel=768, filter_list=[96,96,96,96,96,96,96,96])
        self.ResNextFExtractor4 = ResNext_fExtractor(in_channel=768, filter_list=[128,128,128,128,128,128])
        self.ResNextFExtractor4_4 = ResNext_fExtractor(in_channel=768, filter_list=[160, 160, 160, 160,160,160])
        self.ResNextFExtractor5 = ResNext_fExtractor(in_channel=960, filter_list=[192,192,192,192,192])
        self.ResNextFExtractor5_5 = ResNext_fExtractor(in_channel=960, filter_list=[224,224,224,224])

    def forward(self, input, latent, compute_grad=False, cat_latent=True, epoch=-1):
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
        # if compute_grad and epoch > 5 and epoch <= 500 :
        #     ridx = (torch.rand(int((latent.shape[1])*0.05)) *(latent.shape[1])).long()
        #     latent[:, ridx] = 0
        # elif compute_grad and epoch > 500 and epoch <= 800:
        #     ridx = (torch.rand(int((latent.shape[1])*0.03)) *(latent.shape[1])).long()
        #     latent[:, ridx] = 0  
        # elif compute_grad and epoch > 800 and epoch <= 1500:
        #     ridx = (torch.rand(int((latent.shape[1])*0.02)) *(latent.shape[1])).long()
        #     latent[:, ridx] = 0
        # else:
        #     latent = latent 

        input_con = latent.unsqueeze(1).repeat(1, input.shape[1], 1) if self.latent_size > 0 else input
        input_latent = input_con

        if self.latent_size > 0 and cat_latent:
            x = torch.cat([x, input_con], dim=-1) if len(x.shape) == 3 else torch.cat(
                [x, latent.repeat(input.shape[0], 1)], dim=-1)
            #print('x:',x.size())
        input_con = x
        to_cat = x
        x_grad = None
        x = x.permute(0, 2, 1)
        y = x

        if compute_grad:

            x, x_grad = self.d_conv1d_1(x, x_grad, compute_grad, True)

            x, x_grad = self.softplus(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor1(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor1_1(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor2(x, x_grad, compute_grad)  
 
            x, x_grad = self.ResNextFExtractor2_2(x, x_grad, compute_grad)     

            x, x_grad = self.ResNextFExtractor3(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor3_3(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor4(x, x_grad, compute_grad)
    
            x, x_grad = self.ResNextFExtractor4_4(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor5(x, x_grad, compute_grad)
            x, x_grad = self.ResNextFExtractor5_5(x, x_grad, compute_grad)
           
            x, x_grad = self.d_conv1d_10(x, x_grad, compute_grad, False)

        else:
            x, x_grad = self.d_conv1d_1(x, x_grad, compute_grad, False)
  
            x, x_grad = self.softplus(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor1(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor1_1(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor2(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor2_2(x, x_grad, compute_grad)           

            x, x_grad = self.ResNextFExtractor3(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor3_3(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor4(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor4_4(x, x_grad, compute_grad)

            x, x_grad = self.ResNextFExtractor5(x, x_grad, compute_grad)
            x, x_grad = self.ResNextFExtractor5_5(x, x_grad, compute_grad)
            
            x, x_grad = self.d_conv1d_10(x, x_grad, compute_grad, False)

        return x, x_grad, input_con




def KLD_gauss(mu,logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

class Network(nn.Module):
    def __init__(self, conf, latent_size, auto_decoder):
        super().__init__()
        
        self.latent_size = latent_size
        self.with_normals = conf.get_bool('encoder.with_normals')
        encoder_input_size = 6 if self.with_normals else 3
        self.encoder = ResnetPointnet(hidden_dim= latent_size, c_dim=latent_size, dim=encoder_input_size,normal_channel=False) if not auto_decoder and latent_size > 0 else None

        self.implicit_map = ImplicitMap_fExtrator(latent_size=latent_size, **conf.get_config('decoder_implicit'))
        #print(self.implicit_map)

        self.predict_normals_on_surfce = conf.get_bool('predict_normals_on_surfce')

        # count_parameters(self.encoder)
        # count_parameters(self.implicit_map)

        #import sys 
        #sys.exit()

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
            q_latent_mean, q_latent_std = self.encoder( encoder_input) 

            lq_mean = q_latent_mean
            q_z = dist.Normal(q_latent_mean, (torch.exp(q_latent_std)))
            latent = q_z.rsample() 
            kld = -0.5 *(1 + torch.exp(q_latent_std) - q_latent_mean.pow(2) - torch.exp(q_latent_std).exp())
            output['latent_reg'] = kld

            if only_encoder_forward:
                return latent, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None, None

        if only_decoder_forward:
            return self.implicit_map(manifold_points, latent, False)[0]
        else:

            non_mnfld_pred, non_mnfld_pred_grad, _ = self.implicit_map(sample_nonmnfld, latent, True, epoch=epoch)
            

            output['non_mnfld_pred_grad'] = non_mnfld_pred_grad
            output['non_mnfld_pred'] = non_mnfld_pred

            if not latent is None:
                output['norm_square_latent'] = (latent**2).mean(-1)

            if self.predict_normals_on_surfce:  
                _, grad_on_surface, _ = self.implicit_map(manifold_points, latent,  True)
                output['grad_on_surface'] = grad_on_surface

            return output

    
