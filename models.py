import torch
import torch.nn as nn
import torch.nn.functional as F

# DGCNN implementation based on https://github.com/antao97/dgcnn.pytorch/blob/master/model.py


class DGCNN_cls(nn.Module):
    def __init__(self, input_dims, device, config):
        super(DGCNN_cls, self).__init__()

        self.k = config.k
        self.num_points = config.num_points
        self.num_classes = config.num_classes
        self.dims = input_dims
        self.device = device

        # Elemental info
        if self.dims > 3:
            ele_output = 9
            self.ele_conv1d = nn.Conv1d(in_channels=self.dims-3, out_channels=ele_output, kernel_size=1, bias=False)
            self.combined_dims = 3 + ele_output
        else:
            self.combined_dims = self.dims

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(config.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.combined_dims * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, config.emb_dims, kernel_size=1, bias=False),
                                self.bn5,
                                nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(config.emb_dims, config.emb_dims, bias=False)
        self.bn6 = nn.BatchNorm1d(config.emb_dims)
        self.dp1 = nn.Dropout(p=0) 
        self.linear2 = nn.Linear(config.emb_dims, int(config.emb_dims/2), bias=False)
        self.bn7 = nn.BatchNorm1d(int(config.emb_dims/2))
        self.dp2 = nn.Dropout(p=0) 
        self.linear3 = nn.Linear(int(config.emb_dims/2), self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Ele processing
        if self.dims > 3:
            x0 = self.ele_conv1d(x[:,3:,:]) # using conv1
            x = torch.cat([x[:,0:3,:],x0], dim=1)

        x = get_graph_feature(x, k=self.k, dim4=True, device=self.device)  # (batch_size, x, num_points) -> (batch_size, x*2, num_points, k)
        x = self.conv1(x)  # (batch_size, dims*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, dim4=False, device=self.device)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, dim4=False, device=self.device)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x3, k=self.k, dim4=False, device=self.device)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        point_feat = self.conv5(x) # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        shape_feat = F.adaptive_avg_pool1d(point_feat, 1).view(batch_size, -1) 

        x = F.leaky_relu(self.bn6(self.linear1(shape_feat)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)

        return x, shape_feat

class DGCNN_cls_encoder(nn.Module):
    def __init__(self, input_dims, config, device, return_point_feat=False):
        super(DGCNN_cls_encoder, self).__init__()

        self.k = config.k
        self.num_points = config.num_points
        self.num_classes = config.num_classes
        self.dims = input_dims 
        self.return_point_feat = return_point_feat
        self.device = device

        # Elemental info
        if self.dims > 3:
            ele_output = 9
            self.ele_conv1d = nn.Conv1d(in_channels=self.dims-3, out_channels=ele_output, kernel_size=1, bias=False)
            self.combined_dims = 3 + ele_output
        else:
            self.combined_dims = self.dims

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(self.combined_dims * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, config.emb_dims, kernel_size=1, bias=False))
        
    def forward(self, x): # x: (BS,dim,num_points)
        batch_size = x.shape[0]

        # Ele processing
        if self.dims > 3:
            x0 = self.ele_conv1d(x[:,3:,:]) # using conv1
            x = torch.cat([x[:,0:3,:],x0], dim=1)

        x = get_graph_feature(x, k=self.k, dim4=True, device=self.device)  # (batch_size, x, num_points) -> (batch_size, x*2, num_points, k)
        x = self.conv1(x)  # (batch_size, dims*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, dim4=False, device=self.device)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, dim4=False, device=self.device)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x3, k=self.k, dim4=False, device=self.device)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = torch.mean(x, dim=-1, keepdim=False)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        point_feat = self.conv5(x) # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        shape_feat = F.adaptive_avg_pool1d(point_feat, 1).view(batch_size, -1) 

        if self.return_point_feat:
            return shape_feat, point_feat
        
        else:
            return shape_feat

class MLP_Projection(nn.Module):
    def __init__(self, config):
        super(MLP_Projection, self).__init__()

        self.linear1 = nn.Linear(config.emb_dims, 1024, bias=True)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0)
        self.linear2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=0)
        self.linear3 = nn.Linear(1024, 64)

    def forward(self, x):
            
        x = F.relu(self.bn1(self.linear1(x))) 
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x))) 
        x = self.dp2(x)
        x = self.linear3(x)                    

        return x 

class ContrastiveModel(nn.Module):
    def __init__(self, config, input_dims,device):
        super(ContrastiveModel, self).__init__()

        self.encoder = DGCNN_cls_encoder(config=config,input_dims=input_dims,device=device)
        self.decoder = MLP_Projection(config)

    def forward(self, input, only_enc=False):
        
        encoder_feature = self.encoder(input)

        if only_enc:
            return encoder_feature

        projection_feature = self.decoder(encoder_feature)
            
        return projection_feature, encoder_feature

def get_graph_feature(x, k=20, idx=None, dim4=False, device="cuda"): 
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim4 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 0:3], k=k)
    device = torch.device(device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() 

    return feature  # (batch_size, 2*num_dims, num_points, k)


def knn(x, k):  # helpfer function for dgcnn
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
