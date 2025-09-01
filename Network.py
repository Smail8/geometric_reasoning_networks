import os
import sys
import copy
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from base_gnn import EdgeGATv2Conv

from utils import create_projections, convert_dimensions, convert_pose, get_corners, create_top_projection

FIXED = 0
MOVABLE = 1

# ========================================= Geometric Reasoning Network (GRN) =========================================
# ----------------------------------------- IK Module ------------------------------------------------
class IKModule(nn.Module):
    def __init__(self, num_features=7, hidden_size=512, dropout=0.0):
        super(IKModule, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features=num_features, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout))
                                    
        self.decoder = nn.Linear(in_features=hidden_size, out_features=5)

    def forward(self, x, mode="predict"):
        torch.use_deterministic_algorithms(False)
        enc = self.encoder(self.scale(x))
        pred = self.decoder(enc)
        if mode != "train":
            pred = pred.sigmoid()
        return pred
    
    def scale(self, x):
        x[:, 6] = (x[:, 6] % (2*np.pi))
        return x
    
# ----------------------------------------- GO Module ------------------------------------------------
class GOModule(nn.Module):
    def __init__(self, num_features=14, hidden_size=512, dropout=0.0):
        super(GOModule, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features=num_features, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.decoder = nn.Linear(in_features=hidden_size, out_features=5)

    def forward(self, x, mode="predict", mask=None):
        torch.use_deterministic_algorithms(False)
        enc = self.encoder(self.scale(x))
        pred = self.decoder(enc)
        pred = nn.functional.hardtanh(pred, min_val=0., max_val=1.)
        
        if mode != "train" and mask is not None:
            pred = pred * mask
        return pred
    
    def scale(self, x):
        x[:, 6] = (x[:, 6] % (2*np.pi))
        x[:, 13] = (x[:, 13] % (2*np.pi))
        return x
    
# ------------------------------------------ AGF Module ------------------------------------------------
class AGFModule(nn.Module):
    def __init__(self, num_node_features=7, num_edge_features=7, hidden_size=256):
        super(AGFModule, self).__init__()
        self.node_encoder = nn.Sequential(nn.Linear(num_node_features, hidden_size), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(num_edge_features, hidden_size), nn.ReLU())
        self.conv = EdgeGATv2Conv(in_channels=hidden_size, out_channels=hidden_size, heads=4, edge_dim=hidden_size, add_self_loops=False)
        self.decoder = nn.Sequential(nn.Linear(hidden_size*4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 6))
        
    def forward(self, data, mode):
        node_enc = self.node_encoder(data.x)
        attr_enc = self.edge_encoder(data.edge_attr)
        node_enc = self.conv(node_enc, data.edge_index, attr_enc)
        pred = self.decoder(node_enc)
        
        if mode != "train":
            pred = pred.sigmoid()

        return pred

# ------------------------------------------ GRN ------------------------------------------------
class GRN(nn.Module):
    def __init__(self, device="cuda"):
        super(GRN, self).__init__()
        self.device = device
        self.IKModule = IKModule()
        self.GOModule = GOModule()
        self.AGFModule = AGFModule()

    def forward(self, graph, mode, return_graph = False):
        graph.x = self.scale(graph.x)
        IK_preds = torch.ones((graph.x.shape[0], 5)).to(self.device)
        IK_masks = torch.ones((graph.x.shape[0], 5)).to(self.device)
        GO_preds = torch.zeros((graph.edge_attr.shape[0], 5)).to(self.device)
        
        graph.edge_attr = torch.cat((graph.edge_attr, torch.zeros(graph.edge_attr.shape[0], 5).to(self.device)), dim = 1)
        IK_preds[graph.mask] = self.IKModule(graph.x[graph.mask], "train")
        IK_masks[graph.mask] = IK_preds[graph.mask].sigmoid()
        neighbors = graph.edge_index[:, graph.blocking_mask]
        GO_input_features = torch.cat((graph.x[neighbors[1]], graph.x[neighbors[0]]), dim=1)
        GO_preds[graph.blocking_mask] = self.GOModule(GO_input_features, "predict", IK_masks[neighbors[1]])
        graph.edge_attr[graph.blocking_mask, 2:] = GO_preds[graph.blocking_mask]

        non_blocking_mask = torch.logical_not(graph.blocking_mask)
        graph.edge_attr[non_blocking_mask, 2:] = torch.ones(non_blocking_mask.sum(), 5).to(self.device) - IK_masks[graph.mask]

        F_preds = self.AGFModule(graph, mode)
        if mode != "train":
            IK_preds = IK_preds.sigmoid()
        if return_graph:
            return F_preds, IK_preds, GO_preds, graph
        else:
            return F_preds, IK_preds, GO_preds
    
    def predict_from_json(self, scene):
        graph = self.to_graph(scene)
        feasibility_preds, IK_features, GO_features, graph = self.forward(graph.to(self.device), "predict", return_graph = True)
        return feasibility_preds, IK_features, GO_features, graph
    
    def compute_distance(self, pose1, pose2):
        return np.linalg.norm(np.array(pose1) - np.array(pose2))

    def compute_threshold(self, dim1, dim2):
        return (max(dim1)+max(dim2)+0.6) / 2
    
    def is_neighbor(self, dim1, dim2, pose1, pose2):
        distance = self.compute_distance(pose1[:2], pose2[:2])
        threshold = self.compute_threshold(dim1[:2], dim2[:2])
        if distance > threshold:
            return False
        else:
            return True

    def scale(self, x):
        x[:, 6] = (x[:, 6] % (2*np.pi))
        return x

    def to_graph(self, scene):
        objects = list(scene["objects"].keys())
        indices = {obj: i for i, obj in enumerate(objects)}
        nodes = torch.zeros((len(objects), 7))
        mask = torch.zeros((len(objects)), dtype = bool)
        frame_ids = torch.zeros((len(objects)), dtype = int)
        IK_labels = torch.zeros((len(objects), 5))
        F_labels = torch.zeros((len(objects),6))
        GO_labels = torch.empty((0, 5))
        pos = torch.empty((0,4))
        edges = torch.empty(2, 0)
        blocking_mask = torch.empty(0, 1, dtype = bool)
        edge_features = torch.empty(0, 2)

        for obj in objects:
            scene["objects"] = self.compute_abs_poses(scene["objects"], obj)

        for i, obj in enumerate(objects):
            object_ = scene["objects"][obj]
            frame_id = object_["frame_id"]
            #===================================== Nodes =====================================
            if object_["fixed"]:
                mask[i] = FIXED
            else:
                mask[i] = MOVABLE
            node_features = object_["dimensions"] + object_["abs_pose"][:3] + [object_["abs_pose"][-1]]
            nodes[i] = torch.tensor(node_features).unsqueeze(0)
            if frame_id == "world":
                frame_ids[i] = -1
            else:
                frame_ids[i] = indices[frame_id]
            #===================================== Edges =====================================
            if not object_["fixed"]:
                edge = [indices[frame_id], indices[obj]]
                edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                blocking_mask = torch.cat((blocking_mask, torch.tensor([True]).unsqueeze(0)), dim = 0)
                edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                GO_labels = torch.cat((GO_labels, torch.tensor([-1, -1, -1, -1, -1]).unsqueeze(0)), dim = 0)

                for neighbor in objects:
                    if neighbor == obj or neighbor == frame_id or (frame_id != "base" and neighbor == "base"):    
                        continue
                    if not self.is_neighbor(scene["objects"][obj]["dimensions"], scene["objects"][neighbor]["dimensions"], 
                                            scene["objects"][obj]["abs_pose"], scene["objects"][neighbor]["abs_pose"]):
                        continue
                    edge = [indices[neighbor], indices[obj]]
                    edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                    blocking_mask = torch.cat((blocking_mask, torch.tensor([True]).unsqueeze(0)), dim = 0)
                    edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                    GO_labels = torch.cat((GO_labels, torch.tensor([-1, -1, -1, -1, -1]).unsqueeze(0)), dim = 0)

            if not object_["fixed"]:
                # Self-loop Edges
                edge = [indices[obj], indices[obj]]
                edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                blocking_mask = torch.cat((blocking_mask, torch.tensor([False]).unsqueeze(0)), dim = 0)
                edge_features = torch.cat((edge_features, torch.tensor([0, 1]).unsqueeze(0)), dim = 0)
                GO_labels = torch.cat((GO_labels, torch.tensor([0., 0., 0., 0., 0.]).unsqueeze(0)), dim = 0)

                # Proximity Edges
                edge = [indices[frame_id], indices[obj]]
                edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                blocking_mask = torch.cat((blocking_mask, torch.tensor([True]).unsqueeze(0)), dim = 0)
                edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                GO_labels = torch.cat((GO_labels, torch.tensor([-1, -1, -1, -1, -1]).unsqueeze(0)), dim = 0)
                
                neighbors = filter(lambda x: x != obj and x != frame_id and (frame_id == "base" or x != "base"), objects)
                for neighbor in neighbors:
                    if not self.is_neighbor(scene["objects"][obj]["dimensions"], scene["objects"][neighbor]["dimensions"], 
                                            scene["objects"][obj]["abs_pose"], scene["objects"][neighbor]["abs_pose"]):
                        continue
                    edge = [indices[neighbor], indices[obj]]
                    edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                    blocking_mask = torch.cat((blocking_mask, torch.tensor([True]).unsqueeze(0)), dim = 0)
                    edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                    GO_labels = torch.cat((GO_labels, torch.tensor([-1, -1, -1, -1, -1]).unsqueeze(0)), dim = 0)

        base_index = indices["base"]
        base_mask = torch.tensor([False if n != base_index else True for n in range(len(nodes))])
        graph = Data(x = nodes, mask = mask, frame_ids = frame_ids, 
                    edge_index = edges.long(), blocking_mask = blocking_mask.squeeze(1), edge_attr = edge_features.float(),
                    IK_labels = IK_labels, F_labels = F_labels, GO_labels = GO_labels, pos = pos, base_mask = base_mask)
            
        return graph

    def compute_abs_pose(self, object_rel_pose, frame_abs_pose):
        support_rot = np.array([[np.cos(frame_abs_pose[-1]), -1*np.sin(frame_abs_pose[-1]), 0],
                                [np.sin(frame_abs_pose[-1]), np.cos(frame_abs_pose[-1]), 0],
                                [0, 0, 1]])
        support_trans = np.array(frame_abs_pose[:3])
        object_trans = np.array(object_rel_pose[:3])
        abs_pose = np.matmul(support_rot, object_trans) + support_trans
        abs_yaw = frame_abs_pose[-1] + object_rel_pose[-1]
        return [abs_pose[0], abs_pose[1], abs_pose[2], 0., 0., abs_yaw]

    def compute_abs_poses(self, scene, object_id):
        if scene[object_id]["frame_id"] == "world" or scene[object_id]["frame_id"] == "odom_combined":
            scene[object_id]["abs_pose"] = copy.deepcopy(scene[object_id]["pose"])
        elif "abs_pose" not in scene[object_id]:
            if "abs_pose" not in scene[scene[object_id]["frame_id"]]:
                scene = self.compute_abs_poses(scene, scene[object_id]["frame_id"])
                
            scene[object_id]["abs_pose"] = self.compute_abs_pose(scene[object_id]["pose"], scene[scene[object_id]["frame_id"]]["abs_pose"])
        return scene


#=========================================== Feasibility-GAT / Feasibility-GCN ========================================
class GNN(nn.Module):
    def __init__(self, num_node_features=7, num_edge_features=2, hidden_size=512, gnn_type="GAT"):
        super(GNN, self).__init__()
        self.gnn_type = gnn_type
        self.node_encoder = nn.Sequential(nn.Linear(num_node_features, hidden_size), nn.ReLU())    
        if self.gnn_type == "GAT":
            self.edge_encoder = nn.Sequential(nn.Linear(num_edge_features, hidden_size), nn.ReLU())
            self.conv = gnn.GATv2Conv(in_channels=hidden_size, out_channels=hidden_size, heads=4, edge_dim=hidden_size, add_self_loops=False)
            self.decoder = nn.Sequential(nn.Linear(hidden_size*4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 6))
        elif self.gnn_type == "GCN":
            self.conv = gnn.GCNConv(in_channels=hidden_size, out_channels=hidden_size)
            self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 6))

    def forward(self, data, mode):
        node_enc = self.node_encoder(data.x)
        if self.gnn_type == "GAT":
            attr_enc = self.edge_encoder(data.edge_attr)
            node_enc = self.conv(node_enc, data.edge_index, attr_enc)
        else:
            node_enc = self.conv(node_enc, data.edge_index)
        pred = self.decoder(node_enc)
        
        if mode != "train":
            pred = pred.sigmoid()

# ========================================= CNN-Based Methods =========================================
class customResNet(nn.Module):
    def __init__(self, input_channels=2):
        super(customResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1], nn.Flatten())

    def forward(self, x):
        x = self.resnet(x)
        return x

# ------------------------------------------- AGFPNet ------------------------------------------------
class AGFPNet(nn.Module): 
    def __init__(self, args):
        super(AGFPNet, self).__init__()
        self.args = args
        self.top_view_encoder = customResNet(input_channels=2)
        self.xz_views_encoder = customResNet(input_channels=4)
        self.yz_views_encoder = customResNet(input_channels=4)
        self.scene_encoder = nn.Sequential(nn.Linear(in_features=1536, out_features=512), nn.ReLU(), nn.Dropout(args.dropout))
        self.decoder = nn.Sequential(nn.Linear(in_features=512, out_features=512), nn.ReLU(), nn.Dropout(args.dropout),
                                     nn.Linear(in_features=512, out_features=6))

    def forward(self, projections, mode="predict"):
        encoded_top_view = self.top_view_encoder(projections[:,:2])
        encoded_x_views = self.xz_views_encoder(projections[:,2:6])
        encoded_y_views = self.yz_views_encoder(projections[:,6:])
        encoded_views = torch.cat((encoded_top_view, encoded_x_views, encoded_y_views), dim=1)
        encoded_scene = self.scene_encoder(encoded_views)
        output = self.decoder(encoded_scene)
        if mode != "train":
            output = output.sigmoid()
            
        return output

    def predict_from_json(self, scene):
        preds = []
        scene_projections = create_projections(scene, self.args)/255.
        scene_projections = torch.tensor(scene_projections)
        nb_movable = 0
        for obj in scene["objects"]:
            if scene["objects"][obj]["fixed"]:
                continue
            nb_movable += 1

        for obj in scene["objects"]:
            if scene["objects"][obj]["fixed"]:
                continue
            projections  = torch.zeros(10,256,256, dtype=torch.float32)
            object_index = (scene_projections.shape[0]-nb_movable) + int(obj.replace("object", ""))
            projections_to_keep = [i for i in range(scene_projections.shape[0]) if i != object_index]
            projections[1::2, :, :] = scene_projections[object_index]
            projections[::2, :, :] = torch.max(scene_projections[projections_to_keep], axis=0).values
            projections = projections.to(self.args.device)
            pred = self.forward(projections.unsqueeze(0), "predict")
            preds.append(pred)
        return preds
    
# ------------------------------------------- DVH ------------------------------------------------
class DVH(nn.Module):
    def __init__(self, args):
        super(DVH, self).__init__()
        self.args = args
        self.scene_encoder = customResNet(input_channels=2)
        self.decoder = nn.Sequential(nn.Linear(in_features=512, out_features=100), nn.ReLU(), nn.Dropout(args.dropout),
                                     nn.Linear(in_features=100, out_features=6))
        
    def forward(self, projections, mode="predict"):
        encoded_scene = self.scene_encoder(projections)
        output = self.decoder(encoded_scene)
        if mode != "train":
            output = output.sigmoid()
            
        return output
    
    def predict_from_json(self, scene):
        preds = []
        scene_projections = np.zeros((len(list(scene["objects"].keys())), self.args.image_size, self.args.image_size))
        for i, object_id in enumerate(scene["objects"]):
            object_ = scene["objects"][object_id]
            corners = get_corners(convert_dimensions(object_["dimensions"], self.args.resolution), convert_pose(object_["abs_pose"], self.args))
            scene_projections[i,:,:] = create_top_projection(corners, self.args)
        scene_projections = torch.tensor(scene_projections)/255.
        nb_movable = 0
        for obj in scene["objects"]:
            if scene["objects"][obj]["fixed"]:
                continue
            nb_movable += 1

        for obj in scene["objects"]:
            if scene["objects"][obj]["fixed"]:
                continue

            projections  = torch.zeros(2,256,256, dtype=torch.float32)
            object_index = (scene_projections.shape[0]-nb_movable) + int(obj.replace("object", ""))
            projections_to_keep = [i for i in range(scene_projections.shape[0]) if i != object_index]
            projections[1, :, :] = scene_projections[object_index]
            projections[0, :, :] = torch.max(scene_projections[projections_to_keep], axis=0).values
            projections = projections.to(self.args.device)
            pred = self.forward(projections.unsqueeze(0), "predict")
            preds.append(pred)
        return preds
    
# =============================================== MLP ================================================
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=7, out_features=512), nn.ReLU(), 
                                 nn.Linear(in_features=512, out_features=512), nn.ReLU(),
                                 nn.Linear(in_features=512, out_features=6))
        
    def forward(self, x, mode="predict"):
        output = self.mlp(x)
        if mode != "train":
            output = output.sigmoid()
            
        return output
    
    def predict_from_json(self, scene, device="cuda"):
        preds = []
        for obj in scene["objects"]:
            if scene["objects"][obj]["fixed"]:
                continue
            x = torch.tensor([scene["objects"][obj]["dimensions"] + scene["objects"][obj]["abs_pose"][:3] + [scene["objects"][obj]["abs_pose"][-1]]]).to(device)
            pred = self.forward(x, "predict")
            preds.append(pred)
        return preds
