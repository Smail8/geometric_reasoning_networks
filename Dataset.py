import os
import copy
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import seed
from training_utils import *
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate

from utils import create_projections

seed(0)
torch.manual_seed(0)

FIXED = 0
MOVABLE = 1

# ========================== Dataset for GRN ==========================
class GRNDataset(InMemoryDataset):
    def __init__(self, path, mode, args):
        self.args = args
        self.mode = mode
        super().__init__(root=path)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, "scenes"))

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return ["grouped_augmented_data.pt"]
        else:
            return ["grouped_data.pt"]

    def process(self):
        gnn_dataset = pd.read_json(os.path.join(self.root, "data", "processed_gnn_data.json"))
        dataset = gnn_dataset.groupby("scene_id").agg(list)
        data_list = []
        for scene_name in tqdm(gnn_dataset.scene_id.unique()):
            datapoint = dataset.loc[scene_name]
            movable_indices = {obj: i for i, obj in enumerate(datapoint.object_id)}
            with open(os.path.join(self.root, "scenes", scene_name + ".json")) as f:
                scene = json.load(f)

            objects = list(scene["objects"].keys())
            indices = {obj: i for i, obj in enumerate(objects)}
            nodes = torch.zeros((len(objects), 7))
            mask = torch.zeros((len(objects)), dtype = bool)
            frame_ids = torch.zeros((len(objects)), dtype = int)
            IK_labels = torch.zeros((len(objects), 5))
            F_labels = torch.empty((len(objects),6))
            GO_labels = torch.empty((0, 5))
            pos = torch.empty((0,4))
            edges = torch.empty(2, 0)
            blocking_mask = torch.empty(0, 1, dtype = bool)
            edge_features = torch.empty(0, 2)

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
                #===================================== Labels =====================================
                if object_["fixed"]:
                    IK_labels[i] = torch.tensor([-1, -1, -1, -1, -1])
                    F_labels[i] = torch.tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0)
                else:
                    IK_labels[i] = torch.tensor([datapoint.Top_IK[movable_indices[obj]], datapoint.Front_IK[movable_indices[obj]], 
                                                 datapoint.Rear_IK[movable_indices[obj]], datapoint.Right_IK[movable_indices[obj]], 
                                                 datapoint.Left_IK[movable_indices[obj]]])
                    F_labels[i] = torch.tensor([datapoint.feasibility[movable_indices[obj]], datapoint.Top_F[movable_indices[obj]],
                                                datapoint.Front_F[movable_indices[obj]], datapoint.Rear_F[movable_indices[obj]],
                                                datapoint.Right_F[movable_indices[obj]], datapoint.Left_F[movable_indices[obj]]]).unsqueeze(0)

                pos = torch.cat((pos, torch.tensor(object_["abs_pose"][:3] + [object_["abs_pose"][-1]]).unsqueeze(0)), dim = 0)
                #===================================== Edges =====================================
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
                    ratios = torch.tensor([0., 0., 0., 0., 0.])
                    for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
                        if frame_id in datapoint[grasp+"_obstructors"][movable_indices[obj]]:
                            ratios[g] = datapoint[grasp+"_GO"][movable_indices[obj]][datapoint[grasp+"_obstructors"][movable_indices[obj]].index(frame_id)]
                    GO_labels = torch.cat((GO_labels, ratios.unsqueeze(0)), dim = 0)
                    
                    neighbors = filter(lambda x: x != obj and x != frame_id and (frame_id == "base" or x != "base"), objects)
                    for neighbor in neighbors:
                        if not self.is_neighbor(scene["objects"][obj]["dimensions"], scene["objects"][neighbor]["dimensions"], 
                                                scene["objects"][obj]["abs_pose"], scene["objects"][neighbor]["abs_pose"]):
                            continue
                        edge = [indices[neighbor], indices[obj]]
                        edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                        blocking_mask = torch.cat((blocking_mask, torch.tensor([True]).unsqueeze(0)), dim = 0)
                        edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                        ratios = torch.tensor([0., 0., 0., 0., 0.])
                        for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
                            if neighbor in datapoint[grasp+"_obstructors"][movable_indices[obj]]:
                                ratios[g] = datapoint[grasp+"_GO"][movable_indices[obj]][datapoint[grasp+"_obstructors"][movable_indices[obj]].index(neighbor)]
                        GO_labels = torch.cat((GO_labels, ratios.unsqueeze(0)), dim = 0)

            base_index = indices["base"]
            base_mask = torch.tensor([False if n != base_index else True for n in range(len(nodes))])
            data = Data(x = nodes, mask = mask, frame_ids = frame_ids, 
                        edge_index = edges.long(), blocking_mask = blocking_mask.squeeze(1), edge_attr = edge_features.float(),
                        IK_labels = IK_labels, F_labels = F_labels, GO_labels = GO_labels, pos = pos, base_mask = base_mask,
                        scene = torch.tensor(int(scene_name.replace("scene_", ""))))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data) 
            data_list.append(copy.deepcopy(data))

            # ======================================= Data Augmentation =======================================    
            if self.mode == "train":
                # switch dimensions of fixed objects
                dimswitched = copy.deepcopy(data)
                for _ in range(3):
                    dimswitched = self.dimswitch_fixed(dimswitched)
                    data_list.append(copy.deepcopy(dimswitched))
                
                dimswitched = copy.deepcopy(data)
                for _ in range(3):
                    # switch dimensions of movable objects
                    dimswitched = self.dimswitch_movable(dimswitched)
                    data_list.append(copy.deepcopy(dimswitched))
                    # switch dimensions of fixed objects after dimension switch of movable objects
                    dimswitched_fixed = copy.deepcopy(dimswitched)
                    for _ in range(3):
                        dimswitched_fixed = self.dimswitch_fixed(dimswitched_fixed)
                        data_list.append(copy.deepcopy(dimswitched_fixed))
                    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def compute_distance(self, pose1, pose2):
        return np.linalg.norm(np.array(pose1) - np.array(pose2))

    def compute_threshold(self, dim1, dim2):
        return (max(dim1) + max(dim2)+0.6) / 2
    
    def is_neighbor(self, dim1, dim2, pose1, pose2):
        distance = self.compute_distance(pose1[:2], pose2[:2])
        threshold = self.compute_threshold(dim1[:2], dim2[:2])
        if distance > threshold:
            return False
        else:
            return True
    
    def get(self, idx):
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        if self.args.IK_GO_mode == "gt":
            data.edge_attr = torch.cat((data.edge_attr, data.GO_labels), dim = 1)
            non_blocking_mask = torch.logical_not(data.blocking_mask)
            data.edge_attr[non_blocking_mask, 2:] = torch.ones(non_blocking_mask.sum(), 5) - data.IK_labels[data.mask]

        if hasattr(self.args, "add_noise"):
            if self.args.add_noise:
                metric_noise = torch.normal(mean = 0, std = self.args.position_noise_std, size = (data.x.shape[0]-1, 6))
                angle_noise = torch.normal(mean = 0, std = self.args.orientation_noise_std, size = (data.x.shape[0]-1, 1))
                noise = torch.cat((metric_noise, angle_noise), dim = 1)
                data.x[1:, :] += noise

        return self.scale(data)
    
    def scale(self, data):
        data.x[:, 6] = (data.x[:, 6] % (2*np.pi))
        return data

    def dimswitch_movable(self, data):
        movable_indices = data.mask == MOVABLE
        switched = copy.deepcopy(data)
        switched.x[movable_indices, 0], switched.x[movable_indices, 1] = data.x[movable_indices, 1], data.x[movable_indices, 0]
        switched.x[movable_indices, 6] = (data.x[movable_indices, 6] + np.pi/2) % (2*np.pi)
        comb = [(2, 5), (3, 4), (4, 2), (5, 3)]
        for i, j in comb:
            switched.F_labels[:, i] = data.F_labels[:, j]
            switched.IK_labels[:, i-1] = data.IK_labels[:, j-1]
            switched.GO_labels[:, i-1] = data.GO_labels[:, j-1]
        return switched
    
    def dimswitch_fixed(self, data):
        fixed_indices = torch.logical_and(data.mask == FIXED, data.base_mask == False)
        switched = copy.deepcopy(data)
        switched.x[fixed_indices, 0], switched.x[fixed_indices, 1] = data.x[fixed_indices, 1], data.x[fixed_indices, 0]
        switched.x[fixed_indices, 6] = (data.x[fixed_indices, 6] + np.pi/2) % (2*np.pi)
        return switched

# ========================== Dataset for GO ==========================
class GODataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.mode = mode
        self.args = args
        self.inputs = torch.load(os.path.join(path, "data", "GO_inputs.pt")).to(args.device)
        self.labels = torch.load(os.path.join(path, "data", "GO_labels.pt")).to(args.device).float()
        self.masks = torch.load(os.path.join(path, "data", "GO_masks.pt")).to(args.device).bool()
        # Data augmentation
        if self.mode == "train":
            self.inputs, self.labels, self.masks = self.augment()

        self.scaled_inputs = self.scale()
        self.scaled_labels = self.labels
        if self.mode == "train":
            #shuffle the data
            indices = torch.randperm(self.scaled_inputs.shape[0])
            self.scaled_inputs = self.scaled_inputs[indices]
            self.scaled_labels = self.scaled_labels[indices]
        
    def __len__(self):
        return self.scaled_inputs.shape[0]
    
    def __getitem__(self, index):
        x = self.scaled_inputs[index]
        label = self.scaled_labels[index]
        mask = self.masks[index]
        return x, label, mask

    def get(self, index):
        x = self.scaled_inputs[index]
        label = self.scaled_labels[index]
        mask = self.masks[index]
        datapoint = self.data.iloc[index]
        return x, label, mask, datapoint
    
    def scale(self):
        scaled_inputs = copy.deepcopy(self.inputs)
        scaled_inputs[:, 6] = (self.inputs[:, 6] % (2*np.pi))
        scaled_inputs[:, 13] = (self.inputs[:, 13] % (2*np.pi))
        return scaled_inputs

    def dimswitch(self, inputs, labels, masks, obj_idx):
        length, width = copy.deepcopy(inputs[:, 7*obj_idx]), copy.deepcopy(inputs[:, 7*obj_idx+1])
        inputs[:, 7*obj_idx], inputs[:, 7*obj_idx+1] = width, length
        inputs[:, 7*obj_idx+6] = (inputs[:, 7*obj_idx+6] + np.pi/2) % (2*np.pi)
        if obj_idx == 0:
            f, re, ri, l = copy.deepcopy(labels[:, 1]), copy.deepcopy(labels[:, 2]), copy.deepcopy(labels[:, 3]), copy.deepcopy(labels[:, 4])
            labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4] = l, ri, f, re
            mf, mre, mri, ml = copy.deepcopy(masks[:, 1]), copy.deepcopy(masks[:, 2]), copy.deepcopy(masks[:, 3]), copy.deepcopy(masks[:, 4])
            masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4] = ml, mri, mf, mre
        return inputs, labels, masks
    
    def augment(self):
        augmented_inputs, augmented_labels, augmented_masks = copy.deepcopy(self.inputs), copy.deepcopy(self.labels), copy.deepcopy(self.masks)
        switched_inputs, switched_labels, switched_masks = copy.deepcopy(self.inputs), copy.deepcopy(self.labels), copy.deepcopy(self.masks)
        for i in range(3):
            switched_inputs, switched_labels, switched_masks = self.dimswitch(switched_inputs, switched_labels, switched_masks, 1)
            augmented_inputs = torch.cat((augmented_inputs, copy.deepcopy(switched_inputs)))
            augmented_labels = torch.cat((augmented_labels, copy.deepcopy(switched_labels)))
            augmented_masks = torch.cat((augmented_masks, copy.deepcopy(switched_masks)))
        
        switched_inputs, switched_labels, switched_masks = copy.deepcopy(augmented_inputs), copy.deepcopy(augmented_labels), copy.deepcopy(augmented_masks)
        for i in range(3):
            switched_inputs, switched_labels, switched_masks = self.dimswitch(switched_inputs, switched_labels, switched_masks, 0)
            augmented_inputs = torch.cat((augmented_inputs, copy.deepcopy(switched_inputs)))
            augmented_labels = torch.cat((augmented_labels, copy.deepcopy(switched_labels)))
            augmented_masks = torch.cat((augmented_masks, copy.deepcopy(switched_masks)))

        return augmented_inputs, augmented_labels, augmented_masks
    
# ========================== Dataset for IK ==========================
class IKDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.mode = mode
        self.args = args
        self.data = pd.read_json(os.path.join(path, "data", "processed_gnn_data.json"))
        inputs = torch.zeros((len(self.data), 7))
        labels = torch.zeros((len(self.data), 6))
        inputs[:, :3] = torch.tensor(self.data.dim.values.tolist())
        inputs[:, 3:] = torch.tensor(self.data.pose.values.tolist())
        labels = torch.tensor(self.data[["Top_IK", "Front_IK", "Rear_IK", "Right_IK", "Left_IK"]].values.tolist())
        self.inputs = inputs.to(args.device)
        self.labels = labels.float().to(args.device)
        if self.mode == "train":
            switched_inputs = copy.deepcopy(self.inputs)
            switched_labels = copy.deepcopy(self.labels)
            for i in range(3):
                switched_inputs, switched_labels = self.dimswitch(switched_inputs, switched_labels)
                self.inputs = torch.cat((self.inputs, copy.deepcopy(switched_inputs)))
                self.labels = torch.cat((self.labels, copy.deepcopy(switched_labels)))
            
        self.scaled_inputs = self.scale().to(args.device)
        self.scaled_labels = self.labels.to(args.device)
        if self.mode == "train":
            indices = torch.randperm(self.scaled_inputs.shape[0])
            self.scaled_inputs = self.scaled_inputs[indices]
            self.scaled_labels = self.scaled_labels[indices]
                   
    def __len__(self):
        return self.scaled_inputs.shape[0]
    
    def __getitem__(self, index):
        x = self.scaled_inputs[index]
        label = self.scaled_labels[index]
        return x, label
    
    def get(self, index):
        x = self.scaled_inputs[index]
        label = self.scaled_labels[index]
        datapoint = self.data.iloc[index]
        return x, label, datapoint
    
    def scale(self):
        scaled_inputs = copy.deepcopy(self.inputs)
        scaled_inputs[:, 6] = (self.inputs[:, 6] % (2*np.pi))
        return scaled_inputs

    def dimswitch(self, inputs, labels):
        length, width = copy.deepcopy(inputs[:, 0]), copy.deepcopy(inputs[:, 1])
        inputs[:, 0], inputs[:, 1] = width, length
        inputs[:, 6] = (inputs[:, 6] + np.pi/2) % (2*np.pi)
        f, re, ri, l = copy.deepcopy(labels[:, 1]), copy.deepcopy(labels[:, 2]), copy.deepcopy(labels[:, 3]), copy.deepcopy(labels[:, 4])
        labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4] = l, ri, f, re
        return inputs, labels
    
# ========================== Dataset for GNN baselines ==========================
class GNNDataset(InMemoryDataset):
    def __init__(self, path, mode, args):
        super().__init__(root=path)
        self.args = args
        self.mode = mode
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, "scenes"))

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return ["GNN_augmented_data.pt"]
        else:
            return ["GNN_data.pt"]

    def process(self):
        gnn_dataset = pd.read_json(os.path.join(self.root, "data", "processed_gnn_data.json"))
        dataset = gnn_dataset.groupby("scene_id").agg(list)
        data_list = []
        for scene_name in tqdm(gnn_dataset.scene_id.unique()):
            datapoint = dataset.loc[scene_name]
            movable_indices = {obj: i for i, obj in enumerate(datapoint.object_id)}
            with open(os.path.join(self.root, "scenes", scene_name + ".json")) as f:
                scene = json.load(f)

            objects = list(scene["objects"].keys())
            indices = {obj: i for i, obj in enumerate(objects)}
            nodes = torch.zeros((len(objects), 7))
            mask = torch.zeros((len(objects)), dtype = bool)
            frame_ids = torch.zeros((len(objects)), dtype = int)
            F_labels = torch.empty((len(objects),6))
            pos = torch.empty((0,4))
            edges = torch.empty(2, 0)
            edge_features = torch.empty(0, 2)

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
                if frame_id == "world" or frame_id == "odom_combined":
                    frame_ids[i] = -1
                else:
                    frame_ids[i] = indices[frame_id]
                #===================================== Labels =====================================
                if object_["fixed"]:
                    F_labels[i] = torch.tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0)
                else:
                    F_labels[i] = torch.tensor([datapoint.feasibility[movable_indices[obj]], datapoint.Top_F[movable_indices[obj]],
                                                datapoint.Front_F[movable_indices[obj]], datapoint.Rear_F[movable_indices[obj]],
                                                datapoint.Right_F[movable_indices[obj]], datapoint.Left_F[movable_indices[obj]]]).unsqueeze(0)

                pos = torch.cat((pos, torch.tensor(object_["abs_pose"][:3] + [object_["abs_pose"][-1]]).unsqueeze(0)), dim = 0)
                #===================================== Edges =====================================
                if frame_id == "world":
                    continue

                if not object_["fixed"]:
                    edge = [indices[obj], indices[obj]]
                    edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                    edge_features = torch.cat((edge_features, torch.tensor([0, 0]).unsqueeze(0)), dim = 0)
                
                if not object_["support"]:
                    if "holder" not in obj:
                        edge = [indices[obj], indices[frame_id]]
                    else:
                        edge = [indices[obj], indices["base"]]
                    edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                    edges = torch.cat((edges, torch.tensor(edge[::-1]).unsqueeze(1)), dim=1)
                    edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                    edge_features = torch.cat((edge_features, torch.tensor([0, 1]).unsqueeze(0)), dim = 0)

                elif object_["support"]:
                    if "_" in obj:
                        structure_id, support_id = obj.split("_")
                        holders = [o for o in objects if structure_id in o and "holder" in o]
                    else:
                        holders = []
                        
                    if not holders:
                        holders = ["base"]
                        
                    for holder in holders:
                        edge = [indices[obj], indices[holder]]
                        edges = torch.cat((edges, torch.tensor(edge).unsqueeze(1)), dim=1)
                        edges = torch.cat((edges, torch.tensor(edge[::-1]).unsqueeze(1)), dim=1)
                        edge_features = torch.cat((edge_features, torch.tensor([1, 0]).unsqueeze(0)), dim = 0)
                        edge_features = torch.cat((edge_features, torch.tensor([0, 1]).unsqueeze(0)), dim = 0)

            base_index = indices["base"]
            base_mask = torch.tensor([False if n != base_index else True for n in range(len(nodes))])
            data = Data(x = nodes, mask = mask, frame_ids = frame_ids, 
                        edge_index = edges.long(), edge_attr = edge_features.float(),
                        F_labels = F_labels, pos = pos, base_mask = base_mask,
                        scene = torch.tensor(int(scene_name.replace("scene_", ""))))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data) 
            
            data_list.append(copy.deepcopy(data))
            #------------------------------------------------------------------------------------------    
            if self.mode == "train":
                dimswitched = copy.deepcopy(data)
                for _ in range(3):
                    dimswitched = self.dimswitch_fixed(dimswitched)
                    data_list.append(copy.deepcopy(dimswitched))
                
                dimswitched = copy.deepcopy(data)
                for _ in range(3):
                    dimswitched = self.dimswitch_movable(dimswitched)
                    data_list.append(copy.deepcopy(dimswitched))

                    dimswitched_fixed = copy.deepcopy(dimswitched)
                    for _ in range(3):
                        dimswitched_fixed = self.dimswitch_fixed(dimswitched_fixed)
                        data_list.append(copy.deepcopy(dimswitched_fixed))
                    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def get(self, idx):
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        if hasattr(self.args, "add_noise"):
            if self.args.add_noise:
                metric_noise = torch.normal(mean = 0, std = self.args.position_noise_std, size = (data.x.shape[0]-1, 6))
                angle_noise = torch.normal(mean = 0, std = self.args.orientation_noise_std, size = (data.x.shape[0]-1, 1))
                noise = torch.cat((metric_noise, angle_noise), dim = 1)
                data.x[1:, :] += noise

        return self.scale(data)
    
    def scale(self, data):
        data.x[:, 6] = (data.x[:, 6] % (2*np.pi))
        return data

    def dimswitch_movable(self, data):
        movable_indices = data.mask == MOVABLE
        switched = copy.deepcopy(data)
        switched.x[movable_indices, 0], switched.x[movable_indices, 1] = data.x[movable_indices, 1], data.x[movable_indices, 0]
        switched.x[movable_indices, 6] = (data.x[movable_indices, 6] + np.pi/2) % (2*np.pi)
        comb = [(2, 5), (3, 4), (4, 2), (5, 3)]
        for i, j in comb:
            switched.F_labels[:, i] = data.F_labels[:, j]
        return switched
    
    def dimswitch_fixed(self, data):
        fixed_indices = torch.logical_and(data.mask == FIXED, data.base_mask == False)
        switched = copy.deepcopy(data)
        switched.x[fixed_indices, 0], switched.x[fixed_indices, 1] = data.x[fixed_indices, 1], data.x[fixed_indices, 0]
        switched.x[fixed_indices, 6] = (data.x[fixed_indices, 6] + np.pi/2) % (2*np.pi)
        return switched

# ========================== Dataset for AGFPNet ==========================
class AGFPNetDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.scenes_path = os.path.join(path, "scenes")
        self.projections_path = os.path.join(path, "projections")
        self.data = pd.read_json(os.path.join(path, "data", "processed_gnn_data.json"))
        self.data = self.data.sort_index()
        self.base_value = torch.load(os.path.join(self.projections_path, "base_value.pt"))
        self.args = args
        self.mode = mode
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        datapoint = self.data.loc[index]
        projections  = torch.zeros(10,256,256, dtype=torch.float32)       
        projections[::2, :, :], projections[1::2, :, :] = self.load_scene_projections(datapoint)
        Top_F, Front_F, Rear_F, Right_F, Left_F = self.process_grasp(datapoint)
        labels = torch.tensor([datapoint.feasibility, Top_F, Front_F, Rear_F, Right_F, Left_F], dtype=torch.float32)
        return projections, labels
        
    def load_scene_projections(self, datapoint):
        if os.path.isfile(os.path.join(self.projections_path, datapoint.scene_id+".pt")): 
            scene_projections = torch.load(os.path.join(self.projections_path, datapoint.scene_id+".pt"))
            scene_projections = scene_projections.to_dense()
            scene_projections[0, 0] += self.base_value
        else:
            scene_path = os.path.join(self.scenes_path, datapoint.scene_id+".json")
            with open(scene_path) as f:
                scene = json.load(f)
            scene_projections = create_projections(scene, self.args)/255.
            scene_projections = torch.tensor(scene_projections)

        object_index = (scene_projections.shape[0]-self.args.nb_objects) + int(datapoint.object_id.replace("object", "")) # TODO: find better way to do this (nb_objects)
        projections_to_keep = [i for i in range(scene_projections.shape[0]) if i != object_index]
        object_mask = scene_projections[object_index]
        scene_projections = torch.max(scene_projections[projections_to_keep], axis=0).values
        return scene_projections, object_mask
    
    def process_grasp(self, datapoint):
        Top_F = datapoint.Top_F
        yaw = datapoint.abs_yaw
        while yaw > np.pi:
            yaw = yaw - np.sign(yaw)*2*np.pi
        
        angle_to_robot = np.arctan2(datapoint.abs_y, datapoint.abs_x)
        angle_diff = yaw - angle_to_robot
        if abs(angle_diff) > np.pi:
            angle_diff = angle_diff - np.sign(angle_diff)*2*np.pi
            
        if abs(angle_diff) <= np.pi/4:
            Front_F = datapoint.Front_F
            Rear_F = datapoint.Rear_F
            Right_F = datapoint.Right_F
            Left_F = datapoint.Left_F
            
        elif angle_diff > np.pi/4 and angle_diff <= 3*np.pi/4:
            Front_F = datapoint.Right_F
            Rear_F = datapoint.Left_F
            Right_F = datapoint.Rear_F   
            Left_F = datapoint.Front_F
            
        elif abs(angle_diff) > 3*np.pi/4:
            Front_F = datapoint.Rear_F
            Rear_F = datapoint.Front_F
            Right_F = datapoint.Left_F
            Left_F = datapoint.Right_F
            
        elif angle_diff < -np.pi/4 and angle_diff >= -3*np.pi/4:
            Front_F = datapoint.Left_F
            Rear_F = datapoint.Right_F
            Right_F = datapoint.Front_F
            Left_F = datapoint.Rear_F
                
        return Top_F, Front_F, Rear_F, Right_F, Left_F
    
# ========================== Dataset for DVH ==========================
class DVHDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.scenes_path = os.path.join(path, "scenes")
        self.projections_path = os.path.join(path, "projections")
        self.data = pd.read_json(os.path.join(path, "data", "processed_gnn_data.json"))
        self.data = self.data.sort_index()  
        self.base_value = torch.load(os.path.join(self.projections_path, "base_value.pt"))
        self.args = args
        self.mode = mode
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        datapoint = self.data.loc[index]
        projections  = torch.zeros(2,256,256, dtype=torch.float32)       
        projections[0, :, :], projections[1, :, :] = self.load_scene_projections(datapoint)
        Top_F, Front_F, Rear_F, Right_F, Left_F = self.process_grasp(datapoint)
        labels = torch.tensor([datapoint.feasibility, Top_F, Front_F, Rear_F, Right_F, Left_F], dtype=torch.float32)
        return projections, labels
        
    def load_scene_projections(self, datapoint):
        if os.path.isfile(os.path.join(self.projections_path, datapoint.scene_id+".pt")): 
            scene_projections = torch.load(os.path.join(self.projections_path, datapoint.scene_id+".pt"))
            scene_projections = scene_projections.to_dense()
            scene_projections[0, 0] += self.base_value
        else:
            scene_path = os.path.join(self.scenes_path, datapoint.scene_id+".json")
            with open(scene_path) as f:
                scene = json.load(f)
            scene_projections = create_projections(scene, self.args)/255.
            scene_projections = torch.tensor(scene_projections)

        object_index = (scene_projections.shape[0]-self.args.nb_objects) + int(datapoint.object_id.replace("object", "")) # TODO: find better way to do this (nb_objects)
        projections_to_keep = [i for i in range(scene_projections.shape[0]) if i != object_index]
        object_mask = scene_projections[object_index, 0]
        scene_projections = torch.max(scene_projections[projections_to_keep, 0], axis=0).values
        return scene_projections, object_mask
    
    def process_grasp(self, datapoint):
        Top_F = datapoint.Top_F
        yaw = datapoint.abs_yaw
        while yaw > np.pi:
            yaw = yaw - np.sign(yaw)*2*np.pi
        
        angle_to_robot = np.arctan2(datapoint.abs_y, datapoint.abs_x)
        angle_diff = yaw - angle_to_robot
        if abs(angle_diff) > np.pi:
            angle_diff = angle_diff - np.sign(angle_diff)*2*np.pi
            
        if abs(angle_diff) <= np.pi/4:
            Front_F = datapoint.Front_F
            Rear_F = datapoint.Rear_F
            Right_F = datapoint.Right_F
            Left_F = datapoint.Left_F
            
        elif angle_diff > np.pi/4 and angle_diff <= 3*np.pi/4:
            Front_F = datapoint.Right_F
            Rear_F = datapoint.Left_F
            Right_F = datapoint.Rear_F   
            Left_F = datapoint.Front_F
            
        elif abs(angle_diff) > 3*np.pi/4:
            Front_F = datapoint.Rear_F
            Rear_F = datapoint.Front_F
            Right_F = datapoint.Left_F
            Left_F = datapoint.Right_F
            
        elif angle_diff < -np.pi/4 and angle_diff >= -3*np.pi/4:
            Front_F = datapoint.Left_F
            Rear_F = datapoint.Right_F
            Right_F = datapoint.Front_F
            Left_F = datapoint.Rear_F
                
        return Top_F, Front_F, Rear_F, Right_F, Left_F
    
# ========================== Dataset for MLP ==========================
class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, args):
        self.scenes_path = os.path.join(path, "scenes")
        self.data = pd.read_json(os.path.join(path, "data", "processed_gnn_data.json"))
        self.data = self.data.sort_index()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        datapoint = self.data.loc[index]
        inputs = torch.tensor(datapoint.dim + datapoint.pose)
        labels = torch.tensor([datapoint.feasibility, datapoint.Top_F, datapoint.Front_F, datapoint.Rear_F, datapoint.Right_F, datapoint.Left_F], dtype=torch.float32)
        return inputs, labels
    

