import os
import json
import torch
import argparse
import pandas as pd
from utils import create_projections, WORKSPACE_DIAMETER
from tqdm import tqdm

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Create projections from scenes json files')

    parser.add_argument('--path', type=str, 
                        help='Dataset path',
                        required=True)
    
    parser.add_argument('--image_size', type=int, 
                        help='Image size',
                        required=False,
                        default=256)
    
    args = parser.parse_args()

    return args

def generate_projections(args):
    sets = []
    if os.path.isdir(os.path.join(args.path, "train_set")):
        sets.append("train_set")
    if os.path.isdir(os.path.join(args.path, "val_set")):
        sets.append("val_set")
    if os.path.isdir(os.path.join(args.path, "test_set")):
        sets.append("test_set")
    for set_ in sets:
        scenes_path = os.path.join(args.path, set_, "scenes")
        projections_path = os.path.join(args.path, set_, "projections")
        dataset = pd.read_json(os.path.join(args.path, set_, "data/processed_gnn_data.json"))
        scenes = dataset.scene_id.unique()
        for i, scene_name in enumerate(tqdm(scenes)):
            scene_path = os.path.join(scenes_path, scene_name+".json")
            with open(scene_path) as f:
                scene = json.load(f)
            scene_projections = create_projections(scene, args)/255.
            scene_projections = torch.tensor(scene_projections)
            if i == 0:
                torch.save(scene_projections[0,0].min(), os.path.join(projections_path, "base_value.pt"))
            scene_projections[0,0] -= scene_projections[0,0].min()
            scene_projections = scene_projections.to_sparse()
            torch.save(scene_projections, os.path.join(projections_path, scene_name+".pt"))

if __name__ == '__main__':
    args = parse_args()
    if "panda" in args.path:
        args.robot = "panda"
    elif "pr2" in args.path:
        args.robot = "pr2"
    elif "thiago" in args.path:
        args.robot = "thiago"
    args.resolution = args.image_size/WORKSPACE_DIAMETER[args.robot]
    print("Creating projections for dataset: ", args.path, " with image size: ", args.image_size, " and resolution: ", args.resolution, "...")
    generate_projections(args)
    print("Projections creation completed!")