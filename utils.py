import json
import time
import fcl
import os
import cv2
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


WORKSPACE_DIAMETER = {"panda": 2.0, "pr2": 3.0, "thiago": 3.0}
ROBOT_LINKS = {"panda": {"base": {"dimensions": [0.3, 0.26, 0.6], 
                                  "abs_pose": [-0.05, 0., 0.3, 0., 0., 0.]},
                         "upper_body": {"dimensions": [0.26, 0.25, 0.5], 
                                        "abs_pose": [0.08, 0., 0.85, 0., 0., 0.]},
                         "eef": {"dimensions": [0.31, 0.2, 0.1], 
                                 "abs_pose": [0.23, 0., 1.15, 0., 0., 0.]}},
                "pr2":  {},
                "thiago": {}
              }
                         



def compute_abs_pose(object_rel_pose, frame_abs_pose):
    support_rot = np.array([[np.cos(frame_abs_pose[-1]), -np.sin(frame_abs_pose[-1]), 0],
                            [np.sin(frame_abs_pose[-1]), np.cos(frame_abs_pose[-1]), 0],
                            [0, 0, 1]])
    support_trans = np.array(frame_abs_pose[:3])
    object_trans = np.array(object_rel_pose[:3])
    abs_pose = np.matmul(support_rot, object_trans) + support_trans
    abs_yaw = frame_abs_pose[-1] + object_rel_pose[-1]
    return [abs_pose[0], abs_pose[1], abs_pose[2], 0., 0., abs_yaw]

def compute_abs_poses(scene, object_id):
    if scene[object_id]["frame_id"] == "world":
        scene[object_id]["abs_pose"] = copy.deepcopy(scene[object_id]["pose"])
    elif "abs_pose" not in scene[object_id] or not scene[object_id]["abs_pose"]:
        if "abs_pose" not in scene[scene[object_id]["frame_id"]] or not scene[scene[object_id]["frame_id"]]["abs_pose"]:
            scene = compute_abs_poses(scene, scene[object_id]["frame_id"])
        scene[object_id]["abs_pose"] = compute_abs_pose(scene[object_id]["pose"], scene[scene[object_id]["frame_id"]]["abs_pose"])
    return scene

def get_corners(dimensions, pose):
    half_length = dimensions[0]/2
    half_width = dimensions[1]/2
    half_height = dimensions[2]/2
    Trans = np.array([pose[0], pose[1], pose[2]])
    Rot = np.array([[np.cos(pose[-1]), -np.sin(pose[-1]), 0],
                    [np.sin(pose[-1]), np.cos(pose[-1]), 0],
                    [0, 0, 1]])
    
    corners = np.zeros((8,3))
    corners[0, :] = np.matmul(Rot, np.array([-half_length, -half_width, -half_height])) + Trans   #rear-bottom-left
    corners[1, :] = np.matmul(Rot, np.array([-half_length, half_width, -half_height])) + Trans    #rear-bottom-right
    corners[2, :] = np.matmul(Rot, np.array([-half_length, half_width, half_height])) + Trans     #rear-top-right
    corners[3, :] = np.matmul(Rot, np.array([-half_length, -half_width, half_height])) + Trans    #rear-top-left
    corners[4, :] = np.matmul(Rot, np.array([half_length, -half_width, half_height])) + Trans     #front-top-left
    corners[5, :] = np.matmul(Rot, np.array([half_length, -half_width, -half_height])) + Trans    #front-bottom-left
    corners[6, :] = np.matmul(Rot, np.array([half_length, half_width, -half_height])) + Trans     #front-bottom-right
    corners[7, :] = np.matmul(Rot, np.array([half_length, half_width, half_height])) + Trans      #front-top-right
    return corners

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------- Data Processing -------------------------------------------------------------------------------

def process_gnn_data(dataset_path):
    scenes_path = os.path.join(dataset_path, "scenes")
    data_annotated = pd.read_json(os.path.join(dataset_path, "data/data_annotated.json"))
    data_annotated = data_annotated[["scene_id", "object_id", "generation_type", "frame_id", "abs_x", "abs_y", "abs_z", "abs_yaw", "feasibility", "failing_stage", 
                                     "Top_F", "Front_F", "Rear_F", "Right_F", "Left_F", "Top_IC", "Front_IC", "Rear_IC", "Right_IC", "Left_IC",
                                     "Nb_Top_grasps", "Nb_Front_grasps", "Nb_Rear_grasps", "Nb_Right_grasps", "Nb_Left_grasps", "planning_time"]]
    data_annotated = data_annotated.dropna().sort_values(by=["scene_id"]).reset_index(drop=True)
    data_annotated["dim"] = 0
    data_annotated["pose"] = 0
    data_annotated["Top_IK"] = 0
    data_annotated["Front_IK"] = 0
    data_annotated["Rear_IK"] = 0
    data_annotated["Right_IK"] = 0
    data_annotated["Left_IK"] = 0
    data_annotated["Top_obstructors"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Front_obstructors"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Rear_obstructors"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Right_obstructors"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Left_obstructors"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Top_GO"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Front_GO"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Rear_GO"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Right_GO"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Left_GO"] = [[] for _ in range(data_annotated.shape[0])]
    data_annotated["Top_cause"] = 0
    data_annotated["Front_cause"] = 0
    data_annotated["Rear_cause"] = 0
    data_annotated["Right_cause"] = 0
    data_annotated["Left_cause"] = 0
    data_annotated = data_annotated.parallel_apply(process, args=(scenes_path,), axis=1)
    data_annotated = data_annotated.drop(columns=["Top_IC", "Front_IC", "Rear_IC", "Right_IC", "Left_IC"])
    data_annotated.to_json(os.path.join(dataset_path, "data/processed_gnn_data.json"))
    return data_annotated

def process(row, scenes_path):
    row = add_dim_pose_columns(row, scenes_path)
    for grasp in ["Top", "Front", "Rear", "Right", "Left"]:
        obstructors = []
        ratio_list = []

        # No Infeasibility Causes
        if not row[grasp+"_IC"]:
            row[grasp+"_IK"] = 1
            row[grasp+"_obstructors"] = []
            row[grasp+"_GO"] = []
            row[grasp+"_cause"] = "None"
            continue

        for i, ic in enumerate(row[grasp+"_IC"]):
            # ic is a pair (string, int) where string is the infeasibility cause and int is the number of infeasible grasps
            if ic[0] == "no_ik":
                if ic[1] == row["Nb_" + grasp + "_grasps"]: # all grasps infeasible due to no ik
                    row[grasp+"_IK"] = 0
                    row[grasp+"_cause"] = "IK"
                    break
                else:
                    row[grasp+"_IK"] = 1

            elif ic[0] == "robot" or "link" in ic[0]: # do not consider robot self collisions (supports panda and pr2)
                row[grasp+"_IK"] = 1

            else: # Infeasibility due to obstruction
                row[grasp+"_IK"] = 1
                obstructors.append(ic[0])
                ratio = ic[1] / row["Nb_" + grasp + "_grasps"]
                ratio_list.append(ratio)
                if ratio == 1 and row[grasp+"_cause"] != "IK":
                    row[grasp+"_cause"] = "Collision"

        if row[grasp+"_F"] == 0 and row[grasp+"_cause"] != "IK" and row[grasp+"_cause"] != "Collision":
            if row["failing_stage"] == "motion_planning" or row["failing_stage"] == "approach" or row["failing_stage"] == "lift":
                row[grasp+"_cause"] = "Motion Planning"
            else:
                row[grasp+"_cause"] = "Collision"
        row[grasp+"_obstructors"] = obstructors
        row[grasp+"_GO"] = ratio_list
    return row

def add_dim_pose_columns(row, scenes_path):
    with open(os.path.join(scenes_path, row.scene_id + ".json"), "r") as f:
        scene = json.load(f)
    row["dim"] = scene["objects"][row.object_id]["dimensions"]
    row["pose"] = scene["objects"][row.object_id]["abs_pose"][:3] + [scene["objects"][row.object_id]["abs_pose"][-1]]
    return row

def compute_distance(pose1, pose2):
    return np.linalg.norm(np.array(pose1) - np.array(pose2))

def compute_threshold(dim1, dim2, threshold):
    return (max(dim1) + max(dim2)+threshold) / 2

def process_go_data(dataset_path, proximity_threshold):
    data = pd.read_json(os.path.join(dataset_path, "data/processed_gnn_data.json"))
    go_data = {"scene_id": [], "o1": [], "o2": [], "o1_pose": [], "o2_pose": [], "o1_dim": [], "o2_dim": [],
               "Top_ratio": [], "Front_ratio": [], "Rear_ratio": [], "Right_ratio": [], "Left_ratio": [],
               "mTop": [], "mFront": [], "mRear": [], "mRight": [], "mLeft": []}

    for i in tqdm(range(len(data))):
        with open(os.path.join(dataset_path, "scenes", data.scene_id.iloc[i] + ".json"), "r") as f:
            scene = json.load(f)
        for o in scene["objects"]:
            if o == data.object_id.iloc[i] or o == data.frame_id.iloc[i] or o == "base":
                continue
            distance = compute_distance(scene["objects"][data.object_id.iloc[i]]["abs_pose"][:3], scene["objects"][o]["abs_pose"][:3])
            threshold = compute_threshold(scene["objects"][data.object_id.iloc[i]]["dimensions"][:2], scene["objects"][o]["dimensions"][:2], proximity_threshold)
            if distance > threshold:
                continue
            go_data["scene_id"].append(data.scene_id.iloc[i])
            go_data["o1"].append(data.object_id.iloc[i])
            go_data["o2"].append(o)
            go_data["o1_pose"].append(scene["objects"][data.object_id.iloc[i]]["abs_pose"][:3]+scene["objects"][data.object_id.iloc[i]]["abs_pose"][-1:])
            go_data["o2_pose"].append(scene["objects"][o]["abs_pose"][:3]+scene["objects"][o]["abs_pose"][-1:])
            go_data["o1_dim"].append(scene["objects"][data.object_id.iloc[i]]["dimensions"])
            go_data["o2_dim"].append(scene["objects"][o]["dimensions"])
            for grasp in ["Top", "Front", "Rear", "Right", "Left"]:
                if data[grasp+"_IK"].iloc[i] == 0:
                    go_data[grasp+"_ratio"].append(0)
                    go_data["m"+grasp].append(0)
                elif o in data[grasp+"_obstructors"].iloc[i]:
                    go_data[grasp+"_ratio"].append(data[grasp+"_GO"].iloc[i][data[grasp+"_obstructors"].iloc[i].index(o)])
                    go_data["m"+grasp].append(1)
                else:
                    go_data[grasp+"_ratio"].append(0)
                    go_data["m"+grasp].append(1)

    data = pd.DataFrame(go_data)
    data = data.drop(data[data.parallel_apply(lambda row: row.mTop == 0 and row.mFront == 0 and row.mRear == 0 and row.mLeft == 0 and row.mRight == 0, axis=1)].index)
    data.to_json(os.path.join(dataset_path, "data/processed_go_data.json"))

    inputs, labels, masks = to_tensors(data)
    torch.save(inputs, os.path.join(dataset_path, "data/GO_inputs.pt"))
    torch.save(labels, os.path.join(dataset_path, "data/GO_labels.pt"))
    torch.save(masks, os.path.join(dataset_path, "data/GO_masks.pt"))
    return data
    
def to_tensors(data):
    inputs = torch.zeros((len(data), 14))
    masks = torch.zeros((len(data), 5))
    inputs[:, :3] = torch.tensor(data.o1_dim.values.tolist())
    inputs[:, 3:7] = torch.tensor(data.o1_pose.values.tolist())
    inputs[:, 7:10] = torch.tensor(data.o2_dim.values.tolist())
    inputs[:, 10:14] = torch.tensor(data.o2_pose.values.tolist())
    labels = torch.tensor(data[["Top_ratio", "Front_ratio", "Rear_ratio", "Right_ratio", "Left_ratio"]].values.tolist())
    masks = torch.tensor(data[["mTop", "mFront", "mRear", "mRight", "mLeft"]].values.tolist())
    return inputs, labels, masks

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------- Data Analysis ------------------------------------------------------------------------------

def analyse_data(dataset_path):
    data = pd.read_json(os.path.join(dataset_path, "data/processed_gnn_data.json"))
    nb_scenes = data.scene_id.nunique()

    print("number of scenes : ", nb_scenes)
    print("number of datapoints : ", data.shape[0])

    plt.figure(figsize=(30,30))
    plt.subplot(3,3,1)
    data.frame_id.value_counts().plot.bar()
    plt.title("number of datapoints per support surface")
    plt.subplot(3,3,2)
    data.generation_type.value_counts().plot.bar()
    plt.title("number of place datapoints per generation type")
    plt.subplot(3,3,3)
    data[data.feasibility == 0].failing_stage.value_counts().plot.bar()
    plt.title("number of failing stage occurences")
    plt.subplot(3,3,4)
    data.feasibility.value_counts().plot.bar()
    plt.title("number of feasible vs infeasible datapoints")
    for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
        plt.subplot(3,3,g+5)
        data[grasp+"_F"].value_counts().plot.bar()
        plt.title("number of feasible vs infeasible "+grasp+" grasps")
    plt.show()

    plt.figure(figsize=(30,5))
    for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
        plt.subplot(1,5,g+1)
        data[data[grasp+"_F"] == 0][grasp+"_cause"].value_counts().plot.bar()
        plt.title(grasp+" grasps infeasibility causes")
    plt.show()

    plt.figure(figsize = (30,7))
    plt.subplot(1,4,1)
    plt.scatter(data[(data.frame_id == "base") & (data.feasibility == 1)].abs_x, 
                data[(data.frame_id == "base") & (data.feasibility == 1)].abs_y, c="green")
    plt.title("Coverage of feasible pick positions on ground")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xmin=-1,xmax=1)
    plt.ylim(ymin=-1,ymax=1)

    plt.subplot(1,4,2)
    plt.scatter(data[(data.frame_id == "base") & (data.feasibility == 0)].abs_x, 
                data[(data.frame_id == "base") & (data.feasibility == 0)].abs_y, c="red")
    plt.title("Coverage of infeasible pick positions on ground")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xmin=-1,xmax=1)
    plt.ylim(ymin=-1,ymax=1)

    plt.subplot(1,4,3)
    plt.scatter(data[(data.frame_id != "base") & (data.feasibility == 1)].abs_x, 
                data[(data.frame_id != "base") & (data.feasibility == 1)].abs_y, c="green")
    plt.title("Coverage of feasible pick positions on other support surfaces")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xmin=-1,xmax=1)
    plt.ylim(ymin=-1,ymax=1)

    plt.subplot(1,4,4)
    plt.scatter(data[(data.frame_id != "base") & (data.feasibility == 0)].abs_x, 
                data[(data.frame_id != "base") & (data.feasibility == 0)].abs_y, c="red")
    plt.title("Coverage of infeasible pick positions on other support surfaces")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xmin=-1,xmax=1)
    plt.ylim(ymin=-1,ymax=1)
    plt.show()

    df = data[["abs_x", "abs_y", "abs_z", "feasibility"]]
    fig = px.scatter_3d(df, x='abs_x', y='abs_y', z='abs_z',
                color='feasibility', opacity=0.5)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

    df = data[(data.feasibility == 0)]
    df = df[["abs_x", "abs_y", "abs_z", "failing_stage"]]
    fig = px.scatter_3d(df, x='abs_x', y='abs_y', z='abs_z',
                color='failing_stage', opacity=0.5)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

    df = data[(data.feasibility == 0)]
    df = df[["abs_x", "abs_y", "abs_z", "Front"]]
    fig = px.scatter_3d(df, x='abs_x', y='abs_y', z='abs_z',
                color='Front', opacity=0.5)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

    return data

def analyse_ik_data(dataset_path):
    data = pd.read_json(os.path.join(dataset_path, "data/processed_gnn_data.json"))
    print("number of datapoints : ", data.shape[0])
    plt.figure(figsize=(30,5))
    for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
        plt.subplot(1,5,g+1)
        data[grasp+"_IK"].value_counts().plot.bar()
        plt.title("Value count of "+grasp+" IK")
    plt.show()

def analyse_ic_data(dataset_path):
    data = pd.read_json(os.path.join(dataset_path, "data/processed_ic_data.json"))
    print("number of datapoints : ", data.shape[0])
    plt.figure(figsize=(30,15))
    for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
        plt.subplot(1,5,g+1)
        data[data["m"+grasp] == 1][grasp+"_ratio"].plot.hist(bins=20)
        plt.title("Histogram of blocking ratios of "+grasp+" grasps")
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------- Data Visualization -------------------------------------------------------------------------------
def get_mesh_points(data):
    vertices = []
    faces = []
    lines = data.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                if len(face) > 3: # triangulate the n-polyonal face, n>3
                    faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                else:
                    faces.append([face[j][0]-1 for j in range(len(face))])
            else: pass

    vertices = np.array(vertices)
    faces = np.array(faces)
    I, J, K =  faces.T
    x, y, z = vertices.T
    return x, y, z, I, J, K

def get_robot_mesh_points(robot):
    with open(os.path.join("robot_models", robot+".obj"), 'r') as file:
        data = file.read()
    return get_mesh_points(data)

def visualize_scene(scene, robot="panda", show=True):
    x, y, z, i, j, k = get_robot_mesh_points(robot)
    robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="grey", opacity=0.5, flatshading = True)
    traces = [robot_trace]
    colors = {False: "#E0E0E0", True: "#0C76BD"}
    for obj in scene["objects"]:
        corners = get_corners(scene["objects"][obj]["dimensions"], scene["objects"][obj]["abs_pose"])
        traces.append(go.Mesh3d(
            x=corners[:,0],
            y=corners[:,1],
            z=corners[:,2],
            i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
            j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
            k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
            opacity=1.,
            color=colors[not scene["objects"][obj]["fixed"]],
            flatshading = True,
            name = obj
        ))
    fig = go.Figure(data=traces)
    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
        )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    if show:
        fig.show()
    return fig

def visualize_scene_from_path(scene_path, show=True):
    with open(scene_path) as f:
        scene = json.load(f)
    return visualize_scene(scene, show=show)

def visualize_action_predictions(data, preds, robot="panda", show=True):
    x, y, z, i, j, k = get_robot_mesh_points(robot)
    robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="grey", opacity=0.2, flatshading=True)
    traces = [robot_trace]
    
    data.x = data.x.cpu()
    data.pos = data.pos.cpu()
    preds = preds.cpu()
    preds = torch.where(preds > 0.5, 1., 0.)
    for obj in range(data.pos.shape[0]):
        if not data.mask[obj].item():
            pred_color = "#E9E9E9"
            opacity = 0.2
        else:
            #color dependent on the value of preds[i, 0] from green to red
            pred_color = "#%02X%02X%02X" % (int(255*(1-preds[obj, 0])), int(255*preds[obj, 0]), 0)
            opacity = 1.

        corners = get_corners(data.x[obj, :3].tolist(), data.pos[obj, :].tolist())
        traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                    i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                    j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                    k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                    opacity=opacity, color=pred_color, flatshading=True, showscale=True))

    fig = go.Figure(data=traces)
    fig.update_layout(title="Action Feasibilty")
    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
        )
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    if show:
        fig.show()
    return fig

def visualize_grasp_predictions(data, preds, robot="panda"):
    x, y, z, i, j, k = get_robot_mesh_points(robot)
    robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='grey', opacity=0.2)
    data.x = data.x.cpu()
    data.pos = data.pos.cpu()
    # data.F_labels = data.F_labels.cpu()
    # data.IK_labels = data.IK_labels.cpu()
    # data.IC_labels = data.IC_labels.cpu()
    preds = preds.cpu()
    # targets = data.F_labels
    preds = torch.where(preds > 0.7, 1., 0.)

    traces = [robot_trace]
    for obj in range(data.pos.shape[0]):
        corners = get_corners(data.x[obj, :3].tolist(), data.pos[obj, :].tolist())
        if not data.mask[obj].item():
            pred_color = "#E9E9E9"
            opacity = 0.2
            traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                    i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                    j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                    k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                    opacity=opacity, color=pred_color, flatshading = True, showscale=True))
        else:
            #color dependent on the value of preds[i, 0] from green to red
            opacity = 1.
            facecolor = []
            for g in [1, 3, 2, 5, 4]:
                if preds[obj, g] > 0.5:
                    facecolor.extend(["#00FF00", "#00FF00"])
                else:
                    facecolor.extend(["#FF00000", "#FF00000"])
            facecolor.extend(["E9E9E9", "E9E9E9"])
            #facecolor = [top, top, rear, rear, front, front, left, left, right, right, bottom, bottom]
            traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                    i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                    j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                    k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                    facecolor=facecolor, opacity=opacity, flatshading = True, showscale=True))

    fig = go.Figure(data=traces)
    fig.update_layout(title="Grasp Feasibilty")
    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
        )
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    fig.show()

def visualize_go_predictions(data, IK_preds, IC_preds, main_obj, robot="panda", grasp="Top"):
    x, y, z, i, j, k = get_robot_mesh_points(robot)
    
    data.x = data.x.cpu()
    data.pos = data.pos.cpu()
    data.edge_index = data.edge_index[:, data.blocking_mask[:,0].cpu()].cpu()
    edges = torch.where(data.edge_index[1] == main_obj)[0]
    neighbors = data.edge_index[0, edges].tolist()
    # data.F_labels = data.F_labels.cpu()
    # data.IK_labels = data.IK_labels.cpu()
    # data.IC_labels = data.IC_labels.cpu()
    IC_preds = IC_preds[data.blocking_mask[:,0]][edges].cpu()
    # targets = data.F_labels

    #d = {}
    #for g, grasp in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
    grasp_types = ["Top", "Front", "Rear", "Right", "Left"]
    # grasp = "Top"
    g = grasp_types.index(grasp)
    if IK_preds[main_obj, g] > 0.5:
        robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='grey', opacity=0.2)
        traces = [robot_trace]
        for obj in range(data.pos.shape[0]):
            corners = get_corners(data.x[obj, :3].tolist(), data.pos[obj, :].tolist())
            if obj != main_obj and obj not in neighbors:
                opacity = 0.1
                if not data.mask[obj].item():
                    pred_color = "#E9E9E9"
                else:
                    pred_color = "#0C76BD"
                traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                        i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                        j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                        k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                        opacity=opacity, color=pred_color, flatshading = True, showscale=True))
            elif obj == main_obj:
                opacity = 1.
                pred_color = "#0C76BD"
                traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                        i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                        j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                        k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                        opacity=opacity, color=pred_color, flatshading = True, showscale=True))
            else:
                opacity = 1.
                intensity = [IC_preds[neighbors.index(obj), g].item() for i in range(12)]
                traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                        i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                        j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                        k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                        opacity=opacity, intensity=intensity, cmin=0., cmax=1., 
                                        colorscale='YlOrRd', flatshading = True, showscale=False))
    else:
        robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='red', opacity=0.2)
        traces = [robot_trace]
        for obj in range(data.pos.shape[0]):
            corners = get_corners(data.x[obj, :3].tolist(), data.pos[obj, :].tolist())
            if obj != main_obj:
                opacity = 0.1
            else:
                opacity = 1.
            if not data.mask[obj].item():
                pred_color = "#E9E9E9"
            else:
                pred_color = "#0C76BD"
            traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                    i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                    j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                    k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                    opacity=opacity, color=pred_color, flatshading = True, showscale=True))
        #d[grasp] = traces

    fig = go.Figure(data=traces)
    fig.update_layout(title=grasp + " Grasp Obstructions")
    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
        )
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    fig.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------- Projections Generation ---------------------------------------------------------------------

def convert_dimensions(dimensions, resolution):
        return np.array(dimensions) * resolution

def convert_pose(pose, args):
    p = np.array(pose)
    p[:3] *= args.resolution
    p[:3] += np.array([args.image_size/2, args.image_size/2, args.image_size/4])
    p[:3] = np.round(p[:3], decimals=1)
    return p

def create_top_projection(corners, args):
    horizontal, vertical, normal, direction = 0, 1, 2, -1
    view = np.zeros((args.image_size, args.image_size))
    pts = corners[np.argsort(corners[:, normal])[::direction], :].astype(int)[:4,:]
    pts = pts[np.argsort(pts[:, vertical]), :]
    tmp1 = pts[:2, :]
    tmp2 = pts[2:, :]
    tmp1 = tmp1[np.argsort(tmp1[:, horizontal]), :]
    tmp2 = tmp2[np.argsort(tmp2[:, horizontal]), :]
    pts = np.concatenate((tmp1[0, :].reshape(1,-1), tmp2, tmp1[1,:].reshape(1,-1)))
    depth = np.minimum(np.abs((pts[0,normal])/args.image_size)*255, 255)
    cv2.fillPoly(view, pts=[pts[:, [horizontal, vertical]]], color=int(depth))
    return view

def create_side_view(corners, axes, direction, args):
    horizontal, vertical, normal = axes
    view = np.zeros((args.image_size, args.image_size))
    pts = corners[np.lexsort((corners[:, horizontal], corners[:,normal]))[::direction], :].astype(int)[:6,:]
    center_corners = pts[:2, :]
    left_corners = pts[2:4, :]
    right_corners = pts[4:6, :]
    center_corners = center_corners[np.argsort(center_corners[:,2]), :]
    left_corners = left_corners[np.argsort(left_corners[:,2]), :]
    right_corners = right_corners[np.argsort(right_corners[:,2]), :]
    for side_corners in [left_corners, right_corners]:
        num_lines = np.abs(center_corners[0,horizontal] - side_corners[0,horizontal])
        if num_lines == 0:
            continue
        depth_step = np.abs(center_corners[0,normal] - side_corners[0,normal])/num_lines
        for i in range(num_lines+1):
            delta_depth = depth_step*i
            if center_corners[0,horizontal] < side_corners[0,horizontal]:
                offset = np.array([i,0])
            else:
                offset = np.array([-i,0])

            if direction == -1:
                depth = np.minimum(np.abs((center_corners[0,normal] + (direction*delta_depth))/args.image_size)*255, 255)
            elif direction == 1:
                depth = np.minimum(np.abs((center_corners[0,normal] - args.image_size + (direction*delta_depth))/args.image_size)*255, 255)

            cv2.line(view, center_corners[0,[horizontal,vertical]]+offset, center_corners[1,[horizontal,vertical]]+offset, int(depth))
    return view

def create_projections(scene_, args):
    nb_objects = len(list(scene_["objects"].keys()))
    scene_views = np.zeros((nb_objects, 5, args.image_size, args.image_size))
    for i, object_id in enumerate(scene_["objects"]):
        object_ = scene_["objects"][object_id]
        corners = get_corners(convert_dimensions(object_["dimensions"], args.resolution), convert_pose(object_["abs_pose"], args))
        scene_views[i,0,:,:] = create_top_projection(corners, args)
        scene_views[i,1,:,:] = create_side_view(corners, [1, 2, 0], -1, args)
        scene_views[i,2,:,:] = create_side_view(corners, [1, 2, 0], 1, args)
        scene_views[i,3,:,:] = create_side_view(corners, [0, 2, 1], 1, args)
        scene_views[i,4,:,:] = create_side_view(corners, [0, 2, 1], -1, args)

    return scene_views

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------