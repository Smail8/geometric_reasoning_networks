import os
import sys
import json
import cv2
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import get_robot_mesh_points, get_corners

def plot_confusion_matrix(cm, classes=[0,1], normalize=False, title='Confusion Matrix', cmap=plt.cm.Reds):
    """ 
    Function to plot a sklearn confusion matrix, showing number of cases per prediction condition 
    
    Args:
        cm         an sklearn confusion matrix
        classes    levels of the class being predicted; default to binary outcome
        normalize  apply normalization by setting `normalize=True`
        title      title for the plot
        cmap       color map
    """
    #plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round (cm[i, j],2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    return plt.gcf()

def visualize_scene_from_graph(graph, show=True):
    x, y, z, i, j, k = get_robot_mesh_points()
    robot_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="grey", opacity=0.2, flatshading = True)
    traces = [robot_trace]
    colors = {False: "#E0E0E0", True: "#0C76BD"}
    for i in range(graph.pos.shape[0]):
        corners = get_corners(graph.x[i, :3].tolist(), graph.pos[i, :].tolist())
        traces.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                opacity=1., color=colors[graph.mask[i].item()], flatshading = True))
    if show:
        fig = go.Figure(graph=traces)
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
        fig.show()
    else:
        return traces


def visualize_predictions(data, preds):
    targets = data.F_labels
    for j, output in enumerate(["Action Feasibilty", "Top Grasp Feasibilty", 
                                "Front Grasp Feasibilty", "Rear Grasp Feasibilty", 
                                "Right Grasp Feasibilty", "Left Grasp Feasibilty"]):
        pred_data = []
        target_data = []
        direction_data = []
        for i in range(data.pos.shape[0]):
            if not data.mask[i].item():
                pred_color = "#E9E9E9"
                target_color = "#E9E9E9"
            else:
                #color dependent on the value of preds[i, 0] from green to red
                pred_color = "#%02X%02X%02X" % (int(255*(1-preds[i, j])), int(255*preds[i, j]), 0)
                target_color = "#%02X%02X%02X" % (int(255*(1-targets[i, j])), int(255*targets[i, j]), 0)

            corners = get_corners(data.x[i, :3].tolist(), data.pos[i, :].tolist())
            pred_data.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                       i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                       j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                       k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                       opacity=1., color=pred_color, flatshading = True, showscale=True))
            target_data.append(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                                         i = [7, 2, 0, 0, 4, 4, 6, 6, 4, 0, 0, 0],
                                         j = [3, 3, 1, 2, 5, 6, 7, 2, 0, 3, 6, 1],
                                         k = [4, 7, 2, 3, 6, 7, 2, 1, 5, 4, 5, 6],
                                         opacity=1., color=target_color, flatshading = True, showscale=True))
            
            if data.mask[i].item():
                trace = go.Cone(x=[data.x[i, 3]], y=[data.x[i, 4]], z=[data.x[i, 5] + data.x[i, 2]/2 + 0.1],
                                u=[0.5*np.cos(data.x[i, 6])], v=[0.5*np.sin(data.x[i, 6])], w=[0], showscale=False)
                direction_data.append(trace)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Predictions", "Ground Truth"), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
        for trace in pred_data:
            fig.add_trace(trace, row=1, col=1)
        for trace in direction_data:
            fig.add_trace(trace, row=1, col=1)
        for trace in target_data:
            fig.add_trace(trace, row=1, col=2)
        for trace in direction_data:
            fig.add_trace(trace, row=1, col=2)
        fig.update_layout(title_text=output)
        fig.show()
    
    
def visualize_graph(data, preds=None, show_blocking_edges=True, show=True):
    Xn = [data.pos[k][0].item() for k in range(data.pos.shape[0])]# x-coordinates of nodes
    Yn = [data.pos[k][1].item() for k in range(data.pos.shape[0])]# y-coordinates
    Zn = [data.pos[k][2].item() for k in range(data.pos.shape[0])]# z-coordinates

    node_color = []
    for i in range(data.mask.shape[0]):
        if data.mask[i].item():
            if preds is not None:
                node_color.append("#%02X%02X%02X" % (int(255*(1-preds[i, 0].item())), int(255*preds[i, 0].item()), 0))
            else:
                node_color.append("#%02X%02X%02X" % (int(255*(1-data.F_labels[i, 0].item())), int(255*data.F_labels[i, 0].item()), 0))
        else:
            node_color.append("#E9E9E9")

    node_text = []
    for i in range(len(Xn)):
        if data.mask[i].item():
            text = "Feasibility: <br>"
            for j, task in enumerate(["Action", "Top", "Front", "Rear", "Right", "Left"]):
                text += f"{task : <15}"
                text += f"{data.F_labels[i, j].item()}<br>"
            text += "IK: <br>"
            for j, task in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
                text += f"{task : <15}"
                text += f"{data.IK_labels[i, j].item()}<br>"
        else:
            text = ""
        
        node_text.append(text)

    nodes_trace=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', marker=dict(symbol='circle', size=10, color=node_color, line=dict(color='rgb(50,50,50)', width=0.5)),
                             showlegend=False, hoverinfo='text', hovertext=node_text)

    contact_edges = data.edge_index[:, torch.logical_not(data.blocking_mask[:,0])]
    contact_edges = list(set([tuple(sorted(tuple((int(contact_edges[0,i]), int(contact_edges[1,i]))))) for i in range(contact_edges.shape[1])]))
    
    Xc, Yc, Zc = [], [], []
    for e in contact_edges:
        Xc += [data.pos[e[0]][0].item(), data.pos[e[1]][0].item(), None]# x-coordinates of edge ends
        Yc += [data.pos[e[0]][1].item(), data.pos[e[1]][1].item(), None]
        Zc += [data.pos[e[0]][2].item(),data.pos[e[1]][2].item(), None]

    contact_edges_trace=go.Scatter3d(x=Xc, y=Yc, z=Zc, mode='lines', line=dict(color='rgb(125,125,125)', width=3), showlegend=False, hoverinfo='none')

    blocking_edges = data.edge_index[:, data.blocking_mask[:,0]].t()
    blocking_attributes = torch.max(data.IC_labels[data.blocking_mask[:,0]], dim=1).values
    non_blocking_edges = blocking_edges[blocking_attributes == 0].tolist()
    blocking_edges = blocking_edges[blocking_attributes != 0].tolist()
    blocking_attributes = data.IC_labels[data.blocking_mask[:,0]][blocking_attributes != 0] 

    
    Xnbe, Ynbe, Znbe = [], [], []
    seen = []
    for e in non_blocking_edges:
        if data.mask[e[0]] and data.mask[e[1]]:
            if tuple(e[::-1]) not in seen:
                offset = 0.01
            else:
                offset = -0.01
        else:
            offset = 0.0
        if tuple(e[::-1]) not in seen:
            seen.append(tuple(e))
        Xnbe += [data.pos[e[0]][0].item(), data.pos[e[1]][0].item(), None]
        Ynbe += [data.pos[e[0]][1].item(), data.pos[e[1]][1].item(), None]
        Znbe += [data.pos[e[0]][2].item()+offset,data.pos[e[1]][2].item()+offset, None]

    non_blocking_edges_trace=go.Scatter3d(x=Xnbe, y=Ynbe, z=Znbe, mode='lines', line=dict(color='rgb(0,0,255)', width=3), showlegend=False, hoverinfo='none')

    Xbe, Ybe, Zbe = [], [], []
    Xd, Yd, Zd = [], [], []
    edge_text = []
    if show_blocking_edges:
        for i, e in enumerate(blocking_edges):
            if data.mask[e[0]] and data.mask[e[1]]:
                if tuple(e[::-1]) not in seen:
                    offset = 0.01
                else:
                    offset = -0.01
            else:
                offset = 0.0

            if tuple(e[::-1]) not in seen:
                seen.append(tuple(e))

            Xbe += [data.pos[e[0]][0].item(), data.pos[e[1]][0].item(), None]
            Ybe += [data.pos[e[0]][1].item(), data.pos[e[1]][1].item(), None]
            Zbe += [data.pos[e[0]][2].item()+offset,data.pos[e[1]][2].item()+offset, None]
                
            mid_point = [(data.pos[e[0]][0].item() + data.pos[e[1]][0].item())/2, 
                         (data.pos[e[0]][1].item() + data.pos[e[1]][1].item())/2, 
                         (data.pos[e[0]][2].item() + offset + data.pos[e[1]][2].item() + offset)/2]
            Xd.append(mid_point[0])
            Yd.append(mid_point[1])
            Zd.append(mid_point[2])
            text = "IC: <br>"
            for j, task in enumerate(["Top", "Front", "Rear", "Right", "Left"]):
                text += f"{task : <15}"
                text += f"{blocking_attributes[i, j].item()}<br>"
            edge_text.append(text)

    blocking_edges_trace=go.Scatter3d(x=Xbe, y=Ybe, z=Zbe, mode='lines', line=dict(color='rgb(255,0,0)', width=3), showlegend=False, hoverinfo='none')

    mid_point_trace=go.Scatter3d(x=Xd, y=Yd, z=Zd, mode='markers', marker=dict(symbol='circle', size=2, color="#000000", line=dict(color='rgb(50,50,50)', width=0.5)), 
                                showlegend=False, hoverinfo='text', hovertext=edge_text)
    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(width=500, height=500, showlegend=False, scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis),), hovermode='closest')

    d=[nodes_trace, contact_edges_trace]
    if show_blocking_edges:
        d.append(non_blocking_edges_trace)
        d.append(blocking_edges_trace)
        d.append(mid_point_trace)

    if show:
        fig=go.Figure(data=d, layout=layout)
        fig.show()
    else:
        return d, layout
    
def format_node_info(data):
    pred = "pred"
    label = "label"
    offset = "offset"
    text = ""
    for i, key in enumerate(data):
        title = key
        text += f"{title : <19}" 
        if i == 0:
            text += "Preds  |  labels<br>"
        else:
            text += "<br>"
        
        for task in data[key]:
            text += f"==> {task : <{data[key][task][offset]}}"
            text += f"{data[key][task][pred]:.2f}  |  "
            text += f"{data[key][task][label]:.2f}<br>"
            
    return f"""{text}"""

def format_edge_info(data):
    pred = "pred"
    label = "label"
    offset = "offset"
    text = ""
    title = "IC :"
    text += f"{title : <19}"
    text += "Preds  |  labels<br>"
    for task in data:
        text += f"==> {task : <{data[task][offset]}}"
        text += f"{data[task][pred]:.2f}  |  "
        text += f"{data[task][label]:.2f}<br>"
    return f"""{text}"""
    
def visualize_grn_graph(data, feasibility_preds, IK_preds, IC_preds, show=True):
    Xn=[data.pos[k][0].item() for k in range(data.pos.shape[0])]# x-coordinates of nodes
    Yn=[data.pos[k][1].item() for k in range(data.pos.shape[0])]# y-coordinates
    Zn=[data.pos[k][2].item() for k in range(data.pos.shape[0])]# z-coordinates

    node_color = []
    for i in range(data.mask.shape[0]):
        if data.mask[i].item():
            node_color.append("#000080")
        else:
            node_color.append("#E9E9E9")

    node_text = []
    for i in range(len(Xn)):
        if data.mask[i].item():
            dic = {
                "Feasibility :": 
                {"Action :": {"pred": feasibility_preds[i,0].item(), "label": data.F_labels[i, 0].item(), "offset": 15},
                    "Top :" : {"pred": feasibility_preds[i,1].item(), "label": data.F_labels[i, 1].item(), "offset": 16},
                    "Front :" : {"pred": feasibility_preds[i,2].item(), "label": data.F_labels[i, 2].item(), "offset": 15},
                    "Rear :" : {"pred": feasibility_preds[i,3].item(), "label": data.F_labels[i, 3].item(), "offset": 15},
                    "Right :": {"pred": feasibility_preds[i,4].item(), "label": data.F_labels[i, 4].item(), "offset": 15},
                    "Left :": {"pred": feasibility_preds[i,5].item(), "label": data.F_labels[i, 5].item(), "offset": 16}
                },
                "IK :": 
                {"Top :": {"pred": IK_preds[i, 0].item(), "label": data.IK_labels[i, 0].item(), "offset": 15},
                    "Front :": {"pred": IK_preds[i, 1].item(), "label": data.IK_labels[i, 1].item(), "offset": 14},
                    "Rear :": {"pred": IK_preds[i, 2].item(), "label": data.IK_labels[i, 2].item(), "offset": 14},
                    "Right :": {"pred": IK_preds[i, 3].item(), "label": data.IK_labels[i, 3].item(), "offset": 14},
                    "Left :": {"pred": IK_preds[i, 4].item(), "label": data.IK_labels[i, 4].item(), "offset": 15}
                },
            }    
            
            text = format_node_info(dic)
        else:
            text = ""
        
        node_text.append(text)
            
    nodes_trace=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', marker=dict(symbol='circle', size=6, color=node_color, line=dict(color='rgb(50,50,50)', width=0.5)),
                                showlegend=False, hoverinfo='text', hovertext=node_text)

    non_blocking_edges = data.edge_index[:, torch.logical_not(data.blocking_mask[:,0])]
    blocking_edges = data.edge_index[:, data.blocking_mask[:,0]].t().tolist()
    non_blocking_edges = list(set([tuple(sorted(tuple((int(non_blocking_edges[0,i]), int(non_blocking_edges[1,i]))))) for i in range(non_blocking_edges.shape[1])]))

    Xnbe, Ynbe, Znbe = [], [], []
    for e in non_blocking_edges:
        Xnbe+=[data.pos[e[0]][0].item(), data.pos[e[1]][0].item(), None]# x-coordinates of edge ends
        Ynbe+=[data.pos[e[0]][1].item(), data.pos[e[1]][1].item(), None]
        Znbe+=[data.pos[e[0]][2].item(),data.pos[e[1]][2].item(), None] 

    Xbe, Ybe, Zbe = [], [], []
    Xd, Yd, Zd = [], [], []
    edge_text = []
    seen = {}
    k=0
    for i, e in enumerate(blocking_edges):
        if tuple(e[::-1]) not in seen:
            seen[tuple(e)] = k
            k+=1
            Xbe+=[data.pos[e[0]][0].item(), data.pos[e[1]][0].item(), None]
            Ybe+=[data.pos[e[0]][1].item(), data.pos[e[1]][1].item(), None]
            Zbe+=[data.pos[e[0]][2].item(),data.pos[e[1]][2].item(), None]
            mid_point = [(data.pos[e[0]][0].item() + data.pos[e[1]][0].item())/2, 
                        (data.pos[e[0]][1].item() + data.pos[e[1]][1].item())/2, 
                        (data.pos[e[0]][2].item() + data.pos[e[1]][2].item())/2]
            Xd.append(mid_point[0])
            Yd.append(mid_point[1])
            Zd.append(mid_point[2])
            dic = {
                    "IC": 
                    {"Top :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 0].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 0].item(), "offset": 15},
                        "Front :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 1].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 1].item(), "offset": 14},
                        "Rear :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 2].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 2].item(), "offset": 14},
                        "Right :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 3].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 3].item(), "offset": 14},
                        "Left :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 4].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 4].item(), "offset": 15}
                    },
                }
            edge_text.append(format_edge_info(dic["IC"]))
        else:
            idx = seen[tuple(e[::-1])]
            dic = {
                    "IC2": 
                    {"Top :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 0].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 0].item(), "offset": 15},
                        "Front :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 1].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 1].item(), "offset": 14},
                        "Rear :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 2].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 2].item(), "offset": 14},
                        "Right :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 3].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 3].item(), "offset": 14},
                        "Left :": {"pred": IC_preds[data.blocking_mask[:, 0]][i, 4].item(), "label": data.IC_labels[data.blocking_mask[:, 0]][i, 4].item(), "offset": 15}
                    },
                }
            edge_text[idx] += format_edge_info(dic["IC2"])

        
    non_blocking_edges_trace=go.Scatter3d(x=Xnbe, y=Ynbe, z=Znbe, mode='lines', line=dict(color='rgb(125,125,125)', width=1), showlegend=False, hoverinfo='none')
    blocking_edges_trace=go.Scatter3d(x=Xbe, y=Ybe, z=Zbe, mode='lines', line=dict(color='rgb(255,0,0)', width=3), showlegend=False, hoverinfo='none')
    mid_point_trace=go.Scatter3d(x=Xd, y=Yd, z=Zd, mode='markers', marker=dict(symbol='circle', size=2, color="#FF0000", line=dict(color='rgb(50,50,50)', width=0.5)), 
                                showlegend=False, hoverinfo='text', hovertext=edge_text)

    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')
    layout = go.Layout(width=500, height=500, showlegend=False, scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis),), hovermode='closest')
    d=[nodes_trace, non_blocking_edges_trace, blocking_edges_trace, mid_point_trace]

    if show:
        fig=go.Figure(data=d, layout=layout)
        fig.show()
    else:
        return d, layout