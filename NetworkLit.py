import copy
import torch
import torchmetrics
from Network import *
from training_utils import *
import pytorch_lightning as pl
from lightning.pytorch.utilities import grad_norm
pl.seed_everything(0, workers=True)

ACTION = 0
TOP = 1
FRONT = 2
REAR = 3
RIGHT = 4
LEFT = 5

class ClassificationMetrics(object):
    def __init__(self, device="cuda"):
        self.F1Score = torchmetrics.F1Score(task="binary").to(device)
        self.Precision = torchmetrics.Precision(task="binary").to(device)
        self.Recall = torchmetrics.Recall(task="binary").to(device)
        self.ConfusionMatrix = torchmetrics.ConfusionMatrix(task = "binary").to(device)

class RegressionMetrics(object):
    def __init__(self, device="cuda"):
        self.MSE = torchmetrics.MeanSquaredError().to(device)
        self.MAE = torchmetrics.MeanAbsoluteError().to(device)
        self.R2 = torchmetrics.R2Score().to(device)

class BaseModelLit(pl.LightningModule):
    def __init__(self, args, hyperparameters):
        super().__init__()
        self.model = None
        self.args = args
        self.BCELogitsLoss = nn.BCEWithLogitsLoss()
        self.BCELoss = nn.BCELoss()
        self.MSELoss = nn.MSELoss()
        self.classification_metrics = ClassificationMetrics(args.device)
        self.regression_metrics = RegressionMetrics(args.device)
        self.validation_outputs = []
        self.test_outputs = []
        self.hyperparameters = hyperparameters
        self.save_hyperparameters(ignore=['args'])

    def forward(self, data):
        return self.model(data, "predict")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        if self.args.debug:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)
        else:
            return
        
    def unpack_data(self, batch):
        pass

    def data_to_dict(self, preds, labels, masks):
        pass

    def extract_preds_labels(self):
        pass

    def compute_loss(self, preds, labels, masks):
        pass

    def compute_val_metrics(self, preds, labels, masks):
        pass

    def compute_test_metrics(self, preds, labels, masks):
        pass

    def compute_metric(self, preds, labels, masks, metric):
        values = []
        for i in range(preds.shape[1]):
            if masks.dim() == 1:
                mask = masks
            elif masks.dim() == 2:
                mask = masks[:, i]
            values.append(metric(preds[mask, i].view(-1), labels[mask, i].view(-1)))
        return tuple(values)

    def plot_confusion_matrices(self, preds, labels, suffix):
        tasks = [i for i in range(preds.shape[1])]
        if preds.shape[1] == 6:
            names = ["Action", "Top", "Front", "Rear", "Right", "Left"]
            starting_idx = 1
        else:
            names = ["Top", "Front", "Rear", "Right", "Left"]
            starting_idx = 0

        for task, name in zip(tasks, names):
            cm = self.classification_metrics.ConfusionMatrix(preds[:,task], labels[:,task].long()).cpu().numpy()
            name_w_suffix = name + "_" + suffix
            self.logger.experiment.add_figure(f"{name_w_suffix} Feasibility Confusion matrix", plot_confusion_matrix(cm, ["infeasible", "feasible"]), 0)
        cm = self.classification_metrics.ConfusionMatrix(preds[:,starting_idx:], labels[:,starting_idx:].long()).cpu().numpy()
        self.logger.experiment.add_figure("Grasps Feasibility Confusion matrix", plot_confusion_matrix(cm, ["infeasible", "feasible"]), 0)

    def training_step(self, batch, batch_idx):
        inputs, labels, masks = self.unpack_data(batch)
        preds = self.model(inputs, "train")
        loss = self.compute_loss(preds, labels, masks)
        self.log_dict({"train_loss": loss, "step": float(self.current_epoch)}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, masks = self.unpack_data(batch)
        preds = self.model(inputs, "val")
        loss = self.compute_loss(preds, labels, masks)
        results = self.compute_val_metrics(preds, labels, masks)
        self.validation_outputs.append({'val_loss': loss, **results})
        self.log_dict({"val_loss": loss, **results, "step": float(self.current_epoch)}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        return {'val_loss': loss, **results}

    def test_step(self, batch, batch_idx):
        inputs, labels, masks = self.unpack_data(batch)
        preds = self.model(inputs, "test")
        loss = self.compute_loss(preds, labels, masks)
        preds_labels_dict = self.data_to_dict(preds, labels, masks)
        self.test_outputs.append(preds_labels_dict)
        return {"loss": loss, **preds_labels_dict}

    def on_train_epoch_end(self):
        print()

    def on_validation_epoch_end(self):
        val_loss = torch.stack([output["val_loss"] for output in self.validation_outputs]).mean()
        metrics = {key: torch.stack([output[key] for output in self.validation_outputs]).mean() for key in self.validation_outputs[0].keys() if key != "val_loss"}
        self.validation_outputs.clear()
        
        for key, value in zip(["val_loss", *metrics.keys()], [val_loss, *metrics.values()]):
            self.logger.experiment.add_scalar(key, value, global_step=self.current_epoch)

    def on_test_epoch_end(self):
        preds, labels, masks = self.extract_preds_labels()
        metrics = self.compute_test_metrics(preds, labels, masks)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger = True, batch_size=self.args.batch_size)

# =============================================================== Geometric Reasoning Networks ==========================================================
class GRNLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = GRN(device=args.device).to(args.device)

    def unpack_data(self, batch):
        batch = batch.to(self.args.device)
        return batch, (batch.F_labels, batch.IK_labels, batch.GO_labels), (batch.mask, batch.blocking_mask)
    
    def data_to_dict(self, preds, labels, masks):
        F_preds, IK_preds, GO_preds = preds
        F_labels, IK_labels, GO_labels = labels
        mask, blocking_mask = masks
        return {"F_preds": F_preds, "F_labels": F_labels,
                "IK_preds": IK_preds, "IK_labels": IK_labels,
                "GO_preds": GO_preds, "GO_labels": GO_labels,
                "masks": {"mask": mask, "blocking_mask": blocking_mask}}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        IK_preds = torch.cat([tmp['IK_preds'] for tmp in self.test_outputs])
        IK_labels = torch.cat([tmp['IK_labels'] for tmp in self.test_outputs])
        GO_preds = torch.cat([tmp['GO_preds'] for tmp in self.test_outputs])
        GO_labels = torch.cat([tmp['GO_labels'] for tmp in self.test_outputs])
        masks = torch.cat([tmp['masks']["mask"] for tmp in self.test_outputs])
        blocking_masks = torch.cat([tmp['masks']["blocking_mask"] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return (F_preds, IK_preds, GO_preds), (F_labels, IK_labels, GO_labels), (masks, blocking_masks)

    def compute_loss(self, preds, labels, masks):
        F_preds, IK_preds, GO_preds = preds
        F_labels, IK_labels, GO_labels = labels
        mask, blocking_mask = masks
        F_loss = self.BCELogitsLoss(F_preds[mask].view(-1), F_labels[mask].view(-1))
        IK_loss = self.BCELogitsLoss(IK_preds[mask].view(-1), IK_labels[mask].view(-1))
        GO_loss = self.MSELoss(GO_preds[blocking_mask].view(-1), GO_labels[blocking_mask].view(-1))
        loss = F_loss + IK_loss + 10*GO_loss
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F_preds, IK_preds, GO_preds = preds
        F_labels, IK_labels, GO_labels = labels
        mask, blocking_mask = masks
        F_val = self.classification_metrics.F1Score(F_preds[mask].view(-1), F_labels[mask].view(-1))
        IK_val = self.classification_metrics.F1Score(IK_preds[mask].view(-1), IK_labels[mask].view(-1))
        if GO_labels[blocking_mask].shape[0] > 0:
            GO_val = self.regression_metrics.MAE(GO_preds[blocking_mask].view(-1), GO_labels[blocking_mask].view(-1))
        else:
            GO_val = torch.tensor(0., device=torch.device(self.args.device))
        return {"Feasibility": F_val, "IK": IK_val, "GO": GO_val}

    def compute_test_metrics(self, preds, labels, masks):
        F_preds, IK_preds, GO_preds = preds
        F_labels, IK_labels, GO_labels = labels
        mask, blocking_mask = masks
        F1 = self.classification_metrics.F1Score(F_preds[mask].view(-1), F_labels[mask].view(-1))
        Prec = self.classification_metrics.Precision(F_preds[mask].view(-1), F_labels[mask].view(-1))
        Rec = self.classification_metrics.Recall(F_preds[mask].view(-1), F_labels[mask].view(-1))
        action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(F_preds, F_labels, mask, self.classification_metrics.F1Score)
        top_IK, front_IK, rear_IK, right_IK, left_IK = self.compute_metric(IK_preds, IK_labels, mask, self.classification_metrics.F1Score)
        top_GO, front_GO, rear_GO, right_GO, left_GO = self.compute_metric(GO_preds, GO_labels, blocking_mask, self.regression_metrics.MAE)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_IK_mean = torch.mean(torch.stack([top_IK, front_IK, rear_IK, right_IK, left_IK]))
        grasp_IK_std = torch.std(torch.stack([top_IK, front_IK, rear_IK, right_IK, left_IK]))
        grasp_GO_mean = torch.mean(torch.stack([top_GO, front_GO, rear_GO, right_GO, left_GO]))
        grasp_GO_std = torch.std(torch.stack([top_GO, front_GO, rear_GO, right_GO, left_GO]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Grasp_IK_mean": grasp_IK_mean, "Grasp_IK_std": grasp_IK_std,
                "Grasp_GO_mean": grasp_GO_mean, "Grasp_GO_std": grasp_GO_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F,
                "Top_IK": top_IK, "Front_IK": front_IK, "Rear_IK": rear_IK, "Right_IK": right_IK, "Left_IK": left_IK,
                "Top_GO": top_GO, "Front_GO": front_GO, "Rear_GO": rear_GO, "Right_GO": right_GO, "Left_GO": left_GO}
        
# ----------------------------------------------- Action and Grasp Feasibility Module -----------------------------------------------
class AGFModuleLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = AGFModule().to(args.device)
        
    def unpack_data(self, batch):
        batch = batch.to(self.args.device)
        return batch, batch.F_labels, batch.mask

    def data_to_dict(self, preds, labels, masks):
        return {"F_preds": preds, "F_labels": labels, "masks": masks}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        masks = torch.cat([tmp['masks'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return F_preds, F_labels, masks
    
    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds[masks].view(-1), labels[masks].view(-1))
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds[masks].view(-1), labels[masks].view(-1))
        Prec = self.classification_metrics.Precision(preds[masks].view(-1), labels[masks].view(-1))
        Rec = self.classification_metrics.Recall(preds[masks].view(-1), labels[masks].view(-1))
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}

    def compute_test_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds[masks].view(-1), labels[masks].view(-1))
        Prec = self.classification_metrics.Precision(preds[masks].view(-1), labels[masks].view(-1))
        Rec = self.classification_metrics.Recall(preds[masks].view(-1), labels[masks].view(-1))
        Action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(preds, labels, masks, self.classification_metrics.F1Score)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": Action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F}
        
# ----------------------------------------------- Grasp Obstruction Module -----------------------------------------------
class GOModuleLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        if args.robot == "panda":
            hidden_size = 512
        elif args.robot == "pr2":
            hidden_size = 256
        self.model = GOModule(hidden_size=hidden_size).to(args.device)
        
    def unpack_data(self, batch):
        inputs, labels, masks = batch
        return inputs, labels, masks

    def data_to_dict(self, preds, labels, masks):
        return {"GO_preds": preds, "GO_labels": labels, "masks": masks}
    
    def extract_preds_labels(self):
        GO_preds = torch.cat([tmp['GO_preds'] for tmp in self.test_outputs])
        GO_labels = torch.cat([tmp['GO_labels'] for tmp in self.test_outputs])
        masks = torch.cat([tmp['masks'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return GO_preds, GO_labels, masks
    
    def compute_loss(self, preds, labels, masks):
        loss = self.MSELoss(preds[masks].view(-1), labels[masks].view(-1))
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        MAE = self.regression_metrics.MAE(preds[masks].view(-1), labels[masks].view(-1))
        MSE = self.regression_metrics.MSE(preds[masks].view(-1), labels[masks].view(-1))
        R2 = self.regression_metrics.R2(preds[masks].view(-1), labels[masks].view(-1))
        return {"Val_MAE": MAE, "Val_MSE": MSE, "Val_R2": R2}
    
    def compute_test_metrics(self, preds, labels, masks):
        MAE = self.regression_metrics.MAE(preds[masks].view(-1), labels[masks].view(-1))
        MSE = self.regression_metrics.MSE(preds[masks].view(-1), labels[masks].view(-1))
        R2 = self.regression_metrics.R2(preds[masks].view(-1), labels[masks].view(-1))
        top_GO, front_GO, rear_GO, right_GO, left_GO = self.compute_metric(preds, labels, masks, self.regression_metrics.MAE)
        grasp_GO_mean = torch.mean(torch.stack([top_GO, front_GO, rear_GO, right_GO, left_GO]))
        grasp_GO_std = torch.std(torch.stack([top_GO, front_GO, rear_GO, right_GO, left_GO]))
        return {"Test_MAE": MAE, "Test_MSE": MSE, "Test_R2": R2,
                "Grasp_GO_mean": grasp_GO_mean, "Grasp_GO_std": grasp_GO_std,
                "Top_GO": top_GO, "Front_GO": front_GO, "Rear_GO": rear_GO, "Right_GO": right_GO, "Left_GO": left_GO}
    
# ----------------------------------------------- Inverse Kinematics Networks -----------------------------------------------
class IKModuleLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = IKModule().to(args.device)
        
    def unpack_data(self, batch):
        inputs, labels = batch
        return inputs.to(self.args.device), labels.to(self.args.device), torch.ones(labels.shape, device=self.args.device).bool()
    
    def data_to_dict(self, preds, labels, masks):
        return {"IK_preds": preds, "IK_labels": labels, "masks": masks}

    def extract_preds_labels(self):
        IK_preds = torch.cat([tmp['IK_preds'] for tmp in self.test_outputs])
        IK_labels = torch.cat([tmp['IK_labels'] for tmp in self.test_outputs])
        masks = torch.cat([tmp['masks'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return IK_preds, IK_labels, masks

    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds[masks].view(-1), labels[masks].view(-1))
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds[masks].view(-1), labels[masks].view(-1))
        Prec = self.classification_metrics.Precision(preds[masks].view(-1), labels[masks].view(-1))
        Rec = self.classification_metrics.Recall(preds[masks].view(-1), labels[masks].view(-1))
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}
    
    def compute_test_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds.view(-1), labels.view(-1))
        Prec = self.classification_metrics.Precision(preds.view(-1), labels.view(-1))
        Rec = self.classification_metrics.Recall(preds.view(-1), labels.view(-1))
        top_IK, front_IK, rear_IK, right_IK, left_IK = self.compute_metric(preds, labels, masks, self.classification_metrics.F1Score)
        grasp_IK_mean = torch.mean(torch.stack([top_IK, front_IK, rear_IK, right_IK, left_IK]))
        grasp_IK_std = torch.std(torch.stack([top_IK, front_IK, rear_IK, right_IK, left_IK]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Grasp_IK_mean": grasp_IK_mean, "Grasp_IK_std": grasp_IK_std,
                "Top_IK": top_IK, "Front_IK": front_IK, "Rear_IK": rear_IK, "Right_IK": right_IK, "Left_IK": left_IK}
    
# ========================================================= GNN Network ==========================================================
class GNNLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = GNN(gnn_type=args.gnn_type).to(args.device)

    def unpack_data(self, batch):
        batch = batch.to(self.args.device)
        return batch, batch.F_labels, batch.mask
    
    def data_to_dict(self, preds, labels, masks):
        return {"F_preds": preds[masks].view(-1), "F_labels": labels[masks].view(-1)}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return F_preds, F_labels
    
    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds[masks].view(-1), labels[masks].view(-1))
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds[masks].view(-1), labels[masks].view(-1))
        Prec = self.classification_metrics.Precision(preds[masks].view(-1), labels[masks].view(-1))
        Rec = self.classification_metrics.Recall(preds[masks].view(-1), labels[masks].view(-1))
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}
    
    def compute_test_metrics(self, preds, labels):
        F1 = self.classification_metrics.F1Score(preds.view(-1), labels.view(-1))
        Prec = self.classification_metrics.Precision(preds.view(-1), labels.view(-1))
        Rec = self.classification_metrics.Recall(preds.view(-1), labels.view(-1))
        Action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(preds, labels, self.classification_metrics.F1Score)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": Action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F}
    
# ======================================================= Action and Grasp Feasibility Prediction Network ==========================================================
class AGFPNetLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = AGFPNet(args).to(args.device)
        
    def unpack_data(self, batch):
        projections, labels = batch
        return projections, labels, torch.ones(labels.shape[0], device=self.args.device).bool()
    
    def data_to_dict(self, preds, labels, masks):
        return {"F_preds": preds, "F_labels": labels}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return F_preds, F_labels
    
    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds, labels)
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}
    
    def compute_test_metrics(self, preds, labels):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        Action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(preds, labels, self.classification_metrics.F1Score)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": Action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F}
        
       
# ======================================================= Deep Visual Heuristics ==========================================================
class DVHLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = DVH(args).to(args.device)
        
    def unpack_data(self, batch):
        projections, labels = batch
        return projections, labels, torch.ones(labels.shape[0], device=self.args.device).bool()
    
    def data_to_dict(self, preds, labels, masks):
        return {"F_preds": preds, "F_labels": labels}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return F_preds, F_labels
    
    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds, labels)
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}
    
    def compute_test_metrics(self, preds, labels):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        Action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(preds, labels, self.classification_metrics.F1Score)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": Action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F}
            
# ======================================================= Multi-Layer Perceptron ==========================================================
class MLPLit(BaseModelLit):
    def __init__(self, args, hyperparameters):
        super().__init__(args, hyperparameters)
        self.model = MLPNet().to(args.device)
        
    def unpack_data(self, batch):
        inputs, labels = batch
        return inputs.to(self.args.device), labels.to(self.args.device), torch.ones(labels.shape, device=self.args.device).bool()
    
    def data_to_dict(self, preds, labels, masks):
        return {"F_preds": preds, "F_labels": labels, "masks": masks}
    
    def extract_preds_labels(self):
        F_preds = torch.cat([tmp['F_preds'] for tmp in self.test_outputs])
        F_labels = torch.cat([tmp['F_labels'] for tmp in self.test_outputs])
        masks = torch.cat([tmp['masks'] for tmp in self.test_outputs])
        self.test_outputs.clear()
        return F_preds, F_labels, masks
    
    def compute_loss(self, preds, labels, masks):
        loss = self.BCELogitsLoss(preds, labels)
        return loss
    
    def compute_val_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        return {"Val_F1": F1, "Val_Precision": Prec, "Val_Recall": Rec}
    
    def compute_test_metrics(self, preds, labels, masks):
        F1 = self.classification_metrics.F1Score(preds, labels)
        Prec = self.classification_metrics.Precision(preds, labels)
        Rec = self.classification_metrics.Recall(preds, labels)
        Action_F, top_F, front_F, rear_F, right_F, left_F = self.compute_metric(preds, labels, masks, self.classification_metrics.F1Score)
        grasp_F_mean = torch.mean(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        grasp_F_std = torch.std(torch.stack([top_F, front_F, rear_F, right_F, left_F]))
        return {"Test_F1": F1, "Test_Precision": Prec, "Test_Recall": Rec,
                "Action_F": Action_F, "Grasp_F_mean": grasp_F_mean, "Grasp_F_std": grasp_F_std,
                "Top_F": top_F, "Front_F": front_F, "Rear_F": rear_F, "Right_F": right_F, "Left_F": left_F}
        
    
        