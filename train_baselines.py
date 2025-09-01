import argparse
import torch
import torch_geometric
from Dataset import *
from NetworkLit import MLPLit, DVHLit, AGFPNetLit, GNNLit
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
pl.seed_everything(0, workers=True)

torch.manual_seed(0)
torch.cuda.is_available()
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset_path', type=str,
                        help='Path to dataset',
                        required=True)
    
    parser.add_argument('--model', type=str,
                        help='Model type (MLP, DVH, AGFPNet, GCN, GAT)',
                        required=True)
    
    parser.add_argument('--robot', type=str,
                        help='Robots supported: panda (pr2 and thiago coming soon)',
                        required=False,
                        default='panda')
    
    parser.add_argument('--device', type=str,
                        help='Device',
                        required=False,
                        default="cuda")
    
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers',
                        required=False,
                        default=8)
    
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs',
                        required=False,
                        default=100)
    
    parser.add_argument('--batch_size', type=int,
                        help='Batch size',
                        required=False,
                        default=256)
    
    parser.add_argument('--lr', type=float,
                        help='Learning rate',
                        required=False,
                        default=0.0001)
    
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay',
                        required=False,
                        default=0.0)
    
    parser.add_argument('--dropout', type=float,
                        help='Dropout',
                        required=False,
                        default=0.0)
    
    parser.add_argument('--debug', type=bool,
                        help='Debug',
                        required=False,
                        default=False)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if "panda" in args.dataset_path:
        args.robot = "panda"
    elif "pr2" in args.dataset_path:
        args.robot = "pr2"
    elif "thiago" in args.dataset_path:
        args.robot = "thiago"
    hyperparameters = {"lr": args.lr, "batch_size": args.batch_size, "n_epochs": args.n_epochs}

    if args.model == "MLP":
        Dataset = MLPDataset
        Model = MLPLit
        DataLoader = torch.utils.data.DataLoader
    elif args.model == "DVH":
        Dataset = DVHDataset
        Model = DVHLit
        DataLoader = torch.utils.data.DataLoader
    elif args.model == "AGFPNet":
        Dataset = AGFPNetDataset
        Model = AGFPNetLit
        DataLoader = torch.utils.data.DataLoader
    elif args.model == "GCN" or args.model == "GAT":
        Dataset = GNNDataset
        Model = GNNLit
        DataLoader = torch_geometric.loader.DataLoader

    if not os.path.exists(args.dataset_path):
        raise ValueError("Dataset path does not exist")

    train_set = Dataset(path=os.path.join(args.dataset_path, "train_set"), mode="train", args=args)
    val_set = Dataset(path=os.path.join(args.dataset_path, "val_set"), mode="val", args=args)
    test_set = Dataset(path=os.path.join(args.dataset_path, "test_set"), mode="test", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Number of datapoints: train_set =", len(train_set), " val_set =", len(val_set), " test_set =", len(test_set))
    print("Number of batches : train_set =", len(train_loader), " val_set =", len(val_loader), " test_set =", len(test_loader))

    model = Model(args, hyperparameters)
    logger = TensorBoardLogger("lightning_logs", name=args.model+"/" + args.dataset_path.split("/")[-1])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="Val_F1", mode='max', save_top_k=1, save_last=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if args.debug:
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator=args.device, deterministic=True, callbacks=[checkpoint_callback, lr_monitor], logger=logger, 
                             log_every_n_steps=1, detect_anomaly=True)
    else:
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator=args.device, deterministic=True, callbacks=[checkpoint_callback, lr_monitor], logger=logger)

    trainer.fit(model, train_loader, val_loader)

    model = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args, hyperparameters=hyperparameters)
    trainer.test(model, test_loader)

        

    
    
    
    
