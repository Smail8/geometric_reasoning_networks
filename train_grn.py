import os
import argparse
import torch
from Dataset import GRNDataset
from NetworkLit import GRNLit
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
pl.seed_everything(0, workers=True)

torch.manual_seed(0)
torch.cuda.is_available()
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN')

    parser.add_argument('--dataset_path', type=str,
                        help='Path to dataset',
                        required=True)
    
    parser.add_argument('--robot', type=str,
                        help='Robots supported: panda (pr2 and thiago coming soon)',
                        required=False,
                        default='panda')
    
    parser.add_argument('--IK_GO_mode', type=str,
                        help='IK IC mode',
                        required=False,
                        default="predict")
    
    parser.add_argument('--device', type=str,
                        help='Device',
                        required=False,
                        default="cuda")
    
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs',
                        required=False,
                        default=100)
    
    parser.add_argument('--batch_size', type=int,
                        help='Batch size',
                        required=False,
                        default=512)
    
    parser.add_argument('--lr', type=float,
                        help='Learning rate',
                        required=False,
                        default=0.0001)
    
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay',
                        required=False,
                        default=0.0)
    
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers',
                        required=False,
                        default=8)
    
    parser.add_argument('--dropout', type=float,
                        help='Dropout',
                        required=False,
                        default=0.0)

    parser.add_argument('--debug', type=bool,
                        help='Positive weight',
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

    if not os.path.exists(args.dataset_path):
        raise ValueError("Dataset path does not exist")

    train_set = GRNDataset(path=os.path.join(args.dataset_path, "train_set"), mode="train", args=args)
    val_set = GRNDataset(path=os.path.join(args.dataset_path, "val_set"), mode="val", args=args)
    test_set = GRNDataset(path=os.path.join(args.dataset_path, "test_set"), mode="test", args=args)
    train_loader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_loader = DataLoader(dataset = val_set, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    print("Number of datapoints: train_set =", len(train_set), " val_set =", len(val_set), " test_set =", len(test_set))
    print("Number of batches : train_set =", len(train_loader), " val_set =", len(val_loader), " test_set =", len(test_loader))

    model = GRNLit(args, hyperparameters).to(args.device)
    if not os.path.exists("lightning_logs/IK_module_"+args.robot+".pt") or not os.path.exists("lightning_logs/GO_module_"+args.robot+".pt") or \
    not os.path.exists("lightning_logs/AGF_module_"+args.robot+".pt"):
        raise ValueError("Pretrained IK, GO and AGF models not found! Please train them first.")
    model.model.IKModule.load_state_dict(torch.load("lightning_logs/IK_module_"+args.robot+".pt"))
    model.model.GOModule.load_state_dict(torch.load("lightning_logs/GO_module_"+args.robot+".pt"))
    model.model.AGFModule.load_state_dict(torch.load("lightning_logs/AGF_module_"+args.robot+".pt"))
    logger = TensorBoardLogger("lightning_logs", name="GRN/" + args.dataset_path.split("/")[-1])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="Feasibility", mode='max', save_top_k=1, save_last=False)
    if args.debug:
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator=args.device, deterministic=True, callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=1,
                         detect_anomaly=True)
    else:
        trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator=args.device, deterministic=True, callbacks=[checkpoint_callback], logger=logger)

    trainer.fit(model, train_loader, val_loader)

    model = GRNLit.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args, hyperparameters=hyperparameters)
    torch.save(model.model.state_dict(), "lightning_logs/GRN_"+args.robot+".pt")
    trainer.test(model, test_loader)