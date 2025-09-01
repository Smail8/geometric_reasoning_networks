import os
import argparse
from utils import process_gnn_data, process_go_data


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process data')

    parser.add_argument('--path', type=str, 
                        help='Dataset path',
                        required=True)
    
    parser.add_argument('--proximity_threshold', type=float,
                        help='Proximity threshold',
                        required=False,
                        default=0.6)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    print("Processing data for dataset: ", args.path, "...")
    train_data = process_gnn_data(os.path.join(args.path, "train_set"))
    val_data = process_gnn_data(os.path.join(args.path, "val_set"))
    test_data = process_gnn_data(os.path.join(args.path, "test_set"))
    train_go_data = process_go_data(os.path.join(args.path, "train_set"), args.proximity_threshold)
    val_go_data = process_go_data(os.path.join(args.path, "val_set"), args.proximity_threshold)
    test_go_data = process_go_data(os.path.join(args.path, "test_set"), args.proximity_threshold)
    print("Data processing completed!")