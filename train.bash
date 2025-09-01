#!/usr/bin/env bash
set -euo pipefail

# Usage/help
usage() {
	cat <<EOF
Usage:
	bash train.bash [--dataset_path PATH] [DATASET_NAME]

Examples:
	bash train.bash --dataset_path={PATH_TO_DATASET} # uses default dataset name (panda_3d_4)
EOF
}

# Default
DATASET_PATH="GRN_Datasets/panda_3d_4"

# Parse flags (support: --dataset_path=value or --dataset_path value)
while [[ $# -gt 0 ]]; do
	case "$1" in
		--dataset_path=*)
			DATASET_PATH="${1#*=}"
			shift
			;;
		--dataset_path)
			if [[ $# -lt 2 ]]; then
				echo "Error: --dataset_path requires a value" >&2
				usage
				exit 1
			fi
			DATASET_PATH="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		--)
			shift
			break
			;;
		*)
			# First non-flag: treat as positional dataset name
			break
			;;
	esac
done

# If dataset_path not provided, use positional DATASET (if any), else default
if [[ -z "${DATASET_PATH}" ]]; then
	if [[ $# -gt 0 ]]; then
		DATASET_PATH="$1"
		shift
	fi
fi

python3 train_ik.py --dataset_path="${DATASET_PATH}" --robot="panda" --n_epochs=2
python3 train_go.py --dataset_path="${DATASET_PATH}" --robot="panda" --n_epochs=2
python3 train_agf.py --dataset_path="${DATASET_PATH}" --robot="panda" --n_epochs=2
python3 train_grn.py --dataset_path="${DATASET_PATH}" --robot="panda" --n_epochs=2