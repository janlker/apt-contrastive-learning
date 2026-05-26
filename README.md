# apt-contrastive-learning

Code for the publication: *"An exploration of contrastive self-supervised learning for reconstructed atom probe tomography data"*.

## Models
- **Supervised Classification** (`train_supervised_classification.py`)
- **Self-Supervised Contrastive Learning** (`train_self_supervised.py`) / *coming soon*

## Data Requirements
The models accept APT data in either `.epos` (must be accompanied by a `.rrange` file of the same name) or `.csv` format.
- **Required columns of data:** `x`, `y`, `z`, `element`, `n`.
- **Labels:** An additional `label` column is required for the supervised model

## Repository Structure
- `dataset.py`: PyTorch Datasets (`APT_basic` and `APT_dynamic` (*coming soon*))
- `models.py`: Network architectures including the DGCNN backbone, classification heads, and MLP projection heads for contrastive learning.
- `configs/`: Contains `supervised_config.yaml` and `self_supervised_config.yaml` for setting hyperparameters, data paths, and augmentations.
- `utils/`: Helper scripts for APT data ingestion, point cloud augmentations, metric calculations...

## Usage

All training hyperparameters, dataset paths, and augmentation settings are managed via YAML files. 

1. **Configure settings:**
   Edit the configuration files located in the `configs/` directory

2. **Run the training scripts:**
   ```bash
   # For the supervised model:
   python train_supervised_classification.py

   # For the self-supervised model  (*coming soon*):
   python train_self_supervised.py

## Requirements / tested with
- Python 3.10
- Pytorch 2.1
- Scikit-learn 1.7.2
- Lightly 1.5.3
- Scipy 1.12.0
- Torchmetrics 1.2.1
- Umap-learn 0.5.5
- Open3d 0.18.6
- Omegaconf 2.3.0
- Numpy 1.26.4
- Pandas 2.1.1
