# ML4641 — Detecting AI-Generated Images via Self-Supervised Representations and Transfer Learning

This repository implements a detector for AI-generated images using DINO embeddings, unsupervised analysis (PCA/t-SNE/UMAP/k-means), supervised classifiers (Logistic Regression, SVM, Random Forest, k-NN), and CNN transfer learning (ResNet).

## Dataset

Source: [Kaggle CIFake](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)

Classes: Real and AI-generated images (~100k train, ~20k test)

## Methods

- **Feature Extraction**: DINO Vision Transformer (ViT-S/16) embeddings (label-agnostic)
- **Unsupervised Analysis**: PCA, t-SNE, UMAP, k-means clustering with silhouette/purity metrics
- **Supervised Classifiers**: Logistic Regression, SVM, Random Forest, k-NN on full embeddings and top-k PCs
- **CNN Transfer Learning**: ResNet-18/50 fine-tuning with GradCAM interpretability
- **Robustness**: JPEG compression, Gaussian blur, Gaussian noise perturbations

## Evaluation Metrics

- ROC-AUC (primary)
- PR-AUC, Accuracy
- Silhouette Score, Cluster Purity (k-means)

## Quick Start

```bash
# Download CIFake into data/raw (requires a Kaggle API token)
python scripts/download_dataset.py

# Extract DINO embeddings
python src/extract_embeddings.py --input data/raw --output data/embeddings_vit_small.npz --model-name vit_small_patch16_224

# Run the full analysis pipeline (PCA, UMAP, k-means, classifiers)
python src/full_analysis.py --embeddings data/embeddings_vit_small.npz --outdir outputs/full --n-pca 50 --k-pc 10

# Fine-tune a ResNet (expects train/val/test subfolders)
python src/train_cnn.py --data <processed_dataset_root> --arch resnet18
```

Main dependencies: `torch`, `torchvision`, `timm`, `scikit-learn`, `umap-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`, `joblib`, `kaggle`.

## Repository Structure

`/src/`: Source code
- `/src/preprocess.py`: Image preprocessing — JPEG compression, Gaussian blur, Gaussian noise
- `/src/extract_embeddings.py`: DINO Vision Transformer (ViT-S/16) feature extraction
- `/src/train_classifiers.py`: Train Logistic Regression, SVM, Random Forest classifiers and generate ROC curves
- `/src/train_cnn.py`: ResNet-18/50 fine-tuning for end-to-end classification
- `/src/unsupervised_analysis.py`: PCA, t-SNE, UMAP dimensionality reduction and k-Means clustering
- `/src/pca_analysis.py`: Standalone PCA with explained variance analysis
- `/src/gradcam.py`: GradCAM interpretability — spatial attention heatmaps
- `/src/full_analysis.py`: Complete analysis pipeline orchestrating all methods (incl. k-NN, PR-AUC, silhouette/purity)

`/scripts/`: Helper scripts
- `/scripts/download_dataset.py`: Downloads the CIFake dataset from Kaggle (requires API token)

`/results/`: Generated figures (ROC/AUC comparison, PCA explained variance, t-SNE/UMAP projections, GradCAM grid, training curves, robustness plots)

## Results

Scripts write plots and metrics CSVs to `outputs/` by default. Key figures are committed under `results/`; the full project report is in `docs/index.md`.
