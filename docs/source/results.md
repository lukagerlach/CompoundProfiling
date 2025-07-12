# Results

This section documents experimental results and analysis for compound profiling using different self-supervised learning approaches.

## Model Performance Comparison

The following plot shows the accuracy comparison between different models using cosine similarity as the distance metric:

![Accuracy comparison using cosine similarity](_static/plots/accuracy_comparison_cosine.png)

## Confusion Matrices

Detailed confusion matrices for model evaluation. We show how treatments of each MOA are classified:

![Confusion matrices with cosine similarity](_static/plots/confusion_matrices_cosine.png)

## Similarity Analysis

We compare similarity stats between same and different MOAs here:

![Similarity comparison between models](_static/plots/similarity_comparison.png)

## Dimensionality Reduction Visualization

t-SNE visualization comparing treatment embeddings from different models (Base ResNet, SimCLR variants, and WS-DINO):

![t-SNE comparison of model embeddings](_static/plots/tsne_comparison_base_resnet_simclr_vanilla_ws_neg_simclr_vanilla_simclr_ws_collapsed.png)