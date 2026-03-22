Pedestrian Instance Segmentation (Task-38)This repository contains a high-performance computer vision system for pedestrian instance segmentation, developed as part of Task-38. The project achieves 70%+ $AP_{50}$ on the PennFudanPedestrian dataset using a fine-tuned Mask R-CNN architecture.
Project OverviewThe objective was to build a robust model capable of detecting and segmenting individual pedestrians in diverse environments.Target Metric: $\geq 70\%$ Average Precision ($AP_{50}$).Dataset: PennFudanPedestrian-1024x512 (160 total images).Key Challenge: Migrating from a Streamlit-based deployment to a stable Gradio interface in a Google Colab environment.
Technical Implementation
1. Model Architecture
Base Model: Mask R-CNN with a ResNet-50-FPN backbone.

Training Strategy: Fine-tuned for 50 epochs using a custom PyTorch training loop.

Optimization: * Heavy data augmentation (random horizontal flips, scaling, and color adjustments) to prevent overfitting on a small dataset.

Hyperparameter tuning of the RPN NMS threshold and ROI Align output size to sharpen mask boundaries.

2. Deployment: Streamlit vs. Gradio
A significant part of this project's execution involved overcoming networking hurdles in Google Colab:

Streamlit Challenge: Encountered frequent "503 Tunnel Unavailable" errors and "app.py does not exist" issues due to Streamlit’s file-based execution and localtunnel instability.

Gradio Solution: Successfully deployed using Gradio's memory-based execution, which allows the model to stay "warm" in GPU memory and provides a stable, shareable .live link for immediate peer review.
Live Demo
The project includes a functional UI that provides:

Image Upload: Test the model on any pedestrian image.

Instance Tracking: A real-time log displaying unique IDs and confidence scores for every detected pedestrian.

Visualization: Mask overlays and bounding boxes rendered directly in the browser.

Access the demo here: Gradio Live Link (https://c55a3e9b70a6539dc7.gradio.live)

)


