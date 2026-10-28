# COVID-19 Chest X-Ray Classification and Segmentation

A deep learning project for classifying chest X-rays into **Normal, COVID-19, and Pneumonia** categories while segmenting lung regions and infection areas.

The project explores two multi-task architectures: a Vision Transformer (ViT)-based model and a CNN-based model without ViT. A separate inference notebook contains a Gradio interface intended for uploading chest X-rays, viewing predicted classes and segmentation overlays, and calculating an infection-area-based severity score.
## Models

- **CNN + ViT + U-Net:** Combines a pretrained Vision Transformer, CNN refinement layers, a classification head, and a U-Net segmentation network.
- **CNN + U-Net:** Uses a CNN feature extractor for classification and a U-Net-based segmentation network.

## Dataset

The notebooks use a custom dataset created using 4 separate Kaggle datasets for diversity with appropriate dataset cleaning, containing chest X-ray images and corresponding lung and infection masks.

## Model architectures

<img width="640" height="480" alt="ViT" src="https://github.com/user-attachments/assets/46edc619-a5ff-4aab-9932-4eb8883a224f" />

<img width="640" height="480" alt="CNN_UNET" src="https://github.com/user-attachments/assets/7b3bf92a-d982-42f1-999f-d25a213a408b" />

## Overall workflow flowcharts for each model architecture

<img width="576" height="768" alt="ViTpipeline" src="https://github.com/user-attachments/assets/fef48833-16f4-4579-baf0-ef77aaba7bb4" />

<img width="576" height="768" alt="CNN_UNETpipeline" src="https://github.com/user-attachments/assets/848abd96-202f-48a0-9834-a2da4a72b13d" />

## Repository Contents

- `covid.ipynb` — ViT-based model training and evaluation.
- `covid-no-vit.ipynb` — CNN-based model training and evaluation without ViT.
- `InferenceServer.ipynb` — Gradio inference interface for classification and segmentation.

## Running the Inference Interface

The inference notebook requires the trained model checkpoints and Python dependencies. Update its checkpoint paths to match your environment before running it.
