# Huang-Lab-Work

Welcome to our work on the at the Ngan Huang Lab! Our project focuses on applying AI/ML (specifically computer vision techniques) to synthetic cardiovascular tissue-on-a-chip images. We are currently training object-detection models to predict endothelial-to-mesenchymal transition sites in atherosclerosis-on-a-chip systems. The goal is to generate spatial predictions revealing where vascular plaque forms and stiffens in early-stage atherosclerosis, a leading cause of heart attack and stroke. We are training our models to recognize both ___ features——stiffness of the extracellular environment (ECM), ECM materials and their effect on the dedifferentiation behavior——as well as subtle features such as morpohology, neighborhood interactions, and spatial context. 

This repository contains our dataloader, image data, and model (masked autonencoder, aka an MAE).

MODEL ARCHITECTURE:
------------------------------------------------------------------------------------------------------------------------------------------------
For this project, we focused on an object detection task—predicting the red spots in the overlay (with bounding box coordinates of these red spots as the labels), given the corresponding z-stack, focused-stack, and hbrid images belonging to that environment (inputs). Our thinking is that reliably pinpointing these red-spot regions will help identify where endothelial cells may later de-differentiate into a mesenchymal phenotype.

A Masked Autoencoder (MAE) is a self-supervised model that learns by hiding most of an input image and forcing itself to reconstruct what is missing. Given an image 𝑥, the MAE randomly masks out a large fraction of the image (often around 75%) and trains the model to predict the missing content. Because it learns to reconstruct the image without any human-provided labels, it can learn useful visual structure in a label-free way.

The MAE process works as follows. First, the image is split into small patches, and each patch is converted into a vector representation (a patch embedding). Next, most of these patches are randomly masked so that the model only sees a small subset of the image. The encoder (typically a Vision Transformer, or ViT) processes only the visible patches, which makes training more efficient. After that, mask tokens are inserted for the missing patches. These tokens act like learned placeholders: they tell the model that a patch exists in that location, but its actual contents are hidden. Because the mask tokens are trainable, the model gradually learns how to interpret them during training. Finally, a decoder takes the visible-patch representations plus the mask tokens and attempts to reconstruct the full image. Importantly, the reconstruction loss is computed only on the masked patches, so the model is specifically rewarded for predicting what it could not see.

This setup encourages the model to learn strong visual features. To fill in missing regions correctly, it must understand shapes, textures, edges, spatial relationships, and context—for example, inferring what is likely nearby based on the visible portion of a cell image. In other words, MAE forces the encoder to build useful internal representations of image structure without relying on labels.

In our project, we use MAE as a pretraining method for object detection. In the first stage, we perform self-supervised MAE pretraining using only raw images, with no bounding boxes or class labels. The model is trained to reconstruct masked patches, using a reconstruction loss such as mean squared error (MSE). This produces a pretrained encoder backbone that has already learned strong visual features from the image data.

In the second stage, we move to supervised object detection. We take the pretrained MAE encoder weights and use them as the backbone of an object detection network, then add a detection head on top. During fine-tuning, the input is an image and the model outputs bounding boxes and class labels. The training objective switches from reconstruction loss to a detection loss. By starting from an MAE-pretrained backbone, the detector can leverage the rich visual representations learned during self-supervised training, which can improve performance—especially when labeled detection data is limited.

We drew from this repo for most of our code: https://github.com/facebookresearch/mae.



DATA:
------------------------------------------------------------------------------------------------------------------------------------------------
Dataset link: https://drive.google.com/drive/folders/120bWbdzPsHad-nqh-NdEAk-3PQ-vNyxJ

This dataset contains image data from two cellular-environment stiffness conditions:
- 5 kPa
- 900 kPa

For each stiffness, data is provided in three formats:
- Z-stack (raw)
- Focused stack (focus-stacked overlay)
- Hybrid (processed overlay)

That makes six total top-level folders (3 formats × 2 stiffnesses).

**1) Z-stack Images (raw, highest-volume data)**
-------------------------------------------------

Folders:
- 250811_Athchip_noninflam_900kPa
- 250814_Athchip_non-inflam_5kPa

These are the rawest images in the dataset. Many individual slices are not fully in focus, but this format provides the largest amount of training data.

Structure:
- Each folder contains 222 sample folders (W000, W001, ..., W222)
- Inside each W### folder is a subfolder: P00001
- Inside P00001 are 140 .tif images
- These 140 images together form the z-stack for one sample (one extracellular environment).

Counts
- 222 samples at 900 kPa
- 222 samples at 5 kPa
- 444 total samples (environments)
- 140 images per sample
- 61,600 images total across both stiffnesses

These z-stack images make up roughly 90–95% of the total training data by image count.

**2) Focused Stack Images (high-definition focus overlays)**
------------------------------------------------------------
Folders
- 20251027_2123__FocusStack_250811_Athchip_noninflam_900kPa
- 20251027_2215__FocusStack_250814_Athchip_non-inflam_5kPa

These folders contain focus-stacked overlay images, which are generally more in focus, higher-definition, and more informative than individual z-stack slices.

Structure
- One image per sample (extracellular environment)
- Counts
- 220 images in the 900 kPa folder
- 220 images in the 5 kPa folder
- 440 focused-stack images total

**3) Hybrid Images (high-definition processed overlays)**
---------------------------------------------------------
Folders
- 250811_Athchip_noninflam_900kPa_HybridResults
- 250814_Athchip_non-inflam_5kPa_HybridResults

These folders contain hybrid processed overlay images, also high-definition and highly informative.

Structure
- Each folder contains 222 subfolders:
    - hybrid_results_W001, hybrid_results_W002, ..., hybrid_results_W222
      - Each subfolder contains one .tif image
        - Each image corresponds to one sample (one extracellular environment)

Counts
- 222 samples at 900 kPa
- 222 samples at 5 kPa
- 444 hybrid images total

<br>
<br>

DATALOADER DETAILS:
------------------------------------------------------------------------------------------------------------------------------------------------
We built a custom Colab data-loading pipeline to load this dataset and map all images from each extracellular environment to a single label. For each sample, the label was a .txt file containing bounding boxes, where each line stored one bounding box as four comma-separated coordinates.

The bounding-box labels were stored in Google Drive under two directories, one for each stiffness condition.
Within each directory, the label files were named W001, W002, W003, ..., W222 (as .txt files), with each file corresponding to one extracellular environment.

To organize the image data, we first wrote code that loaded each of the six data directories as its own custom dataset. We then built a class that joined all image files and metadata needed for a given sample, while also returning stiffness (kPa) as an additional model input feature (as a tensor).

Each sample consisted of 142 input images total:
- 140 images from the z-stack
- 1 image from the focus-stacked data
- 1 image from the hybrid data

Next, we collated these into a single custom dataset so that each sample contained:
- the full set of 142 input images,
- the stiffness annotation (5 kPa or 900 kPa), and
- the corresponding bounding-box label file (.txt).

In the end, the final dataset contained 444 total samples (222 at 5 kPa and 222 at 900 kPa), which we shuffled together into one combined dataset and wrapped in a PyTorch DataLoader.

