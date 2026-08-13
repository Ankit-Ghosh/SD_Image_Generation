# 🌾 Stable Diffusion for Agricultural Image Generation

A research-oriented implementation and experimental framework for **text-guided agricultural image generation, Stable Diffusion architecture exploration, SDXL LoRA fine-tuning, and crop-image classification**.

The project brings together a custom implementation of the Stable Diffusion generation pipeline with agricultural crop datasets, SDXL-based LoRA fine-tuning, and downstream crop classification experiments. The primary motivation is to investigate how generative AI can be used to create realistic synthetic agricultural imagery and how generated/augmented images can subsequently support computer-vision tasks.

---

## 📌 Overview

Agricultural computer-vision systems often require large and diverse image datasets. However, collecting and annotating sufficient real-world agricultural imagery can be difficult because of environmental variation, crop availability, disease occurrence, imaging conditions, and geographical differences.

This project explores **diffusion-based generative models as a means of producing additional agricultural imagery from textual descriptions**.

The repository contains three complementary components:

1. **Custom Stable Diffusion Architecture**

   * A PyTorch implementation of the major components involved in Stable Diffusion.
   * Includes CLIP-based text conditioning, attention mechanisms, diffusion sampling, VAE encoding/decoding, model loading, and an end-to-end generation pipeline.

2. **SDXL LoRA Fine-Tuning**

   * A customized SDXL LoRA training script based on the Hugging Face Diffusers training workflow.
   * Designed for fine-tuning Stable Diffusion XL on agricultural/crop-specific datasets.
   * Includes commands for training, validation, checkpointing, mixed-precision training, and optional Weights & Biases / Hugging Face Hub integration.

3. **Crop Classification and Evaluation**

   * ResNet50-based crop classification experiments.
   * Separate workflows are provided for original and augmented agricultural datasets.
   * Evaluation includes validation accuracy, loss curves, confusion matrices, classification reports, and single-image prediction.

---

# 🧠 Project Architecture

The overall project can be viewed as the following workflow:

```text
                    ┌──────────────────────────┐
                    │ Agricultural Datasets    │
                    │                          │
                    │ Crop Images + Captions   │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ Custom Stable       │              │ SDXL LoRA Fine-Tuning │
   │ Diffusion Pipeline  │              │                      │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
              ▼                                     ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ Synthetic           │              │ Fine-Tuned           │
   │ Agricultural Images │              │ Generation Models    │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Classification /      │
                     │ Downstream Evaluation │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Accuracy / Loss /      │
                     │ Confusion Matrix /     │
                     │ Classification Report │
                     └────────────────────────┘
```

---

# ✨ Features

* Text-to-image generation using a custom Stable Diffusion implementation
* Image-to-image generation support
* CLIP-based text conditioning
* Classifier-free guidance
* Configurable guidance scale
* DDPM sampling
* Configurable inference steps
* Seed-controlled image generation
* VAE-based latent representation
* Custom attention implementation
* Stable Diffusion v1.5 checkpoint loading
* SDXL LoRA fine-tuning
* Agricultural crop-specific datasets
* Original vs. augmented dataset classification experiments
* ResNet50-based crop classification
* Confusion matrix generation
* Classification report generation
* Training/validation accuracy and loss visualization
* Sample generated-image organization
* Parquet-based crop-description data for fine-tuning workflows

---

# 📂 Repository Structure

```text
SD_Image_Generation/
│
├── datasets/
│   ├── almond/
│   ├── banana/
│   ├── cherry/
│   ├── clove/
│   ├── coconut/
│   ├── cotton/
│   ├── cucumber/
│   ├── maize/
│   └── sugarcane/
│
├── output/
│   ├── classification_report/
│   ├── plots/
│   └── sample_generated_images/
│       ├── almond/
│       ├── banana/
│       ├── cherry/
│       ├── clove/
│       ├── coconut/
│       ├── cotton/
│       ├── cucumber/
│       ├── maize/
│       └── sugarcane/
│
└── scripts/
    │
    ├── LoRA_finetune/
    │   ├── crop_descriptions_parquet/
    │   ├── README.md
    │   ├── finetune_commands.txt
    │   └── train_text_to_image_lora_sdxl.py
    │
    ├── classification/
    │   ├── augmented/
    │   │   ├── agri_crops_aug/
    │   │   ├── classify.py
    │   │   └── validator.py
    │   │
    │   └── original/
    │       ├── agri_crops_real/
    │       ├── classify.py
    │       └── validator.py
    │
    └── sd_architecture/
        ├── add_noise.ipynb
        ├── attention.py
        ├── clip.py
        ├── ddpm.py
        ├── decoder.py
        ├── demo.py
        ├── diffusion.py
        ├── encoder.py
        ├── model_converter.py
        ├── model_loader.py
        └── pipeline.py
```

The repository currently organizes the crop datasets into nine crop-specific directories, while generated samples follow the same crop-oriented organization.

---

# 🧩 1. Custom Stable Diffusion Architecture

The implementation is located in:

```text
scripts/sd_architecture/
```

This directory contains the core components required to construct and execute the generation pipeline.

## `pipeline.py`

`pipeline.py` provides the high-level generation function that connects the different components of the Stable Diffusion system.

The pipeline supports:

* Text-to-image generation
* Image-to-image generation
* Classifier-free guidance
* Configurable guidance scale
* Configurable sampling method
* Configurable inference steps
* Random or user-provided seeds
* Optional CPU offloading

The current implementation is configured around a **512 × 512** image space, with latent dimensions reduced by a factor of 8.

Conceptually:

```text
Prompt
  │
  ▼
CLIP Tokenizer
  │
  ▼
CLIP Text Encoder
  │
  ▼
Text Conditioning
  │
  ▼
Random Latent Noise
  │
  ▼
Diffusion / UNet Denoising
  │
  ▼
Denoised Latent
  │
  ▼
VAE Decoder
  │
  ▼
Generated Image
```

---

## `clip.py`

`clip.py` implements the text-conditioning component used by the generation pipeline.

The implementation contains:

* Token embeddings
* Positional embeddings
* Transformer-style CLIP layers
* Self-attention
* Layer normalization
* Feed-forward components

The repository's implementation defines a `CLIPEmbedding` module containing learnable token and positional embeddings and uses the custom `SelfAttention` implementation from `attention.py`.

The purpose of this component is to convert a textual prompt into a numerical representation that can condition the diffusion model.

```text
Text Prompt
     │
     ▼
Tokenization
     │
     ▼
Token Embeddings
     │
     +
Positional Embeddings
     │
     ▼
Transformer / Attention Layers
     │
     ▼
Text Conditioning Embedding
```

---

## `attention.py`

This module implements the attention mechanisms used by the architecture.

The repository defines a `SelfAttention` module using:

* Query projection
* Key projection
* Value projection
* Multiple attention heads
* Output projection
* Optional causal masking

The implementation projects the input embedding into Q, K and V representations and performs multi-head attention.

Attention allows the model to determine which parts of a sequence should influence one another.

---

## `diffusion.py`

`diffusion.py` contains the diffusion/denoising neural-network architecture used during the generation process.

It forms the central denoising component of the custom Stable Diffusion implementation.

Its role is to progressively transform a noisy latent representation into a meaningful latent representation conditioned on the input text.

```text
Noisy Latent
     │
     ▼
Diffusion Network
     │
     │ + Text Conditioning
     ▼
Predicted Noise / Denoised Representation
     │
     ▼
Updated Latent
```

The file contains the major neural-network building blocks used by the diffusion model.

---

## `encoder.py`

The VAE encoder converts an image from pixel space into a lower-dimensional latent representation.

```text
RGB Image
   │
   ▼
VAE Encoder
   │
   ▼
Latent Representation
```

This latent-space representation allows diffusion operations to be performed more efficiently than operating directly on full-resolution RGB images.

---

## `decoder.py`

The VAE decoder performs the inverse operation.

```text
Denoised Latent
      │
      ▼
VAE Decoder
      │
      ▼
RGB Image
```

Together, `encoder.py` and `decoder.py` form the VAE portion of the Stable Diffusion pipeline.

---

## `ddpm.py`

This module implements the DDPM sampling process.

The pipeline uses the DDPM sampler as one of its configurable sampling options. The demo configuration currently uses:

```python
sampler = "ddpm"
num_inference_steps = 50
```

The sampler is responsible for determining how the latent representation is updated over successive diffusion steps.

---

## `model_loader.py`

`model_loader.py` handles loading pretrained Stable Diffusion weights into the custom architecture.

The demo uses:

```text
v1-5-pruned-emaonly.ckpt
```

and loads it through:

```python
model_loader.preload_models_from_standard_weights(...)
```

This allows the custom PyTorch implementation to use weights from the Stable Diffusion v1.5 checkpoint rather than training the entire architecture from scratch.

---

## `model_converter.py`

This module is responsible for model/checkpoint conversion utilities required when adapting pretrained model weights to the architecture expected by the custom implementation.

---

## `add_noise.ipynb`

This notebook provides an experimental environment for examining the forward diffusion/noising process.

It is useful for understanding how clean image/latent representations are progressively corrupted with noise before the reverse denoising process reconstructs the underlying representation.

---

## `demo.py`

`demo.py` provides an example entry point for running the custom Stable Diffusion pipeline.

The current example:

* Detects CUDA availability
* Loads the CLIP tokenizer
* Loads the Stable Diffusion v1.5 checkpoint
* Defines an agricultural prompt
* Enables classifier-free guidance
* Uses DDPM sampling
* Generates an image
* Saves the generated image to disk

The example prompt focuses on rice crops affected by downy mildew.

---

# 🌱 2. Agricultural Dataset

The repository contains crop-specific image collections under:

```text
datasets/
```

The current repository includes:

* Almond
* Banana
* Cherry
* Clove
* Coconut
* Cotton
* Cucumber
* Maize
* Sugarcane

For example:

```text
datasets/
└── almond/
    ├── image (1).jpg
    ├── image (2).jpg
    ├── image (3).jpg
    └── ...
```

The datasets provide the agricultural visual material used throughout the project's generation and classification experiments.

---

# 🧪 3. SDXL LoRA Fine-Tuning

The LoRA workflow is located in:

```text
scripts/LoRA_finetune/
```

It contains:

```text
LoRA_finetune/
├── crop_descriptions_parquet/
├── README.md
├── finetune_commands.txt
└── train_text_to_image_lora_sdxl.py
```

## Why LoRA?

Low-Rank Adaptation (LoRA) provides a parameter-efficient method for adapting a pretrained diffusion model to a specialized domain.

Instead of fully retraining the entire SDXL model, trainable low-rank updates are introduced into selected model components.

This makes LoRA particularly useful when adapting large generative models to specialized domains such as agricultural imagery.

---

## SDXL Base Model

The fine-tuning configuration uses:

```text
stabilityai/stable-diffusion-xl-base-1.0
```

with:

```text
madebyollin/sdxl-vae-fp16-fix
```

as the VAE checkpoint.

The training workflow is built around the Hugging Face Diffusers SDXL text-to-image LoRA training setup.

---

## Training Configuration

The supplied command template includes:

```text
resolution = 1024
gradient_accumulation_steps = 4
learning_rate = 5e-5
lr_scheduler = cosine
lr_warmup_steps = 100
snr_gamma = 5.0
mixed_precision = fp16
```

The script also supports:

* Validation prompts
* Validation images
* Periodic checkpointing
* Text-encoder training
* Random horizontal flipping
* Center cropping
* Weights & Biases reporting
* Hugging Face Hub upload

---

## Preparing the Fine-Tuning Environment

The repository's LoRA documentation follows this workflow:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers

pip install -e .

cd examples/text_to_image

pip install -r requirements_sdxl.txt
```

The custom training script can then replace the corresponding Diffusers example script:

```text
diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py
```

with:

```text
scripts/LoRA_finetune/train_text_to_image_lora_sdxl.py
```

This procedure is documented in the repository's LoRA-specific README.

---

# 🖼️ 4. Crop Image Classification

The classification experiments are located in:

```text
scripts/classification/
```

There are two experimental branches:

```text
classification/
├── original/
└── augmented/
```

The original workflow operates on:

```text
agri_crops_real/
```

while the augmented workflow operates on:

```text
agri_crops_aug/
```

Both contain their own `classify.py` and `validator.py` scripts.

---

## ResNet50 Classifier

The classification workflow uses a pretrained ImageNet ResNet50 as the feature extractor.

The base ResNet50 layers are frozen and additional classification layers are added:

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
Pretrained ResNet50
    │
    ▼
Global Average Pooling
    │
    ▼
Flatten
    │
    ▼
Dense(512, ReLU)
    │
    ▼
Dense(number_of_classes, Softmax)
    │
    ▼
Predicted Crop Class
```

The original classifier uses:

* 224 × 224 input images
* Batch size of 32
* ImageNet-pretrained ResNet50
* Frozen pretrained layers
* 512-unit dense layer
* Adam optimizer
* Sparse categorical cross-entropy
* 15 training epochs

---

# 🔬 Data Augmentation

The classification pipeline includes image augmentation operations such as:

```text
Random Horizontal Flip
Random Rotation
Random Zoom
```

These transformations are intended to improve the robustness of the classifier to variations in image orientation and appearance.

---

# 📊 Evaluation

The classification experiments generate several evaluation artifacts.

## Accuracy and Loss

Training and validation curves are generated to examine:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

## Confusion Matrix

A confusion matrix is generated to identify which crop categories are correctly or incorrectly classified.

## Classification Report

The project also generates a classification report containing standard metrics such as:

* Precision
* Recall
* F1-score
* Support

The report is saved as a CSV file by the classification workflow.

## Single-Image Prediction

The classifier includes a helper function for predicting the crop class of an individual image and reporting the predicted class together with its confidence score.

---

# 📁 Output Directory

Generated and evaluation artifacts are organized under:

```text
output/
```

with three major sections:

```text
output/
├── classification_report/
├── plots/
└── sample_generated_images/
```

The generated-image directory follows the crop categories used by the dataset:

```text
sample_generated_images/
├── almond/
├── banana/
├── cherry/
├── clove/
├── coconut/
├── cotton/
├── cucumber/
├── maize/
└── sugarcane/
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Ankit-Ghosh/SD_Image_Generation.git
cd SD_Image_Generation
```

## 2. Create a Python Environment

Using Conda:

```bash
conda create -n sd-agri python=3.10
conda activate sd-agri
```

Or using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

---

# 📦 Dependencies

The project uses multiple Python ecosystems depending on the component being executed.

### Custom Stable Diffusion

The architecture uses packages including:

```text
PyTorch
NumPy
Pillow
tqdm
Transformers
```

### SDXL LoRA

The LoRA workflow relies on:

```text
PyTorch
Hugging Face Diffusers
Transformers
Accelerate
Datasets
PEFT / LoRA-related dependencies
Weights & Biases (optional)
```

### Classification

The classification experiments use:

```text
TensorFlow
Keras
NumPy
Pandas
Pillow
scikit-learn
Matplotlib
Seaborn
```

Because the repository currently separates these workflows rather than providing a single root-level dependency file, it is recommended to create the environment according to the component you intend to run.

---

# 🎨 Running the Custom Stable Diffusion Demo

Navigate to:

```bash
cd scripts/sd_architecture
```

Before running the demo, make sure the required model and tokenizer files are available at the paths expected by the script.

The demo currently expects:

```text
../data/vocab.json
../data/merges.txt
../data/v1-5-pruned-emaonly.ckpt
```

The generation example automatically selects CUDA when available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Then run:

```bash
python demo.py
```

The demo uses an agricultural prompt and generates an image using the custom pipeline.

---

# 📝 Customizing the Prompt

The prompt can be changed directly in `demo.py`.

For example:

```python
prompt = """
A realistic agricultural field containing healthy maize plants
under natural daylight, high-detail botanical photography
"""
```

For disease-oriented generation:

```python
prompt = """
A realistic close-up photograph of rice leaves affected by
brown spot disease, natural field environment, detailed
leaf texture, realistic lighting
"""
```

The repository's pipeline also supports a separate unconditional prompt for classifier-free guidance.

---

# 🎛️ Generation Parameters

Important generation parameters include:

| Parameter           | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `prompt`            | Text description used for generation        |
| `uncond_prompt`     | Unconditional/negative conditioning         |
| `strength`          | Strength used for image-to-image generation |
| `do_cfg`            | Enables classifier-free guidance            |
| `cfg_scale`         | Controls conditioning strength              |
| `sampler_name`      | Selects diffusion sampler                   |
| `n_inference_steps` | Number of denoising steps                   |
| `seed`              | Controls reproducibility                    |
| `input_image`       | Optional image-to-image input               |

The current pipeline defaults include a CFG scale of `7.5`, 50 inference steps, and DDPM sampling.

---

# 🔄 Text-to-Image vs. Image-to-Image

## Text-to-Image

```text
Text Prompt
     │
     ▼
CLIP
     │
     ▼
Text Conditioning
     │
     ▼
Random Latent
     │
     ▼
Diffusion Denoising
     │
     ▼
VAE Decoder
     │
     ▼
Image
```

## Image-to-Image

```text
Input Image
     │
     ▼
VAE Encoder
     │
     ▼
Latent Representation
     │
     ▼
Add Noise
     │
     ▼
Conditioned Diffusion
     │
     ▼
Denoised Latent
     │
     ▼
VAE Decoder
     │
     ▼
Modified Image
```

The pipeline exposes both `input_image` and `strength`, allowing the same generation function to support image-conditioned generation.

---

# 🧪 Experimental Workflow

A typical research workflow using this repository is:

```text
1. Collect agricultural images
          │
          ▼
2. Organize crop-specific datasets
          │
          ▼
3. Create textual descriptions/captions
          │
          ▼
4. Generate synthetic agricultural images
          │
          ▼
5. Fine-tune SDXL using LoRA
          │
          ▼
6. Generate crop-specific synthetic samples
          │
          ▼
7. Combine / compare real and augmented data
          │
          ▼
8. Train crop classifier
          │
          ▼
9. Evaluate classification performance
          │
          ▼
10. Analyze generated images and downstream results
```

This structure makes the repository useful not only as an image-generation implementation but also as an experimental framework for investigating whether synthetic agricultural imagery can contribute to downstream computer-vision tasks.

---

# 🔍 Research Components

The repository can be understood as having three research layers.

### Layer 1 — Generative Modeling

Investigation of the Stable Diffusion architecture and text-conditioned image generation.

### Layer 2 — Domain Adaptation

Adaptation of SDXL to agricultural/crop-specific data using LoRA.

### Layer 3 — Downstream Validation

Evaluation of agricultural imagery using a crop-classification model trained on original and augmented datasets.

This separation makes it possible to study the relationship between:

```text
Generative Model
       ↓
Synthetic Data
       ↓
Data Augmentation
       ↓
Classification
       ↓
Downstream Performance
```

---

# ⚙️ Hardware Considerations

Diffusion models are computationally intensive, particularly during SDXL fine-tuning.

A CUDA-enabled NVIDIA GPU is strongly recommended for:

* Stable Diffusion generation
* SDXL LoRA fine-tuning
* Large-scale image generation

The custom demo automatically selects CUDA when it is available and otherwise falls back to CPU.

For SDXL LoRA training, GPU memory requirements can vary significantly depending on:

* Batch size
* Resolution
* Gradient accumulation
* Mixed precision
* Text-encoder training
* Number of validation images
* Checkpointing configuration

The supplied LoRA configuration uses FP16 mixed precision and gradient accumulation to reduce training memory requirements.

---

# ⚠️ Important Notes

This repository contains research and experimental code rather than a packaged production application.

In particular:

* Model checkpoints are not necessarily included in the repository.
* Some scripts contain local filesystem paths that need to be changed before execution.
* The custom Stable Diffusion demo expects tokenizer/model files at specific relative paths.
* The LoRA workflow is designed to work alongside the Hugging Face Diffusers repository.
* Classification datasets referenced by the scripts may require local path adjustment.
* Dependency versions should be selected carefully because PyTorch, CUDA, TensorFlow, Diffusers, and Transformers versions can affect compatibility.

---

# 📚 References

This project builds upon ideas and implementations from the following areas:

* **Stable Diffusion / Latent Diffusion Models**
* **Denoising Diffusion Probabilistic Models (DDPM)**
* **CLIP**
* **Variational Autoencoders (VAE)**
* **U-Net architectures**
* **Classifier-Free Guidance**
* **Stable Diffusion XL**
* **Low-Rank Adaptation (LoRA)**
* **Hugging Face Diffusers**
* **ResNet50**

Recommended foundational references:

1. Rombach et al. — *High-Resolution Image Synthesis with Latent Diffusion Models*
2. Ho et al. — *Denoising Diffusion Probabilistic Models*
3. Radford et al. — *Learning Transferable Visual Models From Natural Language Supervision*
4. Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models*
5. He et al. — *Deep Residual Learning for Image Recognition*

---

# 🤝 Acknowledgements

This project makes use of concepts and open-source tools from the broader generative-AI and computer-vision ecosystem, particularly:

* PyTorch
* Hugging Face Diffusers
* Hugging Face Transformers
* TensorFlow / Keras
* Stable Diffusion
* Stable Diffusion XL
* CLIP
* LoRA

---

# 📄 License

Please add the license that applies to this project before publishing it as a reusable open-source project.

If this repository is intended primarily for academic/research use, an appropriate open-source license should be selected and added as a `LICENSE` file at the repository root.

---

# 👨‍💻 Author

**Ankit Ghosh**

GitHub: [Ankit-Ghosh](https://github.com/Ankit-Ghosh)

Repository: [SD_Image_Generation](https://github.com/Ankit-Ghosh/SD_Image_Generation)

---

# ⭐ Citation

If you use this repository in academic work, please cite the repository and the associated research work when a formal publication is available.

```text
Ankit Ghosh.
SD_Image_Generation: Stable Diffusion for Agricultural Image Generation.
GitHub Repository.
https://github.com/Ankit-Ghosh/SD_Image_Generation
```

---

# 🚧 Future Work

Potential future extensions include:

* Improved agricultural disease-specific generation
* Larger and more diverse crop datasets
* Systematic FID / IS / LPIPS evaluation
* Quantitative comparison of real and synthetic images
* Better caption generation
* Crop- and disease-specific LoRA adapters
* Improved image-to-image conditioning
* Automated dataset generation pipelines
* More robust classification experiments
* Ablation studies on synthetic-data augmentation
* Comparison of different diffusion samplers
* Automated evaluation of synthetic-data usefulness for downstream agricultural classification
* Integration of generated images into a complete agricultural disease-detection pipeline

---

## 🌾 Project Goal

The broader goal of this project is to investigate how **generative diffusion models can contribute to agricultural computer vision by creating realistic, diverse, and domain-specific synthetic imagery**.

By combining generative modeling, parameter-efficient fine-tuning, and downstream classification, the project provides a foundation for studying synthetic agricultural data generation and its potential value for machine-learning applications.
