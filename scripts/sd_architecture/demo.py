import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import time  # Import time module for generating a time-based seed
import random

# Check for GPU support and set the device accordingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize the tokenizer and model
tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE 

prompt = "A realistic, original and natural image of rice crop plants affected by downy mildew disease. In the background, the contrast between healthy green plants and the infected crops highlights the spread of the disease through the field."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50

# Modify to generate 10 images
output_images = []
n_images = 1  # Number of images to generate

# Specify the desired folder path
output_folder = Path("C:\Users\SUBIR KUMAR GHOSH\OneDrive\Documents\Project_4th_year")  # Update this path as needed
output_folder.mkdir(parents=True, exist_ok=True)  # Create the folder and any necessary parent folders

# Generate a new random seed based on the current time
random_seed = int(time.time())  # Use the current time as the seed for randomness
print(f"Using random seed: {random_seed}")

for i in range(n_images):
    # Use a different seed for each image by adding an offset to the base seed
    current_seed = random_seed + i  # Increment seed to ensure variety in generated images
    
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=current_seed,
        models=models,
        device=DEVICE,  # Set the device to CUDA
        idle_device="cpu",  # Move to CPU when idle (optional, can be set to 'cuda' if always on GPU)
        tokenizer=tokenizer,
    )
    
    # Convert output to PIL image
    pil_image = Image.fromarray(output_image)
    
    # Append the output image to the list
    output_images.append(pil_image)
    
    # Save the image to the specified folder
    output_image_path = output_folder / f"output_image_{i+1}.png"
    pil_image.save(output_image_path)
    print(f"Generated image {i+1} with seed {current_seed} saved to {output_image_path}")

# E:/Ankit/project-aaa/synthetic_dataset/brown_spot