# LoRA Fine-tuning for Stable Diffusion XL

This guide walks you through setting up the **Hugging Face Diffusers** repository and replacing the default LoRA fine-tuning script with your custom one from `scripts/LoRA_finetune`.

---

## 📦 1. Clone the Diffusers Repository

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
```

---

## 🛠 2. Install Diffusers in Editable Mode

```bash
pip install -e .
```

> **Note:** Editable mode (`-e`) allows you to modify the repository's code without reinstalling.

---

## 📂 3. Navigate to the `text_to_image` Examples Folder

```bash
cd examples/text_to_image
```

---

## 📜 4. Install SDXL Example Requirements

```bash
pip install -r requirements_sdxl.txt
```

---

## 🔄 5. Replace the Default LoRA Fine-tuning Script

1. Locate your custom script in:
   ```
   scripts/LoRA_finetune/train_text_to_image_lora_sdxl.py
   ```

2. Replace the existing script in:
   ```
   diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py
   ```

**Example:**
```bash
cp /path/to/scripts/LoRA_finetune/train_text_to_image_lora_sdxl.py    /path/to/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py
```

---

## 📌 Notes
- Ensure your Python environment uses the correct dependencies listed in `requirements_sdxl.txt`.
- If you modify your custom script later, you do **not** need to re-install Diffusers due to editable mode installation.
