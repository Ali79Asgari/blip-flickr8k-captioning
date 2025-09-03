# blip-flickr8k-captioning

A clean, end‑to‑end **image captioning** pipeline that fine‑tunes **[Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)** on the **Flickr8k** dataset.  
This notebook is organized into composable “blocks” with a simple pipeline runner (`ctx` dict), so you can run the whole workflow or just the pieces you need.

> **GPU recommended:** set a CUDA runtime (e.g., Google Colab → *Runtime ▸ Change runtime type* → **GPU**). The trainer automatically uses `bf16` on Ampere+ GPUs.

## ✨ What’s inside
- Reproducible pipeline with lightweight `@block` decorator and `run_pipeline` orchestrator.
- Dataset download via **kagglehub** → parses captions, creates CSV splits (80/10/10).
- **Hugging Face Datasets** loaders + **BLIP** processor/collator.
- Model setup that **freezes the vision encoder** and fine‑tunes the text decoder.
- Training with `transformers.Trainer`, automatic checkpointing to `./outputs/checkpoints`.
- Evaluation using **sacrebleu** with **multi‑reference BLEU** (base vs. fine‑tuned).
- Inference helpers and an optional **ipywidgets** demo for interactive captioning.
- All intermediate results available in an in‑memory context dict: `ctx`.

## 🧱 Pipeline overview
Each numbered function is a block. You can execute the full pipeline or a subset.

```
01) setup_env                      → device, seeds
02) download_flickr8k              → fetch dataset with kagglehub
03) prepare_data                   → parse captions, split images, save CSVs
04) load_ds                        → load HF Datasets from CSVs
05) build_proc_collator            → BLIP processor + data collator
06) build_model                    → load BLIP, freeze vision encoder
07) train                          → Trainer fit + save checkpoints
08) eval_bleu                      → BLEU (base vs. finetuned), save JSON
09) infer_core                     → core caption function for single images
10) demo_widgets (optional)        → notebook UI for quick tests
```

Run everything (inside the notebook):
```python
ctx = run_pipeline(PIPELINE, cfg)
```

Run a subset:
```python
ctx = run_pipeline(PIPELINE, cfg, only=[
    "01_setup_env", "02_download_flickr8k", "03_prepare_data", "04_load_datasets",
])
```

## ⚙️ Defaults & hyperparameters
```python
epochs=2
lr=5e-05
weight_decay=0.01
warmup_ratio=0.05
per_device_train_batch_size=4
per_device_eval_batch_size=4
grad_accum=4
logging_steps=50

# generation
num_beams=5
max_new_tokens=20
no_repeat_ngram_size=2
repetition_penalty=1.05
length_penalty=1.0
```

> Tip: change any field on `cfg` before running a block, e.g. `cfg.epochs = 3`.

## 🛠️ Installation
> Works in Google Colab or locally (Python 3.9+ recommended).

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 2) Install core dependencies
pip install -U datasets ipywidgets kagglehub numpy pandas pillow sacrebleu torch transformers

# (Optional) For faster training / newer GPUs
pip install -U accelerate

# 3) Jupyter (if running locally)
pip install -U jupyter ipykernel && python -m ipykernel install --user --name blip-flickr8k
```

### Flickr8k via Kaggle
This notebook uses **kagglehub** to download `adityajn105/flickr8k`.  
You may need Kaggle API credentials available on the machine (e.g., `~/.kaggle/kaggle.json`).

## 🚀 Usage
Open the notebook and run the cells in order. Key entry points:

```python
# run everything
ctx = run_pipeline(PIPELINE, cfg)

# after training:
ctx['07_train']['metrics']        # final_train_loss, perplexity
ctx['08_eval_bleu']               # BLEU_base, BLEU_finetuned, delta

# caption a single image with the fine-tuned model
caption = ctx['09_infer_core']['caption_image_core']('/path/to/image.jpg', max_new_tokens=24, num_beams=3)
print(caption)
```

To try the interactive uploader (notebook only), run block `10) demo_widgets` and upload an image.

## 📁 Outputs
Artifacts are written under `./outputs` (override with `cfg.outputs_dir`):

- `checkpoints/` – Trainer checkpoints and the fine‑tuned model (also saves processor).
- `train_metrics.json` – training loss and optional perplexity.
- `eval_bleu.json` – BLEU for base vs. fine‑tuned and the improvement (`delta`).
- `splits.json` – split stats (rows and unique images per split).

## 🧪 Evaluation details
- BLEU is computed with **sacrebleu** on up to **500 unique images** from the validation/test split.
- Uses **multi‑reference** scoring (up to **5 refs** per image).
- Compares **base** model vs. **your fine‑tuned** checkpoint and reports both.

## 🧩 How it works (brief)
- **Data prep:** Captions are parsed from `captions.txt` and grouped per image. Images are split 80/10/10 with a fixed RNG seed (`cfg.seed`). We then expand to a row‑wise CSV: `image_path, caption`.
- **Collator:** The BLIP processor tokenizes captions and prepares pixel values; label padding tokens are masked to `-100` for loss.
- **Model:** Loads `BlipForConditionalGeneration` and **freezes** all `vision_model*` parameters—only the text side trains.
- **Training:** Uses `transformers.Trainer` with mixed precision (`bf16` on Ampere+). Best model is restored at the end.
- **Inference:** A tiny wrapper encodes the image, calls `generate`, and decodes the caption.

## 🔧 Customization
- Swap models by changing `cfg.model_id` (e.g., a larger BLIP variant).
- Adjust the train/val/test ratios or provide your own CSVs (skip step 02/03).
- Increase `max_new_tokens` and `num_beams` for higher‑quality captions (slower).
- Unfreeze parts of the vision encoder if you want full‑model fine‑tuning.

## 🐛 Troubleshooting
- **CUDA OOM:** Lower `per_device_train_batch_size`, increase `grad_accum`, or reduce `max_new_tokens`.
- **Kaggle auth:** Ensure your Kaggle API token is configured if downloads fail.
- **PIL errors:** Some images may be truncated/corrupt; re‑download or filter.
- **bf16 not supported:** The script auto‑falls back to `fp16` where needed.

## 📦 Dependencies
```
- datasets
- ipywidgets
- kagglehub
- numpy
- pandas
- pillow
- sacrebleu
- torch
- transformers
```

## 📚 License
Choose a license (e.g., MIT) and add a `LICENSE` file to the repo.

---

*Last updated: 2025-09-03*
