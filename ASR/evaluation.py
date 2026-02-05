# import json
# from datasets import load_from_disk
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import evaluate
# from tqdm import tqdm
# import torch
# import numpy as np
# import torchaudio
# import os

# # Config
# MODEL_DIR = "./whisper-small-yoruba-final"  # or "your_username/whisper-small-yoruba-final" if loaded from Hub
# DATASET_PATH = "whisper_dataset_yor"
# AUDIO_FOLDER = "processed_audio_yor"  # Added for path fixing
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model and processor
# processor = WhisperProcessor.from_pretrained(MODEL_DIR)
# model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")

# # Load test dataset
# dataset = load_from_disk(DATASET_PATH)["test"]

# # Load WER metric
# wer_metric = evaluate.load("wer")

# # Evaluation loop
# results = []
# individual = []
# total_wer = 0.0

# for example in tqdm(dataset, desc="Evaluating"):
#     # Load audio on-the-fly from path (with path fix if relative)
#     path = example["audio"]["path"]
#     if not os.path.isabs(path):
#         path = os.path.join(AUDIO_FOLDER, path)

#     waveform, sr = torchaudio.load(path)
#     audio = waveform.squeeze(0).numpy().astype(np.float32)

#     # Check sample rate
#     if sr != 16000:
#         raise ValueError(f"Sample rate {sr} != 16000 in {path}")

#     # Pad to 30 seconds (480000 samples) if shorter
#     if len(audio) < 480000:
#         audio = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
#     # Truncate if longer (assuming your data is filtered <30s)
#     elif len(audio) > 480000:
#         audio = audio[:480000]

#     # Prepare input
#     input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

#     # Generate prediction
#     predicted_ids = model.generate(input_features)
#     prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

#     # Reference
#     reference = example["text"].lower().strip()

#     # Compute WER
#     wer = 100 * wer_metric.compute(predictions=[prediction], references=[reference])

#     # Save individual
#     individual.append({
#         "prediction": prediction,
#         "reference": reference,
#         "wer": wer
#     })

#     total_wer += wer

# # Average
# average_wer = total_wer / len(dataset)

# # JSON output
# output = {
#     "individual": individual,
#     "average_wer": average_wer
# }

# with open("evaluation_results.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, indent=4, ensure_ascii=False)

# print("Evaluation complete! Results saved to evaluation_results.json")




# =============================================================================
#  EVALUATION SCRIPT — Yoruba Whisper-small Checkpoint — Dec 2025
# =============================================================================

# =============================================================================
#  DETAILED EVALUATION SCRIPT — Yoruba Whisper-small — FIXED & WORKING
# =============================================================================

# import os
# import json
# import argparse
# import numpy as np
# from datasets import load_from_disk
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch
# import torchaudio
# from dataclasses import dataclass
# from tqdm import tqdm
# import evaluate
# import pandas as pd
# import re

# # ==============================
# # CONFIG
# # ==============================
# DATASET_SAVE_PATH = "whisper_dataset_yor_82h"
# AUDIO_FOLDER      = "processed_audio_ig"
# OUTPUT_DIR        = "whisper-small-yoruba-82h"
# RESULTS_JSON      = "evaluation_detailed_test_set.json"
# RESULTS_CSV       = "evaluation_detailed_test_set.csv"

# # ==============================
# # Simple text normalizer (replaces the broken "character" metric)
# # ==============================
# def normalize_text(text: str) -> str:
#     # Lowercase + remove all punctuation + collapse spaces
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ==============================
# # Args
# # ==============================
# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str, required=True,
#                     help="Path to checkpoint, e.g. whisper-small-yoruba-final/checkpoint-4000")
# parser.add_argument("--batch_size", type=int, default=16)
# args = parser.parse_args()

# checkpoint_path = args.checkpoint
# if not os.path.exists(checkpoint_path):
#     raise ValueError(f"Checkpoint not found: {checkpoint_path}")

# print(f"Loading model and processor from {checkpoint_path}")

# # ==============================
# # Load processor & model
# # ==============================
# processor = WhisperProcessor.from_pretrained(OUTPUT_DIR, language="yoruba", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# model.eval()

# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")
# model.generation_config.suppress_tokens = []

# # ==============================
# # Load ONLY the test split
# # ==============================
# print("Loading test dataset...")
# raw_datasets = load_from_disk(DATASET_SAVE_PATH)

# def fix_paths(ex):
#     basename = os.path.basename(ex["audio"]["path"])
#     ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
#     return ex

# test_dataset = raw_datasets["test"].map(fix_paths)
# test_dataset = test_dataset.map(lambda x: {"text": x["text"].lower().strip()})

# print(f"Test set size: {len(test_dataset):,} samples")

# # ==============================
# # Metrics (only official ones that work without extra installs)
# # ==============================
# wer_metric = evaluate.load("wer")
# cer_metric = evaluate.load("cer")

# # ==============================
# # 30-second padding collator (exactly like training)
# # ==============================
# @dataclass
# class EvalCollator:
#     processor: WhisperProcessor

#     def __call__(self, features):
#         audio_paths = [f["audio"]["path"] for f in features]
#         audio_arrays = []

#         for p in audio_paths:
#             wav, sr = torchaudio.load(p)
#             wav = wav.mean(0) if wav.ndim > 1 else wav.squeeze(0)
#             wav = wav.numpy().astype(np.float32)

#             if sr != 16000:
#                 raise ValueError(f"Wrong sample rate {sr} → {p}")

#             if len(wav) < 480000:
#                 wav = np.pad(wav, (0, 480000 - len(wav)))
#             wav = wav[:480000]
#             audio_arrays.append(wav)

#         inputs = processor.feature_extractor(
#             audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
#         )
#         return {"input_features": inputs.input_features}

# collator = EvalCollator(processor)

# # ==============================
# # Inference loop
# # ==============================
# results = []

# print("Starting inference on test set...")
# for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Evaluating"):
#     batch = [test_dataset[j] for j in range(i, min(i + args.batch_size, len(test_dataset)))]
#     collated = collator(batch)

#     with torch.no_grad():
#         predicted_ids = model.generate(
#             collated["input_features"].to(device),
#             max_length=225,
#             num_beams=5,
#             length_penalty=1.0,
#         )

#     predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

#     for pred, example in zip(predictions, batch):
#         ref = example["text"]

#         # Raw (with punctuation)
#         wer_raw  = wer_metric.compute(predictions=[pred], references=[ref]) * 100
#         cer_raw  = cer_metric.compute(predictions=[pred], references=[ref]) * 100

#         # Normalized (no punctuation, single space)
#         pred_norm = normalize_text(pred)
#         ref_norm  = normalize_text(ref)
#         wer_norm = wer_metric.compute(predictions=[pred_norm], references=[ref_norm]) * 100
#         cer_norm = cer_metric.compute(predictions=[pred_norm], references=[ref_norm]) * 100

#         results.append({
#             "file"                : os.path.basename(example["audio"]["path"]),
#             "reference"           : ref,
#             "prediction"          : pred,
#             "prediction_normalized": pred_norm,
#             "wer"                 : round(wer_raw, 3),
#             "cer"                 : round(cer_raw, 3),
#             "wer_normalized"      : round(wer_norm, 3),
#             "cer_normalized"      : round(cer_norm, 3),
#         })

# # ==============================
# # Final summary
# # ==============================
# df = pd.DataFrame(results)
# summary = {
#     "test_samples"       : len(df),
#     "wer"                : round(df["wer"].mean(), 3),
#     "cer"                : round(df["cer"].mean(), 3),
#     "wer_normalized"     : round(df["wer_normalized"].mean(), 3),
#     "cer_normalized"     : round(df["cer_normalized"].mean(), 3),
# }

# final_output = {"summary": summary, "per_sample": results}

# with open(RESULTS_JSON, "w", encoding="utf-8") as f:
#     json.dump(final_output, f, indent=2, ensure_ascii=False)

# df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

# print("\n" + "="*70)
# print("DETAILED EVALUATION ON TEST SET — FINISHED")
# print("="*70)
# print(f"Samples   : {summary['test_samples']}")
# print(f"WER       : {summary['wer']}%")
# print(f"CER       : {summary['cer']}%")
# print(f"WER (norm): {summary['wer_normalized']}%")
# print(f"CER (norm): {summary['cer_normalized']}%")
# print(f"\nResults → {RESULTS_JSON}")
# print(f"          {RESULTS_CSV}")
# print("="*70)


import os
import json
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from dataclasses import dataclass
from tqdm import tqdm
import evaluate
import pandas as pd
import re

# ==============================
# CONFIG - IGBO
# ==============================
DATASET_SAVE_PATH = "whisper_dataset_igbo_82h"   # ← Your Igbo dataset path
AUDIO_FOLDER      = "processed_audio_ig"       # ← Your Igbo audio folder
OUTPUT_DIR        = "whisper-small-igbo-82h"     # ← Folder where you saved processor during training
RESULTS_JSON      = "igbo_evaluation_detailed_test_set.json"
RESULTS_CSV       = "igbo_evaluation_detailed_test_set.csv"

# ==============================
# Simple text normalizer
# ==============================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================
# Args
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint, e.g. whisper-small-igbo-82h/checkpoint-4000")
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

checkpoint_path = args.checkpoint
if not os.path.exists(checkpoint_path):
    raise ValueError(f"Checkpoint not found: {checkpoint_path}")

print(f"Loading model and processor from {checkpoint_path}")

# ==============================
# Load processor & model
# ==============================
processor = WhisperProcessor.from_pretrained(OUTPUT_DIR, language=None, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


model.config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

# Optional: You can try proxy with a close supported language for slight boost
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")  # Some do this

# ==============================
# Load ONLY the test split
# ==============================
print("Loading test dataset...")
raw_datasets = load_from_disk(DATASET_SAVE_PATH)

def fix_paths(ex):
    basename = os.path.basename(ex["audio"]["path"])
    ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
    return ex

test_dataset = raw_datasets["test"].map(fix_paths)
test_dataset = test_dataset.map(lambda x: {"text": x["text"].lower().strip()})

print(f"Test set size: {len(test_dataset):,} samples")

# ==============================
# Metrics
# ==============================
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ==============================
# 30-second padding collator (matches training)
# ==============================
@dataclass
class EvalCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        audio_paths = [f["audio"]["path"] for f in features]
        audio_arrays = []

        for p in audio_paths:
            wav, sr = torchaudio.load(p)
            wav = wav.mean(0) if wav.ndim > 1 else wav.squeeze(0)
            wav = wav.numpy().astype(np.float32)

            if sr != 16000:
                raise ValueError(f"Wrong sample rate {sr} → {p}")

            if len(wav) < 480000:
                wav = np.pad(wav, (0, 480000 - len(wav)))
            wav = wav[:480000]
            audio_arrays.append(wav)

        inputs = processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
        )
        return {"input_features": inputs.input_features}

collator = EvalCollator(processor)

# ==============================
# Inference loop
# ==============================
results = []

print("Starting inference on test set...")
for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Evaluating"):
    batch = [test_dataset[j] for j in range(i, min(i + args.batch_size, len(test_dataset)))]
    collated = collator(batch)

    with torch.no_grad():
        predicted_ids = model.generate(
            collated["input_features"].to(device),
            max_length=225,
            num_beams=5,
            length_penalty=1.0,
        )

    predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    for pred, example in zip(predictions, batch):
        ref = example["text"]

        wer_raw  = wer_metric.compute(predictions=[pred], references=[ref]) * 100
        cer_raw  = cer_metric.compute(predictions=[pred], references=[ref]) * 100

        pred_norm = normalize_text(pred)
        ref_norm  = normalize_text(ref)
        wer_norm = wer_metric.compute(predictions=[pred_norm], references=[ref_norm]) * 100
        cer_norm = cer_metric.compute(predictions=[pred_norm], references=[ref_norm]) * 100

        results.append({
            "file"                : os.path.basename(example["audio"]["path"]),
            "reference"           : ref,
            "prediction"          : pred,
            "prediction_normalized": pred_norm,
            "wer"                 : round(wer_raw, 3),
            "cer"                 : round(cer_raw, 3),
            "wer_normalized"      : round(wer_norm, 3),
            "cer_normalized"      : round(cer_norm, 3),
        })

# ==============================
# Final summary
# ==============================
df = pd.DataFrame(results)
summary = {
    "test_samples"       : len(df),
    "wer"                : round(df["wer"].mean(), 3),
    "cer"                : round(df["cer"].mean(), 3),
    "wer_normalized"     : round(df["wer_normalized"].mean(), 3),
    "cer_normalized"     : round(df["cer_normalized"].mean(), 3),
}

final_output = {"summary": summary, "per_sample": results}

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

print("\n" + "="*70)
print("IGBO DETAILED EVALUATION ON TEST SET — FINISHED")
print("="*70)
print(f"Samples   : {summary['test_samples']}")
print(f"WER       : {summary['wer']}%")
print(f"CER       : {summary['cer']}%")
print(f"WER (norm): {summary['wer_normalized']}%")
print(f"CER (norm): {summary['cer_normalized']}%")
print(f"\nResults → {RESULTS_JSON}")
print(f"          {RESULTS_CSV}")
print("="*70)