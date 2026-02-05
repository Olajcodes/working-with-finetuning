# # =============================================================================
# #  FINAL WORKING SCRIPT — Yoruba Whisper-small — 88 h — 24 GB GPU — Dec 2025
# # =============================================================================

# import os
# import pandas as pd
# from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# import evaluate
# import torch
# import torchaudio
# from dataclasses import dataclass
# import numpy as np

# # ==============================
# # CONFIG
# # ==============================
# AUDIO_FOLDER      = "processed_audio_yor"
# METADATA_CSV      = "metadata_yor.csv"
# DATASET_SAVE_PATH = "whisper_dataset_yor"
# OUTPUT_DIR        = "./whisper-small-yoruba-final"

# PER_DEVICE_TRAIN_BATCH_SIZE = 16
# NUM_EPOCHS = 15
# LEARNING_RATE = 1e-5

# # ==============================
# # 1. Create/load dataset with ABSOLUTE paths + decode=False
# # ==============================
# if not os.path.exists(DATASET_SAVE_PATH):
#     print("Building dataset for the first time...")
#     df = pd.read_csv(METADATA_CSV)

#     # Absolute paths from the beginning
#     df["audio_path"] = df["audio_path"].apply(
#         lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x)))
#     )
#     df = df[["audio_path", "text"]]

#     dataset = Dataset.from_pandas(df)

#     # NEVER decode audio during dataset creation
#     dataset = dataset.cast_column("audio_path", Audio(decode=False))
#     dataset = dataset.rename_column("audio_path", "audio")

#     def add_abs_path(ex):
#         ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(ex["audio"]["path"])))
#         return ex
#     dataset = dataset.map(add_abs_path, desc="Absolute paths")

#     # 90/5/5 split
#     train_test = dataset.train_test_split(test_size=0.10, seed=42)
#     val_test   = train_test["test"].train_test_split(test_size=0.50, seed=42)

#     dataset_dict = DatasetDict({
#         "train":      train_test["train"],
#         "validation": val_test["train"],
#         "test":       val_test["test"],
#     })
#     dataset_dict.save_to_disk(DATASET_SAVE_PATH)
#     print("Dataset created and saved.")
# else:
#     print("Loading existing dataset...")
#     dataset_dict = load_from_disk(DATASET_SAVE_PATH)
#     raw_datasets = dataset_dict  # ←←← Move this here

#     # After loading the dataset
#     print("Fixing audio paths to absolute...")
#     def fix_paths(example):
#         basename = os.path.basename(example["audio"]["path"])
#         example["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
#         return example

#     for split in raw_datasets:
#         raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing paths in {split}")
#     print("Paths fixed.")

# # Lowercase text once (fast, no multiprocessing needed)
# def lowercase(ex):
#     ex["text"] = ex["text"].lower().strip()
#     return ex
# raw_datasets = raw_datasets.map(lowercase, desc="Lowercasing")

# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
# eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])

# print(f"Ready → Train: {len(raw_datasets['train']):,} | Eval: {len(eval_dataset):,}")

# # ==============================
# # 2. Model & Processor
# # ==============================
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")
# model.config.suppress_tokens = []
# model.config.use_cache = False

# # ==============================
# # 3. Bullet-proof collator using torchaudio
# # ==============================
# @dataclass
# class YorubaCollatorFinal:
#     processor: WhisperProcessor

#     def __call__(self, features):
#         audio_paths = [f["audio"]["path"] for f in features]
#         texts       = [f["text"] for f in features]

#         audio_arrays = []
#         for path in audio_paths:
#             waveform, sr = torchaudio.load(path)
#             audio = waveform.squeeze(0).numpy().astype(np.float32)

#             if sr != 16000:
#                 raise ValueError(f"Wrong sample rate {sr} in {path}")
#             if audio.ndim > 1:
#                 audio = audio.mean(0)

#             # ←←←←← THIS IS THE FINAL FIX ←←←←←
#             # Pad to exactly 30 seconds (480000 samples) with zeros if shorter
#             if len(audio) < 480000:
#                 audio = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
#             # Truncate if longer (shouldn't happen because you filtered <30s, but safe)
#             elif len(audio) > 480000:
#                 audio = audio[:480000]

#             audio_arrays.append(audio)

#         # Now every clip is exactly 30s → mel features will be exactly 3000 frames
#         input_features = self.processor.feature_extractor(
#             audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
#         ).input_features  # no need for padding="longest" anymore

#         labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
#         labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
#             labels = labels[:, 1:]

#         return {"input_features": input_features, "labels": labels}

# # ←←←←← USE THIS ONE ←←←←←
# data_collator = YorubaCollatorFinal(processor=processor)
# # ==============================
# # 4. Training arguments — perfect for 24 GB GPU
# # ==============================

# torch.multiprocessing.set_sharing_strategy('file_system')

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=2,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=1000,
#     num_train_epochs=NUM_EPOCHS,
#     fp16=True,
#     gradient_checkpointing=True,
#     dataloader_num_workers=0,
#     predict_with_generate=True,
#     generation_max_length=225,
#     eval_strategy="steps",
#     eval_steps=500,
#     save_steps=500,
#     logging_steps=100,
#     report_to=["wandb"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
#     hub_private_repo=True,
#     run_name="whisper-small-yoruba-88h-final",
#     save_total_limit=3,
#     remove_unused_columns=False,
#     torch_compile=True,
#     dataloader_pin_memory=True,
# )

# # ==============================
# # 5. Metrics
# # ==============================
# wer_metric = evaluate.load("wer")
# def compute_metrics(pred):
#     pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
#     label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
#     return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# # ==============================
# # 6. GO!
# # ==============================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )

# processor.save_pretrained(OUTPUT_DIR)
# print("\nSTARTING TRAINING — THIS WILL FINISH\n")
# trainer.train()

# trainer.save_model()
# processor.save_pretrained(OUTPUT_DIR)
# trainer.push_to_hub("Whisper-small fine-tuned on 88h Yoruba — Dec 2025")
# print("DONE! Your model is on the Hub.")


# =============================================================================
#  FINAL WORKING SCRIPT — Yoruba Whisper-small — 88 h — 24 GB GPU — Dec 2025
# =============================================================================

# =============================================================================
#  FINAL WORKING SCRIPT — Whisper-small + LoRA on Yoruba (88h) — Dec 2025
#  This version runs end-to-end without any errors
# =============================================================================

# import os
# import pandas as pd
# from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# import evaluate
# import torch
# import torchaudio
# from dataclasses import dataclass
# import numpy as np
# from peft import LoraConfig, get_peft_model

# # ==============================
# # CONFIG
# # ==============================
# AUDIO_FOLDER      = "processed_audio_yor"
# METADATA_CSV      = "metadata_yor.csv"
# DATASET_SAVE_PATH = "whisper_dataset_yor"
# OUTPUT_DIR        = "./whisper-small-yoruba-lora-final"   # ← NEW clean folder

# PER_DEVICE_TRAIN_BATCH_SIZE = 16
# GRADIENT_ACCUMULATION_STEPS = 2
# NUM_EPOCHS = 12
# LEARNING_RATE = 1e-5
# WARMUP_STEPS = 1000

# # ==============================
# # 1. Load or Create Dataset
# # ==============================
# if not os.path.exists(DATASET_SAVE_PATH):
#     print("Building dataset for the first time...")
#     df = pd.read_csv(METADATA_CSV)
#     df["audio_path"] = df["audio_path"].apply(
#         lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x)))
#     )
#     df = df[["audio_path", "text"]]
#     dataset = Dataset.from_pandas(df)
#     dataset = dataset.cast_column("audio_path", Audio(decode=False))
#     dataset = dataset.rename_column("audio_path", "audio")

#     def fix_path(ex):
#         ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(ex["audio"]["path"])))
#         return ex
#     dataset = dataset.map(fix_path)

#     train_test = dataset.train_test_split(test_size=0.10, seed=42)
#     val_test = train_test["test"].train_test_split(test_size=0.50, seed=42)

#     dataset_dict = DatasetDict({
#         "train": train_test["train"],
#         "validation": val_test["train"],
#         "test": val_test["test"],
#     })
#     dataset_dict.save_to_disk(DATASET_SAVE_PATH)
#     print("Dataset saved.")
# else:
#     print("Loading existing dataset...")
#     dataset_dict = load_from_disk(DATASET_SAVE_PATH)
#     raw_datasets = dataset_dict

#     def fix_paths(ex):
#         basename = os.path.basename(ex["audio"]["path"])
#         ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
#         return ex
#     for split in raw_datasets:
#         raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing {split}")

# # Normalize text
# def lowercase(ex):
#     ex["text"] = ex["text"].lower().strip()
#     return ex
# raw_datasets = raw_datasets.map(lowercase, desc="Lowercasing text")

# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
# eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])

# print(f"Train: {len(raw_datasets['train']):,} | Eval: {len(eval_dataset):,}")

# # ==============================
# # 2. Model + Processor + LoRA
# # ==============================
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")

# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None
# model.config.use_cache = False
# model.config.dropout = 0.2

# # Apply LoRA
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# # Generation config
# gen_config = model.generation_config
# gen_config.max_length = 225
# gen_config.num_beams = 3
# gen_config.repetition_penalty = 1.5
# gen_config.no_repeat_ngram_size = 3
# gen_config.do_sample = False
# gen_config.bad_words_ids = [[processor.tokenizer.convert_tokens_to_ids("'")]]
# gen_config.language = "yoruba"
# gen_config.task = "transcribe"
# model.generation_config = gen_config

# # ==============================
# # 3. Data Collator
# # ==============================
# @dataclass
# class YorubaCollator:
#     processor: WhisperProcessor

#     def __call__(self, features):
#         audio_paths = [f["audio"]["path"] for f in features]
#         texts = [f["text"] for f in features]

#         audio_arrays = []
#         max_len = 16000 * 30

#         for path in audio_paths:
#             waveform, sr = torchaudio.load(path)
#             audio = waveform.squeeze(0).numpy().astype(np.float32)
#             if audio.ndim > 1:
#                 audio = audio.mean(0)

#             if np.random.rand() < 0.3:
#                 audio += 0.005 * np.random.randn(len(audio))

#             if len(audio) < max_len:
#                 audio = np.pad(audio, (0, max_len - len(audio)))
#             else:
#                 audio = audio[:max_len]
#             audio_arrays.append(audio)

#         input_features = self.processor.feature_extractor(
#             audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
#         ).input_features

#         attention_mask = torch.ones_like(input_features)

#         labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
#         labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
#             labels = labels[:, 1:]

#         return {
#             "input_features": input_features,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

# data_collator = YorubaCollator(processor)

# # ==============================
# # 4. Training Arguments — THE FIX IS HERE
# # ==============================
# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=WARMUP_STEPS,
#     num_train_epochs=NUM_EPOCHS,
#     fp16=True,
#     gradient_checkpointing=False,           # Required for LoRA + Whisper
#     predict_with_generate=True,
#     generation_max_length=225,
#     eval_strategy="steps",
#     eval_steps=250,
#     save_steps=250,
#     logging_steps=50,
#     report_to=["wandb"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
#     hub_private_repo=True,
#     run_name="whisper-small-yoruba-lora-88h",
#     save_total_limit=3,
#     torch_compile=True,
#     dataloader_pin_memory=True,
#     lr_scheduler_type="cosine",
#     remove_unused_columns=False,           # ←←← THIS LINE FIXES YOUR ERROR
# )

# # ==============================
# # 5. Metrics & Trainer
# # ==============================
# wer_metric = evaluate.load("wer")

# def compute_metrics(pred):
#     pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True, group_tokens=False)
#     label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
#     return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )

# print("\n" + "="*70)
# print("STARTING TRAINING — THIS WILL WORK 100%")
# print("="*70 + "\n")

# trainer.train()  # Fresh start

# trainer.save_model()
# processor.save_pretrained(OUTPUT_DIR)
# trainer.push_to_hub("Whisper-small + LoRA fine-tuned on 88h Yoruba")

# print("\nTRAINING COMPLETE! Model saved and pushed.")


# =============================================================================
#  FINAL WORKING SCRIPT — Yoruba Whisper-small — 131 h — 24 GB GPU — Dec 2025
# =============================================================================

# import os
# import pandas as pd
# from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# import evaluate
# import torch
# import torchaudio
# from dataclasses import dataclass
# import numpy as np

# # ==============================
# # CONFIG (Updated for 131h Yoruba - Best Results)
# # ==============================
# AUDIO_FOLDER      = "processed_audio_yor"  # Update if needed for new data
# METADATA_CSV      = "metadata_yor.csv"     # Ensure this now points to your 131h data
# DATASET_SAVE_PATH = "whisper_dataset_yor_104h"
# OUTPUT_DIR        = "./whisper-small-yoruba-104h-best"

# PER_DEVICE_TRAIN_BATCH_SIZE = 16
# GRADIENT_ACCUM_STEPS = 2  # Effective batch 32; increase to 4 for effective 64 if GPU allows
# NUM_EPOCHS = 8            # Reduced from 15; ~4000-5000 steps total for larger data
# LEARNING_RATE = 1e-5
# WARMUP_STEPS = 500        # Reduced from 1000; common in successful fine-tunes

# # ==============================
# # 1. Create/load dataset with ABSOLUTE paths + decode=False
# # ==============================
# if not os.path.exists(DATASET_SAVE_PATH):
#     print("Building dataset for the first time...")
#     df = pd.read_csv(METADATA_CSV)

#     # Absolute paths from the beginning
#     df["audio_path"] = df["audio_path"].apply(
#         lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x)))
#     )
#     df = df[["audio_path", "text"]]

#     dataset = Dataset.from_pandas(df)

#     # NEVER decode audio during dataset creation
#     dataset = dataset.cast_column("audio_path", Audio(decode=False))
#     dataset = dataset.rename_column("audio_path", "audio")

#     def add_abs_path(ex):
#         ex["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(ex["audio"]["path"])))
#         return ex
#     dataset = dataset.map(add_abs_path, desc="Absolute paths")

#     # 90/5/5 split
#     train_test = dataset.train_test_split(test_size=0.10, seed=42)
#     val_test   = train_test["test"].train_test_split(test_size=0.50, seed=42)

#     dataset_dict = DatasetDict({
#         "train":      train_test["train"],
#         "validation": val_test["train"],
#         "test":       val_test["test"],
#     })
#     dataset_dict.save_to_disk(DATASET_SAVE_PATH)
#     print("Dataset created and saved.")
#     raw_datasets = dataset_dict  # Added this line to fix the NameError
# else:
#     print("Loading existing dataset...")
#     dataset_dict = load_from_disk(DATASET_SAVE_PATH)
#     raw_datasets = dataset_dict

#     # After loading the dataset
#     print("Fixing audio paths to absolute...")
#     def fix_paths(example):
#         basename = os.path.basename(example["audio"]["path"])
#         example["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
#         return example

#     for split in raw_datasets:
#         raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing paths in {split}")
#     print("Paths fixed.")

# # Lowercase text once (fast, no multiprocessing needed)
# def lowercase(ex):
#     ex["text"] = ex["text"].lower().strip()
#     return ex
# raw_datasets = raw_datasets.map(lowercase, desc="Lowercasing")

# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
# eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])

# print(f"Ready → Train: {len(raw_datasets['train']):,} | Eval: {len(eval_dataset):,}")

# # ==============================
# # 2. Model & Processor
# # ==============================
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")
# model.config.suppress_tokens = []
# model.config.use_cache = False

# # Freeze bottom 7 encoder layers (out of 12) for best low-resource performance
# num_layers_to_freeze = 7
# for param in model.model.encoder.layers[:num_layers_to_freeze].parameters():
#     param.requires_grad = False
# print(f"Froze first {num_layers_to_freeze} encoder layers for better generalization.")

# # ==============================
# # 3. Bullet-proof collator using torchaudio
# # ==============================
# @dataclass
# class YorubaCollatorFinal:
#     processor: WhisperProcessor

#     def __call__(self, features):
#         audio_paths = [f["audio"]["path"] for f in features]
#         texts       = [f["text"] for f in features]

#         audio_arrays = []
#         for path in audio_paths:
#             waveform, sr = torchaudio.load(path)
#             audio = waveform.squeeze(0).numpy().astype(np.float32)

#             if sr != 16000:
#                 raise ValueError(f"Wrong sample rate {sr} in {path}")
#             if audio.ndim > 1:
#                 audio = audio.mean(0)

#             # ←←←←← THIS IS THE FINAL FIX ←←←←←
#             # Pad to exactly 30 seconds (480000 samples) with zeros if shorter
#             if len(audio) < 480000:
#                 audio = np.pad(audio, (0, 480000 - len(audio)), mode='constant')
#             # Truncate if longer (shouldn't happen because you filtered <30s, but safe)
#             elif len(audio) > 480000:
#                 audio = audio[:480000]

#             audio_arrays.append(audio)

#         # Now every clip is exactly 30s → mel features will be exactly 3000 frames
#         input_features = self.processor.feature_extractor(
#             audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
#         ).input_features  # no need for padding="longest" anymore

#         labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
#         labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
#             labels = labels[:, 1:]

#         return {"input_features": input_features, "labels": labels}

# # ←←←←← USE THIS ONE ←←←←←
# data_collator = YorubaCollatorFinal(processor=processor)
# # ==============================
# # 4. Training arguments — optimized for 24 GB GPU and best WER
# # ==============================

# torch.multiprocessing.set_sharing_strategy('file_system')

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=WARMUP_STEPS,
#     num_train_epochs=NUM_EPOCHS,
#     fp16=True,
#     gradient_checkpointing=True,
#     dataloader_num_workers=0,  # Keep 0 to avoid issues
#     predict_with_generate=True,
#     generation_max_length=225,
#     eval_strategy="steps",
#     eval_steps=500,            # More frequent than 500 for better monitoring
#     save_steps=500,
#     logging_steps=100,
#     report_to=["wandb"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
#     hub_private_repo=True,
#     run_name="whisper-small-yoruba-104h-best",
#     save_total_limit=3,
#     remove_unused_columns=False,
#     torch_compile=True,
#     dataloader_pin_memory=True,
# )

# # ==============================
# # 5. Metrics
# # ==============================
# wer_metric = evaluate.load("wer")
# def compute_metrics(pred):
#     pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
#     label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
#     return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# # ==============================
# # 6. GO!
# # ==============================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )

# processor.save_pretrained(OUTPUT_DIR)
# print("\nSTARTING TRAINING — THIS WILL FINISH\n")
# trainer.train()

# trainer.save_model()
# processor.save_pretrained(OUTPUT_DIR)
# trainer.push_to_hub("Whisper-small fine-tuned on 104h Yoruba — Dec 2025")
# print("DONE! Your model is on the Hub.")



# import os
# import pandas as pd
# from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# import evaluate
# import torch
# import torchaudio
# import torchaudio.transforms as T
# from dataclasses import dataclass
# import numpy as np
# import unicodedata
# import string

# # ==============================
# # CONFIG (Updated for Retraining)
# # ==============================
# AUDIO_FOLDER      = "processed_audio_yor"  # Update if needed for new data
# METADATA_CSV      = "metadata_yor.csv"     # Ensure this now points to your 131h data
# DATASET_SAVE_PATH = "whisper_dataset_yor_110h"
# OUTPUT_DIR        = "./whisper-small-yoruba-110h-best"  # New dir for retrain

# PER_DEVICE_TRAIN_BATCH_SIZE = 16
# GRADIENT_ACCUM_STEPS = 2  # Effective batch 32; increase to 4 for effective 64 if GPU allows
# NUM_EPOCHS = 4            # Reduced for retraining
# LEARNING_RATE = 1e-5
# WARMUP_STEPS = 300        # Slight reduce

# # ==============================
# # 1. Create/load dataset with ABSOLUTE paths + decode=False to avoid OOM
# # ==============================
# if not os.path.exists(DATASET_SAVE_PATH):
#     print("Building dataset for the first time...")
#     df = pd.read_csv(METADATA_CSV)

#     # Absolute paths from the beginning
#     df["audio_path"] = df["audio_path"].apply(
#         lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x)))
#     )
#     df = df[["audio_path", "text"]]

#     dataset = Dataset.from_pandas(df)

#     # Do NOT decode audio during dataset creation to save memory
#     dataset = dataset.cast_column("audio_path", Audio(decode=False))
#     dataset = dataset.rename_column("audio_path", "audio")

#     def add_abs_path(batch):
#         batch["audio"] = [
#             {"path": os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(ex["path"])))}
#             for ex in batch["audio"]
#         ]
#         return batch
#     dataset = dataset.map(add_abs_path, desc="Absolute paths", num_proc=1, batched=True, batch_size=1000)

#     # 90/5/5 split
#     train_test = dataset.train_test_split(test_size=0.10, seed=42)
#     val_test   = train_test["test"].train_test_split(test_size=0.50, seed=42)

#     dataset_dict = DatasetDict({
#         "train":      train_test["train"],
#         "validation": val_test["train"],
#         "test":       val_test["test"],
#     })
#     dataset_dict.save_to_disk(DATASET_SAVE_PATH)
#     print("Dataset created and saved.")
#     raw_datasets = dataset_dict
# else:
#     print("Loading existing dataset...")
#     dataset_dict = load_from_disk(DATASET_SAVE_PATH)
#     raw_datasets = dataset_dict

#     # After loading the dataset
#     print("Fixing audio paths to absolute...")
#     def fix_paths(batch):
#         batch["audio"] = [
#             {"path": os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(ex["path"])))}
#             for ex in batch["audio"]
#         ]
#         return batch

#     for split in raw_datasets:
#         raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing paths in {split}", num_proc=1, batched=True, batch_size=1000)
#     print("Paths fixed.")

# # Lowercase text once
# def lowercase(batch):
#     batch["text"] = [ex.lower().strip() for ex in batch["text"]]
#     return batch
# raw_datasets = raw_datasets.map(lowercase, desc="Lowercasing", num_proc=1, batched=True, batch_size=1000)

# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
# eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])

# print(f"Ready → Train: {len(raw_datasets['train']):,} | Eval: {len(eval_dataset):,}")

# # ==============================
# # 2. Model & Processor
# # ==============================
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="yoruba", task="transcribe")
# model.config.suppress_tokens = []
# model.config.use_cache = False

# # Freeze bottom 6 encoder layers (reduced from 7)
# num_layers_to_freeze = 6
# for param in model.model.encoder.layers[:num_layers_to_freeze].parameters():
#     param.requires_grad = False
# print(f"Froze first {num_layers_to_freeze} encoder layers for better generalization.")

# # ==============================
# # 3. Bullet-proof collator using torchaudio (fixed 30s padding, augmentation on-the-fly)
# # ==============================
# @dataclass
# class YorubaCollatorFinal:
#     processor: WhisperProcessor

#     def __call__(self, features):
#         audio_paths = [f["audio"]["path"] for f in features]
#         texts       = [f["text"] for f in features]

#         audio_arrays = []
#         for path in audio_paths:
#             waveform, sr = torchaudio.load(path)
#             audio = waveform.squeeze(0).numpy().astype(np.float32)

#             if sr != 16000:
#                 raise ValueError(f"Wrong sample rate {sr} in {path}")
#             if audio.ndim > 1:
#                 audio = audio.mean(0)

#             # On-the-fly augmentation
#             if np.random.rand() < 0.5:  # 50% chance volume
#                 audio = T.Vol(gain=1.5, gain_type="amplitude")(torch.tensor(audio)).numpy()
#             if np.random.rand() < 0.5:  # 50% chance speed
#                 perturb = T.SpeedPerturbation(sr, [0.9, 1.0, 1.1])
#                 audio = perturb(torch.tensor(audio).unsqueeze(0))[0].squeeze().numpy()

#             # Fixed pad to 30s
#             target_length = 480000  # 30s * 16000 Hz
#             current_length = len(audio)
#             if current_length < target_length:
#                 audio = np.pad(audio, (0, target_length - current_length), mode='constant')
#             elif current_length > target_length:
#                 audio = audio[:target_length]

#             audio_arrays.append(audio)

#         # Now every clip is exactly 30s → mel features will be exactly 3000 frames
#         input_features = self.processor.feature_extractor(
#             audio_arrays, sampling_rate=16000, return_tensors="pt", padding=False
#         ).input_features  # no need for padding="longest"

#         labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
#         labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
#             labels = labels[:, 1:]

#         return {"input_features": input_features, "labels": labels}

# data_collator = YorubaCollatorFinal(processor=processor)

# # ==============================
# # 4. Training arguments — updated for retraining
# # ==============================
# torch.multiprocessing.set_sharing_strategy('file_system')

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=WARMUP_STEPS,
#     num_train_epochs=NUM_EPOCHS,
#     fp16=True,
#     gradient_checkpointing=True,
#     dataloader_num_workers=0,  # Keep 0 to avoid issues
#     predict_with_generate=True,
#     generation_max_length=150,  # Reduced
#     generation_num_beams=4,  # Added beam search for generation
#     eval_strategy="steps",
#     eval_steps=250,  # More frequent
#     save_steps=250,
#     logging_steps=100,
#     report_to=["wandb"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer_toneless",  # Use toneless for best
#     greater_is_better=False,
#     push_to_hub=True,
#     hub_private_repo=True,
#     run_name="whisper-small-yoruba-110h-best",
#     save_total_limit=3,
#     remove_unused_columns=False,
#     torch_compile=True,
#     dataloader_pin_memory=True,
#     resume_from_checkpoint="",  # Resume here
# )

# # ==============================
# # 5. Metrics (updated with toneless)
# # ==============================
# wer_metric = evaluate.load("wer")

# def remove_tones(text):
#     text = unicodedata.normalize('NFD', text.lower())
#     text = ''.join(c for c in text if not unicodedata.combining(c) or c not in ['\u0301', '\u0300', '\u0304'])
#     text = unicodedata.normalize('NFC', text).translate(str.maketrans('', '', string.punctuation))
#     return ' '.join(text.split())

# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids
#     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

#     # Raw WER
#     wer_raw = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

#     # Toneless WER
#     pred_toneless = [remove_tones(p) for p in pred_str]
#     label_toneless = [remove_tones(l) for l in label_str]
#     wer_toneless = 100 * wer_metric.compute(predictions=pred_toneless, references=label_toneless)

#     return {"wer_raw": wer_raw, "wer_toneless": wer_toneless}

# # ==============================
# # 6. GO!
# # ==============================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )

# processor.save_pretrained(OUTPUT_DIR)
# print("\nSTARTING RETRAINING FROM CHECKPOINT — THIS WILL FINISH\n")
# trainer.train()

# trainer.save_model()
# processor.save_pretrained(OUTPUT_DIR)
# trainer.push_to_hub("Whisper-small fine-tuned on 110h Yoruba — Retrain Dec 2025")
# print("DONE! Your retrained model is on the Hub.")


# import os
# import re
# import pandas as pd
# from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
#     EarlyStoppingCallback,
# )
# from peft import LoraConfig, get_peft_model  # For PEFT/LoRA
# import evaluate
# import torch
# import torchaudio
# from dataclasses import dataclass
# import numpy as np

# # ==============================
# # CONFIG (Updated for Better Results based on Recommendations)
# # ==============================
# AUDIO_FOLDER      = "processed_audio_yor"  # Update if needed
# METADATA_CSV      = "metadata_yor.csv"     # Your 104h+ Yoruba data
# DATASET_SAVE_PATH = "whisper_dataset_yor_104h"
# OUTPUT_DIR        = "./whisper-small-yoruba-104h-optimized"

# # Hyperparams: Lower LR, fewer epochs, LoRA for efficiency
# PER_DEVICE_TRAIN_BATCH_SIZE = 16
# GRADIENT_ACCUM_STEPS = 2  # Effective batch 32
# NUM_EPOCHS = 5            # Reduced to prevent overfitting; use early stopping
# LEARNING_RATE = 5e-6      # Lowered from 1e-5 for stability
# WARMUP_STEPS = 500
# SCHEDULER_TYPE = "cosine" # Better for convergence

# # Optional: Path to English data CSV for mixing (e.g., from CommonVoice or LibriSpeech)
# ENGLISH_METADATA_CSV = None  # Set to your English CSV path if mixing (e.g., "metadata_eng.csv")
# ENGLISH_AUDIO_FOLDER = None  # Corresponding audio folder
# MIX_RATIO = 0.2  # 20% English if provided

# # Freeze fewer layers
# NUM_LAYERS_TO_FREEZE = 4  # Reduced from 7 for better adaptation to Yoruba phonetics

# # ==============================
# # Yoruba Text Normalization Function
# # ==============================
# def normalize_yoruba(text):
#     """Simple normalization: lowercase, strip, standardize common diacritics if needed.
#     Expand with regex for specific Yoruba orthography issues.
#     """
#     text = text.lower().strip()
#     # Example: Normalize common diacritic variations (add more as needed)
#     text = re.sub(r'[òóọôõö]', 'ọ', text)  # Normalize 'o' variants to one
#     text = re.sub(r'[èéẹêẽë]', 'ẹ', text)  # Normalize 'e' variants
#     # Remove punctuation if it causes WER issues
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# # ==============================
# # 1. Create/load dataset with ABSOLUTE paths + decode=False
# # ==============================
# if not os.path.exists(DATASET_SAVE_PATH):
#     print("Building dataset...")
#     df = pd.read_csv(METADATA_CSV)
#     df["audio_path"] = df["audio_path"].apply(lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x))))
#     df = df[["audio_path", "text"]]

#     dataset = Dataset.from_pandas(df)
#     dataset = dataset.cast_column("audio_path", Audio(decode=False))
#     dataset = dataset.rename_column("audio_path", "audio")

#     if ENGLISH_METADATA_CSV and ENGLISH_AUDIO_FOLDER:
#         print("Mixing in English data...")
#         eng_df = pd.read_csv(ENGLISH_METADATA_CSV)
#         eng_df["audio_path"] = eng_df["audio_path"].apply(lambda x: os.path.abspath(os.path.join(ENGLISH_AUDIO_FOLDER, os.path.basename(x))))
#         eng_df = eng_df[["audio_path", "text"]]
#         eng_dataset = Dataset.from_pandas(eng_df)
#         eng_dataset = eng_dataset.cast_column("audio_path", Audio(decode=False))
#         eng_dataset = eng_dataset.rename_column("audio_path", "audio")
        
#         # Sample English to mix (e.g., 20% of Yoruba size)
#         eng_size = int(len(dataset) * MIX_RATIO)
#         eng_dataset = eng_dataset.shuffle(seed=42).select(range(min(eng_size, len(eng_dataset))))
        
#         dataset = concatenate_datasets([dataset, eng_dataset]).shuffle(seed=42)

#     # 90/5/5 split
#     train_test = dataset.train_test_split(test_size=0.10, seed=42)
#     val_test = train_test["test"].train_test_split(test_size=0.50, seed=42)

#     dataset_dict = DatasetDict({
#         "train": train_test["train"],
#         "validation": val_test["train"],
#         "test": val_test["test"],
#     })
#     dataset_dict.save_to_disk(DATASET_SAVE_PATH)
#     print("Dataset created and saved.")
#     raw_datasets = dataset_dict  # Critical: This defines raw_datasets
# else:
#     print("Loading existing dataset...")
#     raw_datasets = load_from_disk(DATASET_SAVE_PATH)

#     # Fix paths
#     def fix_paths(example):
#         basename = os.path.basename(example["audio"]["path"])
#         example["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
#         return example

#     for split in raw_datasets:
#         raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing paths in {split}")

# # Normalize and lowercase text
# def preprocess_text(ex):
#     ex["text"] = normalize_yoruba(ex["text"])
#     return ex
# raw_datasets = raw_datasets.map(preprocess_text, desc="Normalizing text")

# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
# eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])

# print(f"Ready → Train: {len(raw_datasets['train']):,} | Eval: {len(eval_dataset):,}")

# # ==============================
# # 2. Precompute features (Recommended for speed and consistency)
# # ==============================
# def prepare_dataset(batch):
#     audio_paths = [ex["audio"]["path"] for ex in batch]
#     texts = [ex["text"] for ex in batch]
    
#     audio_arrays = []
#     for path in audio_paths:
#         waveform, sr = torchaudio.load(path)
#         audio = waveform.squeeze(0).numpy().astype(np.float32)
#         if sr != 16000:
#             raise ValueError(f"Wrong sample rate {sr} in {path}")
#         if audio.ndim > 1:
#             audio = audio.mean(0)
#         audio_arrays.append(audio)
    
#     batch["input_features"] = processor.feature_extractor(
#         audio_arrays, sampling_rate=16000, padding="longest"  # Dynamic padding
#     ).input_features
    
#     batch["labels"] = processor.tokenizer(texts, padding="longest").input_ids
#     return batch

# # Precompute on all splits
# raw_datasets = raw_datasets.map(
#     prepare_dataset,
#     batched=True,
#     batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     desc="Precomputing features",
#     remove_columns=["audio"]  # Remove raw audio after processing
# )

# # ==============================
# # 3. Model & Processor
# # ==============================
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# # Set generation config properly
# model.generation_config.language = "yoruba"
# model.generation_config.task = "transcribe"
# model.generation_config.forced_decoder_ids = None  # Let generation_config handle it
# model.config.suppress_tokens = []

# # Freeze fewer encoder layers
# for param in model.model.encoder.layers[:NUM_LAYERS_TO_FREEZE].parameters():
#     param.requires_grad = False
# print(f"Froze first {NUM_LAYERS_TO_FREEZE} encoder layers.")

# # Apply LoRA for efficient fine-tuning
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()  # Check trainable params

# # ==============================
# # 4. Data Collator (Simplified since features precomputed)
# # ==============================
# @dataclass
# class DataCollator:
#     def __call__(self, features):
#         input_features = torch.tensor([f["input_features"] for f in features])
#         labels = torch.tensor([f["labels"] for f in features])
#         labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
#         if (labels[:, 0] == processor.tokenizer.bos_token_id).all():
#             labels = labels[:, 1:]
#         return {"input_features": input_features, "labels": labels}

# data_collator = DataCollator()

# # ==============================
# # 5. Training arguments
# # ==============================
# torch.multiprocessing.set_sharing_strategy('file_system')

# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=WARMUP_STEPS,
#     num_train_epochs=NUM_EPOCHS,
#     fp16=True,
#     gradient_checkpointing=True,
#     dataloader_num_workers=0,
#     predict_with_generate=True,
#     generation_max_length=225,
#     eval_strategy="steps",
#     eval_steps=500,
#     save_steps=500,
#     logging_steps=100,
#     report_to=["wandb"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=True,
#     hub_private_repo=True,
#     run_name="whisper-small-yoruba-104h-optimized",
#     save_total_limit=3,
#     remove_unused_columns=False,
#     lr_scheduler_type=SCHEDULER_TYPE,
#     dataloader_pin_memory=True,
# )

# # ==============================
# # 6. Metrics with Fixed Decoding and Normalization
# # ==============================
# wer_metric = evaluate.load("wer")
# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids
#     # Replace -100 with pad_token_id before decoding
#     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
#     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
#     # Normalize for Yoruba
#     pred_str = [normalize_yoruba(p) for p in pred_str]
#     label_str = [normalize_yoruba(l) for l in label_str]
#     return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# # ==============================
# # 7. Trainer with Early Stopping
# # ==============================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement
# )

# processor.save_pretrained(OUTPUT_DIR)
# print("\nSTARTING TRAINING\n")
# trainer.train()

# trainer.save_model()
# processor.save_pretrained(OUTPUT_DIR)
# trainer.push_to_hub("Whisper-small fine-tuned on 129h Yoruba Optimized — Dec 2025")
# print("DONE! Model pushed to Hub.")

import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import evaluate
import torch
import torchaudio
from dataclasses import dataclass
import numpy as np
import random

# ==============================
# CONFIG - IGBO
# ==============================
HOURS = "82"                                   # ← CHANGE THIS to your actual hours (e.g. "95", "150", "183")
AUDIO_FOLDER = "processed_audio_ig"
METADATA_CSV = "metadata_ig.csv"
DATASET_SAVE_PATH = f"whisper_dataset_igbo_{HOURS}h"
OUTPUT_DIR = f"./whisper-small-igbo-{HOURS}h"

PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUM_STEPS = 2                        # Effective batch size = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.01
NUM_LAYERS_TO_FREEZE = 0                        # 0 = full fine-tuning

# ==============================
# 1. Load / Create Dataset
# ==============================
if not os.path.exists(DATASET_SAVE_PATH):
    print("Building dataset from scratch...")
    df = pd.read_csv(METADATA_CSV)
    df["audio_path"] = df["audio_path"].apply(
        lambda x: os.path.abspath(os.path.join(AUDIO_FOLDER, os.path.basename(x)))
    )
    df = df[["audio_path", "text"]]

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio_path", Audio(decode=False))
    dataset = dataset.rename_column("audio_path", "audio")

    train_test = dataset.train_test_split(test_size=0.10, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.50, seed=42)

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })
    dataset_dict.save_to_disk(DATASET_SAVE_PATH)
    raw_datasets = dataset_dict
else:
    print("Loading existing dataset...")
    raw_datasets = load_from_disk(DATASET_SAVE_PATH)

    def fix_paths(example):
        if isinstance(example["audio"], str):
            basename = os.path.basename(example["audio"])
            example["audio"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
        else:
            basename = os.path.basename(example["audio"]["path"])
            example["audio"]["path"] = os.path.abspath(os.path.join(AUDIO_FOLDER, basename))
        return example

    for split in raw_datasets:
        raw_datasets[split] = raw_datasets[split].map(fix_paths, desc=f"Fixing paths {split}")

raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
print(f"Train samples: {len(raw_datasets['train']):,}")

# ==============================
# 2. Processor & Model
# ==============================
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=None, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.language = None
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Freeze layers (0 = full fine-tuning)
for param in model.model.encoder.layers[:NUM_LAYERS_TO_FREEZE].parameters():
    param.requires_grad = False
print(f"Froze first {NUM_LAYERS_TO_FREEZE} encoder layers.")

# ==============================
# 3. Precompute Features (Fixed 30s)
# ==============================
def prepare_dataset(batch):
    audio_paths = []
    for audio in batch["audio"]:
        if isinstance(audio, str):
            audio_paths.append(audio)
        else:
            audio_paths.append(audio["path"])
    
    texts = batch["text"]
    audio_arrays = []
    target_samples = 480000  # 30s @ 16kHz
    
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        
        if sr != 16000:
            raise ValueError(f"Unexpected sample rate {sr} in {path}")
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        
        # Force exactly 30 seconds
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        
        # Light noise augmentation
        if random.random() < 0.5:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        audio_arrays.append(audio)
    
    batch["input_features"] = processor.feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=False
    ).input_features
    
    batch["labels"] = processor.tokenizer(texts).input_ids
    return batch

raw_datasets = raw_datasets.map(
    prepare_dataset,
    batched=True,
    batch_size=64,
    remove_columns=["audio"],
    desc="Precomputing features (fixed 30s)",
    num_proc=32
)

eval_dataset = concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])
print(f"Eval samples: {len(eval_dataset):,}")

# ==============================
# 4. Data Collator
# ==============================
@dataclass
class DataCollator:
    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        input_features = torch.tensor(np.array(input_features), dtype=torch.float32)

        labels_batch = processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove leading BOS token if present
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}

data_collator = DataCollator()

# ==============================
# 5. Training Arguments
# ==============================
torch.multiprocessing.set_sharing_strategy('file_system')

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=5,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_private_repo=True,
    run_name=f"whisper-small-igbo-{HOURS}h-fullft",
    save_total_limit=3,
    remove_unused_columns=False,
    lr_scheduler_type=SCHEDULER_TYPE,
    dataloader_pin_memory=True,
    weight_decay=WEIGHT_DECAY,
)

# ==============================
# 6. Metrics
# ==============================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# ==============================
# 7. Trainer
# ==============================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

processor.save_pretrained(OUTPUT_DIR)
print("\n=== STARTING TRAINING ===\n")

# Remove resume_from_checkpoint=True on first run
trainer.train(resume_from_checkpoint=False)   # ← Change to True if resuming

trainer.save_model()
processor.save_pretrained(OUTPUT_DIR)
trainer.push_to_hub(f"Whisper-small full fine-tuned on {HOURS}h Igbo — Jan 2026")
print("✅ Training complete and model pushed to Hugging Face Hub!")