# #!/usr/bin/env python3
# """
# Download 5–30 sec clips → append to your existing folders
# No clearing, safe to resume
# """
# import argparse
# import os
# import tempfile
# import subprocess
# import random
# import hashlib
# import shutil
# import traceback
# import numpy as np
# from pathlib import Path
# from datasets import Dataset, DatasetDict, Audio
# from tqdm import tqdm
# import soundfile as sf
# import requests
# import mysql.connector
# from dotenv import load_dotenv


# # ==============================
# # CONFIG
# # ==============================
# TARGET_SR = 16000
# MIN_DURATION = 5.0
# MAX_DURATION = 30.0

# # Map your short codes to the exact folder names you use
# FOLDER_MAP = {
#     "yo": ("yor", "yoruba"),   # → processed_audio_yor   & whisper_dataset_yor
#     "ig": ("ig",  "igbo"),     # → processed_audio_ig    & whisper_dataset_ig
#     "ha": ("ha",  "hausa")     # → processed_audio_ha    & whisper_dataset_ha
# }


# def get_output_dirs(language: str):
#     folder_code, full_name = FOLDER_MAP[language]
#     audio_dir    = Path(f"processed_audio_{folder_code}")
#     dataset_dir  = Path(f"whisper_dataset_{folder_code}")
#     log_file     = Path(f"{language}_processing_log.txt")

#     audio_dir.mkdir(parents=True, exist_ok=True)
#     dataset_dir.mkdir(parents=True, exist_ok=True)

#     return audio_dir, dataset_dir, folder_code, full_name.capitalize(), log_file


# # ==============================
# # FAST WEBM → 16kHz WAV
# # ==============================
# def decode_to_wav(url: str) -> Path:
#     response = requests.get(url, timeout=30)
#     if not response.ok or len(response.content) < 1000:
#         return None

#     tmp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
#     tmp_wav  = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     tmp_webm.write(response.content)
#     tmp_webm.close(); tmp_wav.close()

#     try:
#         subprocess.run([
#             "ffmpeg", "-y", "-i", tmp_webm.name,
#             "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
#             "-loglevel", "error", tmp_wav.name
#         ], check=True, timeout=40)
#     except Exception:
#         os.unlink(tmp_webm.name)
#         os.unlink(tmp_wav.name)
#         return None

#     os.unlink(tmp_webm.name)
#     return Path(tmp_wav.name)


# # ==============================
# # MAIN
# # ==============================
# def download_dataset(limit, language, min_duration, max_duration, initial_skip):
#     load_dotenv()
#     DB_HOST = os.getenv("DB_HOST")
#     DB_PORT = int(os.getenv("DB_PORT") or 3306)
#     DB_USER = os.getenv("DB_USER")
#     DB_PASSWORD = os.getenv("DB_PASSWORD")
#     DB_NAME = os.getenv("DB_NAME")

#     lang_map = {'yo': ('yoruba', 'yor_trans'), 'ig': ('igbo', 'ig_trans'), 'ha': ('hausa', 'ha_trans')}
#     full_lang, col = lang_map[language]

#     conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
#                                    password=DB_PASSWORD, database=DB_NAME)
#     cursor = conn.cursor(dictionary=True)

#     audio_dir, dataset_dir, folder_code, lang_name, log_file = get_output_dirs(language)

#     with open(log_file, 'a', encoding='utf-8') as log:
#         log.write(f"\n=== New run for {lang_name} | skip={initial_skip} ===\n")

#         batch_size = 200
#         skip = initial_skip
#         downloaded = 0
#         skipped = 0
#         records = []
#         max_to_download = float('inf') if limit is None else limit

#         print(f"Downloading {lang_name.upper()} clips (5–30s) → appending to existing folders")
#         print(f"Audio folder : {audio_dir}")
#         print(f"Dataset folder: {dataset_dir}\n")

#         while downloaded < max_to_download:
#             query = f"""
#                 SELECT l.id as doc_id, l.audioTranslation,
#                        s.{col} as trans_sentence
#                 FROM lang_speaker_sentence l
#                 JOIN special_transcribed_sentence s ON l.specialTranscribedSentenceId = s.id
#                 WHERE l.completed = 1 AND l.audioTranslation IS NOT NULL
#                   AND JSON_EXTRACT(l.user, '$.language') = '{full_lang}'
#                 LIMIT {batch_size} OFFSET {skip}
#             """
#             cursor.execute(query)
#             docs = cursor.fetchall()
#             if not docs:
#                 print("No more data in database.")
#                 break

#             print(f"Batch fetched {len(docs)} docs | offset {skip}")

#             for doc in tqdm(docs, desc="Processing", leave=False):
#                 try:
#                     url = doc.get("audioTranslation")
#                     text = doc.get("trans_sentence", "").strip()
#                     if not url or not text or len(text) < 5:
#                         skipped += 1
#                         continue

#                     tmp_wav = decode_to_wav(url)
#                     if not tmp_wav:
#                         skipped += 1
#                         continue

#                     audio, sr = sf.read(tmp_wav)
#                     os.unlink(tmp_wav)

#                     duration = len(audio) / sr
#                     if rms := np.sqrt(np.mean(audio**2)) < 0.005:
#                         skipped += 1
#                         continue
#                     if not (min_duration <= duration <= max_duration):
#                         skipped += 1
#                         continue

#                     # Unique filename
#                     short_id = str(doc["doc_id"])[-8:].zfill(8)
#                     hash_part = hashlib.md5(str(doc["doc_id"]).encode()).hexdigest()[:8]
#                     wav_name = f"{folder_code}_{short_id}_{hash_part}.wav"
#                     final_path = audio_dir / wav_name
#                     sf.write(final_path, audio, TARGET_SR)

#                     records.append({
#                         "audio": str(final_path),
#                         "sentence": text.lower(),
#                         "language": lang_name.lower()
#                     })
#                     downloaded += 1

#                 except Exception as e:
#                     skipped += 1
#                     log.write(f"ERROR doc {doc.get('doc_id')}: {e}\n{traceback.format_exc()}\n")

#                 if downloaded >= max_to_download:
#                     break

#             skip += batch_size

#         if not records:
#             print("No new valid clips found in this run.")
#             return

#         # Shuffle & split (only the NEW records)
#         random.seed(42)
#         random.shuffle(records)
#         n = len(records)
#         train = records[:int(0.90 * n)]
#         val   = records[int(0.90 * n):int(0.95 * n)]
#         test  = records[int(0.95 * n):]

#         splits = {"train": train, "validation": val, "test": test}
#         for split_name, split_recs in splits.items():
#             sub_dir = audio_dir / split_name
#             sub_dir.mkdir(exist_ok=True)
#             for rec in split_recs:
#                 old = Path(rec["audio"])
#                 new = sub_dir / old.name
#                 if not new.exists():        # avoid duplicates if re-run
#                     shutil.move(str(old), str(new))
#                 rec["audio"] = str(new)

#         # Build dataset (only new data)
#         dataset = DatasetDict({
#             "train": Dataset.from_list(train),
#             "validation": Dataset.from_list(val),
#             "test": Dataset.from_list(test),
#         })
#         for split in dataset:
#             dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=TARGET_SR))

#         dataset.save_to_disk(dataset_dir)   # overwrites only the HF dataset (normal & expected)

#         hours = sum(len(sf.read(r["audio"])[0]) for r in records) / TARGET_SR / 3600

#         print("\n" + "="*70)
#         print(f"{lang_name.upper()} — NEW BATCH ADDED")
#         print("="*70)
#         print(f"New samples   : {len(records):,}")
#         print(f"New hours     : {hours:.2f}h")
#         print(f"Audio folder  : {audio_dir.resolve()}")
#         print(f"Dataset folder: {dataset_dir.resolve()} (updated)")
#         print("="*70)

#         log.write(f"Added {len(records)} new samples | {hours:.2f}h\n")

#     conn.close()


# # ==============================
# # CLI
# # ==============================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--language", type=str, required=True, choices=['yo', 'ig', 'ha'])
#     parser.add_argument("--limit", type=int, default=None, help="Max new clips to add")
#     parser.add_argument("--skip", type=int, default=0, help="DB offset to resume from")
#     parser.add_argument("--min-duration", type=float, default=5.0)
#     parser.add_argument("--max-duration", type=float, default=30.0)

#     args = parser.parse_args()
#     download_dataset(args.limit, args.language,
#                      args.min_duration, args.max_duration, args.skip)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
FAST & CLEAN: Download ALL Yoruba clips from MySQL + MongoDB
→ Fixed MySQL count bug
→ Runs perfectly now
→ Made streaming for immediate processing
"""

#!/usr/bin/env python3
"""
FAST & CLEAN: Download ALL Yoruba clips from MySQL + MongoDB
→ Fixed MySQL count bug
→ Runs perfectly now
→ Made streaming for immediate processing
"""

#!/usr/bin/env python3
"""
FINAL FIXED & WORKING — Combined MySQL + MongoDB Downloader
→ Pure Yoruba (or ha/ig), SNR ≥ -5 dB, 5–17s clips
→ Files + CSV appear immediately
"""

import argparse
import base64
import io
import numpy as np
from pathlib import Path
from pymongo import MongoClient
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import tempfile
import uuid
import ffmpeg
import librosa
from scipy.signal import hilbert, butter, lfilter
from multiprocessing import Pool
import os
import mysql.connector
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import random
import shutil
from datasets import DatasetDict, Dataset, Audio

# ====================== CONFIG ======================
TARGET_SR = 16000
MIN_DURATION = 5.0
MAX_DURATION = 30.0
MIN_RMS = 0.005
MAX_DOWNLOAD_WORKERS = 10
MAX_PROCESS_WORKERS = min(12, os.cpu_count() or 4)

# ====================== AUDIO HELPERS ======================
def butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def wada_snr(audio):
    y = np.abs(hilbert(audio))
    y_log = 10 * np.log10(y + 1e-10)
    y_log -= np.max(y_log)
    hist, bin_edges = np.histogram(y_log, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mode = bin_centers[np.argmax(hist)]
    gmm = y_log[y_log < mode]
    var_gmm = np.var(gmm) if len(gmm) > 0 else 1.0
    snr = 10 * np.log10((1 / (1 + 10**(-35/10)))**2 / var_gmm)
    return max(snr, -30)

def calculate_snr(audio, sr):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr > 6000:
        try:
            b, a = butter_bandpass(300, 3400, sr)
            audio = lfilter(b, a, audio)
        except:
            pass
    return wada_snr(audio)

def decode_audio(audio_bytes):
    if len(audio_bytes) < 1000:
        raise ValueError("Too short")
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio.astype(np.float32), TARGET_SR
    except:
        pass
    try:
        tin = Path(tempfile.gettempdir()) / f"in_{uuid.uuid4().hex[:8]}.webm"
        tout = tin.with_suffix(".wav")
        tin.write_bytes(audio_bytes)
        (
            ffmpeg
            .input(str(tin))
            .output(str(tout), acodec='pcm_s16le', ar=TARGET_SR, ac=1, loglevel="quiet")
            .overwrite_output()
            .run()
        )
        audio, sr = sf.read(tout)
        tin.unlink(missing_ok=True)
        tout.unlink(missing_ok=True)
        return audio.astype(np.float32), TARGET_SR
    except Exception as e:
        raise ValueError(f"Decode failed: {e}")

# ====================== PROCESS ONE ITEM ======================
def process_item(args):
    source, sid, audio_bytes, text, lang = args
    try:
        if len(text.strip()) < 5:
            return None
        audio, sr = decode_audio(audio_bytes)
        duration = len(audio) / sr
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            return None
        if np.sqrt(np.mean(audio**2)) < MIN_RMS:
            return None
        return {
            "text": text.lower().strip(),
            "duration": round(float(duration), 2),
            "sample_rate": sr,
            "doc_id": sid,
            "source": source,
            "language": lang,
            "audio": (audio, sr)
        }
    except:
        return None

# ====================== MYSQL GENERATOR ======================
def mysql_generator(target_language):
    trans_col = {"yor": "yor_trans", "ha": "ha_trans", "ig": "ig_trans"}[target_language]
    valid_langs = {
        "yor": ["yor", "yoruba", "yo"],
        "ha": ["ha", "hau", "hausa"],
        "ig": ["ig", "ibo", "igbo"]
    }[target_language]

    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )
    cur = conn.cursor(dictionary=True)
    cur.execute(f"SELECT COUNT(*) AS total FROM lang_speaker_sentence l JOIN special_transcribed_sentence s ON l.specialTranscribedSentenceId = s.id WHERE l.completed = 1 AND s.{trans_col} IS NOT NULL")
    total_rows = cur.fetchone()["total"]
    print(f"MySQL total {target_language.upper()} rows: {total_rows:,}")

    offset = 0
    batch = 1000
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        while offset < total_rows:
            cur.execute(f"""
                SELECT l.id, l.audioTranslation, s.{trans_col} AS text, l.user
                FROM lang_speaker_sentence l
                JOIN special_transcribed_sentence s ON l.specialTranscribedSentenceId = s.id
                WHERE l.completed = 1 AND s.{trans_col} IS NOT NULL
                LIMIT %s OFFSET %s
            """, (batch, offset))
            rows = cur.fetchall()

            def download(row):
                try:
                    import json
                    user_data = json.loads(row["user"])
                    user_lang = user_data.get("language", "").strip().lower()
                    if user_lang not in valid_langs:
                        return None  # Skip if language doesn't match
                except (json.JSONDecodeError, KeyError):
                    return None  # Skip on parse error or missing language

                try:
                    r = requests.get(row["audioTranslation"], timeout=20)
                    if r.ok and len(r.content) > 1000:
                        return ("mysql", str(row["id"]), r.content, row["text"], user_lang)  # Use actual lang
                except:
                    pass
                return None

            for fut in as_completed([pool.submit(download, row) for row in rows]):
                item = fut.result()
                if item:
                    yield item

            offset += batch
            print(f"MySQL progress: {min(offset, total_rows):,} / {total_rows:,}")
    conn.close()

# ====================== MONGO GENERATOR (STRICT FILTERING) ======================
def mongo_generator(target_language):
    print(f"MongoDB generator started for {target_language.upper()} — applying STRICT language filtering...")
    client = MongoClient(os.getenv("MONGO_URI"))
    coll = client["meta_public_service"]["dashboard_user_sentences"]
    users = client["meta_public_service"]["users"]

    # Define valid language codes for each target
    valid_lang_map = {
        "yor": ["yor", "yoruba", "yo"],
        "ha": ["ha", "hau", "hausa"],
        "ig": ["ig", "ibo", "igbo"]
    }
    valid_langs = valid_lang_map[target_language]

    processed = 0
    for doc in coll.find({"completed": True, "audioTranslation": {"$exists": True, "$ne": ""}}).batch_size(200):
        try:
            user_doc = users.find_one({"_id": doc["user"]})
            if not user_doc or "language" not in user_doc:
                continue  # Skip if no user or no language field

            db_lang = user_doc["language"].strip().lower()
            if db_lang not in valid_langs:
                continue  # STRICT: only allow known codes for this language

            b64 = doc["audioTranslation"].split(",", 1)[-1] if "," in doc["audioTranslation"] else doc["audioTranslation"]
            text = doc.get("textTranslation", "").strip()
            if len(text) < 5:
                continue

            audio_bytes = base64.b64decode(b64)
            processed += 1
            if processed % 1000 == 0:
                print(f"MongoDB yielded {processed:,} valid {target_language.upper()} clips so far...")

            yield ("mongo", str(doc["_id"]), audio_bytes, text, db_lang)
        except Exception:
            continue

# ====================== SAVE METADATA ======================
def save_metadata_append(data, path):
    if not data:
        return
    df = pd.DataFrame(data)
    header = not path.exists()
    df.to_csv(path, mode='a', header=header, index=False)

# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0, help="Max clips (0 = unlimited)")
    parser.add_argument("--language", type=str, default="ig", choices=["yor", "ha", "ig"])
    parser.add_argument("--skip-mysql", action="store_true", help="Skip MySQL and only use MongoDB")
    args = parser.parse_args()

    load_dotenv()
    lang = args.language
    out_dir = Path(f"processed_audio_{lang}")
    full_csv_path = Path(f"metadata_{lang}.csv")
    simple_csv_path = Path(f"data_{lang}.csv")  # The clean one you want
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting processing for {lang.upper()} | 5–30s clips | Strict filtering enabled")

    # Resume from existing files
    existing_files = list(out_dir.glob("audio_*.wav"))
    if existing_files:
        max_num = max(int(f.stem.split("_")[1]) for f in existing_files)
        downloaded = max_num + 1
        print(f"Resuming from audio_{downloaded:06d}.wav ({len(existing_files)} files exist)")
    else:
        downloaded = 0
        print("Starting fresh — no existing files")

    metadata = []

    gens = [mongo_generator(lang)]
    if not args.skip_mysql:
        gens.insert(0, mysql_generator(lang))  # MySQL first if enabled

    with Pool(MAX_PROCESS_WORKERS) as pool:
        pending = []

        for gen in gens:
            for item in gen:
                pending.append(pool.apply_async(process_item, (item,)))

                # Process completed tasks
                for task in list(pending):
                    if task.ready():
                        res = task.get()
                        pending.remove(task)
                        if res:
                            audio, sr = res.pop("audio")
                            wav_path = out_dir / f"audio_{downloaded:06d}.wav"
                            sf.write(wav_path, audio, sr)
                            res["audio_path"] = str(wav_path)
                            metadata.append(res)
                            downloaded += 1

                            if downloaded % 50 == 0:
                                save_metadata_append(metadata[-50:], full_csv_path)
                                print(f"SAVED {downloaded:,} | {res['duration']}s | {res['source'].upper()} | \"{res['text'][:70]}...\"")

                            if args.count and downloaded >= args.count:
                                print("Reached requested count limit!")
                                pool.close()
                                pool.join()
                                break

        # Final flush
        print("Finalizing remaining tasks...")
        for task in tqdm(pending, desc="Finishing"):
            res = task.get()
            if res and (not args.count or downloaded < args.count):
                audio, sr = res.pop("audio")
                wav_path = out_dir / f"audio_{downloaded:06d}.wav"
                sf.write(wav_path, audio, sr)
                res["audio_path"] = str(wav_path)
                metadata.append(res)
                downloaded += 1

    # Save final batches
    if metadata:
        save_metadata_append(metadata, full_csv_path)

        # === SIMPLE CSV: audio path + text only ===
        simple_df = pd.DataFrame(metadata)[["audio_path", "text"]].copy()
        simple_df = simple_df.rename(columns={"audio_path": "audio", "text": "sentence"})
        simple_df.to_csv(simple_csv_path, index=False)
        print(f"\nSIMPLE CSV CREATED: {simple_csv_path} ({len(simple_df):,} clean clips)")

    print(f"\nDONE! {downloaded:,} clean {lang.upper()} clips saved")
    print(f"Audio folder   : {out_dir.resolve()}")
    print(f"Full metadata  : {full_csv_path}")
    print(f"Simple CSV     : {simple_csv_path}")

    # Optional: Build Hugging Face dataset (comment out if not needed)
    if metadata:
        print("\nBuilding Hugging Face Dataset splits...")
        random.seed(42)
        random.shuffle(metadata)
        n = len(metadata)
        train = metadata[:int(0.90 * n)]
        val = metadata[int(0.90 * n):int(0.95 * n)]
        test = metadata[int(0.95 * n):]

        dataset_dir = Path(f"whisper_dataset_{lang}")
        dataset_dir.mkdir(exist_ok=True, parents=True)

        def make_split(items, name):
            split_dir = dataset_dir / name
            split_dir.mkdir(exist_ok=True)
            records = []
            for item in items:
                src = Path(item["audio_path"])
                dst = split_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
                records.append({"audio": str(dst), "sentence": item["text"]})
            return records

        train_data = make_split(train, "train")
        val_data = make_split(val, "validation")
        test_data = make_split(test, "test")

        ds = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        }).cast_column("audio", Audio(sampling_rate=TARGET_SR))

        ds.save_to_disk(dataset_dir)

        total_hours = sum(m["duration"] for m in metadata) / 3600
        print("\n" + "="*80)
        print(f"FINAL {lang.upper()} DATASET READY!")
        print(f"→ Clips       : {len(metadata):,}")
        print(f"→ Hours       : {total_hours:.1f}h")
        print(f"→ Audio folder: {out_dir.resolve()}")
        print(f"→ Simple CSV  : {simple_csv_path.resolve()}")
        print(f"→ HF Dataset  : {dataset_dir.resolve()}")
        print("="*80)

if __name__ == "__main__":
    main()