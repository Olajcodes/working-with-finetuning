import os
import librosa
import argparse

def calculate_audio_stats(folder_path):
    # Supported audio extensions
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    
    # Initialize counters
    total_files = 0
    total_seconds = 0.0
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                try:
                    # Get duration in seconds
                    duration = librosa.get_duration(path=file_path)
                    total_seconds += duration
                    total_files += 1
                    print(f"Processed: {file_path} ({duration:.2f} seconds)")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Convert total seconds to hours
    total_hours = total_seconds / 3600
    
    print("\nSummary:")
    print(f"Total audio files: {total_files}")
    print(f"Total duration: {total_seconds:.2f} seconds ({total_hours:.2f} hours)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count audio files and calculate total duration in a folder")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    args = parser.parse_args()
    
    calculate_audio_stats(args.folder_path)