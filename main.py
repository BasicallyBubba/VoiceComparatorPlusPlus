###########################################################################################################
#                                                                                                         #
#       This is a project, not a forensics tool. Take everything with a grain of salt.                    #
#                                                                                                         #
###########################################################################################################
# Credit to MehmetYukselSekeroglu for the original code
# This is just a localization + refactor of his code for the most part.

# Imports
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
from pydub import AudioSegment
import argparse
import pydub
import random
import os
import sys
import time
from colorama import *
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk
import torch

###########################################################################################################
# Constants and Style Definitions

bold = "\033[1m"
bold_reset = "\033[0m"
green = Fore.GREEN
blue = Fore.BLUE
color_reset = Fore.RESET
red = Fore.RED
orange = "\033[38;5;208m"

POWERED_BY = "Powered by Redbull and Insomnia"
APP_NAME = "BobaVox"
TEMP_PATH = "temp" + os.sep
VERSION_INFO = "v0.0.3"

###########################################################################################################
# Utility Functions

def GetTime():
    current_time = time.localtime()
    return f"{current_time.tm_hour:02d}:{current_time.tm_min:02d}:{current_time.tm_sec:02d}"

def TitlePrinter(messages: str):
    print(f"{bold}{blue}>> [{messages}]{bold_reset}{color_reset}", end="\n\n")

def InformationPrinter(messages: str):
    print(f"{bold}{blue}[{GetTime()}]{bold}[INFO]: {green}{messages} {color_reset}{bold_reset}")

def ErrorPrinter(messages: str):
    print(f"{bold}{red}[{GetTime()}]{bold}[ERROR]: {green}{messages}{color_reset}{bold_reset}")

###########################################################################################################
# File and Directory Management

# Ensure temp directories exist
if not os.path.exists(TEMP_PATH):
    os.mkdir(TEMP_PATH)

FILE_1_TEMP_DIR = os.path.join(TEMP_PATH, "file1")
FILE_2_TEMP_DIR = os.path.join(TEMP_PATH, "file2")

def clean_temp_dirs():
    if os.path.exists(TEMP_PATH):
        for file in os.listdir(TEMP_PATH):
            file_path = os.path.join(TEMP_PATH, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                for sub_file in os.listdir(file_path):
                    sub_file_path = os.path.join(file_path, sub_file)
                    if os.path.isfile(sub_file_path):
                        os.unlink(sub_file_path)
    else:
        os.makedirs(TEMP_PATH)
    for temp_dir in [FILE_1_TEMP_DIR, FILE_2_TEMP_DIR]:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

clean_temp_dirs()

###########################################################################################################
# Audio Processing Functions

def ConvertAnyAudio_to_wav(target_file_path: str, temp_file_name: str):
    TARGET_FILE_FORMAT = "wav"
    InformationPrinter(f"Converting {target_file_path} to {TARGET_FILE_FORMAT} format...")
    
    if not os.path.exists(target_file_path):
        ErrorPrinter(f"Invalid path: {target_file_path}")
        return {"success": "false", "code": "invalid path"}
    
    target_file_extensions = target_file_path.split(".")[-1]
    supported_formats = ["MP3", "OGG", "FLAC", "AAC", "AIFF", "WMA", "WAV"]
    
    if target_file_extensions.upper() not in supported_formats:
        ErrorPrinter(f"Unsupported file extension: {target_file_extensions}")
        return {"success": "false", "code": "unsupported file extension"}

    LoadedAudio = AudioSegment.from_file(target_file_path, format=target_file_extensions)
    export_name = os.path.join(TEMP_PATH, temp_file_name + "." + TARGET_FILE_FORMAT)
    
    LoadedAudio.export(export_name, format=TARGET_FILE_FORMAT)
    InformationPrinter(f"Exported file to {export_name}")

    if os.path.exists(export_name):
        return {"success": "true", "path": str(export_name)}
    else:
        ErrorPrinter(f"Export error for file: {export_name}")
        return {"success": "false", "code": "export error"}

def split_audio_into_chunks(audio_path, chunk_length, temp_dir_path):
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = chunk_length * 1000
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir_path, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths

###########################################################################################################
# Comparison Functions

def CompareSounds(sound_1_path: str, sound_2_path: str, linear: bool = False, thorough: bool = False, window_size: int = 5, num_chunks: int = 10):
    InformationPrinter(f"Comparing {sound_1_path} and {sound_2_path}...")
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
        ErrorPrinter(f"File not found: {sound_1_path} or {sound_2_path}")
        return {"success": "false", "code": "file not found"}
    
    device = torch.device("cpu")
    sound_encoder = VoiceEncoder(verbose=False)
    file_1 = torch.tensor(preprocess_wav(sound_1_path)).to(device)
    file_2 = torch.tensor(preprocess_wav(sound_2_path)).to(device)

    if not linear and not thorough:
        encoded_sound1 = torch.tensor(sound_encoder.embed_utterance(file_1.cpu().numpy())).to(device)
        encoded_sound2 = torch.tensor(sound_encoder.embed_utterance(file_2.cpu().numpy())).to(device)

        dot_product_size = torch.dot(encoded_sound1, encoded_sound2).item()
        norm_sound1 = torch.norm(encoded_sound1).item()
        norm_sound2 = torch.norm(encoded_sound2).item()

        GetSimilarity = (dot_product_size / (norm_sound1 * norm_sound2)) * 100
        InformationPrinter(f"Similarity: {int(GetSimilarity)}%")
        return {"success": "true", "similarity": str(int(GetSimilarity))}

    min_length = min(len(file_1), len(file_2))
    window_sizes = list(range(1, min_length // 16000 + 1)) if linear else [2**i for i in range(int(np.log2(min_length // 16000)) + 1)]
    
    similarities = []
    control_similarities_1 = []
    control_similarities_2 = []

    def compare_chunk(start_1, start_2):
        chunk_1 = torch.tensor(sound_encoder.embed_utterance(file_1[start_1:start_1 + window_size * 16000].cpu().numpy())).to(device)
        chunk_2 = torch.tensor(sound_encoder.embed_utterance(file_2[start_2:start_2 + window_size * 16000].cpu().numpy())).to(device)

        dot_product_size = torch.dot(chunk_1, chunk_2).item()
        norm_chunk1 = torch.norm(chunk_1).item()
        norm_chunk2 = torch.norm(chunk_2).item()

        return (dot_product_size / (norm_chunk1 * norm_chunk2)) * 100

    for window_size in window_sizes:
        chunk_size = window_size * 16000
        chunks_1 = [file_1[i:i + chunk_size].to(device) for i in range(0, len(file_1), chunk_size)]
        chunks_2 = [file_2[i:i + chunk_size].to(device) for i in range(0, len(file_2), chunk_size)]

        if thorough:
            for chunk_1 in chunks_1:
                for chunk_2 in chunks_2:
                    encoded_chunk1 = torch.tensor(sound_encoder.embed_utterance(chunk_1.cpu().numpy())).to(device)
                    encoded_chunk2 = torch.tensor(sound_encoder.embed_utterance(chunk_2.cpu().numpy())).to(device)
                    dot_product_size = torch.dot(encoded_chunk1, encoded_chunk2).item()
                    norm_chunk1 = torch.norm(encoded_chunk1).item()
                    norm_chunk2 = torch.norm(encoded_chunk2).item()
                    similarities.append((dot_product_size / (norm_chunk1 * norm_chunk2)) * 100)
        else:
            with ThreadPoolExecutor(max_workers=num_threads_var.get()) as executor:
                futures = [executor.submit(compare_chunk, random.randint(0, len(file_1) - chunk_size), random.randint(0, len(file_2) - chunk_size)) for _ in range(num_chunks)]
                similarities.extend([future.result() for future in futures])

        if control_var.get():
            control_similarities_1.append(CompareSounds(sound_1_path, sound_1_path, linear, thorough, window_size, num_chunks))
            control_similarities_2.append(CompareSounds(sound_2_path, sound_2_path, linear, thorough, window_size, num_chunks))

    return {"success": "true", "window_sizes": window_sizes, "similarities": similarities, "control_similarities_1": control_similarities_1, "control_similarities_2": control_similarities_2}

def plot_results(window_sizes, avg_similarities, min_similarities, max_similarities):
    avg_similarities = np.array(avg_similarities)
    min_similarities = np.array(min_similarities)
    max_similarities = np.array(max_similarities)
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(window_sizes, avg_similarities, yerr=[avg_similarities - min_similarities, max_similarities - avg_similarities], fmt='-o', capsize=5)
    plt.xlabel('Chunk Length (seconds)')
    plt.ylabel('Similarity (%)')
    plt.title('Similarity vs. Chunk Length')
    plt.grid(True)
    plt.show()

###########################################################################################################
# GUI Functions

def select_file_1():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_1_path.set(file_path)

def select_file_2():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_2_path.set(file_path)

def clear_temp_subdirs():
    for temp_dir in [FILE_1_TEMP_DIR, FILE_2_TEMP_DIR]:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

def run_comparison():
    file1 = file_1_path.get()
    file2 = file_2_path.get()
    deep_compare = deep_compare_var.get()
    linear_compare = linear_compare_var.get()
    num_threads = int(num_threads_var.get())
    
    if not file1 or not file2:
        messagebox.showerror("Error", "Please select both files.")
        return
    
    result1 = ConvertAnyAudio_to_wav(file1, "file1")
    result2 = ConvertAnyAudio_to_wav(file2, "file2")
    
    if result1["success"] == "false" or result2["success"] == "false":
        messagebox.showerror("Error", "Error converting files to WAV format.")
        return
    
    wav_file1 = result1["path"]
    wav_file2 = result2["path"]
    
    if not deep_compare:
        comparison_result = CompareSounds(wav_file1, wav_file2)
        if comparison_result["success"] == "true":
            similarity = comparison_result["similarity"]
            messagebox.showinfo("Result", f"Similarity: {similarity}%")
        else:
            messagebox.showerror("Error", "Error comparing files.")
        return
    
    file1_length = len(AudioSegment.from_wav(wav_file1)) / 1000  # in seconds
    file2_length = len(AudioSegment.from_wav(wav_file2)) / 1000  # in seconds
    min_length = min(file1_length, file2_length)
    
    window_sizes = list(range(1, int(min_length) + 1)) if deep_compare and linear_compare else [2**i for i in range(int(np.log2(min_length)) + 1)] if deep_compare else [5]
    
    avg_similarities = []
    min_similarities = []
    max_similarities = []
    
    for chunk_length in window_sizes:
        clear_temp_subdirs()
        
        chunk_paths1 = split_audio_into_chunks(wav_file1, chunk_length, FILE_1_TEMP_DIR)
        chunk_paths2 = split_audio_into_chunks(wav_file2, chunk_length, FILE_2_TEMP_DIR)
        
        similarities = []
        
        def compare_chunks(chunk1, chunk2):
            comparison_result = CompareSounds(chunk1, chunk2, window_size=chunk_length)
            if comparison_result["success"] == "true":
                similarities.append(int(comparison_result["similarity"]))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(compare_chunks, chunk1, chunk2) for chunk1 in chunk_paths1 for chunk2 in chunk_paths2]
            for future in futures:
                future.result()
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            min_similarity = min(similarities)
            max_similarity = max(similarities)
            avg_similarities.append(avg_similarity)
            min_similarities.append(min_similarity)
            max_similarities.append(max_similarity)
        else:
            avg_similarities.append(0)
            min_similarities.append(0)
            max_similarities.append(0)
    
    plot_results(window_sizes, avg_similarities, min_similarities, max_similarities)
    
    if avg_similarities:
        overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
        overall_min_similarity = min(min_similarities)
        overall_max_similarity = max(max_similarities)
        messagebox.showinfo("Result", f"Average Similarity: {overall_avg_similarity:.2f}%\nMinimum Similarity: {overall_min_similarity}%\nMaximum Similarity: {overall_max_similarity}%")
    else:
        messagebox.showerror("Error", "Error comparing files.")

def toggle_linear_compare(*args):
    if deep_compare_var.get():
        linear_compare_checkbutton.state(["!disabled"])
    else:
        linear_compare_checkbutton.state(["disabled"])

###########################################################################################################
# GUI Setup

root = tk.Tk()
root.title("BobaVox Audio Comparison")

file_1_path = tk.StringVar()
file_2_path = tk.StringVar()
deep_compare_var = tk.BooleanVar()
linear_compare_var = tk.BooleanVar()
num_threads_var = tk.StringVar(value="8")

deep_compare_var.trace_add("write", toggle_linear_compare)

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Select first audio file:").grid(row=0, column=0, sticky=tk.W)
ttk.Entry(frame, textvariable=file_1_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
ttk.Button(frame, text="Browse", command=select_file_1).grid(row=0, column=2, sticky=tk.E)

ttk.Label(frame, text="Select second audio file:").grid(row=1, column=0, sticky=tk.W)
ttk.Entry(frame, textvariable=file_2_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E))
ttk.Button(frame, text="Browse", command=select_file_2).grid(row=1, column=2, sticky=tk.E)

deep_compare_checkbutton = ttk.Checkbutton(frame, text="Deep Compare", variable=deep_compare_var)
deep_compare_checkbutton.grid(row=2, column=0, sticky=tk.W)

linear_compare_checkbutton = ttk.Checkbutton(frame, text="Linear", variable=linear_compare_var)
linear_compare_checkbutton.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
linear_compare_checkbutton.state(["disabled"])

ttk.Label(frame, text="Number of threads:").grid(row=3, column=0, sticky=tk.W)
num_threads_entry = ttk.Entry(frame, textvariable=num_threads_var, width=10)
num_threads_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

ttk.Button(frame, text="Compare", command=run_comparison).grid(row=4, column=0, columnspan=3)

device = torch.device("cpu")
ttk.Label(frame, text=f"Running on: CPU").grid(row=5, column=0, columnspan=3, sticky=tk.W)

root.mainloop()
