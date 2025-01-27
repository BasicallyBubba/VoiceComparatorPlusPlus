###########################################################################################################
#                                                                                                         #
#       This is a project, not a forensics tool. Take everything with a grain of salt.                    #
#                                                                                                         #
###########################################################################################################
# Credit to MehmetYukselSekeroglu for the original code
# This is just a localization + refactor of his code for the most part.


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


# Style colors and whatnot
bold ="\033[1m"
bold_reset ="\033[0m"
green = Fore.GREEN
blue = Fore.BLUE
color_reset = Fore.RESET
red = Fore.RED
orange = "\033[38;5;208m"



POWERED_BY = "PRIME"
APP_NAME = "Voice Comparator"
TEMP_PATH = "temp"+os.sep
VERSION_INFO = "v0.0.1"


if not os.path.exists(TEMP_PATH):
    os.mkdir(TEMP_PATH)



###########################################################################################################

# Output functions


def GetTime():
    """
    Returns the current system time as hour:minute:second

    Returns:
        str: hour:minute:second
    """
    current_time = time.localtime()
    hour = current_time.tm_hour
    minute = current_time.tm_min
    second = current_time.tm_sec
    formatted_time = f"{hour:02d}:{minute:02d}:{second:02d}"
    
    return formatted_time

# Fancy heading stylization
def TitlePrinter(messages:str):
    print(f"{bold}{blue}>> [{messages}]{bold_reset}{color_reset}",end="\n\n")

    

# Fancy information stylization
def InformationPrinter(messages:str):
    print(f"{bold}{blue}[{GetTime()}]{bold}[INFO]: {green}{messages} {color_reset}{bold_reset}")


# Fancy error stylization
def ErrorPrinter(messages:str):
    print(f"{bold}{red}[{GetTime()}]{bold}[ERROR]: {green}{messages}{color_reset}{bold_reset}")

###########################################################################################################

# The Actually Main Program Stuff (tm)


def ConvertAnyAudio_to_wav(target_file_path:str, temp_dir_path:str=TEMP_PATH):
    TARGET_FILE_FORMAT = "wav"
    InformationPrinter(f"Converting {target_file_path} to {TARGET_FILE_FORMAT} format...")
    
    if not os.path.exists(target_file_path) or not os.path.exists(temp_dir_path):
        ErrorPrinter(f"Invalid path: {target_file_path} or {temp_dir_path}")
        return {"success":"false", "code":"invalid path"}
    
    target_file_extensions = target_file_path.split(".")
    target_file_extensions = target_file_extensions[len(target_file_extensions)-1]

    supported_formats = ["MP3","OGG","FLAC","AAC","AIFF","WMA","WAV"]
    
    if target_file_extensions.upper() not in supported_formats:
        ErrorPrinter(f"Unsupported file extension: {target_file_extensions}")
        return {"success":"false", "code":"unsupported file extension"}

    LoadedAudio = AudioSegment.from_file(target_file_path, format=target_file_extensions)
    export_name = temp_dir_path+"exported_file_"+str(random.randint(1,999))+"."+TARGET_FILE_FORMAT
    
    LoadedAudio.export(export_name, format=TARGET_FILE_FORMAT)
    InformationPrinter(f"Exported file to {export_name}")

    if os.path.exists(export_name):
        return {"success":"true", "path":str(export_name)}
    else:
        ErrorPrinter(f"Export error for file: {export_name}")
        return { "success":"false", "code":"export error"}



def CompareSounds(sound_1_path:str, sound_2_path:str):
    InformationPrinter(f"Comparing {sound_1_path} and {sound_2_path}...")
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
        ErrorPrinter(f"File not found: {sound_1_path} or {sound_2_path}")
        return { "success":"false", "code":"file not found" }
    
    sound_encoder = VoiceEncoder(verbose=False)
    file_1 = preprocess_wav(sound_1_path)
    file_2 = preprocess_wav(sound_2_path)

    encoded_sound1 = sound_encoder.embed_utterance(file_1)
    encoded_sound2 = sound_encoder.embed_utterance(file_2)

    dot_product_size = np.dot(encoded_sound1, encoded_sound2)
    norm_sound1 = np.linalg.norm(encoded_sound1)
    norm_sound2 = np.linalg.norm(encoded_sound2)

    # calculate cosine similarity
    GetSimilarity = dot_product_size / (norm_sound1 * norm_sound2)
    GetSimilarity = GetSimilarity * 100
    GetSimilarity = int(GetSimilarity)
    InformationPrinter(f"Similarity: {GetSimilarity}%")
    return { "success":"true" ,"similarity":str(GetSimilarity) }


def CompareSoundsInChunks(sound_1_path:str, sound_2_path:str, num_splits:int):
    InformationPrinter(f"Comparing {sound_1_path} and {sound_2_path} with {num_splits} splits...")
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
        ErrorPrinter(f"File not found: {sound_1_path} or {sound_2_path}")
        return { "success":"false", "code":"file not found" }
    
    sound_encoder = VoiceEncoder(verbose=False)
    file_1 = preprocess_wav(sound_1_path)
    file_2 = preprocess_wav(sound_2_path)

    if num_splits == 1:
        return CompareSounds(sound_1_path, sound_2_path)

    min_length = min(len(file_1), len(file_2))
    chunk_size = min_length // num_splits
    similarities = []

    for i in range(num_splits):
        start_1 = random.randint(0, len(file_1) - chunk_size)
        start_2 = random.randint(0, len(file_2) - chunk_size)
        chunk_1 = file_1[start_1:start_1 + chunk_size]
        chunk_2 = file_2[start_2:start_2 + chunk_size]

        encoded_chunk1 = sound_encoder.embed_utterance(chunk_1)
        encoded_chunk2 = sound_encoder.embed_utterance(chunk_2)

        dot_product_size = np.dot(encoded_chunk1, encoded_chunk2)
        norm_chunk1 = np.linalg.norm(encoded_chunk1)
        norm_chunk2 = np.linalg.norm(encoded_chunk2)

        # calculate cosine similarity
        similarity = dot_product_size / (norm_chunk1 * norm_chunk2)
        similarities.append(similarity * 100)

    average_similarity = int(np.mean(similarities))
    InformationPrinter(f"Average similarity: {average_similarity}%")
    return { "success":"true" ,"similarity":str(average_similarity) }

def CompareRandomChunks(sound_1_path:str, sound_2_path:str, window_size:int, num_chunks:int, random_order:bool):
    order_type = "random" if random_order else "sequential"
    InformationPrinter(f"Comparing {sound_1_path} and {sound_2_path} with {num_chunks} {order_type} {window_size}-second chunks...")
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
        ErrorPrinter(f"File not found: {sound_1_path} or {sound_2_path}")
        return { "success":"false", "code":"file not found" }
    
    sound_encoder = VoiceEncoder(verbose=False)
    file_1 = preprocess_wav(sound_1_path)
    file_2 = preprocess_wav(sound_2_path)

    chunk_size = window_size * 16000  # window_size in seconds to samples (16kHz)
    similarities = []

    for _ in range(num_chunks):
        if random_order:
            start_1 = random.randint(0, len(file_1) - chunk_size)
            start_2 = random.randint(0, len(file_2) - chunk_size)
        else:
            start_1 = _ * chunk_size
            start_2 = _ * chunk_size
        chunk_1 = file_1[start_1:start_1 + chunk_size]
        chunk_2 = file_2[start_2:start_2 + chunk_size]

        encoded_chunk1 = sound_encoder.embed_utterance(chunk_1)
        encoded_chunk2 = sound_encoder.embed_utterance(chunk_2)

        dot_product_size = np.dot(encoded_chunk1, encoded_chunk2)
        norm_chunk1 = np.linalg.norm(encoded_chunk1)
        norm_chunk2 = np.linalg.norm(encoded_chunk2)

        # calculate cosine similarity
        similarity = dot_product_size / (norm_chunk1 * norm_chunk2)
        similarities.append(similarity * 100)

    average_similarity = np.mean(similarities)
    InformationPrinter(f"Window size: {window_size} seconds, Average similarity: {average_similarity}%")
    return average_similarity

def CompareSoundsWithRollingWindow(sound_1_path:str, sound_2_path:str, linear:bool, random_order:bool):
    InformationPrinter(f"Comparing {sound_1_path} and {sound_2_path} with rolling windows ({'linear' if linear else 'exponential'}), {'random' if random_order else 'sequential'} order...")
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
        ErrorPrinter(f"File not found: {sound_1_path} or {sound_2_path}")
        return { "success":"false", "code":"file not found" }
    
    sound_encoder = VoiceEncoder(verbose=False)
    file_1 = preprocess_wav(sound_1_path)
    file_2 = preprocess_wav(sound_2_path)

    min_length = min(len(file_1), len(file_2))
    if linear:
        window_sizes = list(range(1, min_length // 16000 + 1))
    else:
        window_sizes = [2**i for i in range(int(np.log2(min_length // 16000)) + 1)]
    
    similarities = []
    control_similarities_1 = []
    control_similarities_2 = []

    for window_size in window_sizes:
        num_chunks = min(len(file_1), len(file_2)) // (window_size * 16000)
        average_similarity = CompareRandomChunks(sound_1_path, sound_2_path, window_size, num_chunks, random_order)
        similarities.append(average_similarity)
        if control_var.get():
            control_similarity_1 = CompareRandomChunks(sound_1_path, sound_1_path, window_size, num_chunks, random_order)
            control_similarity_2 = CompareRandomChunks(sound_2_path, sound_2_path, window_size, num_chunks, random_order)
            control_similarities_1.append(control_similarity_1)
            control_similarities_2.append(control_similarity_2)

    return { "success":"true", "window_sizes": window_sizes, "similarities": similarities, "control_similarities_1": control_similarities_1, "control_similarities_2": control_similarities_2 }

def plot_similarities(window_sizes, similarities, control_similarities_1, control_similarities_2):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(window_sizes, similarities, marker='o', label='Comparison')
    if control_var.get():
        plt.plot(window_sizes, control_similarities_1, marker='o', linestyle='--', label='Control 1')
        plt.plot(window_sizes, control_similarities_2, marker='o', linestyle='--', label='Control 2')
    plt.xscale('log')
    plt.xlabel('Window Size (seconds)')
    plt.ylabel('Similarity (%)')
    plt.title('Similarity vs. Window Size')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.axis('off')
    stats_text = (
        f"Number of data points: {len(similarities)}\n"
        f"Peak similarity rate: {max(similarities):.2f}%\n"
        f"Average similarity rate: {np.mean(similarities):.2f}%\n"
        f"Minimum similarity rate: {min(similarities):.2f}%"
    )
    if control_var.get():
        stats_text += (
            f"\nControl 1 - Peak similarity rate: {max(control_similarities_1):.2f}%\n"
            f"Control 1 - Average similarity rate: {np.mean(control_similarities_1):.2f}%\n"
            f"Control 1 - Minimum similarity rate: {min(control_similarities_1):.2f}%"
            f"\nControl 2 - Peak similarity rate: {max(control_similarities_2):.2f}%\n"
            f"Control 2 - Average similarity rate: {np.mean(control_similarities_2):.2f}%\n"
            f"Control 2 - Minimum similarity rate: {min(control_similarities_2):.2f}%"
        )
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.show()

def compare_files(file1, file2):
    InformationPrinter("Converting files to 'wav' format...")
    raw_file_1_convert_status = ConvertAnyAudio_to_wav(target_file_path=file1)
    raw_file_2_convert_status = ConvertAnyAudio_to_wav(target_file_path=file2)

    if raw_file_1_convert_status["success"] == "false" or raw_file_2_convert_status["success"] == "false":
        ErrorPrinter("File conversion failed.")
        display_results("N/A", "File conversion failed.", "red")
        return

    wav_file_1 = raw_file_1_convert_status["path"]
    wav_file_2 = raw_file_2_convert_status["path"]

    InformationPrinter("Comparing voice similarity...")

    if rolling_window_var.get():
        final_status = CompareSoundsWithRollingWindow(wav_file_1, wav_file_2, linear_var.get(), random_order_var.get())
    else:
        final_status = CompareSounds(wav_file_1, wav_file_2)

    if not final_status["success"] == "true":
        ErrorPrinter("Audio comparison failed.")
        os.remove(wav_file_1)
        os.remove(wav_file_2)
        display_results("N/A", "Audio comparison failed.", "red")
        return

    if rolling_window_var.get():
        InformationPrinter("Comparison finished. Displaying results...")
        plot_similarities(final_status["window_sizes"], final_status["similarities"], final_status["control_similarities_1"], final_status["control_similarities_2"])
    else:
        voice_similarity_rate = final_status["similarity"]
        if int(voice_similarity_rate) < 60:
            color = "red"
            text = "Low similarity - Not a likely match."
        elif int(voice_similarity_rate) < 70:
            color = "orange"
            text = "Moderate similarity - Possible match."
        else:
            color = "green"
            text = "High similarity - Likely match."
        display_results(voice_similarity_rate, text, color)

        if control_var.get():
            control_similarity_1 = CompareSounds(wav_file_1, wav_file_1)
            control_similarity_2 = CompareSounds(wav_file_2, wav_file_2)
            control_text = (
                f"Control 1 - Similarity rate: {control_similarity_1['similarity']}%\n"
                f"Control 2 - Similarity rate: {control_similarity_2['similarity']}%"
            )
            InformationPrinter(control_text)
            result_frame = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
            result_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
            control_label = tk.Label(result_frame, text=control_text, pady=10)
            control_label.pack()

    os.remove(wav_file_1)
    os.remove(wav_file_2)

def toggle_linear_switch():
    if rolling_window_var.get():
        linear_switch.config(state=tk.NORMAL)
        random_order_switch.config(state=tk.NORMAL)
    else:
        linear_switch.config(state=tk.DISABLED)
        random_order_switch.config(state=tk.DISABLED)

###########################################################################################################

# MAIN EXECUTION BLOCK

def create_file_input_box(root, label_text, row, column, columnspan):
    frame = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
    frame.grid(row=row, column=column, columnspan=columnspan, padx=10, pady=10, sticky="nsew")
    
    label = tk.Label(frame, text=label_text, pady=10)
    label.pack()
    
    entry = tk.Entry(frame, width=50)
    entry.pack(padx=10, pady=10)
    
    button = tk.Button(frame, text="Browse", command=lambda: browse_file(entry))
    button.pack(pady=10)
    
    return entry

def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.ogg *.flac *.aac *.aiff *.wma *.wav")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def select_files():
    file1 = file1_entry.get()
    file2 = file2_entry.get()
    if not file1 or not file2:
        messagebox.showerror("Error", "Please select two audio files.")
        return
    compare_files(file1, file2)

def display_results(voice_similarity_rate, text, color):
    InformationPrinter(f"Displaying results: {voice_similarity_rate}% - {text}")
    result_frame = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
    result_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    
    result_label = tk.Label(result_frame, text=f"Similarity rate: {voice_similarity_rate}%", fg=color, pady=10)
    result_label.pack()
    
    info_label = tk.Label(result_frame, text=f"Detection information: {text}", pady=10)
    info_label.pack()

if __name__ == "__main__":
    init()
    TitlePrinter(f"{APP_NAME} {VERSION_INFO} | {POWERED_BY}")

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("600x600")

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    file1_entry = create_file_input_box(root, "Select first audio file", 0, 0, 1)
    file2_entry = create_file_input_box(root, "Select second audio file", 0, 1, 1)

    control_var = tk.BooleanVar()
    control_switch = tk.Checkbutton(root, text="Enable Control", variable=control_var)
    control_switch.grid(row=1, column=0, columnspan=2, pady=10)

    rolling_window_var = tk.BooleanVar()
    rolling_window_checkbox = tk.Checkbutton(root, text="Enable Rolling Window", variable=rolling_window_var, command=toggle_linear_switch)
    rolling_window_checkbox.grid(row=2, column=0, columnspan=2, pady=10)

    linear_var = tk.BooleanVar()
    linear_switch = tk.Checkbutton(root, text="Linear Window", variable=linear_var)
    linear_switch.grid(row=3, column=0, columnspan=2, pady=10)
    linear_switch.config(state=tk.DISABLED)

    random_order_var = tk.BooleanVar()
    random_order_switch = tk.Checkbutton(root, text="Random Order", variable=random_order_var)
    random_order_switch.grid(row=4, column=0, columnspan=2, pady=10)
    random_order_switch.config(state=tk.DISABLED)

    compare_button = tk.Button(root, text="Compare Files", command=select_files, padx=20, pady=10)
    compare_button.grid(row=5, column=0, columnspan=2, pady=20)

    root.mainloop()
