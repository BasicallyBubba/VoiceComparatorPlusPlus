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
    
    if not os.path.exists(target_file_path) or not os.path.exists(temp_dir_path):
        return {"success":"false", "code":"invalid path"}
    
    target_file_extensions = target_file_path.split(".")
    target_file_extensions = target_file_extensions[len(target_file_extensions)-1]

    supported_formats = ["MP3","OGG","FLAC","AAC","AIFF","WMA","WAV"]
    
    if target_file_extensions.upper() not in supported_formats:
        return {"success":"false", "code":"unsupported file extension"}

    LoadedAudio = AudioSegment.from_file(target_file_path, format=target_file_extensions)
    export_name = temp_dir_path+"exported_file_"+str(random.randint(1,999))+"."+TARGET_FILE_FORMAT
    
    LoadedAudio.export(export_name, format=TARGET_FILE_FORMAT)

    if os.path.exists(export_name):
        return {"success":"true", "path":str(export_name)}
    else:
        return { "success":"false", "code":"export error"}



def CompareSounds(sound_1_path:str, sound_2_path:str):
    if not os.path.exists(sound_1_path) or not os.path.exists(sound_2_path):
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
    return { "success":"true" ,"similarity":str(GetSimilarity) }



###########################################################################################################

# MAIN EXECUTION BLOCK

def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.ogg *.flac *.aac *.aiff *.wma *.wav")])
    if len(file_paths) != 2:
        messagebox.showerror("Error", "Please select exactly two audio files.")
        return
    compare_files(file_paths[0], file_paths[1])

def compare_files(file1, file2):
    InformationPrinter("Converting files to 'wav' format...")
    raw_file_1_convert_status = ConvertAnyAudio_to_wav(target_file_path=file1)
    raw_file_2_convert_status = ConvertAnyAudio_to_wav(target_file_path=file2)

    if raw_file_1_convert_status["success"] == "false" or raw_file_2_convert_status["success"] == "false":
        ErrorPrinter("File conversion failed.")
        messagebox.showerror("Error", "File conversion failed.")
        return

    wav_file_1 = raw_file_1_convert_status["path"]
    wav_file_2 = raw_file_2_convert_status["path"]

    InformationPrinter("Comparing voice similarity...")

    final_status = CompareSounds(wav_file_1, wav_file_2)

    if not final_status["success"] == "true":
        ErrorPrinter("Audio comparison failed.")
        os.remove(wav_file_1)
        os.remove(wav_file_2)
        messagebox.showerror("Error", "Audio comparison failed.")
        return

    InformationPrinter("Comparison finished. Displaying results...")
    voice_similarity_rate = final_status["similarity"]

    if int(voice_similarity_rate) < 60:
        color = red
        text = "Low similarity - Not a likely match."
    elif int(voice_similarity_rate) < 70:
        color = orange
        text = "Moderate similarity - Possible match."
    else:
        color = green
        text = "High similarity - Likely match."

    result_message = f"Similarity rate: {voice_similarity_rate}%\nDetection information: {text}"
    messagebox.showinfo("Results", result_message)

    os.remove(wav_file_1)
    os.remove(wav_file_2)

if __name__ == "__main__":
    init()
    TitlePrinter(f"{APP_NAME} {VERSION_INFO} | {POWERED_BY}")

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("400x200")

    label = tk.Label(root, text="Drag and drop two audio files to compare", pady=20)
    label.pack()

    compare_button = tk.Button(root, text="Select Files", command=select_files, padx=20, pady=10)
    compare_button.pack()

    root.mainloop()
