# VoiceComparator

VoiceComparator is an educational tool in the field of computer forensics and OSINT, designed to determine whether existing voice samples belong to the same person. Based on the Python3 Resemblyzer library, the tool analyzes audio files in supported formats, calculates their similarities using cosine similarity, and provides results.

---

## Supported Audio Formats

The tool supports the following audio formats:

`MP3`, `OGG`, `FLAC`, `AAC`, `AIFF`, `WMA`, `WAV`

---

## Required Libraries

All Python dependencies are listed in `requirements.txt`. You can install them with the following command:

```bash
pip install -r requirements.txt
```

Additionally, you will need C++ Build Tools to avoid issues with the Torch library. You can download them from [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## Usage

Using VoiceComparator is straightforward:

1. Add two audio clips you want to compare.
2. Click "Run."

### Features

- **Overall Similarity**: Provides the Resemblyzer similarity score for the full audio inputs.
- **Chunk Comparison**: Breaks audio files into smaller chunks and compares them.
- **Deep Compare**: Compares audio samples using a rolling window of varying sizes:
  - Starts with a 1-second window and increases exponentially until exceeding the input length.
  - Thoroughly compares every sample to every other sample at every window size.
- **Fast Deep Compare**: Limits the number of comparisons by sampling `n` random chunks from each input (adjustable with a slider).
- **Linear Window Option**: Toggles between exponential and linear window size increases for deep searches. **Note**: Using linear mode with longer samples may significantly impact performance.

If you have questions about the semantics, accuracy, or algorithm of Resemblyzer, the source code is available [here](https://github.com/resemble-ai/Resemblyzer).

---

