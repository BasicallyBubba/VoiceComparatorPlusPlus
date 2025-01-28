# VoiceComparator

<p>An educational tool in the field of computer forensics and osint to check whether existing voices belong to the same person. Based on the Python3 resemblyzer library, the tool analyzes sounds given in supported formats (albeit in very short periods of time), calculates their similarities using cosine similarity and gives the results..
<br>
<hr>
<br>
<h2>Supported audio formats:</h2>

`["MP3","OGG","FLAC","AAC","AIFF","WMA","WAV"]`


<br>
<h2>Required libraries:</h2> 
<br>
All python dependencies are listed in requirements.txt, which you can automagically install with

```bash
pip install -r requirements.txt

```
<br>
You will also need C++ build tools otherwise torch will get angy at you.
https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

<br>
<h2>Usage:</h2>
<br>
It is super simple. You add two sound clips that you want to compare, and click run! <br>
This gives you the overall resemblyzer similarity of the full audio inputs. <br>
If you have any questions on the semantics of resemblizer or the accuracy or algorithm and whatnot, the resemblyzer source code can be found [here](https://github.com/resemble-ai/Resemblyzer).

<br> <br>
You can also break down the audio files into smaller chunks, and compare those chunks to each other. <br>
Deep compare by default takes a rolling window of samples from the input file. It starts at 1 second window size and increases exponentially up to the point where it exceeds the length of the input. This is very thorough, since it compares every sample to every sample at every window size. <br>
Fast deep compare is deep compare with a consideration limit. Instead of comparing EVERY sample, it compares n random samples to n random samples (adjustable with the slider). <br>
The 'Linear' window option toggles whether the rolling window for deep search increases in size exponentially or linearly. It is STRONGLY advised not to use this with longer samples if you value having a working computer. <br>
</p>
