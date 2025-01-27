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
This gives you the overall similarity. <br> <br>
You can also break down the audio files into smaller chunks, and compare those chunks to each other. <br>
The 'Linear Window' option toggles whether the rolling window increases in size exponentially or linearly. <br>

</p>
