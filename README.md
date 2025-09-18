# Goal
This project attempts to analyze the various verses making up the Old Testament (Hebrew Bible) and attribute them to one of several hypothesized editors. 
This is accomplished by pre-training an LLM, training on a subset with a known editor label, and then classifying on a final, test set. 

Importantly, the tokens correspond to Hebrew words, not English ones, so the results should be somewhat novel. 

# Data 
The data is pulled from https://github.com/openscriptures/morphhb/blob/master/oxlos-import/wlc.txt

# Code
The code is only slightly modified from the example code provided by Sebastian Raschka's _Building LLMs from Scratch_ repository:
https://github.com/rasbt/LLMs-from-scratch

