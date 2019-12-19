#!/bin/sh

# Update the slides whenever the notebooks change.
# Requires `entr`, available for Linux and Mac.

ls *.ipynb | entr -r ./slides.sh 

