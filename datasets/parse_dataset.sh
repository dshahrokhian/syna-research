#!/bin/bash       
#==============================================================================
# Title: parse_dataset.sh
# Description: This script will parse the CK+ dataset for its use with OpenFace
#              ,which involves first transforming the images into video and
#              then feeding these videos to OpenFace.
# Author: Daniyal Shahrokhian <daniyal@kth.se>
# Date: 20170310
# Version : 1.0
# Usage: bash parse_dataset.sh {ck+, afew} <dataset directory>
#                              <output directory for parsed dataset>
# Notes: 'ffmpeg' is a dependency
#==============================================================================

# Transforms the images in a directory from the CK+ dataset into videos
function images2video {
  cat "$1"/*.png | ffmpeg -y -loglevel error -r 30 -f image2pipe -i - "$2"
}

# For some reason, OpenFace doesn't like certain .avi files from the AFEW dataset
function avi2mkv {
  ffmpeg -i "$1" -vcodec ffv1 -acodec pcm_s16le "$2"
}

# Create Landmark video using OpenFace
function landmark_video {
  ../OpenFace/build/bin/FaceLandmarkVid -f "$1" -ov "$2" >> parse_dataset.log
}

# Extract features using OpenFace
function feature_extraction {
  ../OpenFace/build/bin/FeatureExtraction -rigid -f "$1" -of "$2" >> parse_dataset.log
}

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# != 3 ] || ([ "$1" != "ck+" ] && [ "$1" != "afew" ])
then
  echo "Usage: parse_dataset.sh {ck+, afew} <dataset directory> <output directory for the videos>"
  exit 1
fi

INPUT_DIRECTORY="$2"
OUTPUT_DIRECTORY="$3"

if [ -f "parse_dataset.log" ]
then
  rm "parse_dataset.log"
fi

# Process Cohn-Kanade extended dataset
if [ $1 == "ck+" ]
then
  # The directory files encodes the different emotions with numbers. 
  # For unification with other datasets, we transform this format.
  declare -A EMOTION_CODES=(["000"]="Neutral" ["001"]="Angry" ["002"]="Contempt" ["003"]="Disgust" ["004"]="Fear" ["005"]="Happy" ["006"]="Sad" ["007"]="Surprise")
  
  for SUBJECT in ${INPUT_DIRECTORY}/* # Directory containing all the subjects
  do
    if [[ -d "${SUBJECT}" ]]
    then
      for EMOTION in ${SUBJECT}/* # Directory containing all the emotions shown by the subject
      do
        if [[ -d "${EMOTION}" ]]
        then
          OUTPUT_FILES="${OUTPUT_DIRECTORY}/${EMOTION_CODES[$(basename ${EMOTION})]}"
          mkdir -p ${OUTPUT_FILES}/videos ${OUTPUT_FILES}/features

          echo "Processing ${EMOTION}"
          VIDEO="${OUTPUT_FILES}/videos/original_$(basename ${SUBJECT}).mkv"
          images2video "${EMOTION}" "${VIDEO}"
          landmark_video "${VIDEO}" "${OUTPUT_FILES}/videos/openface_$(basename ${SUBJECT}).mkv"
          feature_extraction "${VIDEO}" "${OUTPUT_FILES}/features/$(basename ${SUBJECT}).txt"
        fi
      done
    fi
  done

# Process Acted Facial Expressions In The Wild dataset
elif [ $1 == "afew" ]
then
  for EMOTION in ${INPUT_DIRECTORY}/* # Directory containing all the emotion-types
  do
    if [[ -d "${EMOTION}" ]]
    then
      for FILE in ${EMOTION}/* # Directory containing all the videos of the emotions
      do
        OUTPUT_FILES="${OUTPUT_DIRECTORY}/$(basename ${EMOTION})"
        mkdir -p ${OUTPUT_FILES}
        
        echo "Processing ${FILE}"
        VIDEO="${OUTPUT_FILES}/original_$(basename ${FILE} .avi).mkv"
        avi2mkv "${FILE}" "${VIDEO}"
        process_emotion "${VIDEO}" "${OUTPUT_FILES}/openface_${SUBJECT}$(basename ${FILE} .avi).txt"
      done
    fi
  done
fi