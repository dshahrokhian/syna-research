#!/bin/bash       
#==============================================================================
# Title: parse_dataset.sh
# Description: This script will parse the CK+ and afew datasets for its use 
#              with OpenFace, which involves first transforming the images into
#              video and then feeding these videos to OpenFace.
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
  ../deepmotion/OpenFace/build/bin/FaceLandmarkVid -q -f "$1" -ov "$2" >> parse_dataset.log
}

# Extract features using OpenFace
function feature_extraction {
  ../deepmotion/OpenFace/build/bin/FeatureExtraction -q -rigid -wild -multi-view 1 -f "$1" -of "$2" >> parse_dataset.log
}

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# != 3 ] || ([ "$1" != "ck+" ] && [ "$1" != "afew" ])
then
  echo "Usage: parse_dataset.sh {ck+, afew} <dataset directory> <output directory for the videos>"
  exit 1
fi

# Robust paths
INPUT_DIRECTORY="${2%/}"
OUTPUT_DIRECTORY="${3%/}"

if [ -f "parse_dataset.log" ]
then
  rm "parse_dataset.log"
fi

# Process Cohn-Kanade extended dataset
if [ $1 == "ck+" ]
then
  INPUT_DIRECTORY="$INPUT_DIRECTORY/cohn-kanade-images"

  for SUBJECT in ${INPUT_DIRECTORY}/* # Directory containing all the subjects
  do
    if [[ -d "${SUBJECT}" ]]
    then
      for RECORD in ${SUBJECT}/* # Directory containing all the recordings from the subject
      do
        if [[ -d "${RECORD}" ]]
        then
          OUTPUT_FILES=${OUTPUT_DIRECTORY}/$(basename ${SUBJECT})/$(basename ${RECORD})
          mkdir -p ${OUTPUT_FILES}
          
          echo "Processing ${RECORD}"
          VIDEO="${OUTPUT_FILES}/$(basename ${SUBJECT})_$(basename ${RECORD})_original.mp4"
          images2video "${RECORD}" "${VIDEO}"
          landmark_video "${VIDEO}" "${OUTPUT_FILES}/$(basename ${SUBJECT})_$(basename ${RECORD})_openface.mkv"
          feature_extraction "${VIDEO}" "${OUTPUT_FILES}/$(basename ${SUBJECT})_$(basename ${RECORD})_features.txt"
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
        VIDEO="${OUTPUT_FILES}/$(basename ${FILE} .avi)_original.mkv"
        avi2mkv "${FILE}" "${VIDEO}"
        landmark_video "${VIDEO}" "${OUTPUT_FILES}/$(basename ${FILE} .avi)_openface.mkv"
        feature_extraction "${VIDEO}" "${OUTPUT_FILES}/$(basename ${FILE} .avi)_features.txt"
      done
    fi
  done
fi