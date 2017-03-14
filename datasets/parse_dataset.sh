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
#                              <output directory for the videos>
# Notes: 'ffmpeg' is required
#==============================================================================

function images_to_video {
  cat "$1"/*.png | ffmpeg -y -loglevel error -r 30 -f image2pipe -i - "$2"
}

function process_emotion {
  # Create Landmark video using OpenFace
  ../OpenFace/build/bin/FaceLandmarkVid -f "$1" -ov "$2/openface_video.mkv" >> parse_dataset.log

  # Extract features using OpenFace
  ../OpenFace/build/bin/FeatureExtraction -rigid -f "$1" -of "$2/openface_features.txt" >> parse_dataset.log
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
  for SUBJECT in ${INPUT_DIRECTORY}/* # Directory containing all the subjects
  do
    if [[ -d "${SUBJECT}" ]]
    then
      for EMOTION in ${SUBJECT}/* # Directory containing all the emotions shown by the subject
      do
        if [[ -d "${EMOTION}" ]]
        then
          OUTPUT_FILES=${OUTPUT_DIRECTORY}/$(basename ${SUBJECT})/$(basename ${EMOTION})
          mkdir -p ${OUTPUT_FILES}

          echo "Processing ${EMOTION}"
          images_to_video ${EMOTION} "${OUTPUT_FILES}/original_video.mkv"
          process_emotion "${OUTPUT_FILES}/original_video.mkv" ${OUTPUT_FILES}
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
        OUTPUT_FILES=${OUTPUT_DIRECTORY}/$(basename ${EMOTION})/$(basename ${FILE})
        mkdir -p ${OUTPUT_FILES}
        
        echo "Processing ${FILE}"
        process_emotion ${FILE} ${OUTPUT_FILES}
      done
    fi
  done
fi