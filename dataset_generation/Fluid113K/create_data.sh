#!/bin/bash

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
OUTPUT_SCENES_DIR=generate_scene
OUTPUT_DATA_DIR=generate_data

mkdir $OUTPUT_SCENES_DIR


for seed in `seq 1 250`; do
        python create_physics_scenes.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed \
                                        --default-viscosity \
                                        --default-density \
                                        --default-box \
                                        --num-objects 1 
done



python create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR 
