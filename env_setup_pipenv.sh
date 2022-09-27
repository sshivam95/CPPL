#!/bin/bash

ml load system/singularity/3.9.9
rm -rf .venv/
export IMG_FILE=$PFS_FOLDER/CPPL/singularity/pbcppl.sif
singularity exec -B $PFS_FOLDER/CPPL/ --nv $IMG_FILE pipenv install