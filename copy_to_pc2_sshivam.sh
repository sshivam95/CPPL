#!/bin/bash

rsync -avz -P --rsh="ssh -J sshivam@fe.noctua1.pc2.uni-paderborn.de" --exclude=".git" \
--exclude="build"  --exclude="dist" --exclude=".egg-info" --exclude=".~lock." \
--exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" --exclude="__pycache__" --exclude="\*\sandbox" \
/home/shivam/PycharmProjects/CPPL sshivam@ln-0001:/scratch/hpc-prf-pbrac/sshivam/