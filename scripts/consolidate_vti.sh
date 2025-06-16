#!/bin/bash
#SBATCH --job-name=cons
#SBATCH --output=cons-%j.out
#SBATCH --error=cons-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=1:00:00


source .env
model="bruss"
dataset_id="default_bruss"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

for dir in "$DATAPATH/out-*"; do
    python scripts/gather_vti.py --path $dir --output "$DATAPATH/_merged.nc"
    rm dir/*.vti
done