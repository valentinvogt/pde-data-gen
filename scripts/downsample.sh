#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=downsample-%j.out
#SBATCH --error=downsample-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16384
#SBATCH --time=1:00:00
#SBATCH --mail-type=END

module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6 nco/5.1.6
MODEL="bruss"
DATASET="final"
DATA_DIR="/cluster/scratch/vogtva/data/$MODEL/$DATASET"
IN_FILE="$DATA_DIR/_dataset.nc"
WORK_DIR="$DATA_DIR/work"
mkdir -p $WORK_DIR
OUT_FILE="$DATA_DIR/_dataset_processed_new.nc"

mkdir -p $WORK_DIR
snapshots=10
for ((i=0; i<$snapshots; i++)); do
    idx=$((5 + 10 * i))
    out_file="$WORK_DIR/tmp${i}.nc"
    if [[ -f "$out_file" ]]; then
        echo "Skipping existing file: $out_file"
        continue
    fi
    ncks -O -d snapshot,$idx,$idx -v data $IN_FILE $out_file
    echo "Sampled step $i (snapshot $idx)"
done
ncecat -O -u sampled_snapshot $WORK_DIR/tmp*.nc $WORK_DIR/data_only.nc
ncks -O -x -v data $IN_FILE $WORK_DIR/static.nc

ncwa -O -a snapshot $WORK_DIR/static.nc $WORK_DIR/static.nc
ncwa -O -a snapshot $WORK_DIR/data_only.nc $WORK_DIR/data_only.nc

# Merge static and downsampled dynamic parts
ncks -A $WORK_DIR/static.nc $WORK_DIR/data_only.nc
ncks -O -x -v snapshot $WORK_DIR/data_only.nc $WORK_DIR/data_only.nc
# Move final file to destination
ncpdq -O -a trajectory,sampled_snapshot,component,x,y $WORK_DIR/data_only.nc $WORK_DIR/data_transposed.nc

# Rename sampled_snapshot to snapshot
ncrename -d sampled_snapshot,snapshot $WORK_DIR/data_transposed.nc

mv $WORK_DIR/data_transposed.nc $OUT_FILE