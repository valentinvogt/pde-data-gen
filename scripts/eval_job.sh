#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16384
#SBATCH --time=04:00:00
# #SBATCH --mail-type=END

module load stack/2024-06
module load gcc/12.2.0
module load hdf5/1.14.3
module load netcdf-c/4.9.2
module load python/3.11.6

source .env
HDF5_USE_FILE_LOCKING=FALSE

MODEL=gray_scott
DATASET=gs_int

# python3 scripts/rechunk.py
#--- Consolidate
# python3 scripts/consolidate_old_format.py $SCRATCHDIR/data/$MODEL/$DATASET
# python3 analysis/ms.py
#--- Classify 
python3 src/classify.py --model $MODEL --ds_id $DATASET --time_ratio 0.2 --directory_var SCRATCHDIR --mode new

#--- Process for training

# NUM_SNAPSHOTS=10
# python3 scripts/process_dataset_for_training.py $SCRATCHDIR/data/$MODEL/$DATASET/_dataset.nc  --output_file $SCRATCHDIR/data/$MODEL/$DATASET/_dataset_processed_new.nc --num_snapshots $NUM_SNAPSHOTS

#--- Analysis
# python3 analysis/test.py --model $MODEL --ds_id $DATASET --time_ratio 0.2 --directory_var SCRATCHDIR
# python3 analysis/time.py

#--- Move to tmp directory -- not faster
# cp /cluster/scratch/vogtva/data/bruss/final/_dataset.nc $TMPDIR
# echo "Copied to tmp"
# python3 scripts/process_dataset_for_training.py $TMPDIR/_dataset.nc  --output_file $TMPDIR/_dataset_processed.nc
# echo "Performed computation"
# cp $TMPDIR/_dataset_processed.nc /cluster/work/math/vogtva/data/bruss/final/_dataset_processed_full.nc
