#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8192
#SBATCH --time=23:00:00
# #SBATCH --mail-type=END

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
# module load cuda/12.1.1
module load hdf5/1.14.3
# module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

source .env
HDF5_USE_FILE_LOCKING=FALSE

MODEL=bruss
DATASET=default_bruss

#--- Consolidate
# python3 scripts/consolidate_old_format.py $WORKDIR/data/$MODEL/$DATASET

#--- Classify 
python3 src/classify.py --model $MODEL --ds_id $DATASET --time_ratio 0.2 --directory_var SCRATCHDIR

#--- Process for training
NUM_SNAPSHOTS=10
# --num_snapshots $NUM_SNAPSHOTS
# python3 scripts/process_dataset_for_training.py $SCRATCHDIR/data/$MODEL/$DATASET/_dataset.nc  --output_file $SCRATCHDIR/data/$MODEL/$DATASET/_dataset_processed.nc --num_snapshots $NUM_SNAPSHOTS

# python3 src/classify_pca.py