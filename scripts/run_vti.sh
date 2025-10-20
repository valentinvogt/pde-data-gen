#!/bin/bash
#SBATCH --job-name=input-runner-gs
#SBATCH --output=input-runner-gs-%j.out
#SBATCH --error=input-runner-gs-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=1-4:00:00

# module load stack/2024-05 gcc/13.2.0
source .env
model="bruss"
dataset_id="rean"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

./build/rdy -i data/test.vti || { echo "Ready not working!"; exit 1; }

for file in $DATAPATH/*.vti; do
    start=$(date +%s.%N)
    # remove .vti extension and add _out.nc
    file_name=$(basename "$file" .vti)
    out_dir="$DATAPATH/out-$file_name"
    mkdir -p "$out_dir"
    build/rdy -i "$file" -o "tmp.vti" -n 100000 --snapshot-path "$out_dir" --num-snapshots 100
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$file took $duration seconds"
done

rm tmp.vti