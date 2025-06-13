#!/bin/bash
#SBATCH --job-name=input-runner-gs
#SBATCH --output=input-runner-gs-%j.out
#SBATCH --error=input-runner-gs-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=1:00:00

# conda activate
source .env
model="gray_scott"
dataset_id="gs_pt"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

for file in "$DATAPATH"/*.vti; do
    start=$(date +%s.%N)
    # remove .vti extension and add _out.nc
    file_name=$(basename "$file" .vti)
    file_name_out="${file_name}_out.nc"
    mkdir -p "$DATAPATH/out-$out_file_name"
    build/rdy -i "$file" -o "tmp.vti" -n 200000 --snapshot-path "$DATAPATH/out-$out_file_name"
    # python scripts/gather_vti.py --path "$DATAPATH/out/" --output "$DATAPATH/${file_name_out}"
    # rm $DATAPATH/out/*.vti
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$file took $duration seconds"
done

rm tmp.vti