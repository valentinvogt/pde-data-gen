#!/bin/bash
#SBATCH --job-name=run-remainder
#SBATCH --output=run-remainder-%j.out
#SBATCH --error=run-remainder-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=0:30:00
#SBATCH --mail-type=END

module load stack/2024-05 gcc/13.2.0
source .env
model="gray_scott"
dataset_id="gs_final"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

./build/rdy -i data/test.vti || { echo "Ready not working!"; exit 1; }

for file in $DATAPATH/*.vti; do
    start=$(date +%s.%N)
    # remove .vti extension and add _out.nc
    file_name=$(basename "$file" .vti)
    nc_file="$DATAPATH/$file_name.nc"
    out_dir="$DATAPATH/out-$file_name"
    
    if [[ -f "$nc_file" ]]; then
        echo "Skipping $file_name: $nc_file already exists"
        continue
    fi
    
    if [[ -d "$out_dir" ]]; then
        echo "Skipping $file_name: $out_dir already exists"
        continue
    fi
    
    mkdir -p "$out_dir"
    build/rdy -i "$file" -o "tmp.vti" -n 100000 --snapshot-path "$out_dir" --num-snapshots 100
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$file took $duration seconds"
done

# rm tmp.vti