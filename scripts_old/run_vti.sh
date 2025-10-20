#!/bin/bash
#SBATCH --job-name=run-fz
#SBATCH --output=run-fz-%j.out
#SBATCH --error=run-fz-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=2:00:00
#SBATCH --mail-type=END

module load stack/2024-05 gcc/13.2.0
source .env
model="bruss"
dataset_id="transfer2"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

./build/rdy -i data/test.vti || { echo "Ready not working!"; exit 1; }

count=0
for file in $DATAPATH/*.vti; do
    # if (( count > 20)); then
    #     echo "Processed 10 files, stopping."
    #     break
    # fi
    # count=$((count + 1))
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
    build/rdy -i "$file" -o "tmp.vti" -n 60000 --snapshot-path "$out_dir" --num-snapshots 100
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$file took $duration seconds"
done

# rm tmp.vti