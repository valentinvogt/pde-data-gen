#!/bin/bash
#SBATCH --job-name=run-vti
#SBATCH --output=run-vti-%j.out
#SBATCH --error=run-vti-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=2:00:00
#SBATCH --mail-type=END

module load stack/2024-05 gcc/13.2.0 jq/1.6
source .env
model="bruss"
dataset_id="default_bruss"
DATAPATH="./data/$model/$dataset_id"
CONFIG_FILE="$DATAPATH/_config.json"

# test Ready availability 
./build/rdy -i data/test.vti || { echo "Ready not working!"; exit 1; }

Nt=$(jq -r '.sim_params.Nt' "$CONFIG_FILE")
n_snapshots=$(jq -r '.sim_params.n_snapshots' "$CONFIG_FILE")

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
    build/rdy -i "$file" -o "tmp.vti" -n $Nt --snapshot-path "$out_dir" --num-snapshots $n_snapshots
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$file took $duration seconds"
done