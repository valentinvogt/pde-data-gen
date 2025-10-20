source .env
model="bruss"
dataset_id="$1"

DATAPATH="./data/$model/$dataset_id"
echo "$DATAPATH"
# Check if we're using the consolidated file approach
if [[ -f "$DATAPATH/_dataset.nc" ]]; then
    echo "Using consolidated output approach"

    for file in "$DATAPATH"/*.nc; do
        # Skip files that aren't input files
        if [[ "$file" == *_output.nc || "$file" == *_dataset.nc ]]; then
            continue
        fi

        # echo "Processing $file"
        build/run_from_netcdf "$file" 1
    done
    python scripts/consolidate_outputs.py $DATAPATH/_dataset.nc

else
    # Original approach - process each file individually
    echo "Using original individual files approach"

    for file in "$DATAPATH"/*.nc; do
        # Skip output files
        if [[ "$file" == *_output.nc ]]; then
            continue
        fi

        build/run_from_netcdf "$file" 1
    done
    python3 scripts/consolidate_old_format.py $DATAPATH
fi
