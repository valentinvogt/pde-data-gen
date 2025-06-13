source .env
model="gray_scott"
dataset_id="gs_pt"

DATAPATH="./data/$model/$dataset_id"

mkdir ./data/$model/$dataset_id/out/
for file in "$DATAPATH"/*.vti; do
    echo "Processing file: $file"
    build/rdy -i "$file" -o "tmp.vti" -n 200000 -s "./data/$model/$dataset_id/out/"
    # remove .vti extension and add _out.nc
    file_name=$(basename "$file" .vti)
    file_name_out="${file_name}_out.nc"
    uv run scripts/gather_vti.py --path "./data/$model/$dataset_id/out/" --output "./data/$model/$dataset_id/${file_name_out}"
    rm ./data/$model/$dataset_id/out/*.vti
done

rm tmp.vti