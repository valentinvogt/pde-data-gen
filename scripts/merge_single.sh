#!/bin/bash
module load stack/2024-06  gcc/12.2.0  openmpi/4.1.6 nco/5.1.6

# Overwrite existing files
set -o noclobber
ncks_options="-O"
ncrcat_options="-O"

DIR="/cluster/scratch/vogtva/data/gray_scott/gs_final"
output_file="$DIR/_dataset.nc"
    
single_file="/cluster/scratch/vogtva/data/gray_scott/gs_final/cd6cac1a-5ffb-466f-be96-b945c2dd422f.nc"
file_mod="${single_file%.nc}_mod.nc"
ncks ${ncks_options} --mk_rec_dmn trajectory "$single_file" "$file_mod"
ncrcat ${ncrcat_options} "$output_file" "$file_mod" "$output_file"
ncatted -a history,global,d,, ${output_file} ${output_file} -O
rm "$file_mod"

# rm "$output_file"

# # Step 1: Add 'trajectory' dimension to each file
# for file_in in $DIR/*.nc; do
#     echo $file_in
#     # skip file beginning with '_'
#     if [[ $(basename "$file_in") == _* ]]; then
#         echo "Skipping file: $file_in"
#         continue
#     fi
#     file_mod="${file_in%.nc}_mod.nc"
#     mod_files+=("$file_mod")
#     ncks ${ncks_options} --mk_rec_dmn trajectory "$file_in" "$file_mod"
# done
# echo "Done appending dims"

# # Step 2: Concatenate the modified files along the 'trajectory' dimension
# output_file="$DIR/_dataset.nc"
# ncrcat ${ncrcat_options} "${mod_files[@]}" "${output_file}"
# ncatted -a history,global,d,, ${output_file} ${output_file} -O
# # Optional: Clean up temporary files
# rm "${mod_files[@]}"

# echo "Merged file created: ${output_file}"