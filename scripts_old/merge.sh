#!/bin/bash

# Overwrite existing files
set -o noclobber
ncks_options="-O"
ncrcat_options="-O"

# Temporary files array
mod_files=()

# Step 1: Add 'trajectory' dimension to each file
for file_in in out/data/bruss/default_bruss/*.nc; do
    file_mod="${file_in%.nc}_mod.nc"
    mod_files+=("$file_mod")
    ncks ${ncks_options} --mk_rec_dmn trajectory "$file_in" "$file_mod"
done

# Step 2: Concatenate the modified files along the 'trajectory' dimension
output_file="output_merged.nc"
ncrcat ${ncrcat_options} "${mod_files[@]}" "${output_file}"
ncatted -a history,global,d,, ${output_file} ${output_file} -O
# Optional: Clean up temporary files
# rm "${mod_files[@]}"

echo "Merged file created: ${output_file}"