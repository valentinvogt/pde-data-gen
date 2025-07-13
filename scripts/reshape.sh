#!/bin/bash

cd /cluster/scratch/vogtva
# Input and output files
INFILE="/cluster/scratch/vogtva/data/gray_scott/gs_new/_dataset_processed.nc"
OUTFILE="output.nc"

# Step 1: Collapse sampled_snapshot from variables where it's spurious
ncwa -O -a sampled_snapshot -v A,B,Du,Dv,input_filename "$INFILE" reduced.nc

# # Step 2: Extract variables where sampled_snapshot is meaningful
# ncks -O -v data,component,x,y,trajectory,sampled_snapshot "$INFILE" rest.nc

# # Step 3: Merge collapsed variables into rest.nc
# ncks -A reduced.nc rest.nc

# # Step 4: Remove length-1 snapshot dimension from 'data'
# ncwa -O -a snapshot rest.nc tmp1.nc

# # Step 5: Rename sampled_snapshot → snapshot
# ncrename -O -d sampled_snapshot,snapshot tmp1.nc tmp2.nc

# # Step 6: Reorder dimensions to trajectory,snapshot,component,x,y
# ncpdq -O -a trajectory,snapshot,component,x,y tmp2.nc "$OUTFILE"

# # Clean up intermediates
# rm reduced.nc rest.nc tmp1.nc tmp2.nc
