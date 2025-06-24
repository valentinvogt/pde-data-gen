module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6 nco/5.1.6
IN_FILE="/cluster/scratch/vogtva/data/bruss/final/_dataset.nc"
WORK_DIR="/cluster/scratch/vogtva/data/bruss/final/work"

mkdir -p $WORK_DIR
snapshots=10
for ((i=0; i<$snapshots; i++)); do
    idx=$((5 + 2 * i))
    ncks -O -d snapshot,$idx,$idx $IN_FILE $WORK_DIR/tmp${i}.nc
    echo "$i"
done
ncecat -O -u sampled_snapshot $WORK_DIR/tmp*.nc $WORK_DIR/tmp_combined.nc