# A: float = 5
# B: float = 9
# Nx: int = 100
# dx: float = 1.0
# Nt: int = 1000
# dt: float = 0.01
# Du: float = 2.0
# Dv: float = 22.0
# n_snapshots: int = 100

PYTHON=".venv/bin/python"
DATAPATH="/Users/vv/eth/bachelor/pde-solvers-cuda/data"

A=5
B=9
Nx=32
dx=1.0
Nt=10_000
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=100
model="bruss"
run_id="abd_big"

mkdir -p $DATAPATH/$model/$run_id

for A in 0.5 1 2; do
        for B_mult in 2 3; do
                for Du in 1.0 2.0; do
                        for D_mult in 4 8; do
                                for seed in $(seq 1 5); do
                                        start=`date +%s`
                                        B=$($PYTHON -c "print($A * $B_mult)")
                                        Dv=$($PYTHON -c "print($Du * $D_mult)")
                                        FILENAME="${DATAPATH}/${model}/${run_id}/$(uuidgen).nc"
                                        echo "(A, B, Du, Dv) = ($A, $B, $Du, $Dv)"
                                        FILE=$($PYTHON scripts/rd_runner.py --model $model --A $A --B $B \
                                                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                                                --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id \
                                                --random_seed $seed)
                                        build/run_from_netcdf $FILE 1
                                        end=`date +%s`
                                        runtime=$((end-start))
                                        echo "Took $runtime seconds"
                                done
                        done
                done
        done
done