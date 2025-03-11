set -a && source .env
for dt in 0.0025 0.002 0.0015 0.001 0.0005 0.0001; do
    echo "Running with dt = $dt"
    Nt=$(python -c "print(int(150.0/$dt))")
    python scripts/make_inputs.py --config-name blowup sim_params.dt=$dt sim_params.Nt=$Nt
done