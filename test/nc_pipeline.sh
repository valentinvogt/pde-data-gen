TEST_CONFIG=test_nc
TEST_DATASET_ID="test_nc_pipeline"
TEST_MODEL="bruss"

set -eo pipefail
source .env
rm -rf "$WORK_DIR/$TEST_MODEL/$TEST_DATASET_ID"
./scripts/create_inputs_job.sh --config-name=$TEST_CONFIG
./scripts/run_from_nc.sh $TEST_MODEL $TEST_DATASET_ID