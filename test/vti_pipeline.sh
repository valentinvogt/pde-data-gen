TEST_CONFIG=test_vti
TEST_DATASET_ID="test_vti_pipeline"
TEST_MODEL="gray_scott"

set -eo pipefail
source .env
rm -rf "$WORK_DIR/$TEST_MODEL/$TEST_DATASET_ID"
./scripts/create_inputs_job.sh --config-name=$TEST_CONFIG
./scripts/run_from_vti.sh $TEST_MODEL $TEST_DATASET_ID
./scripts/consolidate_vti.sh $TEST_MODEL $TEST_DATASET_ID