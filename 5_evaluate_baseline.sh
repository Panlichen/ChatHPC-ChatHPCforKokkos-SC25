#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Generate timestamp in format yyyymmddhhmmss
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo '*** Running 5_evaluate_baseline.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

# Define test files array
    # "C2_Kokkos_Dataset/kokkos_testing.yaml"
TEST_FILES=(
    "C2_Kokkos_Dataset/kokkos_create_context_initial.yaml"
    "C2_Kokkos_Dataset/kokkos_create_context_refinement.yaml"
)

# Loop through each test file
for TEST_FILE in "${TEST_FILES[@]}"; do
    # Extract filename without path for better display
    TEST_FILENAME=$(basename "$TEST_FILE")
    
    # Add visually prominent status output
    echo "\n**********************************************************************"
    echo "*** PROCESSING TEST FILE: $TEST_FILENAME ***"
    echo "*** FILE PATH: $TEST_FILE ***"
    echo "*** TIMESTAMP: $TIMESTAMP ***"
    echo "**********************************************************************\n"
    
    # Extract filename without extension for use in output filenames
    TEST_NAME=$(basename "$TEST_FILE" .yaml)
    # Replace underscores and slashes with hyphens
    TEST_NAME=${TEST_NAME//_/-}
    TEST_NAME=${TEST_NAME//\//-}
    
    echo '*** Evaluate code-llama base model ***'
    echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.json $TEST_FILE
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.json $TEST_FILE
    echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.json > evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.md"
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.json > evaluation/code_llama_base_${TEST_NAME}_results_${TIMESTAMP}.md
    
    # Add completion message for each test file
    echo "\n**********************************************************************"
    echo "*** COMPLETED TEST FILE: $TEST_FILENAME ***"
    echo "**********************************************************************\n"
done
