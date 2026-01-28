#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Generate timestamp in format yyyymmddhhmmss
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo '*** Running 6_evaluate_remote.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR
echo
echo '*** Check for UV ***'
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "please install uv following https://docs.astral.sh/uv/getting-started/installation/."
    echo "or by running:"
    echo "`curl -LsSf https://astral.sh/uv/install.sh | sh`"
    exit 1
else
    echo "uv is installed."
fi
echo

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using system environment variables."
fi

echo

# Define test files array
    # "C2_Kokkos_Dataset/kokkos_testing.yaml"
TEST_FILES=(
    "C2_Kokkos_Dataset/kokkos_create_context_initial.yaml"
    "C2_Kokkos_Dataset/kokkos_create_context_refinement.yaml"
)

echo '*** Evaluate SiliconFlow model ***'
if [ -z "$SILICONFLOW_API_KEY" ]; then
    echo "Error: SILICONFLOW_API_KEY is not set."
    echo "Please create a .env file with your SiliconFlow API key."
    echo "Example:"
    echo "SILICONFLOW_API_KEY=your-api-key-here"
    echo "SILICONFLOW_MODEL=Pro/deepseek-ai/DeepSeek-V3.2"
    exit 1
else
    echo "SILICONFLOW_API_KEY is set."
    # Use the model from environment variable or default to Pro/deepseek-ai/DeepSeek-V3.2
    SILICONFLOW_MODEL=${SILICONFLOW_MODEL:-Pro/deepseek-ai/DeepSeek-V3.2}
    echo "Using SiliconFlow model: $SILICONFLOW_MODEL"
    # Create filename with model name (replacing slashes with underscores)
    MODEL_FILENAME=$(echo "$SILICONFLOW_MODEL" | tr '/' '_')
    
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
        
        echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json siliconflow test --model "$SILICONFLOW_MODEL" --save_results_file evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.json $TEST_FILE  > evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.out
        uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json siliconflow test --model "$SILICONFLOW_MODEL" --save_results_file evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.json $TEST_FILE > evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.out
        echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.json > evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.md"
        uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.json > evaluation/siliconflow_${MODEL_FILENAME}_${TEST_NAME}_results_${TIMESTAMP}.md
        
        # Add completion message for each test file
        echo "\n**********************************************************************"
        echo "*** COMPLETED TEST FILE: $TEST_FILENAME ***"
        echo "**********************************************************************\n"
    done
fi

