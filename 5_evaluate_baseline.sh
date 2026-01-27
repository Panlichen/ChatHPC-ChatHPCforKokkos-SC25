#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo '*** Running 5_evaluate_baseline.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Evaluate code-llama base model ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/code_llama_base_results.json > evaluation/code_llama_base_results.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/code_llama_base_results.json > evaluation/code_llama_base_results.md

# echo
# 
# echo '*** Evaluate OpenAI gpt-4o base model ***'
# if [ -z "$OPENAI_API_KEY" ]; then
#     echo "Error: OPENAI_API_KEY is not set."
#     echo "Please export your key if you would like to evaluate the OpenAI gpt-4o base model."
#     echo "export OPENAI_API_KEY=your-key-here"
#     exit 1
# else
#     echo "OPENAI_API_KEY is set."
#     echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json openai test --model gpt-4o --save_results_file evaluation/openai_gpt-4o_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
#     uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json openai test --model gpt-4o --save_results_file evaluation/openai_gpt-4o_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
#     echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/openai_gpt-4o_base_results.json > evaluation/openai_gpt-4o_base_results.md"
#     uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/openai_gpt-4o_base_results.json > evaluation/openai_gpt-4o_base_results.md
# fi

echo

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using system environment variables."
fi

echo

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
    echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json siliconflow test --model "$SILICONFLOW_MODEL" --save_results_file evaluation/siliconflow_${MODEL_FILENAME}_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json siliconflow test --model "$SILICONFLOW_MODEL" --save_results_file evaluation/siliconflow_${MODEL_FILENAME}_results.json C2_Kokkos_Dataset/kokkos_testing.yaml
    echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/siliconflow_${MODEL_FILENAME}_results.json > evaluation/siliconflow_${MODEL_FILENAME}_results.md"
    uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/siliconflow_${MODEL_FILENAME}_results.json > evaluation/siliconflow_${MODEL_FILENAME}_results.md
fi

