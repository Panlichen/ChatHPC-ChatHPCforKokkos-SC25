#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Generate timestamp in format yyyymmddhhmmss
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo '*** Running 7_verify_raw.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Verifying code-llama base model ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base verify --save_results_file verify/code_llama_base_verify_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base verify --save_results_file verify/code_llama_base_verify_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md verify/code_llama_base_verify_results_${TIMESTAMP}.json > verify/code_llama_base_verify_results_${TIMESTAMP}.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md verify/code_llama_base_verify_results_${TIMESTAMP}.json > verify/code_llama_base_verify_results_${TIMESTAMP}.md
