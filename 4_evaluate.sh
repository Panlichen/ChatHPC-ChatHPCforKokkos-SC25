#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Generate timestamp in format yyyymmddhhmmss
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo '*** Running 4_evaluate.sh ***'
echo
echo '*** Ensure commands are running in correct directory. ***'
echo cd $SCRIPT_DIR
cd $SCRIPT_DIR

echo
echo '*** Evaluating ChatHPC for Kokkos Initial ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json test --save_results_file evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json test --save_results_file evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.json > evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.json > evaluation/ChatHPCforKokkos_initial_results_${TIMESTAMP}.md

echo
echo '*** Evaluating ChatHPC for Kokkos Refinement ***'
echo uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_refinement.json test --save_results_file evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_refinement.json test --save_results_file evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.json C2_Kokkos_Dataset/kokkos_testing.yaml
echo "uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.json > evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.md"
uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc-data-to-md evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.json > evaluation/ChatHPCforKokkos_refinement_results_${TIMESTAMP}.md