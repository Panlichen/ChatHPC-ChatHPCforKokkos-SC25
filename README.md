# ChatHPC: ChatHPC for Kokkos SC25 Artifacts

[![DOI](https://zenodo.org/badge/967029187.svg)](https://doi.org/10.5281/zenodo.15226006)

This repository holds the artifacts for the ChatHPC SC'25 submission. Contained in this repo is the ChatHPC Library and corresponding CLI application and the Kokkos training and verification datasets used to train and validate ChatHPC for Kokkos.

See [ChatHPC App README](C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/README.md) for more details on how to use the ChatHPC Library CLI Application.

## Dependencies

### Software

This repository's scripts depend on [uv](https://docs.astral.sh/uv/) to build the python virtual environment and to run the software with all the correct dependencies installed. A full list of the dependencies can be found in the `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/pyproject.toml` file. Please install uv, using the standard instructions, [installing uv](https://docs.astral.sh/uv/getting-started/installation/). This repository was developed on an Ubuntu 22.04.5 LTS system and should work on any modern Linux system.

### Hardware

This repository was tested on systems with Ampere A100 and Hopper H100 GPUs. However, this repository should work on any system supported by the upstream Hugging Face Trainer and PyTorch Libraries. 

## Directory Structure

```txt
ChatHPC-ChatHPCforKokkos-SC25
├── 1_setup.sh — Set up the program in a python virtual environment and download the base code-llama model.
├── 2_train.sh — Train ChatHPC for Kokkos Initial and ChatHPC for Kokkos Refinement.
├── 3_verify.sh — Verify trained models on training data.
├── 4_evaluate.sh — Test trained models on validation data.
├── 5_evaluate_baseline.sh — Test baseline models on validation data.
├── basemodels — Location for base models.
├── C1_ChatHPC_Lib — ChatHPC Library contribution artifact.
│   └── ChatHPC-app-v25.7.1 — Copy of ChatHPC-app at version 25.7.1
├── C2_Kokkos_Dataset — Kokkos data contribution artifact.
│   ├── kokkos_create_context_initial.json — Dataset for training the initial  model.
│   ├── kokkos_create_context_refinement.json — Dataset for training the refined model.
│   └── kokkos_testing.yaml — validation testing data.
├── output — Trained models.
├── config_initial.json — Config for training/running the initial model.
├── config_refinement.json — Config for training/running the refined model.
├── prompt_template.txt — Prompt template used for ChatHPC for Kokkos.
└── run_all.sh — Run all the reproduction scripts.
```

## Output Artifacts

```txt
ChatHPC-ChatHPCforKokkos-SC25
├── output
│   ├── 0_ChatKokko_initial_training_checkpoints — Training checkpoints for initial model.
│   ├── 0_ChatKokko_refinement_training_checkpoints — Training checkpoints for refined model.
│   ├── 1_ChatKokko_initial_peft_adapter — Trained adapter weights for initial model.
│   ├── 1_ChatKokko_refinement_peft_adapter — Trained adapter weights for refined model.
│   ├── 2_ChatKokko_initial_merged_adapters — Merged full initial model.
│   └── 2_ChatKokko_refinement_merged_adapters — Merged full refined model.
└── evaluation
    ├── ChatHPCforKokkos_initial_results.json — ChatHPC for Kokkos initial results in JSON.
    ├── ChatHPCforKokkos_initial_results.md — ChatHPC for Kokkos initial results converted to Markdown.
    ├── ChatHPCforKokkos_refinement_results.json — ChatHPC for Kokkos refinement results in JSON.
    ├── ChatHPCforKokkos_refinement_results.md — ChatHPC for Kokkos refinement results converted to Markdown.
    ├── code_llama_base_results.json — CodeLlama baseline results in JSON.
    ├── code_llama_base_results.md — CodeLlama baseline results converted to Markdown.
    ├── openai_gpt-4o_base_results.json — GPT-4o baseline results in JSON.
    └── openai_gpt-4o_base_results.md — GPT-4o baseline results converted to Markdown.
```

## Reproduction Quick Steps

1. Download base model.
    - Please register your SSH key with Hugging Face, and request access to https://huggingface.co/meta-llama/CodeLlama-7b-hf.
    - Alternatively, you can manually download the CodeLlama-7b-hf model from the hugging face website and place it in the basemodels directory at `basemodels/CodeLlama-7b-hf`
2. Run `run_all.sh` which will call `1_setup.sh`, `2_train.sh`, `3_verify.sh`, `4_evaluate.sh`, and `5_evaluate_baseline.sh` in order.
3. Review the created output artifacts from training and evaluating ChatHPC for Kokkos. See [Output Artifacts](#output-artifacts).

> [!NOTE]
> Note: An example of the expected output is found in the `example_evaluation_output` folder. This folder contains the expected output stored in the `evaluation` folder after running the `run_all.sh` script. Additionally, the expected console output from the `run_all.sh` script is provided in the `example_evaluation_output/run_all_output.txt` file.