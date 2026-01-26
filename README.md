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


## 2_train

### 执行流程分解

脚本 `2_train.sh` 负责训练两个版本的 ChatHPC for Kokkos 模型：初始模型和精细化模型。其完整执行步骤如下：

1. **脚本初始化**：
   - 打印当前正在执行的脚本名称
   - 确保在脚本所在目录下执行后续命令

2. **初始模型训练**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json train`
   - 使用 `config_initial.json` 配置文件
   - 训练数据：`C2_Kokkos_Dataset/kokkos_create_context_initial.json`
   - 输出路径：`output/0_ChatKokko_initial_training_checkpoints`

3. **精细化模型训练**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_refinement.json train`
   - 使用 `config_refinement.json` 配置文件
   - 训练数据：`C2_Kokkos_Dataset/kokkos_create_context_refinement.json`
   - 输出路径：`output/0_ChatKokko_refinement_training_checkpoints`

### 依赖文件

执行 `2_train.sh` 过程中会调用或依赖以下文件：

- **配置文件**：
  - `config_initial.json`：初始模型训练配置
  - `config_refinement.json`：精细化模型训练配置

- **数据文件**：
  - `C2_Kokkos_Dataset/kokkos_create_context_initial.json`：初始训练数据集
  - `C2_Kokkos_Dataset/kokkos_create_context_refinement.json`：精细化训练数据集

- **模型文件**：
  - 基础模型：`basemodels/CodeLlama-7b-hf`

- **模板文件**：
  - `prompt_template.txt`：训练和推理使用的提示模板

- **Python核心代码**：
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/__main__.py`：命令行入口
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py`：核心训练逻辑实现

### 训练核心逻辑

训练的核心逻辑位于 Python 文件 `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py` 中的 `App` 类的 `train` 方法。该方法实现了完整的模型训练流程：

1. **配置 LoRA 参数**：设置 LoRA（低秩适应）微调的各项参数
2. **模型准备**：将模型转换为训练模式并准备进行 k-bit 训练
3. **训练参数设置**：配置训练批次大小、学习率、训练步数等参数
4. **Trainer 创建**：使用 Hugging Face Trainer 初始化训练器
5. **模型编译**：如果条件允许，使用 PyTorch 2.0+ 的编译功能优化模型
6. **训练执行**：调用 Trainer 的 train 方法执行训练
7. **模型保存**：
   - 保存微调后的 PEFT 适配器
   - 合并基础模型和适配器权重
   - 保存完整的合并模型

### 核心文件调用方式

`chathpc` 命令通过以下方式调用核心训练逻辑：

1. **命令行入口**：`uv run` 命令执行 `chathpc` 命令，该命令映射到 `chathpc.app.__main__.py` 中的 `cli` 函数
2. **命令解析**：`cli` 函数解析命令行参数，识别到 `train` 子命令后调用 `train` 函数
3. **App 实例创建**：`train` 函数使用指定的配置文件创建 `App` 实例
4. **训练方法调用**：`App` 实例调用 `train` 方法，执行实际的训练流程
5. **训练流程执行**：`App.train()` 方法按顺序执行模型加载、数据集处理、训练和模型保存等步骤



## 3_verify

### 执行流程分解

脚本 `3_verify.sh` 负责验证两个版本的 ChatHPC for Kokkos 模型在训练数据上的表现。其完整执行步骤如下：

1. **脚本初始化**：
   - 打印当前正在执行的脚本名称
   - 确保在脚本所在目录下执行后续命令

2. **初始模型验证**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json verify`
   - 使用 `config_initial.json` 配置文件
   - 验证数据：`C2_Kokkos_Dataset/kokkos_create_context_initial.json`

3. **精细化模型验证**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_refinement.json verify`
   - 使用 `config_refinement.json` 配置文件
   - 验证数据：`C2_Kokkos_Dataset/kokkos_create_context_refinement.json`

### 依赖文件

执行 `3_verify.sh` 过程中会调用或依赖以下文件：

- **配置文件**：
  - `config_initial.json`：初始模型验证配置
  - `config_refinement.json`：精细化模型验证配置

- **数据文件**：
  - `C2_Kokkos_Dataset/kokkos_create_context_initial.json`：初始训练数据集（用于验证）
  - `C2_Kokkos_Dataset/kokkos_create_context_refinement.json`：精细化训练数据集（用于验证）

- **模型文件**：
  - 基础模型：`basemodels/CodeLlama-7b-hf`
  - 微调适配器：`output/1_ChatKokko_initial_peft_adapter` 和 `output/1_ChatKokko_refinement_peft_adapter`

- **Python核心代码**：
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/__main__.py`：命令行入口
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py`：核心验证逻辑实现

### 验证核心逻辑

验证的核心逻辑位于 Python 文件 `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py` 中的 `App` 类的 `verify` 方法。该方法实现了完整的模型验证流程：

1. **加载数据集**：从配置的数据源加载训练数据集
2. **遍历验证数据**：对数据集中的每个样本进行模型推理
3. **生成响应**：使用模型生成针对每个样本的响应
4. **比较结果**：将模型响应与预期答案进行比较
5. **计算误差**：统计模型响应与预期答案不匹配的数量

### 核心文件调用方式

`chathpc` 命令通过以下方式调用核心验证逻辑：

1. **命令行入口**：`uv run` 命令执行 `chathpc` 命令，该命令映射到 `chathpc.app.__main__.py` 中的 `cli` 函数
2. **命令解析**：`cli` 函数解析命令行参数，识别到 `verify` 子命令后调用 `verify` 函数
3. **App 实例创建**：`verify` 函数使用指定的配置文件创建 `App` 实例
4. **模型加载**：`App` 实例调用 `load_finetuned_model` 方法加载未合并的微调模型
5. **验证方法调用**：`App` 实例调用 `verify` 方法，执行实际的验证流程

## 4_evaluate

### 执行流程分解

脚本 `4_evaluate.sh` 负责在验证数据集上评估两个版本的 ChatHPC for Kokkos 模型，并将结果转换为 Markdown 格式。其完整执行步骤如下：

1. **脚本初始化**：
   - 打印当前正在执行的脚本名称
   - 确保在脚本所在目录下执行后续命令

2. **初始模型评估**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json test --save_results_file evaluation/ChatHPCforKokkos_initial_results.json C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 使用 `config_initial.json` 配置文件
   - 评估数据：`C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 结果保存：`evaluation/ChatHPCforKokkos_initial_results.json`
   - 将 JSON 结果转换为 Markdown：`chathpc-data-to-md evaluation/ChatHPCforKokkos_initial_results.json > evaluation/ChatHPCforKokkos_initial_results.md`

3. **精细化模型评估**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_refinement.json test --save_results_file evaluation/ChatHPCforKokkos_refinement_results.json C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 使用 `config_refinement.json` 配置文件
   - 评估数据：`C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 结果保存：`evaluation/ChatHPCforKokkos_refinement_results.json`
   - 将 JSON 结果转换为 Markdown：`chathpc-data-to-md evaluation/ChatHPCforKokkos_refinement_results.json > evaluation/ChatHPCforKokkos_refinement_results.md`

### 依赖文件

执行 `4_evaluate.sh` 过程中会调用或依赖以下文件：

- **配置文件**：
  - `config_initial.json`：初始模型评估配置
  - `config_refinement.json`：精细化模型评估配置

- **数据文件**：
  - `C2_Kokkos_Dataset/kokkos_testing.yaml`：验证测试数据集

- **模型文件**：
  - 基础模型：`basemodels/CodeLlama-7b-hf`
  - 微调适配器：`output/1_ChatKokko_initial_peft_adapter` 和 `output/1_ChatKokko_refinement_peft_adapter`

- **Python核心代码**：
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/__main__.py`：命令行入口
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py`：核心评估逻辑实现
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/json_to_markdown.py`：JSON 转 Markdown 功能

### 评估核心逻辑

评估的核心逻辑位于 Python 文件 `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py` 中的 `App` 类的 `test` 方法。该方法实现了完整的模型评估流程：

1. **加载测试数据**：从指定的测试数据文件加载评估数据
2. **遍历测试数据**：对数据集中的每个样本进行模型推理
3. **生成响应**：使用模型生成针对每个样本的响应
4. **保存结果**：将模型响应与预期答案保存到指定文件
5. **计算准确率**：统计模型响应与预期答案匹配的比例

### 核心文件调用方式

`chathpc` 命令通过以下方式调用核心评估逻辑：

1. **命令行入口**：`uv run` 命令执行 `chathpc` 命令，该命令映射到 `chathpc.app.__main__.py` 中的 `cli` 函数
2. **命令解析**：`cli` 函数解析命令行参数，识别到 `test` 子命令后调用 `test` 函数
3. **App 实例创建**：`test` 函数使用指定的配置文件创建 `App` 实例
4. **模型加载**：`App` 实例调用 `load_finetuned_model` 方法加载未合并的微调模型
5. **评估方法调用**：`App` 实例调用 `test` 方法，执行实际的评估流程
6. **结果转换**：通过 `chathpc-data-to-md` 命令将 JSON 结果转换为 Markdown 格式

## 5_evaluate_baseline

### 执行流程分解

脚本 `5_evaluate_baseline.sh` 负责在验证数据集上评估基线模型，并将结果转换为 Markdown 格式。其完整执行步骤如下：

1. **脚本初始化**：
   - 打印当前正在执行的脚本名称
   - 确保在脚本所在目录下执行后续命令

2. **CodeLlama 基础模型评估**：
   - 执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json base test --save_results_file evaluation/code_llama_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 使用 `config_initial.json` 配置文件
   - 评估数据：`C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 结果保存：`evaluation/code_llama_base_results.json`
   - 将 JSON 结果转换为 Markdown：`chathpc-data-to-md evaluation/code_llama_base_results.json > evaluation/code_llama_base_results.md`

3. **OpenAI GPT-4o 模型评估**：
   - 检查 `OPENAI_API_KEY` 环境变量是否设置
   - 如果未设置，输出错误信息并退出
   - 如果已设置，执行命令：`uv run --project C1_ChatHPC_Lib/ChatHPC-app-v25.7.1 chathpc --config config_initial.json openai test --model gpt-4o --save_results_file evaluation/openai_gpt-4o_base_results.json C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 使用 `config_initial.json` 配置文件
   - 评估数据：`C2_Kokkos_Dataset/kokkos_testing.yaml`
   - 结果保存：`evaluation/openai_gpt-4o_base_results.json`
   - 将 JSON 结果转换为 Markdown：`chathpc-data-to-md evaluation/openai_gpt-4o_base_results.json > evaluation/openai_gpt-4o_base_results.md`

### 依赖文件

执行 `5_evaluate_baseline.sh` 过程中会调用或依赖以下文件：

- **配置文件**：
  - `config_initial.json`：基线模型评估配置

- **数据文件**：
  - `C2_Kokkos_Dataset/kokkos_testing.yaml`：验证测试数据集

- **模型文件**：
  - 基础模型：`basemodels/CodeLlama-7b-hf`（用于 CodeLlama 评估）
  - OpenAI GPT-4o：通过 API 调用，无需本地模型文件

- **Python核心代码**：
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/__main__.py`：命令行入口
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py`：核心评估逻辑实现
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/openai_interface.py`：OpenAI API 调用实现
  - `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/json_to_markdown.py`：JSON 转 Markdown 功能

### 评估核心逻辑

基线模型评估的核心逻辑位于 Python 文件 `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/src/chathpc/app/app.py` 中的 `App` 类的 `test` 方法，以及 `openai_interface.py` 中的 `ChatHPCOpenAI` 类的 `openai_chat_evaluate` 方法。该方法实现了完整的基线模型评估流程：

1. **加载测试数据**：从指定的测试数据文件加载评估数据
2. **遍历测试数据**：对数据集中的每个样本进行模型推理
3. **生成响应**：
   - 对于 CodeLlama 基础模型：使用本地模型生成响应
   - 对于 OpenAI GPT-4o：通过 API 调用生成响应
4. **保存结果**：将模型响应与预期答案保存到指定文件
5. **计算准确率**：统计模型响应与预期答案匹配的比例

### 核心文件调用方式

`chathpc` 命令通过以下方式调用核心评估逻辑：

1. **命令行入口**：`uv run` 命令执行 `chathpc` 命令，该命令映射到 `chathpc.app.__main__.py` 中的 `cli` 函数
2. **命令解析**：`cli` 函数解析参数，识别到 `base test` 或 `openai test` 子命令后调用相应函数
3. **App 实例创建**：`base_test` 或 `openai_test` 函数使用指定的配置文件创建 `App` 实例
4. **模型加载**：
   - 对于 CodeLlama 基础模型：调用 `load_base_model` 方法加载基础模型
   - 对于 OpenAI GPT-4o：无需本地模型加载
5. **评估方法调用**：`App` 实例调用 `test` 方法，执行实际的评估流程
6. **结果转换**：通过 `chathpc-data-to-md` 命令将 JSON 结果转换为 Markdown 格式