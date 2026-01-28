"""App Module: used to construct an app for training ChatHPC LLMs."""

# 从未来版本导入类型注解支持
from __future__ import annotations

# 导入标准库模块
import atexit
import os
import readline
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

# 导入第三方库
import jinja2
import torch
from loguru import logger
from peft import (
    LoraConfig,  # type: ignore
    PeftModel,  # type: ignore
    get_peft_model,  # type: ignore
    prepare_model_for_kbit_training,  # type: ignore
)
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
from pytz import timezone
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

# 导入ChatHPC相关模块
import chathpc
import chathpc.app
from chathpc.app.json_to_markdown import json_yaml_to_markdown
from chathpc.app.ollama_interface import ollama_chat_evaluate
from chathpc.app.openai_interface import ChatHPCOpenAI
from chathpc.app.siliconflow_interface import ChatHPCSiliconFlow
from chathpc.app.utils import template_utils
from chathpc.app.utils.common_utils import load_json_yaml_arg, run
from chathpc.app.utils.datastore import save_json, save_md
from chathpc.app.utils.template_utils import map_keywords
from chathpc.app.utils.verify_utils import ignore_minor

# 定义默认应用配置文件路径
DEFAULT_APP_CONFIG_FILE = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "config/default_app_settings.json"))
)


class AppConfig(BaseSettings):
    """Configuration settings for the ChatHPC application.

    This class inherits from Pydantic BaseSettings to manage application configuration
    through multiple sources with a defined priority order.

    Attributes:
        data_file (Path): Training data JSON file path.
        base_model_path (Path): Pre-trained base LLM model directory.
        finetuned_model_path (Path): Directory for fine-tuned model layers.
        merged_model_path (Path): Directory for complete merged model.
        training_output_dir (Path): Directory for training output and checkpoints.
        max_training_tokens (int): Maximum tokens for training set tokenization.
        max_response_tokens (int): Maximum tokens for model response generation.
        prompt_history_file (Path): File path for interactive chat history.
        prompt_template_file (Path): File containing prompt template for training/inference.
        prompt_template (str): Direct string template for prompts.
        use_wandb (bool): Enable/disable Weights & Biases logging.

    Configuration Priority:
        1. Environment variables (CHATHPC_ prefix)
        2. .env file
        3. Direct initialization
        4. JSON config file
        5. File secrets

    Example:
        ```python
        config = AppConfig(base_model_path="/path/to/model")
        config = AppConfig.from_json("config.json")
        ```

    Note:
        - All paths are handled as Path objects
        - UTF-8 encoding used for all file operations
        - Either prompt_template_file or prompt_template must be set
    """

    # 训练数据文件路径
    data_file: Path = Field(..., description="Path to the JSON file containing training data for model fine-tuning.")
    # 基础模型路径
    base_model_path: Path = Field(
        Path("/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf"),
        description="Path to the pre-trained base LLM model directory.",
    )
    # 微调模型路径
    finetuned_model_path: Path = Field(
        Path("peft_adapter"), description="Path where fine-tuned model layers will be saved."
    )
    # 合并模型路径
    merged_model_path: Path = Field(
        Path("merged_adapters"), description="Path where the complete merged model will be saved."
    )
    # 训练输出目录
    training_output_dir: Path = Field(
        Path("training_checkpoints"), description="Path where training output will be saved."
    )
    # 最大训练 tokens
    max_training_tokens: int = Field(
        512, gt=0, description="Maximum number of tokens to use to tokenize the training sets."
    )
    # 最大响应 tokens
    max_response_tokens: int = Field(600, gt=0, description="Maximum number of tokens to generate in model responses.")
    # 提示历史文件
    prompt_history_file: Path = Field(
        Path("~/.chathpc_history"), description="Path to the file containing interactive prompt history."
    )
    # 提示模板文件
    prompt_template_file: Path | None = Field(
        None, description="Path to the prompt template to use for training and inference."
    )
    # 提示模板字符串
    prompt_template: str | None = Field(
        None, description="Path to the prompt template to use for training and inference."
    )
    # 自动导出为 Markdown
    auto_export_markdown: bool = Field(False, description="Auto export output files to markdown.")
    # 使用 Weights & Biases
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases for logging.")

    # 模型配置
    model_config = SettingsConfigDict(
        # cli_parse_args=True,
        env_prefix="CHATHPC_",  # 环境变量前缀
        env_file=".env",  # 环境变量文件
        env_file_encoding="utf-8",  # 环境变量文件编码
        # json_file=DEFAULT_APP_CONFIG_FILE,
        json_file_encoding="utf-8",  # JSON文件编码
        extra="allow",  # 允许额外的配置项
    )

    # 提示模板验证器
    @model_validator(mode="before")
    @classmethod
    def check_for_prompt_template(cls, values):
        """Validate prompt template configuration.

        This validator ensures that exactly one of prompt_template_file or prompt_template
        is set in the configuration. Having both or neither is invalid.

        Args:
            values (dict): Dictionary of configuration values to validate.

        Returns:
            dict: The validated configuration values.

        Raises:
            ValueError: If neither or both prompt template options are set.

        Example:
            Valid configurations:
            - prompt_template_file set, prompt_template None
            - prompt_template set, prompt_template_file None

            Invalid configurations:
            - Both prompt_template and prompt_template_file set
            - Neither prompt_template nor prompt_template_file set
        """
        # 检查是否设置了提示模板文件或提示模板字符串
        if not (bool(values.get("prompt_template_file")) | bool(values.get("prompt_template"))):
            raise ValueError("Either prompt_template_file or prompt_template must be set.")
        # 检查是否同时设置了提示模板文件和提示模板字符串
        if bool(values.get("prompt_template_file")) & bool(values.get("prompt_template")):
            raise ValueError("prompt_template_file and prompt_template should not both be set.")
        return values

    # 自定义配置源顺序
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 返回配置源顺序：环境变量 > .env文件 > 初始化参数 > JSON配置文件 > 文件密钥
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # 从JSON创建配置实例
    @classmethod
    def from_json(cls, json_or_file: str | Path | dict, extra_params: str | Path | dict | None = None) -> AppConfig:
        """Create an AppConfig instance from JSON configuration sources.

        This class method creates an AppConfig instance by combining settings from a primary
        JSON source and optional additional parameters.

        Args:
            json_or_file (Union[str, Path, dict]): Primary configuration source - either a
                path to a JSON file or a dictionary with configuration values.
            extra_params (Union[str, Path, dict], optional): Additional configuration source
                to override or supplement primary settings.

        Returns:
            AppConfig: A new AppConfig instance initialized with combined settings.

        Example:
            ```python
            # From JSON file
            config = AppConfig.from_json("config.json")

            # With extra parameters
            config = AppConfig.from_json("config.json", {"max_response_tokens": 800})

            # From dictionary
            config = AppConfig.from_json({"data_file": "data.json"})
            ```

        Note:
            When both sources are provided, settings from extra_params override
            corresponding values from the primary source.
        """
        # 加载JSON配置
        json_config = load_json_yaml_arg(json_or_file)
        # 加载额外参数
        extra_config = load_json_yaml_arg(extra_params)
        # 更新配置
        json_config.update(extra_config)
        # 创建并返回配置实例
        return cls(**json_config)


class App:
    """Main application class for ChatHPC Application.

    This class handles the initialization, loading, and management of models,
    datasets, and training processes for the ChatHPC application. It provides
    methods for loading different types of models, evaluating prompts, and
    fine-tuning the model.

    Attributes:
        config (AppConfig): Configuration settings for the application.
        tokenizer: Tokenizer for processing input text.
        model: The language model used for text generation and fine-tuning.
        train_dataset: Dataset used for training.
        eval_dataset: Dataset used for evaluation.
        tokenized_train_dataset: Tokenized version of training dataset.
        tokenized_val_dataset: Tokenized version of validation dataset.
        peft_config: Configuration for LoRA fine-tuning.
        training_args: Arguments for model training.

    Methods:
        load_base_model(): Loads the base LLM model.
        load_finetuned_model(): Loads a model with fine-tuned layers.
        load_merged_model(): Loads a complete merged model.
        load_datasets(): Loads training and evaluation datasets.
        evaluate_model(): Generates responses for given prompts.
        chat_prompt(): Creates formatted prompts for questions.
        chat_evaluate(): Evaluates questions with context.
        tokenize_training_set(): Prepares datasets for training.
        train(): Executes model fine-tuning process.
        interactive(): Starts interactive chat session.
        print_config(): Displays current configuration settings.
    """

    # 初始化应用实例
    def __init__(self, app_config: AppConfig | None = None):
        """Initialize the ChatHPC application instance.

        This method sets up a new application instance with configuration settings
        and initializes the Jinja2 environment for template processing.

        Args:
            app_config (AppConfig, optional): Application configuration settings.
                If None, creates default AppConfig instance.

        Sets:
            - self.config: Application configuration settings
            - self.jinja: Jinja2 environment for template processing

        Example:
            ```python
            # With default settings
            app = App()

            # With custom settings
            config = AppConfig(base_model_path="/path/to/model")
            app = App(app_config=config)
            ```

        Note:
            Model loading and other initializations must be performed explicitly
            by calling the appropriate methods after initialization.
        """
        # 如果未提供配置，创建默认配置
        if app_config is None:
            app_config = AppConfig()  # type: ignore

        # 保存配置
        self.config = app_config

        # 初始化Jinja2环境
        self.jinja = jinja2.Environment(autoescape=False, keep_trailing_newline=True)  # noqa: S701
        # 加载模板
        self._load_templates()

    # 从JSON创建应用实例
    @classmethod
    def from_json(cls, json_or_file: str | Path | dict, extra_params: str | Path | dict | None = None) -> App:
        """Create an App instance from JSON configuration sources.

        This class method creates an App instance by combining settings from a primary
        JSON source and optional additional parameters.

        Args:
            json_or_file (Union[str, Path, dict]): Primary configuration source - either a
                path to a JSON file or a dictionary with configuration values.
            extra_params (Union[str, Path, dict], optional): Additional configuration source
                to override or supplement primary settings.

        Returns:
            App: A new App instance initialized with combined settings.

        Example:
            ```python
            # From JSON file
            app = App.from_json("config.json")

            # With extra parameters
            app = App.from_json("config.json", {"max_response_tokens": 800})

            # From dictionary
            app = App.from_json({"data_file": "data.json"})
            ```

        Note:
            When both sources are provided, settings from extra_params override
            corresponding values from the primary source.
        """
        # 从JSON创建配置
        config = AppConfig.from_json(json_or_file, extra_params=extra_params)
        # 创建并返回应用实例
        return cls(app_config=config)

    # 加载模板
    def _load_templates(self):
        """Load and initialize prompt templates for training and inference.

        This method loads prompt templates either from a file or a string configuration,
        processes them for training and inference use, and initializes Jinja2 templates.

        The templates are split into prefix and postfix components around the response
        section for proper formatting during training and inference.

        Raises:
            ValueError: If neither prompt_template nor prompt_template_file is properly configured
            ValueError: If the specified prompt template file does not exist

        Sets:
            - self.training_template: Complete Jinja2 template for training
            - self.inference_template: Prefix template for inference
            - self.postfix_template: Postfix template for inference
            - self._prompt_prefix: Raw prefix string
            - self._prompt_postfix: Raw postfix string

        Example:
            ```python
            app = App(config)
            app._load_templates()  # Templates are loaded during initialization
            ```

        Note:
            This method is called automatically during App initialization and should
            not typically be called directly.
        """
        # 初始化相对路径
        relative_path = None
        # 检查是否需要使用相对路径
        if (
            hasattr(self.config, "filename")
            and self.config.prompt_template is None
            and self.config.prompt_template_file is not None
            and not self.config.prompt_template_file.is_absolute()
        ):
            # 获取配置文件所在目录
            filename = Path(self.config.filename)  # type: ignore
            # 计算相对路径
            relative_path = filename.parent / self.config.prompt_template_file

        # 检查是否直接提供了模板字符串
        if self.config.prompt_template is not None:
            # 使用直接提供的模板字符串
            prompt_template_string = self.config.prompt_template

        else:
            # 检查模板文件是否设置
            if self.config.prompt_template_file is None:
                raise ValueError("Unexpected Error: Prompt template file is not set.")

            # 检查模板文件是否存在
            if self.config.prompt_template_file.is_file():
                # 加载模板文件
                logger.info("Loading prompt template from {file}", file=self.config.prompt_template_file)
                with open(self.config.prompt_template_file) as f:
                    prompt_template_string = f.read()
            # 检查相对路径是否存在
            elif relative_path is not None and relative_path.is_file():
                # 加载相对路径的模板文件
                logger.info("Loading prompt template from {file}", file=relative_path)
                with open(relative_path) as f:
                    prompt_template_string = f.read()
            else:
                # 模板文件不存在
                raise ValueError("Prompt template file not found.")

        # 标准化模板
        prompt_template_string = template_utils.normalize_template(prompt_template_string)
        # 保存标准化后的模板
        self.config.prompt_template = prompt_template_string

        # 创建训练模板
        self.training_template = self.jinja.from_string(prompt_template_string)
        # 分割模板为前缀和后缀
        self._prompt_prefix, self._prompt_postfix = template_utils.split_on_response(prompt_template_string)
        # 创建推理模板
        self.inference_template = self.jinja.from_string(self._prompt_prefix)
        # 创建后缀模板
        self.postfix_template = self.jinja.from_string(self._prompt_postfix)

    def load_base_model(self) -> None:
        """Load and initialize the base Large Language Model.

        This method initializes both the tokenizer and model from the base model path
        specified in the application preferences. The model is loaded with specific
        configurations for optimal performance.

        Requires:
            - preferences.base_model_path must be set to a valid model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM in float16 precision

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/model"
            >>> app.load_base_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        # 加载基础模型
        logger.info("Loading the base model from {path}", path=self.config.base_model_path)

        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        # 初始化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            load_in_8bit=False,  # 不使用8位量化
            torch_dtype=torch.float16,  # 使用float16精度
            device_map="auto",  # 自动设备映射
            # device_map={'':torch.cuda.current_device()}
        )

    def load_finetuned_model(self) -> None:
        """Load and initialize the finetuned Large Language Model.

        This method loads a finetuned model by first initializing the base model and tokenizer,
        then loading the finetuned layers on top of it using PeftModel.

        Requires:
            - preferences.base_model_path must be set to a valid base model path
            - preferences.finetuned_model_path must be set to a valid finetuned model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized PeftModel with finetuned layers

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.finetuned_model_path = "path/to/finetuned/model"
            >>> app.load_finetuned_model()
            ```

        Note:
            This method first calls load_base_model() to initialize the foundation model
            before applying the finetuned layers.
        """

        # 加载微调模型
        logger.info("Loading the finetuned model from {path}", path=self.config.finetuned_model_path)

        # 先加载基础模型
        self.load_base_model()

        # 加载微调层
        self.model = PeftModel.from_pretrained(self.model, self.config.finetuned_model_path)  # type: ignore

    def load_merged_model(self) -> None:
        """Load and initialize the merged Large Language Model.

        This method loads a complete merged model that combines the base model with
        finetuned layers into a single model file. The tokenizer is initialized from
        the base model path while the full model is loaded from the merged model path.

        Requires:
            - preferences.base_model_path must be set to a valid base model path for tokenizer
            - preferences.merged_model_path must be set to a valid merged model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM with merged weights

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.merged_model_path = "path/to/merged/model"
            >>> app.load_merged_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        # 加载合并模型
        logger.info("Loading the merged model from {path}", path=self.config.merged_model_path)

        # 从基础模型加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        # 加载合并后的模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.merged_model_path,
            load_in_8bit=False,  # 不使用8位量化
            torch_dtype=torch.float16,  # 使用float16精度
            device_map="auto",  # 自动设备映射
            # device_map={'':torch.cuda.current_device()}
        )

    def load_datasets(self) -> None:
        """Load training and evaluation datasets from a JSON file.

        This method loads datasets from the JSON file specified in the application preferences.
        The datasets are loaded using the Hugging Face datasets library and split into
        training and evaluation sets.

        Config:
            preferences.data_file (str): Path to the JSON file containing the datasets.

        Sets:
            self.train_dataset: Dataset object for training
            self.eval_dataset: Dataset object for evaluation

        Requires:
            - The data file must be in JSON format
            - The data file path must be set in preferences.data_file
        """
        # 加载数据集
        logger.info("Loading the dataset from {path}", path=self.config.data_file)

        # 导入datasets库
        from datasets import load_dataset

        # 加载训练数据集
        self.train_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")
        # 加载评估数据集
        self.eval_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")

    def evaluate_model(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate a model response for a given input prompt.

        Args:
            prompt (str): Input text prompt for model evaluation.
            max_new_tokens (int|None): Maximum tokens to generate. Defaults to config.max_response_tokens.

        Returns:
            str: Generated text response from the model.

        Requires:
            - Initialized model via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - Initialized tokenizer

        Example:
            ```python
            app = App()
            app.load_base_model()
            response = app.evaluate_model("What is Kokkos?", max_new_tokens=100)
            print(response)  # "Kokkos is a programming model..."
            ```

        Note:
            Uses evaluation mode and torch.no_grad() for inference.
            Input is processed on CUDA if available.
        """
        # 处理输入提示
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # 确定最大生成token数
        if max_new_tokens is None:
            max_new_tokens = self.config.max_response_tokens

        # 设置模型为评估模式
        self.model.eval()  # type: ignore
        # 禁用梯度计算
        with torch.no_grad():
            # 生成响应
            output = self.model.generate(  # type: ignore
                **model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
            )[0]
            # 解码响应
            return self.tokenizer.decode(output)

    def chat_prompt(self, **kwargs) -> str:
        """Create a formatted prompt for chat questions.

        This method generates a structured prompt using the inference template by combining
        provided keyword arguments according to the template defined in the application
        configuration.

        Args:
            **kwargs: Keyword arguments to be passed to the template.
                Common arguments include:
                - question (str): The question to be answered
                - context (str): Supporting context or documentation
                Additional arguments can be used if defined in the template.

        Returns:
            str: A formatted prompt string following the inference template.

        Requires:
            - Initialized inference_template via _load_templates()
            - Template must be properly formatted with expected variables

        Example:
            ```python
            app = App()
            prompt = app.chat_prompt(
                question="How do I use Views?",
                context="Views are memory spaces in Kokkos...",
            )
            print(prompt)  # Returns formatted prompt based on template
            ```

        Note:
            - The actual prompt format is determined by the inference template loaded during initialization
            - Keywords are automatically mapped using template_utils.map_keywords()
            - This method is typically used internally by chat_evaluate()
        """

        # 生成格式化的提示
        return self.inference_template.render(**template_utils.map_keywords(kwargs))

    def chat_evaluate(self, **kwargs) -> str:
        """Evaluate a question with provided context using the model.

        This method processes a question-context pair through the model by:
        1. Formatting the input using the inference template
        2. Generating a response using the model
        3. Returning both the response and original prompt

        Args:
            question (str): The question to be answered by the model.
            **kwargs: Additional keyword arguments passed to evaluate_model().
                Common arguments include:
                - max_new_tokens (int): Override default token generation limit
                - Other template variables defined in prompt template

        Returns:
            str: Generated model response.

        Requires:
            - Initialized model via one of load methods:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - Initialized tokenizer and templates

        Example:
            ```python
            app = App()
            app.load_merged_model()
            response = app.chat_evaluate(
                question="What is Kokkos?",
                context="Kokkos is a performance portable programming model...",
                max_new_tokens=200,
            )
            print(response)  # Prints model's explanation of Kokkos
            ```

        Note:
            - Uses chat_prompt() for template-based input formatting
            - Uses evaluate_model() for response generation
            - Response format follows inference template structure
            - Template variables can be passed via kwargs
        """
        # 生成格式化提示
        prompt = self.chat_prompt(**kwargs)
        # 评估模型并返回响应
        return self.evaluate_model(prompt)

    def chat_evaluate_extract(self, **kwargs) -> str:
        """Extract the model's answer from a chat evaluation response.

        This method combines chat_evaluate() with answer extraction, removing template
        formatting and returning only the model's direct response.

        Args:
            **kwargs: Keyword arguments passed to chat_evaluate().
                Common arguments include:
                - question (str): The question to be answered
                - context (str): Supporting context or documentation
                - max_new_tokens (int): Override default token generation limit
                Additional arguments can be used if defined in the template.

        Returns:
            str: The extracted answer from the model's response, without template formatting.

        Example:
            ```python
            app = App()
            app.load_merged_model()
            answer = app.chat_evaluate_extract(
                question="What is Kokkos?", context="Kokkos is a programming model..."
            )
            print(answer)  # Prints just the model's answer without template
            ```

        Note:
            - Uses chat_evaluate() for response generation
            - Automatically extracts the answer portion using template structure
            - More concise than chat_evaluate() for direct answer retrieval
        """
        # 生成完整响应
        chat_response = self.chat_evaluate(**kwargs)
        # 提取答案
        return self.extract_answer(chat_response, **kwargs)

    def training_prompt(self, **kwargs) -> str:
        """Create a formatted prompt for training data.

        This method generates a structured prompt using the training template by combining
        provided keyword arguments according to the template defined in the application
        configuration.

        Args:
            **kwargs: Keyword arguments to be passed to the template.
                Common arguments include:
                - question (str): The question to be used in training
                - context (str): Supporting context or documentation
                - answer (str): The expected answer or response
                Additional arguments can be used if defined in the template.

        Returns:
            str: A formatted prompt string following the training template.

        Requires:
            - Initialized training_template via _load_templates()
            - Template must be properly formatted with expected variables

        Example:
            ```python
            app = App()
            prompt = app.training_prompt(
                question="How do I use Views?",
                context="Views are memory spaces in Kokkos...",
                answer="To use Views in Kokkos...",
            )
            print(prompt)  # Returns formatted prompt based on template
            ```

        Note:
            - The actual prompt format is determined by the training template loaded during initialization
            - Keywords are automatically mapped using template_utils.map_keywords()
            - This method is typically used internally by tokenize_training_set()
        """

        # 生成训练提示
        return self.training_template.render(**template_utils.map_keywords(kwargs))

    def tokenize_training_set(self) -> None:
        """Tokenize the training and validation datasets.

        This method processes the loaded datasets by tokenizing text data for model training.
        It handles padding configuration and EOS token management during tokenization.

        Requires:
            - Initialized datasets via load_datasets()
            - Initialized tokenizer via loading a model

        Sets:
            - self.tokenized_train_dataset: Processed training dataset
            - self.tokenized_val_dataset: Processed validation dataset

        Example:
            ```python
            app = App()
            app.load_base_model()
            app.load_datasets()
            app.tokenize_training_set()
            ```

        Note:
            - The method uses the training_prompt template from config to format inputs before tokenization.
            - This method also handles padding token configuration and adds/removes EOS tokens as needed for the tokenization process.
        """
        # 设置填充token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        # 定义分词函数
        def tokenize(prompt):
            # 分词并截断
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_training_tokens,
                padding=False,
                return_tensors=None,
            )
            # 完整分词（用于比较）
            result_full = self.tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            # 检查是否被截断
            if result != result_full:
                # 记录警告
                logger.warning(
                    "Training tokenizer needs {token_count} tokens to fully tokenize the training input and max training tokens is set to {max_training_tokens}. \nPrompt: {prompt}\nCropped to: {prompt_cropped}",
                    token_count=len(result_full.data["input_ids"]),
                    max_training_tokens=self.config.max_training_tokens,
                    prompt=prompt,
                    prompt_cropped=self.tokenizer.decode(result.data["input_ids"]),
                )

            # 自监督学习：标签就是输入
            result["labels"] = result["input_ids"].copy()  # type: ignore

            return result

        # 定义生成和分词提示的函数
        def generate_and_tokenize_prompt(data_point):
            # 生成完整提示
            full_prompt = self.training_prompt(**data_point)
            # 分词
            return tokenize(full_prompt)

        # 添加EOS token
        self.tokenizer.add_eos_token = True

        # 分词训练数据集
        self.tokenized_train_dataset = self.train_dataset.map(generate_and_tokenize_prompt)
        # 分词验证数据集
        self.tokenized_val_dataset = self.eval_dataset.map(generate_and_tokenize_prompt)

        # 恢复EOS token设置
        self.tokenizer.add_eos_token = False

    def train(self):
        """Train the model using fine-tuning layers.

        训练流程详解：
        1. 数据预处理：
           - 加载训练数据集和验证数据集（通过load_datasets()方法）
           - 对数据集进行标记化处理（通过tokenize_training_set()方法）
           - 准备数据加载器和数据碰撞器

        2. 模型初始化：
           - 配置LoRA参数（LoraConfig）
           - 设置模型为训练模式
           - 准备模型用于k位训练
           - 获取PEFT模型

        3. 训练循环：
           - 配置训练参数（TrainingArguments）
           - 创建Trainer实例
           - 编译模型（如果支持）
           - 执行训练过程

        4. 评估验证：
           - 在训练过程中定期评估模型性能
           - 基于评估结果调整模型参数

        5. 文件生成机制：
           - 触发条件：训练完成后自动生成
           - 数据来源：训练过程中的模型参数和配置信息
           - 保存机制：
             * 微调模型参数保存到config.finetuned_model_path
             * 生成README.md文件，记录训练信息
             * 注释掉的代码显示了合并模型的保存路径（config.merged_model_path）

        6. 补丁参数文件结构：
           - 由PEFT库自动生成，包含以下字段：
             * lora_alpha: LoRA缩放因子，类型为整数
             * lora_dropout: 丢弃率，类型为浮点数
             * r: LoRA秩，类型为整数
             * bias: 偏置处理方式，类型为字符串
             * task_type: 任务类型，类型为字符串
             * target_modules: 目标模块列表，类型为字符串数组
             * modules_to_save: 要保存的模块，类型为字符串数组
             * inference_mode: 推理模式，类型为布尔值

        7. PEFT（Parameter-Efficient Fine-Tuning）详解：
           - 含义：参数高效微调，一种只微调模型部分参数的技术
           - 核心原理：
             * 冻结原始模型参数
             * 在模型的关键层（如注意力机制）中插入小型可训练组件
             * 只训练这些新增的小型组件
             * 显著减少可训练参数数量
           - 应用方式：
             * 使用LoRA（Low-Rank Adaptation）方法
             * 在模型的q_proj、k_proj、v_proj、o_proj等关键模块中应用
             * 通过prepare_model_for_kbit_training和get_peft_model实现

        训练依赖关系分析：
        1. 执行2_train.sh时，第二步refinement训练基于：
           - 直接使用基础模型参数重新开始训练
           - 原因：每次训练都会调用load_base_model()加载基础模型

        2. 代码实现中的训练依赖关系：
           - 体现：每次训练都从基础模型开始，没有加载之前训练的模型
           - 工作原理：
             * 训练前通过load_base_model()加载原始基础模型
             * 应用LoRA配置创建新的PEFT模型
             * 训练完成后保存到指定路径，但后续训练不会自动加载

        Requires:
            - App.load_datasets() must be called first to load training data
            - App.load_base_model() must be called first to load the base model
            - Tokenizer and model must be properly initialized

        Sets:
            - self.peft_config: LoRA configuration for fine-tuning
            - self.training_args: Training arguments for the Trainer
            - self.model: Updated model after training

        Saves:
            - Finetuned model layers to preferences.finetuned_model_path
            - Complete merged model to preferences.merged_model_path

        Note:
            This method uses Hugging Face's Trainer for the training process and
            supports multi-GPU training when available. It also integrates with
            Weights & Biases (wandb) for experiment tracking.
        """

        # 配置LoRA
        self.peft_config = LoraConfig(
            lora_alpha=16,  # LoRA alpha参数
            lora_dropout=0.05,  # LoRA dropout参数
            r=16,  # LoRA秩
            bias="none",  # 无偏置
            task_type="CAUSAL_LM",  # 任务类型
            target_modules=[  # 目标模块
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )
        # 设置模型为训练模式
        self.model.train()  # type: ignore # put model back into training mode
        # 准备模型用于k位训练
        self.model = prepare_model_for_kbit_training(self.model)
        # 获取PEFT模型
        self.model = get_peft_model(self.model, self.peft_config)
        # 打印可训练参数
        self.model.print_trainable_parameters()

        # 设置批处理大小
        batch_size = 128
        per_device_train_batch_size = 32
        # 计算梯度累积步数
        gradient_accumulation_steps = batch_size // per_device_train_batch_size
        # 获取输出目录
        output_dir = self.config.training_output_dir.as_posix()

        # 设置wandb项目
        wandb_project = "ChatHPC"
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project

        # 处理多GPU情况
        if torch.cuda.device_count() > 1:
            # 防止Trainer尝试自己的DataParallelism
            print("multiple gpus detected!")
            self.model.is_parallelizable = True  # type: ignore
            self.model.model_parallel = True  # type: ignore

        # 设置训练参数
        self.training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,  # 预热步数
            max_steps=400,  # 最大步数
            # max_steps=20,
            learning_rate=3e-4,  # 学习率
            fp16=True,  # 使用float16
            logging_steps=10,  # 日志步数
            optim="adamw_torch",  # 优化器
            eval_strategy="steps",  # 评估策略
            save_strategy="steps",  # 保存策略
            eval_steps=20,  # 评估步数
            save_steps=20,  # 保存步数
            output_dir=output_dir,  # 输出目录
            # save_total_limit=3,
            load_best_model_at_end=False,  # 不加载最佳模型
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True,  # 按长度分组以加速训练
            report_to="wandb" if self.config.use_wandb else "none",  # 报告目标
            run_name=f"codellama-{datetime.now(tz=timezone('EST')).strftime('%Y-%m-%d-%H-%M')}",  # 运行名称
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,  # type: ignore
            eval_dataset=self.tokenized_val_dataset,  # type: ignore
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        # 禁用缓存
        self.model.config.use_cache = False  # type: ignore

        # 编译模型（如果支持）
        if torch.__version__ >= "2" and sys.platform != "win32":
            print("compiling the model")
            self.model = torch.compile(self.model)

        # 开始训练
        trainer.train()

        # 保存模型
        print("Saving Model...")
        trainer.model.save_pretrained(self.config.finetuned_model_path)  # type: ignore
        # 保存README
        self.save_readme(self.config.finetuned_model_path)
        # 注释掉的模型合并代码
        # print("Merging model...")
        # self.model = trainer.model.merge_and_unload()  # type: ignore
        # print("Saving merged model...")
        # self.tokenizer.save_pretrained(self.config.merged_model_path)
        # self.model.save_pretrained(self.config.merged_model_path)
        # print("Saving README.md...")
        # self.save_readme(self.config.merged_model_path)

    def interactive(self, args, prompt="chathpc") -> None:
        """Start an interactive chat session with the model.

        This method provides a command-line interface for interacting with the model.
        It maintains a command history and supports context setting for conversations.

        Commands:
            /bye: Exit the interactive session
            /context: Set a new context for subsequent questions

        Args:
            prompt (str, optional): The prompt prefix to display. Defaults to "chathpc".

        Requires:
            - A model must be loaded via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - The tokenizer must be initialized

        Example:
            ```python
            >>> app = App()
            >>> app.load_merged_model()
            >>> app.interactive()
            chathpc ()> What is Kokkos?
        """
        # 处理历史文件
        history_file = self.config.prompt_history_file.expanduser().as_posix()
        try:
            # 读取历史文件
            readline.read_history_file(history_file)
            # 获取历史长度
            h_len = readline.get_current_history_length()
        except FileNotFoundError:
            # 创建历史文件
            open(history_file, "wb+").close()
            # 添加默认命令
            readline.add_history("/context")
            readline.add_history("/bye")
            # 获取历史长度
            h_len = readline.get_current_history_length()

        # 定义保存历史的函数
        def save_history(prev_h_len, histfile):
            # 获取新历史长度
            new_h_len = readline.get_current_history_length()
            # 设置历史长度
            readline.set_history_length(1000)
            # 追加新历史
            readline.append_history_file(new_h_len - prev_h_len, histfile)

        # 注册退出时保存历史
        atexit.register(save_history, h_len, history_file)

        # 初始化上下文
        context = None
        # 打印提示信息
        print("Use '/bye' to exit.\nUse '/context' to set context.")
        # 主循环
        while True:
            # 构建提示行
            prompt_line = f"{prompt} ({context})> " if context is not None else f"{prompt}> "
            # 获取用户输入
            user_input = input(prompt_line)
            # 处理退出命令
            if user_input == "/bye":
                print("Goodbye!")
                break
            # 处理上下文命令
            if user_input.startswith("/context"):
                # 提取上下文
                context = user_input.replace("/context", "").strip()
                # 如果上下文为空，提示用户输入
                if context == "":
                    context = input("Context: ")
                # 如果上下文为空，设置为None
                if context.strip() == "":
                    context = None
                continue
            # 处理普通输入
            if args.extract:
                # 提取回答
                print(self.chat_evaluate_extract(question=user_input, context=context))
            else:
                # 直接回答
                print(self.chat_evaluate(question=user_input, context=context))

    def verify(
        self,
        save_verify_data_path: str | Path | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
        siliconflow_model: str | None = None,
    ) -> int:
        """验证模型输出与训练数据集的匹配度

        此方法通过将模型输出与训练数据集进行比较，对模型进行验证测试。
        支持使用本地模型或外部模型（Ollama、OpenAI、SiliconFlow）进行验证。

        参数:
            save_verify_data_path (Union[str, Path, None]): 保存验证结果的可选路径。如果提供，
                验证结果将保存为JSON文件，并在配置了auto_export_markdown时保存为Markdown文件。
            ollama_model (str, optional): Ollama模型名称，如果使用Ollama而不是应用程序的模型。
            openai_model (str, optional): OpenAI模型名称，如果使用OpenAI而不是应用程序的模型。
            siliconflow_model (str, optional): SiliconFlow模型名称，如果使用SiliconFlow而不是应用程序的模型。

        返回值:
            int: 验证过程中发现的错误数量。值为0表示所有测试用例都通过验证，
                大于0的值表示有对应数量的测试用例未通过验证。

        异常:
            RuntimeError: 当同时指定了多个外部模型时抛出，例如同时设置了ollama_model和openai_model。
            ValueError: 当使用外部模型但缺少必要的配置（如API密钥）时，可能由客户端初始化时抛出。

        关键实现逻辑:
            1. 验证模型选择的唯一性（只能选择一个外部模型）
            2. 初始化所选外部模型的客户端（如果有）
            3. 遍历训练数据集中的每个数据点
            4. 对每个数据点进行字段映射，确保格式标准化
            5. 使用选定的模型生成响应
            6. 构建包含提示、上下文、问题、预期答案和模型响应的数据点
            7. 保存验证结果（如果指定了保存路径）
            8. 比较模型响应与预期答案，统计错误数量
            9. 打印错误详情和总数
            10. 返回错误数量

        使用示例:
            ```python
            # 使用本地微调模型进行验证
            app = App()
            app.load_merged_model()
            errors = app.verify(save_verify_data_path="verification_results.json")
            print(f"验证完成，发现 {errors} 个错误")

            # 使用Ollama模型进行验证
            errors = app.verify(
                save_verify_data_path="ollama_verification.json",
                ollama_model="llama3"
            )

            # 使用OpenAI模型进行验证
            errors = app.verify(
                save_verify_data_path="openai_verification.json",
                openai_model="gpt-4"
            )

            # 使用SiliconFlow模型进行验证
            errors = app.verify(
                save_verify_data_path="siliconflow_verification.json",
                siliconflow_model="Pro/deepseek-ai/DeepSeek-V3.2"
            )
            ```

        注意事项:
            - 验证使用的是训练数据集，因此结果可能存在过拟合的情况
            - 外部模型需要相应的配置（如API密钥）才能正常工作
            - 保存路径如果不包含.json扩展名，会自动添加
            - 错误比较使用ignore_minor函数，会忽略轻微的差异
        """
        # 初始化验证数据
        verify_data = []

        # 验证模型选择
        if ollama_model is not None and openai_model is not None:
            raise RuntimeError("Both Ollama model and OpenAI model cannot both be set. Only one should be set.")
        if ollama_model is not None and siliconflow_model is not None:
            raise RuntimeError("Both Ollama model and SiliconFlow model cannot both be set. Only one should be set.")
        if openai_model is not None and siliconflow_model is not None:
            raise RuntimeError("Both OpenAI model and SiliconFlow model cannot both be set. Only one should be set.")

        # 初始化客户端
        openai_client = ChatHPCOpenAI(self.config) if openai_model is not None else None
        siliconflow_client = ChatHPCSiliconFlow(self.config) if siliconflow_model is not None else None

        # 遍历训练数据集
        for i, item in tqdm(enumerate(self.train_dataset), "Verify", total=len(self.train_dataset)):  # type: ignore
            # 映射关键字
            item_mapped = map_keywords(item)
            # 根据模型类型生成响应
            if ollama_model is not None:
                # 使用Ollama模型
                response = ollama_chat_evaluate(self.config, ollama_model, **item_mapped)
            elif openai_model is not None and openai_client is not None:
                # 使用OpenAI模型
                response = openai_client.openai_chat_evaluate(openai_model, **item_mapped)
            elif siliconflow_model is not None and siliconflow_client is not None:
                # 使用SiliconFlow模型
                response = siliconflow_client.siliconflow_chat_evaluate(siliconflow_model, **item_mapped)
            else:
                # 使用本地模型
                response = self.chat_evaluate_extract(**item_mapped)
            # 生成提示
            prompt = self.chat_prompt(**item_mapped)
            # 生成训练提示
            training_prompt = self.training_prompt(**item_mapped)

            # 创建数据点
            datapoint = OrderedDict()
            datapoint["index"] = i
            datapoint["prompt"] = prompt
            datapoint["training_prompt"] = training_prompt
            # 添加上下文
            if "context" in item_mapped and item_mapped["context"] is not None:
                datapoint["context"] = item_mapped["context"]
            # 添加问题
            datapoint["question"] = item_mapped["prompt"]
            # 添加答案
            datapoint["answer"] = item_mapped["response"]
            # 添加响应
            datapoint["response"] = response
            # 添加到验证数据
            verify_data.append(datapoint)

        # 保存验证结果
        if save_verify_data_path is not None:
            # 分离路径和扩展名
            save_verify_data_path_name, ext = os.path.splitext(save_verify_data_path)
            # 检查扩展名
            if ext not in [".json", ""]:
                # 记录警告
                logger.warning(
                    'Expected save path extension to be ".json", but got "{}" ("{}"). Saving to "{}".',
                    save_verify_data_path,
                    ext,
                    save_verify_data_path_name + ".json",
                )
            # 保存为JSON
            save_json(save_verify_data_path_name, verify_data)
            # 记录保存成功
            logger.info("Saved verify results to {file}", file=save_verify_data_path_name + ".json")
            # 自动导出为Markdown
            if self.config.auto_export_markdown:
                # 转换为Markdown
                md = json_yaml_to_markdown(verify_data)
                # 保存为Markdown
                save_md(save_verify_data_path_name, md)
                # 记录保存成功
                logger.info("Saved verify results as markdown to {file}", file=save_verify_data_path_name + ".md")

        # 计算错误
        errors = 0
        for d in verify_data:
            # 比较响应和答案
            if ignore_minor(d["response"]) != ignore_minor(d["answer"]):
                # 增加错误计数
                errors += 1
                # 打印错误信息
                print("Error: answer mismatch")
                print(f"Index: {d['index']}")
                print(f"Answer:\n{d['answer']}")
                print(f"Response:\n{d['response']}")
                print("**********************************************************")
                print()

        # 打印错误总数
        print(f"Total mismatches: {errors}")
        # 返回错误数
        return errors

    def test(
        self,
        test_dataset: str,
        save_test_data_path: str | Path | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
        siliconflow_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """测试模型在提供的测试数据集上的性能

        此方法评估模型在测试文件上的表现，为每个测试用例生成响应，并在测试数据中提供预期答案时将其与模型响应进行比较。

        参数:
            test_dataset (str): 测试数据集文件路径（JSON或YAML格式）
            save_test_data_path (Union[str, Path, None]): 保存测试结果的可选路径
            ollama_model (str, optional): 使用Ollama模型时的模型名称
            openai_model (str, optional): 使用OpenAI模型时的模型名称
            siliconflow_model (str, optional): 使用SiliconFlow模型时的模型名称

        返回值:
            list[dict[str, Any]]: 测试结果列表，每个结果包含提示、上下文、问题、响应和可选的答案

        执行流程:
            1. 验证模型选择（只能指定一个外部模型）
            2. 初始化所选外部模型的客户端（如果有）
            3. 从指定文件加载测试数据集
            4. 对每个测试用例:
               a. 将测试用例中的关键字映射到标准格式
               b. 使用所选模型（本地或外部）生成模型响应
               c. 创建包含测试用例详细信息和模型响应的数据点
               d. 将数据点添加到结果列表
            5. 将测试结果保存到指定文件（如果提供）
            6. 如果测试数据包含预期答案，计算准确率:
               a. 将模型响应与预期答案进行比较
               b. 统计不匹配数量并计算正确百分比
               c. 打印准确率统计信息

        数据结构:
            测试数据集应为JSON或YAML格式，包含以下内容:
            - "context": 测试用例的可选上下文信息
            - "question"或"prompt": 测试问题/提示
            - "response": 用于比较的可选预期答案

        提示模板:
            使用AppConfig中配置的提示模板生成模型输入

        示例:
            ```python
            app = App()
            test_results = app.test(
                test_dataset="test_data.json", save_test_data_path="test_results.json"
            )
            print(f"Test completed with {len(test_results)} cases")
            ```
        """
        # 初始化结果列表
        results = []

        # 验证模型选择
        if ollama_model is not None and openai_model is not None:
            raise RuntimeError("Both Ollama model and OpenAI model cannot both be set. Only one should be set.")
        if ollama_model is not None and siliconflow_model is not None:
            raise RuntimeError("Both Ollama model and SiliconFlow model cannot both be set. Only one should be set.")
        if openai_model is not None and siliconflow_model is not None:
            raise RuntimeError("Both OpenAI model and SiliconFlow model cannot both be set. Only one should be set.")

        # 初始化OpenAI客户端（如果指定了OpenAI模型）
        openai_client = ChatHPCOpenAI(self.config) if openai_model is not None else None
        # 初始化SiliconFlow客户端（如果指定了SiliconFlow模型）
        siliconflow_client = ChatHPCSiliconFlow(self.config) if siliconflow_model is not None else None

        # 加载测试数据集（支持JSON或YAML格式）
        test_data = load_json_yaml_arg(test_dataset, False)
        # 计算测试数据集的长度
        test_data_len = len(test_data)

        # 使用tqdm显示测试进度条
        for i, item in tqdm(enumerate(test_data), "Test", total=test_data_len):  # type: ignore
            # 将测试数据中的关键字映射到标准格式
            # 核心功能：将不同来源的数据字段名统一映射到标准化的字段名，确保后续处理的一致性
            # 实现逻辑：
            # 1. 遍历输入字典的每个键值对
            # 2. 对每个键，调用keyword_alias函数获取其标准化别名
            # 3. 构建新的字典，使用标准化的键名和原始值
            # 输入：原始测试数据字典，例如：{"Context": "OpenMP to Kokkos translation", "Question": "Can you translate this OpenACC code to Kokkos? ..."}
            # 输出：标准化后的字典，例如：{"context": "OpenMP to Kokkos translation", "prompt": "Can you translate this OpenACC code to Kokkos? ..."}
            # 技术优势：
            # 1. 灵活性：支持多种数据格式，如YAML、JSON等不同来源的数据
            # 2. 一致性：确保后续模板渲染和模型评估使用统一的字段名
            # 3. 可扩展性：通过修改ALIASES字典即可支持新的字段映射
            # 4. 代码简洁：使用字典推导式和辅助函数，代码简洁易维护
            # 映射示例（基于kokkos_testing.yaml）：
            # 映射前：{"Context": "OpenMP to Kokkos translation", "Question": "Can you translate this OpenACC code to Kokkos? ..."}
            # 映射后：{"context": "OpenMP to Kokkos translation", "prompt": "Can you translate this OpenACC code to Kokkos? ..."}
            # 应用场景：
            # - 处理不同格式的测试数据集
            # - 确保模板渲染时使用正确的变量名
            # - 统一模型评估的输入格式
            item_mapped = map_keywords(item)
            
            # 根据指定的模型类型生成响应
            if ollama_model is not None:
                # 使用Ollama模型生成响应
                response = ollama_chat_evaluate(self.config, ollama_model, **item_mapped)
            elif openai_model is not None and openai_client is not None:
                # 使用OpenAI模型生成响应
                response = openai_client.openai_chat_evaluate(openai_model, **item_mapped)
            elif siliconflow_model is not None and siliconflow_client is not None:
                # 使用SiliconFlow模型生成响应
                response = siliconflow_client.siliconflow_chat_evaluate(siliconflow_model, **item_mapped)
            else:
                # 使用本地微调模型生成响应
                response = self.chat_evaluate_extract(**item_mapped)
            
            # 生成用于模型输入的提示
            prompt = self.chat_prompt(**item_mapped)
            
            # 创建有序字典存储测试结果数据点
            datapoint = OrderedDict()
            # 添加索引
            datapoint["index"] = i
            # 添加提示
            datapoint["prompt"] = prompt
            # 如果有上下文信息，添加上下文
            if "context" in item_mapped and item_mapped["context"] is not None:
                datapoint["context"] = item_mapped["context"]
            # 添加问题
            datapoint["question"] = item_mapped["prompt"]
            # 如果有预期答案，添加答案
            # 这就懂了，如果测试数据集里提供了预期的输出，那么模型生成的就是answer，然后会算一下匹配程度，否则就不算了
            if "response" in item_mapped and item_mapped["response"] is not None:
                datapoint["answer"] = item_mapped["response"]
            # 添加模型响应
            datapoint["response"] = response
            # 将数据点添加到结果列表
            results.append(datapoint)

        # 如果指定了保存路径，保存测试结果
        if save_test_data_path is not None:
            # 分离文件路径和扩展名
            save_test_data_path_name, ext = os.path.splitext(save_test_data_path)
            # 检查扩展名是否为.json
            if ext not in [".json", ""]:
                # 如果不是.json扩展名，记录警告并使用.json扩展名
                logger.warning(
                    'Expected save path extension to be ".json", but got "{}" ("{}"). Saving to "{}".',
                    save_test_data_path,
                    ext,
                    save_test_data_path_name + ".json",
                )
            # 保存测试结果为JSON文件
            save_json(save_test_data_path_name, results)
            # 记录保存成功的日志
            logger.info("Saved test results to {file}", file=save_test_data_path_name + ".json")
            # 如果配置了自动导出Markdown，保存为Markdown格式
            if self.config.auto_export_markdown:
                # 将测试结果转换为Markdown格式
                md = json_yaml_to_markdown(results)
                # 保存Markdown文件
                save_md(save_test_data_path_name, md)
                # 记录保存成功的日志
                logger.info("Saved test results as markdown to {file}", file=save_test_data_path_name + ".md")

        # 检查测试结果中是否包含答案字段
        if "answer" in next(iter(results), {}):  # type: ignore
            # 初始化错误计数
            errors = 0
            # 遍历所有测试结果
            for d in results:
                # 比较模型响应与预期答案（忽略轻微差异）
                if ignore_minor(d["response"]) != ignore_minor(d["answer"]):
                    # 如果不匹配，增加错误计数
                    errors += 1
                    # 打印不匹配的测试信息
                    print("Missed test:")
                    print(f"Index: {d['index']}")
                    print(f"Answer:\n{d['answer']}")
                    print(f"Response:\n{d['response']}")
                    print("**********************************************************")
                    print()

            # 计算正确的测试数量
            correct = test_data_len - errors
            # 打印准确率统计信息
            print(f"Total correct: {correct} out of {test_data_len} ({(float(correct)/test_data_len) * 100:.2f}%)")
        # 返回测试结果列表
        return results

    def print_config(self) -> None:
        """Print the current configurations of the application in a formatted table.

        This method displays all configuration settings from self.config in a
        formatted table using the tabulate library.

        The table includes all configuration parameters like:
        - File paths (data, models, checkpoints)
        - Model parameters
        - Training settings
        - Other application settings

        Example:
            ```python
            app = App()
            app.print_config()
            # Outputs:
            # =====================  ==========================
            # Setting               Value
            # =====================  ==========================
            # data_file             /path/to/data.json
            # base_model_path       /path/to/base/model
            # max_response_tokens   600
            # use_wandb            False
            # =====================  ==========================
            ```

        Note:
            The output format uses the 'simple' table format from the tabulate library
            for clean and readable presentation of settings.
        """
        # Get configuration as dict, excluding internal pydantic fields
        config_dict = self.config.model_dump()

        # Format as table rows
        table_data = [[setting, value] for setting, value in config_dict.items()]

        # Define table headers
        headers = ["Setting", "Value"]

        # Print formatted table
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

    def save_readme(self, filename: Path | str) -> None:
        if type(filename) is not Path:
            filename = Path(filename)

        if filename.is_dir():
            filename = filename / "README.md"

        # Get configuration as dict, excluding internal pydantic fields
        config_dict = self.config.model_dump()

        # Replace newlines with newline char.
        config_dict["prompt_template"] = config_dict["prompt_template"].replace("\n", "\\n")

        # Add version
        version_dict = {
            "commit": run("git rev-parse --short HEAD", verbose=False),
            "version": chathpc.app.version,
        }

        # Format as table rows
        table_data = [[setting, value] for setting, value in config_dict.items()]
        version_table_data = [[setting, value] for setting, value in version_dict.items()]

        # Define table headers
        headers = ["Setting", "Value"]

        # Print formatted table
        config_table = tabulate(table_data, headers=headers, tablefmt="github")
        version_table = tabulate(version_table_data, headers=headers, tablefmt="github")

        with open(filename, "w") as fd:
            project_name = Path(run("git rev-parse --show-toplevel", verbose=False)).name.strip()
            fd.write(f"# {project_name} Model Info\n\n## ChatHPC Version Info\n\n")
            fd.write(version_table)
            fd.write("\n\n## Configuration\n\n")
            fd.write(config_table)

    def extract_answer(self, chat_response: str, **kwargs):
        """Extract the model's answer from a complete response string.

        This method processes the full model response to extract just the answer portion,
        removing any template formatting or context that was part of the prompt.

        Args:
            response (str): The complete response string from the model evaluation
            **kwargs: Additional keyword arguments that may be used for template-specific extraction

        Returns:
            str: The extracted answer portion of the response

        Example:
            ```python
            app = App()
            response = app.chat_evaluate("What is Kokkos?")
            answer = app.extract_answer(response)
            print(answer)  # Prints just the model's answer without template formatting
            ```

        Note:
            The exact extraction logic depends on the prompt template structure
            defined in the application configuration.
        """
        chat_answer = chat_response
        chat_answer = chat_answer.replace("<s> ", "").replace("</s>", "")

        prefix = self.inference_template.render(**template_utils.map_keywords(kwargs))
        postfix = self.postfix_template.render(**template_utils.map_keywords(kwargs))
        if chat_answer.startswith(prefix):
            chat_answer = chat_answer[len(prefix) :]

        if chat_answer.endswith(postfix):
            chat_answer = chat_answer[: -len(postfix)]

        return chat_answer
