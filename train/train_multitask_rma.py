import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from rma_model_profiles import (
        RMA_MODEL_PROFILES,
        resolve_profile_max_length,
        resolve_profile_model_name,
    )
except ImportError:
    from train.rma_model_profiles import (
        RMA_MODEL_PROFILES,
        resolve_profile_max_length,
        resolve_profile_model_name,
    )

from gemma_prompts import (
    SFT_REWRITE_INFERENCE_GEMMA,
    SFT_REWRITE_TRAIN_GEMMA,
    SFT_RMA_INFERENCE_GEMMA,
    SFT_RMA_TRAIN_GEMMA,
)
from llama_prompts import (
    SFT_REWRITE_INFERENCE_LLAMA,
    SFT_REWRITE_TRAIN_LLAMA,
    SFT_RMA_INFERENCE_LLAMA,
    SFT_RMA_INFERENCE_PHI4,
    SFT_RMA_INFERENCE_QWEN3,
    SFT_RMA_TRAIN_LLAMA,
    SFT_RMA_TRAIN_PHI4,
    SFT_RMA_TRAIN_QWEN3,
)
from train_sentence_rewriter import (
    DEFAULT_LLAMA_CPP_DIR,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    DEFAULT_OUTPUT_ROOT,
    build_lora_config,
    collect_training_files as collect_rewrite_training_files,
    postprocess_trained_model,
    resolve_model_storage_paths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
DEFAULT_ADDITIONAL_DIR = DEFAULT_TRAIN_DIR / "additional"
PLANNING_EXTRA_FILE = "it2_NR_train.tsv"
DEFAULT_OUTPUT_TAG = "multitask-rewrite-planning"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
RMA_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the query clearly by replacing ambiguous pronouns (like \"it\", "
    "\"that\") with explicit information from the conversation history. Keep "
    "exactly the same sentence structure. Do NOT generate or include any "
    "information, words, or values outside of the provided conversation_history "
    "and query."
)
PLANNING_SYSTEM_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values "
    "that are explicitly stated or can be reasonably inferred from the query.\n "
    "<|tool|>{tools}<|/tool|>"
)
KNOWN_UNSUPPORTED_GGUF_ARCHITECTURES = {
    "Exaone4ForCausalLM",
    "Lfm2ForCausalLM",
    "SmolLM3ForCausalLM",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multitask LoRA adapter for rewrite + planning."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RMA_MODEL_PROFILES.keys()),
        default=None,
        help="Named model profile. --model_name overrides the profile model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Base model name to load from Hugging Face.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for HF cache, epoch checkpoints, and final adapter.",
    )
    parser.add_argument(
        "--output_tag",
        default=DEFAULT_OUTPUT_TAG,
        help="Suffix used to distinguish this multitask run under output_root.",
    )
    parser.add_argument(
        "--train_dir",
        default=str(DEFAULT_TRAIN_DIR),
        help="Directory containing root TSV files.",
    )
    parser.add_argument(
        "--additional_dir",
        default=str(DEFAULT_ADDITIONAL_DIR),
        help="Directory containing additional TSV files.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--mix_strategy",
        choices=["balanced", "concat"],
        default="balanced",
        help="How to mix rewrite and planning datasets.",
    )
    parser.add_argument(
        "--rewrite_sampling_prob",
        type=float,
        default=0.5,
        help="Rewrite sampling probability when mix_strategy=balanced.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle/interleave seed.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Token truncation length.",
    )
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for tokenizers without chat_template.",
    )
    parser.add_argument(
        "--epoch_eval_path",
        default=None,
        help="Optional comma-separated TSV path(s) to evaluate planning generation at each epoch end.",
    )
    parser.add_argument(
        "--epoch_eval_sample_size",
        default=50,
        type=int,
        help="Number of epoch-eval examples to use. Use <=0 for all rows.",
    )
    parser.add_argument(
        "--epoch_eval_max_new_tokens",
        default=128,
        type=int,
        help="Max new tokens for epoch-end planning generation evaluation.",
    )
    parser.add_argument(
        "--skip_postprocess",
        action="store_true",
        help="Skip merge, GGUF conversion, and Ollama registration after training.",
    )
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help="Skip training and run only merge, GGUF conversion, and Ollama registration from existing outputs.",
    )
    parser.add_argument(
        "--export_checkpoint_epoch",
        type=int,
        default=None,
        help="Export the Nth epoch checkpoint instead of the final checkpoint.",
    )
    parser.add_argument(
        "--export_source",
        choices=["final_checkpoint", "final_adapter"],
        default="final_checkpoint",
        help="Use the final epoch checkpoint or the final saved adapter directory for export.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base model ID instead of reading adapter_config.json.",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Optional explicit local directory containing the base model snapshot.",
    )
    parser.add_argument(
        "--merged_dir",
        default=None,
        help="Output directory for the merged HF model.",
    )
    parser.add_argument(
        "--gguf_path",
        default=None,
        help="Output path for the merged GGUF file.",
    )
    parser.add_argument(
        "--modelfile",
        default=None,
        help="Path to write the Ollama Modelfile.",
    )
    parser.add_argument(
        "--ollama_model_name",
        default=None,
        help="Model name to register in Ollama.",
    )
    parser.add_argument(
        "--ollama_host",
        default=DEFAULT_OLLAMA_HOST,
        help="OLLAMA_HOST value for ollama create.",
    )
    parser.add_argument(
        "--ollama_models_dir",
        default=DEFAULT_OLLAMA_MODELS_DIR,
        help="OLLAMA_MODELS directory for ollama create.",
    )
    parser.add_argument(
        "--ollama_bin",
        default=DEFAULT_OLLAMA_BIN,
        help="Path to the ollama binary.",
    )
    parser.add_argument(
        "--ollama_lib_path",
        default=DEFAULT_OLLAMA_LIB_PATH,
        help="LD_LIBRARY_PATH to use for ollama create.",
    )
    parser.add_argument(
        "--llama_cpp_dir",
        default=DEFAULT_LLAMA_CPP_DIR,
        help="Path to llama.cpp containing convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--merge_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype to use when merging the LoRA adapter.",
    )
    parser.add_argument(
        "--gguf_outtype",
        default="bf16",
        help="GGUF output type for llama.cpp convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--trust_remote_code",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to trust remote model/tokenizer code while merging.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of the base model when download is needed.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token for private or gated repos.",
    )
    parser.add_argument(
        "--strict_postprocess",
        action="store_true",
        help="Attempt GGUF/Ollama postprocess even for model architectures known to be unsupported locally.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging interval.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=8,
        help="Checkpoint retention limit.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            path = cwd_path
        else:
            path = PROJECT_ROOT / path
    return path.resolve()


def read_simple_apis(api_path: Path) -> Dict:
    with open(api_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_trust_remote_code_arg(args, default: bool = True) -> bool:
    if args.trust_remote_code == "true":
        return True
    if args.trust_remote_code == "false":
        return False
    return default


def get_config_architecture(config) -> str | None:
    architectures = getattr(config, "architectures", None)
    if architectures:
        return architectures[0]
    return None


def converter_supports_architecture(llama_cpp_dir: str | Path, architecture: str) -> bool:
    llama_cpp_path = Path(llama_cpp_dir).expanduser().resolve()
    converter_path = llama_cpp_path / "convert_hf_to_gguf.py"
    if not converter_path.exists():
        return False

    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(llama_cpp_path)!r})\n"
        "import convert_hf_to_gguf as c\n"
        f"c.ModelBase.from_model_architecture({architecture!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(llama_cpp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def maybe_auto_skip_unsupported_postprocess(args, config):
    architecture = get_config_architecture(config)
    if (
        architecture in KNOWN_UNSUPPORTED_GGUF_ARCHITECTURES
        and not args.skip_postprocess
        and not args.strict_postprocess
    ):
        if converter_supports_architecture(args.llama_cpp_dir, architecture):
            print(
                "[postprocess] selected llama.cpp converter supports "
                f"{architecture}; postprocess will proceed."
            )
            return
        args.skip_postprocess = True
        print(
            "[postprocess] auto-skipped because selected llama.cpp converter does not "
            f"register {architecture}: {args.llama_cpp_dir}. Use a newer "
            "--llama_cpp_dir or --strict_postprocess to attempt it anyway."
        )


def configure_tokenizer_padding(tokenizer) -> int:
    added_tokens = 0
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    return added_tokens


def collect_planning_training_files(train_dir: Path, additional_dir: Path) -> List[Path]:
    planning_files = sorted(
        path
        for path in train_dir.glob("*.tsv")
        if "nonNR" in path.name
    )
    extra_file = train_dir / PLANNING_EXTRA_FILE
    if extra_file.exists():
        planning_files.append(extra_file)
    return planning_files


def dataframe_to_dataset(files: List[Path], task_name: str) -> Dataset:
    if not files:
        raise ValueError(f"No TSV files found for task={task_name}")

    dataframes = []
    for file in files:
        dataframe = pd.read_csv(file, sep="\t", dtype=str)
        dataframe["task_name"] = task_name
        dataframe["source_file"] = file.name
        dataframes.append(dataframe)

    return Dataset.from_pandas(
        pd.concat(dataframes, ignore_index=True, sort=False),
        preserve_index=False,
    )


def build_api_str(example, apis: Dict) -> str:
    api_str = ""
    candidates = ast.literal_eval(example["candidates"])
    for plan in candidates:
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
    return api_str


def parse_answer_value(value):
    if isinstance(value, dict):
        return value

    parsed = value
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        try:
            parsed = ast.literal_eval(parsed)
        except Exception:
            try:
                parsed = json.loads(parsed)
            except Exception:
                break

    return parsed if isinstance(parsed, dict) else {}


def parse_generated_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.replace("```json", "```")
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    match = None
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                match = text[start:index + 1]
                break

    candidate = match or text
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        parsed = json.loads(candidate)
    return parsed if isinstance(parsed, dict) else {}


def resolve_epoch_eval_paths(path_arg: str | None) -> list[str]:
    if not path_arg:
        return []
    return [
        str(resolve_project_path(path))
        for path in path_arg.split(",")
        if path.strip()
    ]


def load_epoch_eval_dataset(path_arg: str | None, sample_size: int):
    data_files = resolve_epoch_eval_paths(path_arg)
    if not data_files:
        return None

    dataframes = []
    for file_path in data_files:
        dataframe = pd.read_csv(file_path, sep="\t", dtype=str)
        dataframe["task_name"] = "epoch_eval_planning"
        dataframe["source_file"] = Path(file_path).name
        dataframes.append(dataframe)
    dataset = Dataset.from_pandas(
        pd.concat(dataframes, ignore_index=True, sort=False),
        preserve_index=False,
    )
    if sample_size > 0:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    return dataset


def render_chat_template(tokenizer, messages, add_generation_prompt: bool, model_name: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_messages_as_plain_text(messages, add_generation_prompt: bool) -> str:
    sections = []
    for message in messages:
        role = str(message.get("role", "")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    if add_generation_prompt:
        sections.append("Assistant:\n")
    return "\n\n".join(sections)


def render_model_messages(
    tokenizer,
    messages,
    add_generation_prompt: bool,
    model_name: str,
    chat_template_fallback: str,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer=tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            model_name=model_name,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "for base-model multitask SFT."
        )
    return render_messages_as_plain_text(messages, add_generation_prompt=add_generation_prompt)


def tokenize_with_labels(tokenizer, prompt: str, prompt_prefix: str, max_length: int):
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]
    label_start = len(tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"])
    label_start = min(label_start, len(input_ids))
    labels = [-100] * label_start + input_ids[label_start:]
    return input_ids, labels


def select_rewrite_templates(model_name: str) -> Tuple[str | None, str | None]:
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_RMA_TRAIN_LLAMA, SFT_RMA_INFERENCE_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_RMA_TRAIN_GEMMA, SFT_RMA_INFERENCE_GEMMA
    if model_name == "microsoft/Phi-4-mini-instruct":
        return SFT_RMA_TRAIN_PHI4, SFT_RMA_INFERENCE_PHI4
    if "Qwen/" in model_name:
        return SFT_RMA_TRAIN_QWEN3, SFT_RMA_INFERENCE_QWEN3
    return None, None


def select_planning_templates(model_name: str) -> Tuple[str | None, str | None]:
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_REWRITE_TRAIN_LLAMA, SFT_REWRITE_INFERENCE_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_REWRITE_TRAIN_GEMMA, SFT_REWRITE_INFERENCE_GEMMA
    return None, None


def build_rewrite_preprocess_fn(
    tokenizer,
    model_name: str,
    max_length: int,
    chat_template_fallback: str,
):
    train_template, inference_template = select_rewrite_templates(model_name)

    def preprocess(example):
        data = {
            "conversation_history": example["conversation_history"],
            "query": example["query"],
        }
        data_json = json.dumps(data, ensure_ascii=False, indent=2)
        assistant_content = {"rewrited_query": example["rewrited_query"]}

        if train_template is not None and inference_template is not None:
            prompt_prefix = inference_template.format(data=data_json)
            prompt = train_template.format(
                data=data_json,
                answer=assistant_content,
            )
        else:
            messages = [
                {"role": "system", "content": RMA_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": data_json},
            ]
            prompt_prefix = render_model_messages(
                tokenizer=tokenizer,
                messages=messages,
                add_generation_prompt=True,
                model_name=model_name,
                chat_template_fallback=chat_template_fallback,
            )
            prompt = render_model_messages(
                tokenizer=tokenizer,
                messages=messages + [{"role": "assistant", "content": str(assistant_content)}],
                add_generation_prompt=False,
                model_name=model_name,
                chat_template_fallback=chat_template_fallback,
            )

        input_ids, labels = tokenize_with_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            max_length=max_length,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": example["rewrited_query"],
        }

    return preprocess


def build_planning_preprocess_fn(
    tokenizer,
    model_name: str,
    apis: Dict,
    max_length: int,
    chat_template_fallback: str,
):
    train_template, inference_template = select_planning_templates(model_name)

    def preprocess(example):
        api_str = build_api_str(example, apis)
        assistant_content = (
            json.dumps(example["answer"], ensure_ascii=False)
            if isinstance(example["answer"], dict)
            else example["answer"]
        )

        user_query = example["rewrited_query"]

        if train_template is not None and inference_template is not None:
            prompt_prefix = inference_template.format(
                tools=api_str,
                data=user_query,
            )
            prompt = train_template.format(
                tools=api_str,
                data=user_query,
                answer=assistant_content,
            )
        else:
            system_msg = PLANNING_SYSTEM_PROMPT.format(tools=api_str)
            user_content = f"User Query: {user_query}"

            prompt_prefix = render_model_messages(
                tokenizer=tokenizer,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                add_generation_prompt=True,
                model_name=model_name,
                chat_template_fallback=chat_template_fallback,
            )
            prompt = render_model_messages(
                tokenizer=tokenizer,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                add_generation_prompt=False,
                model_name=model_name,
                chat_template_fallback=chat_template_fallback,
            )

        input_ids, labels = tokenize_with_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            max_length=max_length,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": assistant_content,
        }

    return preprocess


def build_planning_inference_prompt(
    tokenizer,
    model_name: str,
    apis: Dict,
    example,
    chat_template_fallback: str,
) -> str:
    train_template, inference_template = select_planning_templates(model_name)
    api_str = build_api_str(example, apis)
    user_query = example["rewrited_query"]

    if train_template is not None and inference_template is not None:
        return inference_template.format(
            tools=api_str,
            data=user_query,
        )

    system_msg = PLANNING_SYSTEM_PROMPT.format(tools=api_str)
    user_content = f"User Query: {user_query}"
    return render_model_messages(
        tokenizer=tokenizer,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
        model_name=model_name,
        chat_template_fallback=chat_template_fallback,
    )


class EpochPlanningEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        apis: Dict,
        model_name: str,
        max_length: int,
        chat_template_fallback: str,
        max_new_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.apis = apis
        self.model_name = model_name
        self.max_length = max_length
        self.chat_template_fallback = chat_template_fallback
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return control

        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        plan_pass = 0
        arg_pass = 0
        all_pass = 0
        parse_fail = 0
        total = 0

        with torch.no_grad():
            for example in self.eval_dataset:
                total += 1
                prompt = build_planning_inference_prompt(
                    tokenizer=self.tokenizer,
                    model_name=self.model_name,
                    apis=self.apis,
                    example=example,
                    chat_template_fallback=self.chat_template_fallback,
                )
                model_inputs = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_inputs = {
                    key: value.to(device)
                    for key, value in model_inputs.items()
                }
                input_length = model_inputs["input_ids"].shape[-1]
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.decode(
                    generated_ids[0][input_length:],
                    skip_special_tokens=False,
                )

                try:
                    prediction = parse_generated_json(generated_text)
                except Exception:
                    prediction = {}
                    parse_fail += 1

                gold = parse_answer_value(example["answer"])
                plan_ok = prediction.get("plan") == gold.get("plan")
                args_ok = prediction.get("arguments") == gold.get("arguments")
                plan_pass += int(plan_ok)
                arg_pass += int(args_ok)
                all_pass += int(plan_ok and args_ok)

        if model_was_training:
            model.train()

        if total:
            epoch_value = state.epoch if state.epoch is not None else -1
            print(
                "[epoch_eval] "
                f"epoch={epoch_value:.2f} samples={total} "
                f"plan={plan_pass / total * 100:.2f}% "
                f"arguments={arg_pass / total * 100:.2f}% "
                f"all={all_pass / total * 100:.2f}% "
                f"parse_fail={parse_fail}"
            )
        return control


def build_epoch_eval_callbacks(args, tokenizer, apis: Dict, model_name: str, max_length: int):
    eval_dataset = load_epoch_eval_dataset(args.epoch_eval_path, args.epoch_eval_sample_size)
    if eval_dataset is None:
        return []
    print(
        "[epoch_eval] loaded "
        f"{len(eval_dataset)} planning examples from {resolve_epoch_eval_paths(args.epoch_eval_path)}"
    )
    return [
        EpochPlanningEvalCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            apis=apis,
            model_name=model_name,
            max_length=max_length,
            chat_template_fallback=args.chat_template_fallback,
            max_new_tokens=args.epoch_eval_max_new_tokens,
        )
    ]


def max_total_length(dataset: Dataset) -> int:
    max_len = 0
    for example in dataset:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_len:
            max_len = total_length
    return max_len


def mix_processed_datasets(rewrite_ds: Dataset, planning_ds: Dataset, args) -> Dataset:
    if args.mix_strategy == "concat":
        return concatenate_datasets([rewrite_ds, planning_ds]).shuffle(seed=args.shuffle_seed)

    rewrite_prob = args.rewrite_sampling_prob
    if not 0.0 < rewrite_prob < 1.0:
        raise ValueError("--rewrite_sampling_prob must be between 0 and 1.")

    return interleave_datasets(
        [rewrite_ds, planning_ds],
        probabilities=[rewrite_prob, 1.0 - rewrite_prob],
        seed=args.shuffle_seed,
        stopping_strategy="all_exhausted",
    )


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        label_pad_token_id = -100
        max_label_length = max(len(feature["labels"]) for feature in features)
        for feature in features:
            padding_length = max_label_length - len(feature["labels"])
            feature["labels"] = feature["labels"] + [label_pad_token_id] * padding_length
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")


def main():
    args = parse_args()
    args.model_name = resolve_profile_model_name(
        profile=args.profile,
        model_name=args.model_name,
        default_model_name=DEFAULT_MODEL_NAME,
    )
    args.max_length = resolve_profile_max_length(
        profile=args.profile,
        max_length=args.max_length,
        default_max_length=1536,
    )

    train_dir = resolve_path(args.train_dir)
    additional_dir = resolve_path(args.additional_dir)
    tools_path = resolve_path(args.tools_path)
    output_root = Path(args.output_root).expanduser()
    model_name = args.model_name

    model_root_dir, output_dir, cache_dir = resolve_model_storage_paths(
        output_root,
        model_name,
        args.output_tag,
    )

    trust_remote_code = resolve_trust_remote_code_arg(args, default=True)
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )
    maybe_auto_skip_unsupported_postprocess(args, config)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    if args.postprocess_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Training output directory not found for --postprocess_only: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }

    print(f"profile: {args.profile}")
    print(f"model_name: {model_name}")
    print(f"model_architecture: {get_config_architecture(config)}")
    print(f"train_dir: {train_dir}")
    print(f"additional_dir: {additional_dir}")
    print(f"tools_path: {tools_path}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"training_output_dir: {output_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"mix_strategy: {args.mix_strategy}")
    print(f"chat_template_fallback: {args.chat_template_fallback}")

    if args.postprocess_only:
        postprocess_trained_model(
            args=args,
            model_name=model_name,
            output_dir=output_dir,
            output_root=output_root,
            preexisting_checkpoint_names=preexisting_checkpoint_names,
            num_train_epochs=args.num_train_epochs,
            final_global_step=None,
            cache_dir=cache_dir,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )
    added_tokens = configure_tokenizer_padding(tokenizer)

    if "google/gemma" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )

    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    rewrite_files = collect_rewrite_training_files(train_dir, additional_dir)
    planning_files = collect_planning_training_files(train_dir, additional_dir)
    print("rewrite_files:", [str(path) for path in rewrite_files])
    print("planning_files:", [str(path) for path in planning_files])

    rewrite_raw = dataframe_to_dataset(rewrite_files, "rewrite")
    planning_raw = dataframe_to_dataset(planning_files, "planning")
    apis = read_simple_apis(tools_path)

    rewrite_processed = rewrite_raw.map(
        build_rewrite_preprocess_fn(
            tokenizer,
            model_name,
            args.max_length,
            args.chat_template_fallback,
        ),
        remove_columns=rewrite_raw.column_names,
        desc="Preprocessing rewrite task",
    )
    planning_processed = planning_raw.map(
        build_planning_preprocess_fn(
            tokenizer=tokenizer,
            model_name=model_name,
            apis=apis,
            max_length=args.max_length,
            chat_template_fallback=args.chat_template_fallback,
        ),
        remove_columns=planning_raw.column_names,
        desc="Preprocessing planning task",
    )

    print(f"rewrite_examples: {len(rewrite_processed)}")
    print(f"planning_examples: {len(planning_processed)}")
    print("rewrite sample prompt:")
    print(rewrite_processed[0]["strprompt"])
    print()
    print("planning sample prompt:")
    print(planning_processed[0]["strprompt"])
    print()
    print(f"rewrite max total length: {max_total_length(rewrite_processed)}")
    print(f"planning max total length: {max_total_length(planning_processed)}")

    processed_train = mix_processed_datasets(rewrite_processed, planning_processed, args)
    print(f"mixed_examples: {len(processed_train)}")
    print(f"mixed max total length: {max_total_length(processed_train)}")

    model = get_peft_model(model, build_lora_config(model_name))
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Trainable Parameters: "
        f"{trainable_params}, Total Parameters: {total_params}, "
        f"Training Rate: {100 * trainable_params / total_params:.2f}%"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        logging_dir=str(output_dir / "logs"),
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
    )
    callbacks = build_epoch_eval_callbacks(
        args=args,
        tokenizer=tokenizer,
        apis=apis,
        model_name=model_name,
        max_length=args.max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
