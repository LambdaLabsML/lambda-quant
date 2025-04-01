import argparse
import os
import logging
import shutil
import subprocess
import sys
import json
import faulthandler

import huggingface_hub
import torch
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import cpu_offload

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=[
            "AWQ-Int4",
            "GPTQ-Int8",
            "GPTQ-Int4",
            "Static-F8",
            "Dynamic-F8",
        ],
        required=True,
        help="The type of quantization to apply",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        required=True,
        help="The base model. Should be huggingface tag.",
    )
    parser.add_argument(
        "--dataset", default="HuggingFaceH4/ultrachat_200k", help="Calibration data"
    )
    parser.add_argument(
        "--dataset-split", default="train_sft", help="Split for calibration data"
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name for calibration data, passed to datasets.load_dataset.",
    )
    parser.add_argument(
        "--num-samples",
        default=512,
        type=int,
        help="Number of items from dataset to use for calibration",
    )
    parser.add_argument(
        "--seq-length",
        default=2048,
        type=int,
        help="Sequence length for calibration data",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Number of calibration samples to process at the same time.",
    )
    parser.add_argument("--update-metadata-only", default=False, action="store_true")
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-{args.quantization}"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)
    LOGGER.info(os.environ)

    target_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    LOGGER.info(f"Using cuda:{target_device}")

    # NOTE: `0` index means the first **visible** cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if args.update_metadata_only:
        with open(f"{quant_name}/lambda-quant-args.json") as fp:
            args = argparse.Namespace(**json.load(fp))
        write_metadata(args, quant_name)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = datasets.load_dataset(
        args.dataset, args.dataset_name, split=args.dataset_split
    )
    ds = ds.shuffle(seed=0).select(range(args.num_samples))

    if args.quantization == "AWQ-Int4":
        from awq import AutoAWQForCausalLM
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) > Version("4.47.1"):
            raise ValueError(
                "There's currently a bug in autoawq with newer transformers versions. Please downgrade to 4.47.1"
            )

        model = AutoAWQForCausalLM.from_pretrained(
            args.model, low_cpu_mem_usage=True, use_cache=False
        )

        ds = ds.map(preprocess, remove_columns=ds.column_names)
        ds = [q["text"] for q in ds]

        model.quantize(
            tokenizer,
            calib_data=ds,
            max_calib_samples=args.num_samples,
            max_calib_seq_len=args.seq_length,
            n_parallel_calib_samples=args.batch_size,
        )
        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)

    elif args.quantization in ["GPTQ-Int4", "GPTQ-Int8"]:
        from gptqmodel import GPTQModel, QuantizeConfig

        model = GPTQModel.load(
            args.model,
            QuantizeConfig(
                bits=4 if "4" in args.quantization else 8,
                group_size=128,
                device=device,
            ),
        )
        ds = ds.map(preprocess)
        ds = ds.map(tokenize, remove_columns=ds.column_names)
        model.quantize(ds.to_list(), batch_size=args.batch_size, tokenizer=tokenizer)
        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)

    elif args.quantization in ["Static-F8", "Dynamic-F8"]:
        """See https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8"""
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        model = AutoModelForCausalLM.from_pretrained(args.model, use_cache=False)
        cpu_offload(model, execution_device=device)

        oneshot(
            model=model,
            recipe=QuantizationModifier(
                targets="Linear",
                scheme={
                    "Static-F8": "FP8",
                    "Dynamic-F8": "FP8_DYNAMIC",
                }[args.quantization],
                ignore=["lm_head"],
            ),
        )
        model.save_pretrained(quant_name)
        tokenizer.save_pretrained(quant_name)

    else:
        raise NotImplementedError(args.quantization)

    write_metadata(args, quant_name)


def write_metadata(args, metdata_dir):
    os.makedirs(metdata_dir, exist_ok=True)

    LOGGER.info("Downloading base model readmes.")
    hf_cache_dir = huggingface_hub.snapshot_download(args.model, allow_patterns="*.md")
    for fname in os.listdir(hf_cache_dir):
        if fname.endswith("md"):
            LOGGER.info(f"Copying {hf_cache_dir}/{fname} into {metdata_dir}")
            shutil.copy(
                os.path.join(hf_cache_dir, fname),
                os.path.join(metdata_dir, fname),
            )

    if "AWQ" in args.quantization:
        import awq

        quantization_library = (
            f"[autoawq=={awq.__version__}](https://github.com/casper-hansen/AutoAWQ)"
        )
    elif "GPTQ" in args.quantization:
        import gptqmodel

        quantization_library = f"[gptqmodel=={gptqmodel.__version__}](https://github.com/ModelCloud/GPTQModel)"
    else:
        import llmcompressor

        quantization_library = f"[llmcompressor=={llmcompressor.__version__}](https://github.com/vllm-project/llm-compressor)"
    LOGGER.info(f"Using {quantization_library}")

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    LOGGER.info(f"lambda-quant commit {commit_hash}")

    new_lines = [
        "# Quantization",
        f"Created with [lambda-quant](https://github.com/LambdaLabsML/lambda-quant/tree/{commit_hash}) on `Python {sys.version}`\n",
        f"Base Model: [{args.model}](https://huggingface.co/{args.model})\n",
        f"Quantized using {quantization_library}\n",
        "Steps to create:",
        f"1. `git clone https://github.com/LambdaLabsML/lambda-quant`",
        f"2. `git checkout {commit_hash}`",
        f"3. `python {' '.join(sys.argv)}`",
        "## Evaluation",
        "TODO",
        "## Benchmarks",
        "TODO",
        "# Base Model README.md\n",
    ]
    new_content = "\n".join(new_lines)
    LOGGER.info(f"Writing {new_content} into README.md")
    with open(f"{metdata_dir}/README.md") as fp:
        readme_content = fp.read()
    with open(f"{metdata_dir}/README.md", "w") as fp:
        fp.write(new_content + readme_content)

    LOGGER.info(f"Dumping `pip freeze` to {metdata_dir}/requirements-lambda-quant.txt")
    freeze = subprocess.check_output(["pip", "freeze"]).decode()
    with open(f"{metdata_dir}/requirements-lambda-quant.txt", "w") as fp:
        fp.write(freeze)

    LOGGER.info(f"Dumping `args` to {metdata_dir}/args-lambda-quant.json")
    with open(f"{metdata_dir}/args-lambda-quant.json", "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    main()
