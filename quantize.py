import argparse
import os
import logging
import shutil
import subprocess
import sys
import json

import huggingface_hub
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            "W4A16-Int4",
            "W8A8-Int8",
            "W8A8-F8",
        ],
        required=True,
    )
    parser.add_argument("-m", "--model", default=None, required=True)
    parser.add_argument("-d", "--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--num-samples", default=128, type=int)
    parser.add_argument("--seq-length", default=512, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--metadata-only", default=False, action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    LOGGER.info(args)

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-{args.quantization}"

    write_metadata(args, quant_name)
    if args.metadata_only:
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

    ds = datasets.load_dataset(args.dataset, args.dataset_name, split=args.split)
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    if args.quantization == "AWQ-Int4":
        from awq import AutoAWQForCausalLM

        with device:
            model = AutoAWQForCausalLM.from_pretrained(
                args.model,
                low_cpu_mem_usage=True,
                use_cache=False,
                attn_implementation="flash_attention_2",
            )

        ds = ds.map(preprocess, remove_columns=ds.column_names)
        ds = [q["text"] for q in ds]

        model.quantize(
            tokenizer,
            calib_data=ds,
            max_calib_samples=args.num_samples,
            max_calib_seq_len=args.seq_length,
        )
        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)
    elif args.quantization in ["GPTQ-Int4", "GPTQ-Int8"]:
        from gptqmodel import GPTQModel, QuantizeConfig

        model = GPTQModel.load(
            args.model, dict(bits=4 if "4" in args.quantization else 8, group_size=128)
        )
        ds = ds.map(preprocess)
        ds = ds.map(tokenize, remove_columns=ds.column_names)
        model.quantize(ds.to_list(), batch_size=32, tokenizer=tokenizer)
        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)
    elif args.quantization in ["W4A16-Int4", "W8A8-Int8", "W8A8-F8"]:
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        with device:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="flash_attention_2",
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        oneshot(
            model=model,
            dataset=ds,
            recipe=[
                SmoothQuantModifier(smoothing_strength=0.7),
                QuantizationModifier(
                    targets="Linear",
                    scheme={
                        "W4A16-Int4": "W4A16",
                        "W8A8-Int8": "W8A8",
                        "W8A8-F8": "FP8_DYNAMIC",
                    }[args.quantization],
                    ignore=["lm_head"],
                ),
            ],
            max_seq_length=args.seq_length,
            num_calibration_samples=args.num_samples,
        )

        model.save_pretrained(quant_name)
        tokenizer.save_pretrained(quant_name)
    else:
        raise NotImplementedError(args.quantization)


def write_metadata(args, metdata_dir):
    os.makedirs(metdata_dir, exist_ok=True)
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

        quantization_library = f"autoawq=={awq.__version__}"
    elif "GPTQ" in args.quantization:
        import gptqmodel

        quantization_library = f"gptqmodel=={gptqmodel.__version__}"
    else:
        import llmcompressor

        quantization_library = f"llmcompressor=={llmcompressor.__version__}"
    LOGGER.info(f"Using {quantization_library}")

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    LOGGER.info(f"Commit {commit_hash}")

    new_lines = [
        "# Quantization",
        f"Created with [lambda-quant](https://github.com/LambdaLabsML/lambda-quant/tree/{commit_hash})\n",
        f"Quantized using `{quantization_library}`\n",
        f"`Python {sys.version}`\n",
        f"Steps to create:",
        f"1. `git clone https://github.com/LambdaLabsML/lambda-quant`",
        f"2. `git checkout {commit_hash}`",
        f"3. `python {' '.join(sys.argv)}`",
        "# Original README.md:\n",
    ]
    new_content = "\n".join(new_lines)
    LOGGER.info(f"Writing {new_content} into README.md")
    with open(f"{metdata_dir}/README.md") as fp:
        readme_content = fp.read()
    with open(f"{metdata_dir}/README.md", "w") as fp:
        fp.write(new_content + readme_content)

    LOGGER.info(f"Dumping `pip freeze` to {metdata_dir}/requirements.txt")
    freeze = subprocess.check_output(["pip", "freeze"]).decode()
    with open(f"{metdata_dir}/requirements.txt", "w") as fp:
        fp.write(freeze)

    LOGGER.info(f"Dumping `args` to {metdata_dir}/lambda-quant-args.json")
    with open(f"{metdata_dir}/lambda-quant-args.json", "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    main()
