import argparse
import os
import logging

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    LOGGER.info(args)

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

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-{args.quantization}"

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

        model = GPTQModel.load(args.model, dict(bits=4, group_size=128))
        ds = ds.map(preprocess)
        ds = ds.map(tokenize, remove_columns=ds.column_names)
        model.quantize(ds.to_list(), batch_size=args.num_samples)
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


if __name__ == "__main__":
    main()
