import argparse
import os
import logging

import torch
import datasets
from transformers import AutoTokenizer


LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=["awq-int4", "gptq-int8", "gptq-int4", "fp8", "fp8-dynamic"],
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

    # ds = ds.map(tokenize, remove_columns=ds.column_names)

    ds = datasets.load_dataset(args.dataset, args.dataset_name, split=args.split)
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    model_name = os.path.basename(args.model)

    if args.quantization == "awq-int4":
        from awq import AutoAWQForCausalLM

        ds = [q["text"] for q in ds.map(preprocess, remove_columns=ds.column_names)]

        with device:
            model = AutoAWQForCausalLM.from_pretrained(
                args.model,
                low_cpu_mem_usage=True,
                use_cache=False,
                attn_implementation="flash_attention_2",
            )

        model.quantize(
            tokenizer,
            quant_config={
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            },
            calib_data=ds,
            max_calib_samples=args.num_samples,
            max_calib_seq_len=args.seq_length,
        )

        quant_name = f"{model_name}-AWQ"
        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)
    elif args.quantization == "gptq-int4":
        raise NotImplementedError(args.quantization)
    elif args.quantization == "gptq-int8":
        raise NotImplementedError(args.quantization)
    else:
        raise NotImplementedError(args.quantization)


if __name__ == "__main__":
    main()
