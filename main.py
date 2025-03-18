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
            "awq-int4",
            "gptq-int8",
            "gptq-int4",
            "w4a16-int4",
            "w8a8-int8",
            "w8a8-fp8",
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

    # ds = ds.map(tokenize, remove_columns=ds.column_names)

    ds = datasets.load_dataset(args.dataset, args.dataset_name, split=args.split)
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-{args.quantization}"

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

        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)
    elif args.quantization in ["gptq-int4", "gptq-int8"]:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quant_config = BaseQuantizeConfig(
            bits=4 if "int4" in args.quantize else 8,
            group_size=128,
            desc_act=False,
        )

        with device:
            model = AutoGPTQForCausalLM.from_pretrained(
                args.model,
                quant_config,
                use_cache=False,
                attn_implementation="flash_attention_2",
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names).to_list()
        # TODO what should batch size be?
        model.quantize(ds, batch_size=1)

        model.save_quantized(quant_name)
        tokenizer.save_pretrained(quant_name)
    elif args.quantization in ["w4a16-int4", "w8a8-int8", "w8a8-fp8"]:
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
                        "w4a16-int4": "W4A16",
                        "w8a8-int8": "W8A8",
                        "w8a8-fp8": "FP8_DYNAMIC",
                    }[args.quantization],
                    ignore=["lm_head"],
                    dampening_frac=0.1,
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
