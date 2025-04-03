# Lambda Quant

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install torch
pip install -r requirements.txt
```

## Quantizing Llama-3.3 70B

```bash
python quantize.py -m meta-llama/Llama-3.3-70B-Instruct -q {AWQ-Int4,GPTQ-Int4,GPTQ-Int8,Dynamic-F8}
```
