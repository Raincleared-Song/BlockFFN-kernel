# Inference

This code is modified based on [FR-Spec](https://github.com/thunlp/FR-Spec).

## Install

Change ARCH environment variable to your GPU architecture.
Default 80 for A100.

- A100: `ARCH=80`
- RTX 3090: `ARCH=86`
- RTX 4090: `ARCH=89`
- Jetson Orin NX: `ARCH=87`
- H100: `ARCH=90`

Then run the following command to install the package.

```bash
ARCH=80 python3 setup.py install
```

## Simple generate example

```bash
cd tests/
# echo "=== baseline ==="
python3 test_generate.py --model-path SparseLLM/BlockFFN-v2-4.5B-A1.xB
# echo "=== baseline + ffn kernel ==="
python3 test_generate.py --model-path SparseLLM/BlockFFN-v2-4.5B-A1.xB --use-kernel
```

## Run the benchmark

```bash
pip install -r other_requirements.txt
bash scripts/eval/spec_bench/run_all.sh --model-path SparseLLM/BlockFFN-v2-4.5B-A1.xB 
```