# Fine-Tuning an LLM with LoRA

This project demonstrates parameter-efficient fine-tuning of
`Qwen/Qwen2.5-0.5B-Instruct` with LoRA. The notebook converts question-and-answer
rows into training examples, trains LoRA adapters, saves them, and runs a sample
inference.

> The current notebook uses LoRA with FP16 weights. It does not load a quantized
> model, so it does not currently perform QLoRA training.

## Project structure

- `llm_fine_tuning_with_lora.ipynb` contains the training and inference workflow.
- `training_data/lora_training_data.csv` contains the `question` and `response`
  columns used for training.
- `lora_directory/` stores trainer checkpoints.
- `lora_peft_model/` stores the final LoRA adapter and tokenizer files.

## Requirements

- Python 3.10 or newer
- A CUDA-capable GPU with FP16 support
- PyTorch
- Transformers
- TRL
- PEFT
- Datasets
- pandas
- Jupyter Notebook or JupyterLab

Install the Python packages with:

```bash
pip install torch transformers trl peft datasets pandas jupyter
```

## Run the notebook

1. Open `llm_fine_tuning_with_lora.ipynb` from the repository root.
2. Select a Python environment containing the required packages.
3. Run the training cell. It loads the CSV data, creates the LoRA model, trains
   for three epochs, and saves the adapter to `lora_peft_model/`.
4. Run the inference cell. It reloads the base model and saved adapter, generates
   an answer for the sample prompt, and prints the result.

The first run downloads the base model from Hugging Face. Authentication is
optional for public models but can improve download rate limits.

## How the training works

Each CSV row is formatted as:

```text
Question: <question text>

Answer: <response text>
```

LoRA adds small trainable matrices to the model's attention projection layers:
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. The base model remains unchanged,
which makes the saved adapter much smaller than a complete model checkpoint.

The effective batch size is 6 examples: a device batch size of 2 multiplied by
3 gradient-accumulation steps.

## Using the saved adapter

The inference section first loads the same Qwen base model, then attaches the
adapter from `lora_peft_model/`. Both are required: the adapter directory does
not contain a standalone copy of the base model.
