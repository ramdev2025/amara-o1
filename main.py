"""
Fine-tune Google FunctionGemma-270M-IT as Amara-O1 for code generation on Modal.com
Auto-pushes to HuggingFace: https://huggingface.co/ramdev12345/amara-o1

Datasets: github-code, gsm8k, livecodebench
Training: LoRA fine-tuning with 4-bit quantization on A100
"""

import modal
from modal import Image, Secret, gpu

# Define Modal app
app = modal.App("amara-o1")

# Create custom image with dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "datasets==2.16.0",
        "peft==0.7.1",
        "bitsandbytes==0.41.3",
        "accelerate==0.25.0",
        "trl==0.7.10",
        "scipy",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
)

# Training configuration
WANDB_PROJECT = "amara-o1-code-generation"
MODEL_NAME = "google/functiongemma-270m-it"
OUTPUT_DIR = "/results/amara-o1"
HF_REPO = "ramdev12345/amara-o1"  # Your HuggingFace repo

@app.function(
    image=image,
    gpu=gpu.A100(count=1, memory=40),  # A100 40GB
    timeout=86400,  # 24 hours
    secrets=[Secret.from_name("huggingface-secret")],  # Create this in Modal dashboard
    volumes={"/results": modal.Volume.from_name("model-storage", create_if_missing=True)},
)
def train_model(
    max_steps: int = 10000,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_samples_per_dataset: int = 50000,
):
    """
    Fine-tune FunctionGemma with LoRA on code generation datasets
    """
    import os
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, concatenate_datasets
    from trl import SFTTrainer
    
    print("🚀 Starting FunctionGemma fine-tuning...")
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure 4-bit quantization for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    print("🤖 Loading FunctionGemma model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and preprocess datasets
    print("📚 Loading datasets...")
    
    def format_instruction_code(example, source_type):
        """Format examples into instruction-following format"""
        if source_type == "github":
            # GitHub code - use as completion task
            code = example.get("code", "")
            if len(code) > 100:
                # Split code for completion task
                split_point = len(code) // 2
                prompt = code[:split_point]
                completion = code[split_point:]
                text = f"<start_of_turn>user\nComplete this code:\n```python\n{prompt}\n```<end_of_turn>\n<start_of_turn>model\n```python\n{completion}\n```<end_of_turn>"
                return {"text": text}
            return None
        
        elif source_type == "gsm8k":
            # Math reasoning - convert to code generation
            question = example.get("question", "")
            answer = example.get("answer", "")
            text = f"<start_of_turn>user\nWrite a Python function to solve this problem:\n{question}<end_of_turn>\n<start_of_turn>model\n```python\ndef solve():\n    # {answer}\n    pass\n```<end_of_turn>"
            return {"text": text}
        
        elif source_type == "livecode":
            # Code generation benchmark
            problem = example.get("question_title", "") or example.get("problem", "")
            code = example.get("code", "") or example.get("solution", "")
            text = f"<start_of_turn>user\nSolve this coding problem:\n{problem}<end_of_turn>\n<start_of_turn>model\n```python\n{code}\n```<end_of_turn>"
            return {"text": text}
        
        return None
    
    # Load GitHub code dataset
    print("Loading codeparrot/github-code...")
    try:
        github_data = load_dataset(
            "codeparrot/github-code",
            languages=["Python"],
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        github_data = github_data.take(num_samples_per_dataset)
        github_data = [format_instruction_code(ex, "github") for ex in github_data]
        github_data = [ex for ex in github_data if ex is not None]
        print(f"✅ Loaded {len(github_data)} GitHub samples")
    except Exception as e:
        print(f"⚠️ Error loading GitHub dataset: {e}")
        github_data = []
    
    # Load GSM8K dataset
    print("Loading openai/gsm8k...")
    try:
        gsm8k_data = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_data = gsm8k_data.select(range(min(num_samples_per_dataset, len(gsm8k_data))))
        gsm8k_data = [format_instruction_code(ex, "gsm8k") for ex in gsm8k_data]
        gsm8k_data = [ex for ex in gsm8k_data if ex is not None]
        print(f"✅ Loaded {len(gsm8k_data)} GSM8K samples")
    except Exception as e:
        print(f"⚠️ Error loading GSM8K dataset: {e}")
        gsm8k_data = []
    
    # Load LiveCodeBench dataset
    print("Loading livecodebench/code_generation_lite...")
    try:
        livecode_data = load_dataset("livecodebench/code_generation_lite", split="train")
        livecode_data = livecode_data.select(range(min(num_samples_per_dataset, len(livecode_data))))
        livecode_data = [format_instruction_code(ex, "livecode") for ex in livecode_data]
        livecode_data = [ex for ex in livecode_data if ex is not None]
        print(f"✅ Loaded {len(livecode_data)} LiveCodeBench samples")
    except Exception as e:
        print(f"⚠️ Error loading LiveCodeBench dataset: {e}")
        livecode_data = []
    
    # Combine all datasets
    from datasets import Dataset
    all_data = github_data + gsm8k_data + livecode_data
    print(f"📊 Total training samples: {len(all_data)}")
    
    if len(all_data) == 0:
        raise ValueError("No data loaded! Check dataset availability.")
    
    train_dataset = Dataset.from_list(all_data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",  # Set to "wandb" if you want W&B logging
    )
    
    # Initialize trainer
    print("🏋️ Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    # Train!
    print("🔥 Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving model locally...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Push to HuggingFace Hub
    print(f"🚀 Pushing model to HuggingFace: {HF_REPO}")
    try:
        # Merge LoRA weights with base model for easier inference
        print("🔀 Merging LoRA adapters with base model...")
        model = model.merge_and_unload()
        
        # Create model card
        model_card = f"""---
license: gemma
base_model: {MODEL_NAME}
tags:
- code
- code-generation
- python
- function-calling
- gemma
- lora
library_name: transformers
---

# Amara-O1: Fine-tuned FunctionGemma for Code Generation

Amara-O1 is a fine-tuned version of Google's FunctionGemma-270M-IT, specialized for Python code generation and completion.

## Model Details

- **Base Model**: {MODEL_NAME}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 
  - codeparrot/github-code (Python)
  - openai/gsm8k (Math reasoning)
  - livecodebench/code_generation_lite (Coding benchmarks)
- **Training Steps**: {max_steps}
- **Parameters**: ~270M (base) + LoRA adapters

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "{HF_REPO}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Write a Python function to calculate fibonacci numbers"
inputs = tokenizer(
    f"<start_of_turn>user\\n{{prompt}}<end_of_turn>\\n<start_of_turn>model\\n",
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- **GPU**: NVIDIA A100 40GB
- **Batch Size**: {batch_size} per device
- **Gradient Accumulation**: {gradient_accumulation_steps} steps
- **Learning Rate**: {learning_rate}
- **Max Sequence Length**: {max_seq_length}
- **LoRA Config**:
  - r={lora_r}
  - alpha={lora_alpha}
  - dropout=0.05

## Intended Use

This model is designed for:
- Python code completion
- Function generation from natural language descriptions
- Code problem solving
- Programming assistance

## Limitations

- Specialized for Python code generation
- May not perform well on other programming languages
- Based on a 270M parameter model (smaller than GPT-3.5/4)
- Should be used as a coding assistant, not for production-critical code without review

## License

This model inherits the Gemma license from the base model.

## Citation

```bibtex
@misc{{amara-o1,
  author = {{Your Name}},
  title = {{Amara-O1: Fine-tuned FunctionGemma for Code Generation}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{{HF_REPO}}}}}
}}
```
"""
        
        # Save model card
        with open(f"{OUTPUT_DIR}/README.md", "w") as f:
            f.write(model_card)
        
        # Push to hub
        model.push_to_hub(
            HF_REPO,
            token=hf_token,
            commit_message="Upload Amara-O1 fine-tuned model"
        )
        
        tokenizer.push_to_hub(
            HF_REPO,
            token=hf_token,
            commit_message="Upload tokenizer"
        )
        
        print(f"✅ Successfully pushed to https://huggingface.co/{HF_REPO}")
        
    except Exception as e:
        print(f"⚠️ Error pushing to HuggingFace: {e}")
        print("Model saved locally but not pushed to hub")
    
    print("✨ Training complete!")
    print(f"Model saved locally: {OUTPUT_DIR}")
    print(f"Model available at: https://huggingface.co/{HF_REPO}")
    
    return OUTPUT_DIR


@app.function(
    image=image,
    secrets=[Secret.from_name("huggingface-secret")],
    volumes={"/results": modal.Volume.from_name("model-storage")},
)
def test_model(
    prompt: str = "Write a Python function to calculate fibonacci numbers",
    use_hub: bool = True
):
    """
    Test the fine-tuned model
    
    Args:
        prompt: The prompt to test
        use_hub: If True, load from HuggingFace Hub. If False, load from local storage
    """
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("🧪 Testing Amara-O1 model...")
    
    if use_hub:
        print(f"📦 Loading from HuggingFace Hub: {HF_REPO}")
        model = AutoModelForCausalLM.from_pretrained(
            HF_REPO,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        print(f"📦 Loading from local storage: {OUTPUT_DIR}")
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    # Generate
    inputs = tokenizer(
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("PROMPT:", prompt)
    print("="*50)
    print("RESPONSE:", response)
    print("="*50)
    
    return response


@app.local_entrypoint()
def main(
    action: str = "train",
    test_prompt: str = "Write a Python function to reverse a string",
    use_hub: bool = False
):
    """
    Main entry point for Amara-O1 training and testing
    
    Usage:
        # Train and push to HuggingFace
        modal run gemma_finetune.py --action train
        
        # Test from local storage
        modal run gemma_finetune.py --action test --test-prompt "Your prompt here"
        
        # Test from HuggingFace Hub (after pushing)
        modal run gemma_finetune.py --action test --use-hub true
    """
    if action == "train":
        print("🚀 Starting Amara-O1 training job on Modal...")
        result = train_model.remote()
        print(f"✅ Training complete!")
        print(f"📦 Model saved locally: {result}")
        print(f"🌐 Model available at: https://huggingface.co/{HF_REPO}")
    
    elif action == "test":
        print("🧪 Testing Amara-O1...")
        response = test_model.remote(prompt=test_prompt, use_hub=use_hub)
        print(f"\n✨ Test complete!")
    
    else:
        print("❌ Invalid action. Use 'train' or 'test'")


if __name__ == "__main__":
    main()