import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = "./models"

def run_inference(model_name: str, message: str, max_new_tokens: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODELS_DIR,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    prompt = f"User: {message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = generated[len(prompt):].strip() if generated.startswith(prompt) else generated

    print("\n=== Basic Inference Test ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Message: {message}")
    print(f"Response: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic single-message inference test.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--message",
        default="Hello! Can you briefly explain what fine-tuning is?",
        help="Single input message for inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        message=args.message,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
