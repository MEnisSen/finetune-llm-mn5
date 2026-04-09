import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "results")


def run_inference(model_path: str, message: str, max_new_tokens: int) -> None:
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    input_text = f"User: {message}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(input_text):].strip() if response.startswith(input_text) else response

    print(f"\nInput : {message}")
    print(f"Output: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model with a single prompt.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to fine-tuned model directory (default: ./results).",
    )
    parser.add_argument(
        "--message",
        default="What are the symptoms of type 2 diabetes?",
        help="Input message for inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()
    run_inference(args.model, args.message, args.max_new_tokens)


if __name__ == "__main__":
    main()
