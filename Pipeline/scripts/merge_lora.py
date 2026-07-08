import argparse, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading base: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained(args.adapter)
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging...")
    model = model.merge_and_unload()
    print(f"Saving to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("Done.")

if __name__ == "__main__":
    main()
