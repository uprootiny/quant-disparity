#!/usr/bin/env python3
"""
quant-fairness CLI: Find critical layers for fair multilingual quantization.

Usage:
    quant-fairness sweep --model gpt2
    quant-fairness sweep --model facebook/opt-125m --langs en,he,ar,zh
    quant-fairness recommend --model gpt2 --top 3
"""

import argparse
import sys
import json


def cmd_sweep(args):
    """Run full layer sweep."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .sweep import layer_sweep, DEFAULT_TEXTS

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    # Parse languages
    if args.langs:
        langs = args.langs.split(',')
        texts = {l: DEFAULT_TEXTS.get(l, DEFAULT_TEXTS['en']) for l in langs}
    else:
        texts = DEFAULT_TEXTS

    print(f"Languages: {list(texts.keys())}")
    print(f"Running layer sweep...")
    print()

    results = layer_sweep(model, tokenizer, texts, verbose=True)

    print()
    print("=" * 50)
    print("RESULTS (sorted by criticality)")
    print("=" * 50)
    print(f"{'Rank':<6} {'Layer':<8} {'Disparity':<12} {'Assessment'}")
    print("-" * 50)

    for rank, (layer_idx, disp) in enumerate(results, 1):
        if disp < 1.0:
            assessment = "CRITICAL"
        elif disp < 10.0:
            assessment = "Important"
        elif disp < 50.0:
            assessment = "Moderate"
        else:
            assessment = "Low priority"

        print(f"{rank:<6} L{layer_idx:<7} {disp:<11.2f}x {assessment}")

    print()
    top3 = [r[0] for r in results[:3]]
    print(f"RECOMMENDATION: Protect layers {top3}")

    if args.json:
        output = {
            "model": args.model,
            "results": [{"layer": l, "disparity": d} for l, d in results],
            "recommended": top3
        }
        print()
        print(json.dumps(output, indent=2))


def cmd_recommend(args):
    """Quick recommendation of layers to protect."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .sweep import quick_sweep, DEFAULT_TEXTS

    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    layers = quick_sweep(model, tokenizer, top_n=args.top)

    if args.json:
        print(json.dumps({"model": args.model, "protect": layers}))
    else:
        print(",".join(str(l) for l in layers))


def main():
    parser = argparse.ArgumentParser(
        description="Find critical layers for fair multilingual quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quant-fairness sweep --model gpt2
  quant-fairness sweep --model gpt2 --langs en,he,ar,zh --json
  quant-fairness recommend --model gpt2 --top 3

Based on 80 experiments showing that protecting specific "gateway" layers
(typically L0 + L11 for GPT-2-like architectures) eliminates most quantization
disparity between high-resource and low-resource languages.

For GPT-2: Protect L0, L9, L11 (~17% overhead, 0.59x disparity)
For OPT: Protect L4, L6, L11 (architecture-specific)
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Full layer sweep')
    sweep_parser.add_argument('--model', required=True, help='Model name/path')
    sweep_parser.add_argument('--langs', help='Comma-separated language codes (default: en,he,ar,zh)')
    sweep_parser.add_argument('--json', action='store_true', help='Output as JSON')
    sweep_parser.set_defaults(func=cmd_sweep)

    # recommend command
    rec_parser = subparsers.add_parser('recommend', help='Quick layer recommendation')
    rec_parser.add_argument('--model', required=True, help='Model name/path')
    rec_parser.add_argument('--top', type=int, default=3, help='Number of layers (default: 3)')
    rec_parser.add_argument('--json', action='store_true', help='Output as JSON')
    rec_parser.set_defaults(func=cmd_recommend)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
