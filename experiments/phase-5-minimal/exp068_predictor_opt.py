#!/usr/bin/env python3
"""
Exp-068: Test combined predictor on OPT-125M
Goal: Does position+stats predict OPT's critical layers?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}
num_layers = 12

TEXTS = {
    'en': 'Fox.',
    'he': 'שועל.',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def get_layer_stats(layer_idx):
    """Get statistics for ranking"""
    prefix = f"model.decoder.layers.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if prefix in name and 'weight' in name:
            weights.append(param.data.flatten())

    if not weights:
        return {'variance': 0, 'outlier_ratio': 0, 'sparsity': 0}

    all_w = torch.cat(weights)
    std = all_w.std().item()

    return {
        'variance': all_w.var().item(),
        'outlier_ratio': (all_w.abs() > 3 * std).float().mean().item() if std > 0 else 0,
        'sparsity': (all_w.abs() < 0.01).float().mean().item(),
    }

def compute_score(layer_idx, stats):
    """Composite score: higher = more critical to protect"""
    # Position score
    pos_dist = min(layer_idx, num_layers - 1 - layer_idx)
    position_score = 1.0 / (1 + pos_dist)

    # Statistics scores (normalized 0-1)
    var_ranks = sorted(range(12), key=lambda i: -stats[i]['variance'])
    out_ranks = sorted(range(12), key=lambda i: -stats[i]['outlier_ratio'])
    spa_ranks = sorted(range(12), key=lambda i: -stats[i]['sparsity'])

    var_score = 1 - var_ranks.index(layer_idx) / 11
    out_score = 1 - out_ranks.index(layer_idx) / 11
    spa_score = 1 - spa_ranks.index(layer_idx) / 11

    return 0.4 * position_score + 0.2 * var_score + 0.2 * out_score + 0.2 * spa_score

def protect_and_measure(layers):
    """Protect specified layers and measure disparity"""
    restore()

    # Get baseline first
    baseline_en = ppl(TEXTS['en'])
    baseline_he = ppl(TEXTS['he'])

    # Quantize
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'layers.{l}.' in name for l in layers):
                continue
            if 'final_layer_norm' in name:
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    q_en = ppl(TEXTS['en'])
    q_he = ppl(TEXTS['he'])

    en_deg = (q_en - baseline_en) / baseline_en * 100
    he_deg = (q_he - baseline_he) / baseline_he * 100

    if en_deg <= 0:
        return float('inf')

    return he_deg / en_deg

print("=" * 70)
print("OPT-125M: Testing Combined Predictor")
print("=" * 70)

# Get stats for all layers
stats = {i: get_layer_stats(i) for i in range(num_layers)}

print("\n1. OPT Layer Statistics")
print("-" * 70)
print(f"{'Layer':<8} {'Variance':>12} {'Outlier%':>12} {'Sparsity%':>12}")
print("-" * 50)

for i in range(12):
    s = stats[i]
    print(f"L{i:<7} {s['variance']:>12.6f} {s['outlier_ratio']*100:>11.3f}% {s['sparsity']*100:>11.1f}%")

# Compute combined scores
print("\n2. Combined Scores")
print("-" * 40)

scores = []
for i in range(12):
    score = compute_score(i, stats)
    scores.append((i, score))
    print(f"L{i}: {score:.3f}")

sorted_scores = sorted(scores, key=lambda x: -x[1])

print("\n3. Ranking by Combined Score")
print("-" * 30)
for rank, (layer, score) in enumerate(sorted_scores, 1):
    marker = " <-- predicted" if rank <= 2 else ""
    print(f"#{rank}: Layer {layer} (score={score:.3f}){marker}")

# Test predictions
print("\n4. Testing Predictions")
print("-" * 60)

# Predicted (top-2 combined)
top2_combined = [l for l, s in sorted_scores[:2]]
disp_combined = protect_and_measure(top2_combined)

# Known best for OPT (from exp-062)
disp_known = protect_and_measure([4, 9])

# First + Last (position)
disp_pos = protect_and_measure([0, 11])

print(f"{'Method':<30} {'Layers':<15} {'Disparity':>12}")
print("-" * 60)
print(f"{'Combined predictor':<30} {str(top2_combined):<15} {disp_combined:>11.1f}x")
print(f"{'Known best (L4+L9)':<30} {'[4, 9]':<15} {disp_known:>11.1f}x")
print(f"{'Position only (L0+L11)':<30} {'[0, 11]':<15} {disp_pos:>11.1f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if top2_combined == [0, 11]:
    print("Predictor selected first+last layers")
    if disp_combined < disp_known:
        print("SUCCESS: Predictor beats known best!")
    else:
        print(f"PARTIAL: Predictor works ({disp_combined:.1f}x) but known L4+L9 is better ({disp_known:.1f}x)")
else:
    print(f"Predictor selected: {top2_combined}")
    print(f"vs Known best: [4, 9]")
