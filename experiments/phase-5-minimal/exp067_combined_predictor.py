#!/usr/bin/env python3
"""
Exp-067: Combined predictor for critical layer identification
Goal: Create a composite score using position + statistics
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}
num_layers = 12

TEXTS = {
    'en': 'The quick brown fox.',
    'he': 'השועל החום המהיר.',
    'zh': '敏捷的棕色狐狸。',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def get_layer_stats(layer_idx):
    """Get statistics for ranking"""
    prefix = f"transformer.h.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            weights.append(param.data.flatten())

    if not weights:
        return {}

    all_w = torch.cat(weights)
    std = all_w.std().item()

    return {
        'variance': all_w.var().item(),
        'outlier_ratio': (all_w.abs() > 3 * std).float().mean().item(),
        'sparsity': (all_w.abs() < 0.01).float().mean().item(),
    }

def protect_and_measure(layers):
    """Protect specified layers and measure disparity"""
    restore()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'h.{l}.' in name for l in layers):
                continue
            if 'ln_f' in name:
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    restore()  # Fresh baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    restore()  # Quantize again for measurement
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'h.{l}.' in name for l in layers):
                continue
            if 'ln_f' in name:
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg <= 0:
        return float('inf')

    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    return sum(non_en) / len(non_en)

print("=" * 70)
print("Combined Predictor for Critical Layer Identification")
print("=" * 70)

# Get stats for all layers
stats = {i: get_layer_stats(i) for i in range(num_layers)}

# Define scoring function
def compute_score(layer_idx):
    """Composite score: higher = more critical to protect"""
    s = stats[layer_idx]

    # Position score: first and last layers are gateways
    pos_dist = min(layer_idx, num_layers - 1 - layer_idx)
    position_score = 1.0 / (1 + pos_dist)  # 1.0 for L0/L11, decreases inward

    # Statistics scores (normalized 0-1)
    var_ranks = sorted(range(12), key=lambda i: -stats[i]['variance'])
    out_ranks = sorted(range(12), key=lambda i: -stats[i]['outlier_ratio'])
    spa_ranks = sorted(range(12), key=lambda i: -stats[i]['sparsity'])

    var_score = 1 - var_ranks.index(layer_idx) / 11
    out_score = 1 - out_ranks.index(layer_idx) / 11
    spa_score = 1 - spa_ranks.index(layer_idx) / 11

    # Weighted combination
    # Position is most important (verified by L0's unique role)
    return 0.4 * position_score + 0.2 * var_score + 0.2 * out_score + 0.2 * spa_score

print("\n1. Combined Scores")
print("-" * 50)
print(f"{'Layer':<8} {'Position':>10} {'Variance':>10} {'Outlier':>10} {'Sparsity':>10} {'Combined':>10}")
print("-" * 70)

scores = []
for i in range(12):
    s = stats[i]
    pos_dist = min(i, num_layers - 1 - i)
    pos_score = 1.0 / (1 + pos_dist)

    var_ranks = sorted(range(12), key=lambda x: -stats[x]['variance'])
    out_ranks = sorted(range(12), key=lambda x: -stats[x]['outlier_ratio'])
    spa_ranks = sorted(range(12), key=lambda x: -stats[x]['sparsity'])

    var_score = 1 - var_ranks.index(i) / 11
    out_score = 1 - out_ranks.index(i) / 11
    spa_score = 1 - spa_ranks.index(i) / 11

    combined = compute_score(i)
    scores.append((i, combined))

    print(f"L{i:<7} {pos_score:>10.3f} {var_score:>10.3f} {out_score:>10.3f} {spa_score:>10.3f} {combined:>10.3f}")

# Sort by combined score
sorted_scores = sorted(scores, key=lambda x: -x[1])

print("\n2. Layers Ranked by Combined Score")
print("-" * 30)
for rank, (layer, score) in enumerate(sorted_scores, 1):
    print(f"#{rank}: Layer {layer} (score={score:.3f})")

# Test predictor
print("\n3. Testing Predictor")
print("-" * 60)

# Get baseline disparity with no protection
restore()
baseline_disp = protect_and_measure([])  # Empty = no protection except biases

print(f"\n{'Selection':<30} {'Layers':<15} {'Disparity':>12}")
print("-" * 60)

# Top-2 by combined score
top2_combined = [l for l, s in sorted_scores[:2]]
disp_combined = protect_and_measure(top2_combined)
print(f"{'Top-2 combined score':<30} {str(top2_combined):<15} {disp_combined:>11.2f}x")

# Top-2 by variance only
var_ranks = sorted(range(12), key=lambda i: -stats[i]['variance'])
top2_var = var_ranks[:2]
disp_var = protect_and_measure(top2_var)
print(f"{'Top-2 variance only':<30} {str(top2_var):<15} {disp_var:>11.2f}x")

# First + Last (position only)
disp_pos = protect_and_measure([0, 11])
print(f"{'First + Last (position)':<30} {'[0, 11]':<15} {disp_pos:>11.2f}x")

# Top-2 by outlier ratio
out_ranks = sorted(range(12), key=lambda i: -stats[i]['outlier_ratio'])
top2_out = out_ranks[:2]
disp_out = protect_and_measure(top2_out)
print(f"{'Top-2 outlier ratio':<30} {str(top2_out):<15} {disp_out:>11.2f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if top2_combined == [0, 11]:
    print("SUCCESS: Combined predictor identifies L0+L11 as most critical!")
else:
    print(f"Combined predictor selected: {top2_combined}")

winner = min([
    ("Combined", disp_combined),
    ("Variance", disp_var),
    ("Position", disp_pos),
    ("Outlier", disp_out),
], key=lambda x: x[1])
print(f"Best method: {winner[0]} with {winner[1]:.2f}x disparity")
