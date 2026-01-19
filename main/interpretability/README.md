# Interpretability Framework for FightLadder SF2

## Design Philosophy

This framework implements **rigorous interpretability analysis** for peer-reviewed publication. It avoids hand-crafted domain concepts that would introduce circular reasoning in probing analysis.

### Key Principles

1. **No Hand-Crafted Concepts**: We do NOT define categories like "close/mid/far spacing" with arbitrary thresholds. Instead, we use raw RAM values as ground truth.

2. **Information-Theoretic Analysis**: We compute mutual information I(action; feature) to measure what the policy has learned to depend on, without assuming what it *should* encode.

3. **Unsupervised Discovery**: Concept discovery uses clustering on activations to find what the network naturally organizes, not what we expect it to encode.

---

## Modules

### Core Analysis Modules

| Module | File | Purpose |
|--------|------|---------|
| Ground Truth Extraction | `ground_truth.py` | Extract raw RAM values (positions, HP, timer) |
| Mutual Information Analysis | `mi_analysis.py` | Compute I(action; feature) for all features |
| Concept Discovery | `concept_discovery.py` | Unsupervised clustering of activations |
| Geometry Analysis | `geometry_analysis.py` | PCA, separability, linear probes |
| Representation Metrics | `representation_metrics.py` | CKA, KL divergence, intrinsic dimensionality |

### Runtime Integration

| Module | File | Purpose |
|--------|------|---------|
| Runtime Callback | `callbacks/runtime_interpretability.py` | Collect activations during training |
| Analysis Runner | `analysis_runner.py` | Unified post-training analysis pipeline |

---

## Ground Truth Features

These are the ONLY features used for analysis, extracted directly from RAM:

| Feature | Source | Description |
|---------|--------|-------------|
| `agent_x` | RAM | Agent horizontal position (0-256) |
| `enemy_x` | RAM | Enemy horizontal position (0-256) |
| `distance` | Derived | `abs(agent_x - enemy_x)` |
| `relative_position` | Derived | `agent_x - enemy_x` (signed) |
| `agent_hp` | RAM | Agent health points (0-176) |
| `enemy_hp` | RAM | Enemy health points (0-176) |
| `hp_diff` | Derived | `agent_hp - enemy_hp` |
| `hp_ratio` | Derived | `agent_hp / (agent_hp + enemy_hp)` |
| `timer` | RAM | Round countdown (0-99) |

**No arbitrary thresholds are applied to these values.**

---

## Usage

### Training with Runtime Interpretability

```bash
cd main
python train.py \
    --state Champion.Level1.RyuVsGuile \
    --total-steps 100000 \
    --enable-interpretability \
    --interp-log-dir interpretability_logs \
    --interp-log-freq 1000 \
    --interp-probe-freq 50000
```

This will:
1. Collect policy activations during training
2. Periodically compute MI between actions and ground truth features
3. Log results to TensorBoard under `interpretability/`
4. Save detailed analysis to JSON files

### Post-Training Analysis

```bash
cd main
python interpretability/analysis_runner.py \
    --model-path trained_models/ppo_ryu_final_steps.zip \
    --output interpretability_results \
    --n-episodes 20
```

This runs the full analysis pipeline:
1. **MI Analysis**: Which features does the policy depend on?
2. **Concept Discovery**: What clusters emerge in activation space?
3. **Geometry Analysis**: PCA, linear separability, probing accuracy

### Programmatic Usage

```python
from interpretability import (
    RolloutCollector,
    MutualInformationAnalyzer,
    ConceptDiscoverer,
    GeometryAnalyzer,
)

# Collect data
collector = RolloutCollector()
# ... collect rollouts ...
data = collector.get_data()

# MI Analysis: What features matter to the policy?
mi_analyzer = MutualInformationAnalyzer(n_bins=10)
mi_results = mi_analyzer.analyze(data)
print(mi_results.get_ranking())
# Output: [('distance', 0.45), ('hp_diff', 0.32), ...]

# Concept Discovery: What does the network organize?
discoverer = ConceptDiscoverer(n_clusters=8)
concepts = discoverer.discover(data.activations)
print(concepts.cluster_profiles)

# Geometry Analysis: How is the representation structured?
geometry = GeometryAnalyzer()
geo_results = geometry.analyze(data.activations, data.features)
print(f"Effective dimensionality: {geo_results.effective_dim}")
```

---

## TensorBoard Metrics

When training with `--enable-interpretability`, these metrics are logged:

| Metric | Interpretation |
|--------|----------------|
| `interpretability/mi_action_distance` | How much action depends on distance |
| `interpretability/mi_action_hp_diff` | How much action depends on HP difference |
| `interpretability/mi_action_timer` | How much action depends on timer |

**High MI values** indicate the policy has learned to use that feature.
**Low MI values** indicate the feature has little influence on decisions.

---

## Output Files

### MI Analysis JSON
```json
{
  "timestep": 50000,
  "n_samples": 10000,
  "mi_results": {
    "mi_action_distance": 0.423,
    "mi_action_hp_diff": 0.312,
    "mi_action_agent_x": 0.156,
    ...
  }
}
```

### Concept Discovery
- Cluster profiles with mean/std of each feature
- Activation-to-cluster assignments
- Silhouette scores for cluster quality

### Geometry Analysis
- PCA explained variance ratios
- Linear probe accuracies for each feature
- Effective dimensionality estimates

---

## File Structure

```
main/interpretability/
├── __init__.py                 # Package exports
├── ground_truth.py             # Raw feature extraction (NO discretization)
├── mi_analysis.py              # Mutual information computation
├── concept_discovery.py        # Unsupervised clustering
├── geometry_analysis.py        # PCA, separability, probes
├── representation_metrics.py   # CKA, KL divergence
├── analysis_runner.py          # Unified analysis pipeline
├── callbacks/
│   ├── __init__.py
│   └── runtime_interpretability.py  # Training integration
└── README.md                   # This file
```

---

## Why No Hand-Crafted Concepts?

Traditional interpretability approaches define concepts like "close range" as `distance < 60 pixels`. This creates problems:

1. **Circular Reasoning**: If we train a probe to detect "close range" (defined by distance), high probe accuracy just means the network encodes distance—not that it has a "concept" of close range.

2. **Arbitrary Thresholds**: Why 60 pixels? Why not 50 or 70? The threshold is researcher-imposed, not learned by the agent.

3. **Missing Novel Strategies**: The agent might discover effective strategies at 47 pixels that don't fit our "close/mid/far" categories.

**Our Approach**: Use raw values, let MI analysis discover what matters, and let unsupervised clustering reveal how the network organizes its representations.

---

## Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib
```

Optional for visualization:
```bash
pip install pandas seaborn
```
