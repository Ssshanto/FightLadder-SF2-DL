"""
Interpretability Framework for FightLadder SF2

Rigorous analysis of learned representations using information-theoretic,
unsupervised, and geometric methods - without hand-coded domain concepts.

Core Modules:
- Module A: Mutual Information Analysis (mi_analysis.py)
  Computes I(action; feature) to measure what game state features
  the policy depends on for decision-making.

- Module B: Unsupervised Concept Discovery (concept_discovery.py)
  Discovers emergent clusters in activation space without predefined concepts.

- Module C: Representation Geometry Analysis (geometry_analysis.py)
  PCA, separability metrics, and linear probes for representation structure.

Supporting Modules:
- Ground Truth Extraction (ground_truth.py): RAM-based feature extraction
- Representation Metrics (representation_metrics.py): CKA, KL divergence, etc.
- Analysis Runner (analysis_runner.py): Unified analysis pipeline

Usage:
    from interpretability import (
        RolloutCollector,
        MutualInformationAnalyzer,
        ConceptDiscoverer,
        GeometryAnalyzer,
    )

    # Collect rollouts with ground truth features
    collector = RolloutCollector(env, model)
    rollout_data = collector.collect(n_steps=10000)

    # Analyze what features the policy uses
    mi_analyzer = MutualInformationAnalyzer()
    mi_results = mi_analyzer.analyze(rollout_data)

    # Discover emergent concepts
    concept_discoverer = ConceptDiscoverer()
    concepts = concept_discoverer.discover(rollout_data.activations)

    # Or run full analysis pipeline:
    python interpretability/analysis_runner.py --model-path model.zip --output results/
"""

# =============================================================================
# Ground Truth Extraction (RAM-based features)
# =============================================================================
from .ground_truth import (
    GroundTruthExtractor,
    RolloutCollector,
    RolloutData,
    GROUND_TRUTH_FEATURES,
)

# =============================================================================
# Module A: Mutual Information Analysis
# =============================================================================
from .mi_analysis import (
    MutualInformationAnalyzer,
    MIAnalysisResult,
)

# =============================================================================
# Module B: Unsupervised Concept Discovery
# =============================================================================
from .concept_discovery import (
    ConceptDiscoverer,
    ConceptDiscoveryResult,
    ClusterProfile,
)

# =============================================================================
# Module C: Representation Geometry Analysis
# =============================================================================
from .geometry_analysis import (
    GeometryAnalyzer,
    GeometryAnalysisResult,
)

# =============================================================================
# Supporting: Representation Metrics
# =============================================================================
from .representation_metrics import (
    RepresentationMetrics,
    compute_all_metrics,
)

__all__ = [
    # Ground truth
    'GroundTruthExtractor',
    'RolloutCollector',
    'RolloutData',
    'GROUND_TRUTH_FEATURES',
    # Module A: MI Analysis
    'MutualInformationAnalyzer',
    'MIAnalysisResult',
    # Module B: Concept Discovery
    'ConceptDiscoverer',
    'ConceptDiscoveryResult',
    'ClusterProfile',
    # Module C: Geometry Analysis
    'GeometryAnalyzer',
    'GeometryAnalysisResult',
    # Representation Metrics
    'RepresentationMetrics',
    'compute_all_metrics',
]
