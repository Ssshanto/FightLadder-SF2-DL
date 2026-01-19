import stable_retro as retro
import os
import numpy as np


SF_BONUS_LEVEL = [4, 8, 12]

SF_DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

retro_directory = os.path.dirname(retro.__file__)
sf_game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis-v0"
SF_STATE_DIR = os.path.join(retro_directory, sf_game_dir)

sf_game = "StreetFighterIISpecialChampionEdition-Genesis-v0"

START_STATUS = 0

END_STATUS = 1

BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

SF_COMBOS_BUTTONS = [
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['X']], # 'Hadouken-R'
    [['RIGHT'], ['DOWN'], ['DOWN', 'RIGHT'], ['X']], # 'Shoryuken-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['A']], # 'Tatsumaki-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['X']], # 'Hadouken-L'
    [['LEFT'], ['DOWN'], ['DOWN', 'LEFT'], ['X']], #'Shoryuken-L'
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['A']], # 'Tatsumaki-L'
]
SF_COMBOS = []
for combo_buttons in SF_COMBOS_BUTTONS:
    action_seq = []
    for combo_button in combo_buttons:
        button = [int(b in combo_button) for b in BUTTONS]
        for _ in range(2):
            action_seq.append(np.array(button))
    SF_COMBOS.append(action_seq)

DIRECTIONS_BUTTONS = [
    [], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], 
    ['UP', 'LEFT'], ['UP', 'RIGHT'], ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'], 
]

ATTACKS_BUTTONS = [
    [], ['B'], ['A'], ['C'], ['Y'], ['X'], ['Z'],
]

SELECT_CHARACTER_MOVEMENTS = {
    'NO_OP': [],
    'START': ['START'],
    'LEFT': ['LEFT'],
    'RIGHT': ['RIGHT'],
    'UP': ['UP'],
    'DOWN': ['DOWN'],
}
SELECT_CHARACTER_BUTTONS = {}
for k, v in SELECT_CHARACTER_MOVEMENTS.items():
    SELECT_CHARACTER_BUTTONS[k] = np.array([int(b in v) for b in BUTTONS])
SELECT_CHARACTER_SEQUENCES = {
    'Ryu': [],
    'Honda': ['RIGHT'],
    'Blanka': ['RIGHT'] * 2,
    'Guile': ['RIGHT'] * 3,
    'Balrog': ['RIGHT'] * 4,
    'Vega': ['RIGHT'] * 5,
    'Ken': ['DOWN'],
    'Chunli': ['DOWN'] + ['RIGHT'],
    'Zangief': ['DOWN'] + ['RIGHT'] * 2,
    'Dhalsim': ['DOWN'] + ['RIGHT'] * 3,
    'Sagat': ['DOWN'] + ['RIGHT'] * 4,
    'Bison': ['DOWN'] + ['RIGHT'] * 5,
}


# =============================================================================
# Interpretability Framework Constants
# =============================================================================
#
# NOTE: The hand-crafted concept definitions have been removed for publication.
# See main/interpretability/README.md for the rigorous approach using raw RAM values.
#
# The threshold constants below are kept for backward compatibility but are NOT
# used by the new interpretability framework.
#
# References:
# - Fighting Game Theory: Footsies Handbook (Sonichurricane), EventHubs, Infil's Glossary
# - ML Interpretability: Concept Bottleneck Models (Koh et al., ICML 2020),
#   Linear Representation Hypothesis (Mikolov et al., 2013; Park et al., 2024)

# =============================================================================
# SPACING THRESHOLDS (SF2 Genesis calibrated)
# =============================================================================
# Derived from SF2 hitbox data and stage dimensions
# - Throw range: ~35-45 pixels
# - Light attacks: ~50-70 pixels
# - Heavy/sweep: ~80-110 pixels
# - Projectile optimal: 150+ pixels

CONCEPT_SPACING_CLOSE = 55     # Inside throw range - immediate mixup threat
CONCEPT_SPACING_MID = 100      # Normal attack range - primary footsies zone
CONCEPT_SPACING_FAR = 180      # Outside normals - projectile/approach decisions

# =============================================================================
# STAGE CONTROL THRESHOLDS
# =============================================================================
CONCEPT_CORNER_THRESHOLD = 35  # Within this distance from edge = cornered
CONCEPT_STAGE_WIDTH = 256      # Approximate visible stage width

# =============================================================================
# TEMPORAL WINDOWS (frames at 60fps)
# =============================================================================
CONCEPT_MOMENTUM_WINDOW = 30   # ~0.5 seconds for momentum calculation
CONCEPT_ADVANTAGE_WINDOW = 10  # ~0.17 seconds for neutral state estimation

# Health values (SF2 specific)
CONCEPT_FULL_HP = 176

# =============================================================================
# CONCEPT DEFINITIONS - Grounded in FGC Theory
# =============================================================================
# Each concept corresponds to documented fighting game concepts:
# - Spacing: Footsies/neutral game (Sonichurricane's Footsies Handbook)
# - Neutral State: Frame advantage proxy (PlayStation Fighting Game Guide)
# - Momentum: Exchange differential (PeerJ Adaptive AI, 2025)
# - Stage Control: Corner pressure (EventHubs Footsies 101)
# - Life Lead: HP advantage (universal game concept)
# - Round Timer: Time pressure (universal game mechanic)
# - Offense State: Play style indicator (observable behavior)

CONCEPT_NAMES = [
    'spacing',        # Footsies: distance management (close/mid/far)
    'neutral_state',  # Frame advantage proxy (disadvantage/neutral/advantage)
    'momentum',       # Recent HP exchange differential
    'stage_control',  # Corner/positional advantage
    'life_lead',      # HP advantage ratio
    'round_timer',    # Time remaining normalized
    'offense_state',  # Current offensive intensity
]

# Number of classes for classification probes
# Follows Concept Bottleneck Model design principles
CONCEPT_NUM_CLASSES = {
    'spacing': 3,        # close, mid, far (footsies zones)
    'neutral_state': 3,  # disadvantage, neutral, advantage
    'momentum': 3,       # losing, neutral, winning
    'stage_control': 3,  # cornered, neutral, opponent_cornered
    'life_lead': 3,      # behind, even, ahead
    'round_timer': 3,    # early, mid, late
    'offense_state': 2,  # passive, aggressive
}

# =============================================================================
# TACTICAL PRIMITIVES (Action Clustering)
# =============================================================================
# Based on fighting game strategy analysis - common action patterns
# Reference: Cornell Game Theory Blog "Fighting Games and Game Theory" (2021)
# - Atewaza (approach attack), Okiwaza (pre-emptive), Sashigaeshi-waza (punish)

TACTICAL_PRIMITIVES = {
    0: 'approach',    # Moving towards opponent (atewaza setup)
    1: 'pressure',    # Attacking at close range (okizeme/pressure)
    2: 'punish',      # Responding to opponent mistake (sashigaeshi-waza)
    3: 'retreat',     # Moving away, defensive spacing
    4: 'neutral',     # Footsies, spacing, waiting for opportunity
}

# =============================================================================
# STRATEGY LABELS (Opponent Analysis)
# =============================================================================
# High-level play style categories for opponent modeling
STRATEGY_NAMES = [
    'aggressive',   # Rushdown, constant offense
    'defensive',    # Reactive, punish-focused
    'spacing',      # Footsies-focused, mid-range control
    'rushdown',     # Close-range mixup heavy
    'adaptive',     # Changes strategy based on opponent
]