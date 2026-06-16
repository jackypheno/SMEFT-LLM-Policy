# ============================================================
# smeft_rl/config.py
# All hyperparameters in one place.
# Edit this file to change any training setting.
# ============================================================

from pathlib import Path
import torch

REPO            = "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly"

SAVE_DIR        = Path("/home/kumarj/links/scratch/anomaly/models/rl-policy")
#PRETRAINED_PATH = "/home/kumarj/links/scratch/anomaly/models/policy4-claude/smeft_policy_best.pt"
PRETRAINED_PATH = "/home/kumarj/links/scratch/anomaly/models/policy4-claude-0f-2f/smeft_policy_best.pt"

#VOCAB_PATH      = "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly/src/vocab_noindex.txt"
VOCAB_PATH      = "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly/smeft_rl/vocab_0f2f.txt"
vocab= Path("/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly/smeft_rl/vocab_0f2f.txt").read_text().split()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B              = 20       # sequences per generation round
G              = 5        # generation rounds per batch
N_BATCHES      = 60


MAX_NEW_TOKENS = 12
N_UPDATE_STEPS = 4
PLOT_EVERY     = 10

#SEARCH_SPACE_SIZE  = 6884 ###FIXME
ROLLOUTS_PER_BATCH = B * G

EPS         = 0.2
ADV_STD_MIN = 0.1

TOP_K = 5

KL_COEF_START = 0.00
KL_COEF_END   = 0.05
KL_WARMUP     = 15        # batches over which KL ramps up

LR_PEAK = 1e-4

EPSILON_START = 0.8
EPSILON_END   = 0.05
EPSILON_DECAY = 0.88

REPLAY_BUFFER_SIZE = 100
REPLAY_MIN_REWARD  = 0.3
REPLAY_FRAC        = 0.25

ENTROPY_COEF = 0.1

FREE_OPS          = 3
JUNK_PENALTY      = 0.10
LENGTH_PENALTY    = 0.1
TERMINATION_BONUS = 0.05
"""
OP_SCORES = {
    "G":          0.40,
    "phi":        0.200,
    "W":          0.300,
    "phiWB":      0.800,
    "phiD":       1.000,
    "phiW":       0.01,
    "phiB":       0.04,
    "phiWtilde":  0.08,
    "phiBox":     0.09,
}
"""
OP_SCORES = {
    "phil3_11":   1.0,
    "G":          0.9,
    "phi":        0.2,
    "W":          0.05,
    "phiWB":      0.01,
    "phiD":       0.01,
    "phiW":       0.01,
    "phiB":       0.02,
    "phiWtilde":  0.03,
    "phiBox":     0.04,
}

WC_NAME_MAP = {
    "G":             "G",
    "Gtilde":        "Gtilde",
    "W":             "W",
    "Wtilde":        "Wtilde",
    "phi":           "phi",
    "phiB":          "phiB",
    "phiBtilde":     "phiBtilde",
    "phiBox":        "phiBox",
    "phiD":          "phiD",
    "phiG":          "phiG",
    "phiGtilde":     "phiGtilde",
    "phiW":          "phiW",
    "phiWB":         "phiWB",
    "phiWtilde":     "phiWtilde",
    "phiWtildeB":    "phiWtildeB",
    "phil1_11":      "phil1_11",
    "phil1_22":      "phil1_22",
    "phil1_33":      "phil1_33",
    "phil1_12":      "phil1_12",
    "phil1_13":      "phil1_13",
    "phil1_23":      "phil1_23",
    "phil3_11":      "phil3_11",
    "phil3_22":      "phil3_22",
    "phil3_33":      "phil3_33",
    "phil3_12":      "phil3_12",
    "phil3_13":      "phil3_13",
    "phil3_23":      "phil3_23",
    "phiq1_11":      "phiq1_11",
    "phiq1_12":      "phiq1_12",
    "phiq1_13":      "phiq1_13",
    "phiq1_22":      "phiq1_22",
    "phiq1_23":      "phiq1_23",
    "phiq1_33":      "phiq1_33",
    "phiq3_11":      "phiq3_11",
    "phiq3_12":      "phiq3_12",
    "phiq3_13":      "phiq3_13",
    "phiq3_22":      "phiq3_22",
    "phiq3_23":      "phiq3_23",
    "phiq3_33":      "phiq3_33",
    "phie_11":       "phie_11",
    "phie_22":       "phie_22",
    "phie_33":       "phie_33",
    "phiu_11":       "phiu_11",
    "phiu_12":       "phiu_12",
    "phiu_13":       "phiu_13",
    "phiu_22":       "phiu_22",
    "phiu_23":       "phiu_23",
    "phiu_33":       "phiu_33",
    "phid_11":       "phid_11",
    "phid_12":       "phid_12",
    "phid_13":       "phid_13",
    "phid_22":       "phid_22",
    "phid_23":       "phid_23",
    "phid_33":       "phid_33",
    "phiud_11":      "phiud_11",
    "phiud_12":      "phiud_12",
    "phiud_13":      "phiud_13",
    "phiud_21":      "phiud_21",
    "phiud_22":      "phiud_22",
    "phiud_23":      "phiud_23",
    "phiud_31":      "phiud_31",
    "phiud_32":      "phiud_32",
    "phiud_33":      "phiud_33",
    "ll_1111":       "ll_1111",
    "ll_1122":       "ll_1122",
    "ll_1133":       "ll_1133",
    "ll_1221":       "ll_1221",
    "ll_1331":       "ll_1331",
    "ll_2222":       "ll_2222",
    "ll_2233":       "ll_2233",
    "ll_2332":       "ll_2332",
    "ll_3333":       "ll_3333",
}


MAX_LEN=5
HIGH_REWARD_THRESHOLD = 0.90
#HIGH_REWARD_THRESHOLD = -1


