import sweep as sweep
from sweep import hyperparam
import math

def get_grid(args):

    grid = []

    grid += [
        hyperparam("--max-positions", 2048),
        hyperparam("--update-freq", 1),
        hyperparam("--user-dir", "user_dir"),
        hyperparam("--task", "feature_based_sentence_prediction"),
        hyperparam("--reset-optimizer"),
        hyperparam("--reset-dataloader"),
        hyperparam("--reset-meters"),
        hyperparam("--arch", "feature_based_linear_model"),
        hyperparam("--criterion", "sentence_prediction"),
        hyperparam("--classification-head-name", "imdb_head"),
        hyperparam("--num-classes", 2),
        hyperparam("--optimizer", "mixed_adam"),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--adam-eps", 1e-06),
        hyperparam("--clip-norm", 1.0),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--total-num-update", [None]),
        hyperparam("--warmup-updates", [None]),
        hyperparam("--max-epoch", [None]),
        hyperparam("--best-checkpoint-metric", "accuracy"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--find-unused-parameters"),
        # hyperparam("--fp16"),
        # hyperparam("--fp16-init-scale", 4),
        # hyperparam("--threshold-loss-scale", 1),
        # hyperparam("--fp16-scale-window", 128),
    ]

    grid += [
        hyperparam("--weight-decay", [0.0, 1e-2, 1e-1], save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--lr", [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], save_dir_key=lambda val: f"lr{val}"),
    ]

    grid += [
        hyperparam("--feature-dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], save_dir_key=lambda val: f"featdrop{val}"),
        hyperparam("--pooler-dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], save_dir_key=lambda val: f"pooldrop{val}"),
    ]

    grid += [
        hyperparam("--encoder-embed-dim", [10, 20, 50, 100], save_dir_key=lambda val: f"hdim{val}")
    ]

    grid += [
        hyperparam("--batch-size", [32, 64, 128, 256], save_dir_key=lambda val: f"bsz{val}"),
    ]

    return grid

def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    config["--max-epoch"].current_value = 100 * int(config["--batch-size"].current_value / 32)
    config["--total-num-update"].current_value = math.ceil(25000 / config["--batch-size"].current_value) * config["--max-epoch"].current_value
    config["--warmup-updates"].current_value = int(config["--total-num-update"].current_value * 0.06)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)