import csv
import os
from ast import literal_eval


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
NORMALIZER = os.path.join(BASE_DIR, "data/model/max_EV_by_condition_empirical.csv")
TRIAL_FILES = [
    "data/model/exp1/processed/trials.csv",
    "data/model/exp1/processed/trials_exclude.csv",
    "data/model/exp1_fitcost/processed/trials.csv",
    "data/model/exp1_fitcost_exclude/processed/trials_exclude.csv",
    "data/human/1.0/processed/trials.csv",
    "data/human/1.0/processed/trials_exclude.csv",
]
MAX_TRIAL_FILE_SIZE = 100 * 1024 * 1024
MODEL_STRATEGY_SAMPLES = 10
TOLERANCE = 1e-9


def condition_key(row):
    return float(row["sigma"]), round(float(row["alpha"]), 1)


def read_normalizer():
    with open(NORMALIZER, newline="") as f:
        return {condition_key(row): float(row["mean"]) for row in csv.DictReader(f)}


def assert_close(actual, expected, path, row_number, column):
    if abs(actual - expected) > TOLERANCE:
        raise AssertionError(
            f"{path}:{row_number} {column}={actual} does not match expected {expected}"
        )


def audit_trial_file(path, normalizer):
    full_path = os.path.join(BASE_DIR, path)
    size = os.path.getsize(full_path)
    if size > MAX_TRIAL_FILE_SIZE:
        raise AssertionError(f"{path} is {size / 1024 / 1024:.1f} MB; expected under 100 MB")

    with open(full_path, newline="") as f:
        rows = csv.DictReader(f)
        for row_number, row in enumerate(rows, start=2):
            denom = normalizer[condition_key(row)]
            payoff_gross = float(row["payoff_gross"])
            payoff_net = float(row["payoff_net"])
            assert_close(float(row["payoff_perfect"]), denom, path, row_number, "payoff_perfect")
            assert_close(
                float(row["payoff_gross_relative"]),
                payoff_gross / denom,
                path,
                row_number,
                "payoff_gross_relative",
            )
            assert_close(
                float(row["payoff_net_relative"]),
                payoff_net / denom,
                path,
                row_number,
                "payoff_net_relative",
            )
            if path.startswith("data/model/"):
                strategy_samples = literal_eval(row["strategy"])
                if len(strategy_samples) != MODEL_STRATEGY_SAMPLES:
                    raise AssertionError(
                        f"{path}:{row_number} has {len(strategy_samples)} strategy samples; "
                        f"expected {MODEL_STRATEGY_SAMPLES}"
                    )


def main():
    normalizer = read_normalizer()
    for path in TRIAL_FILES:
        audit_trial_file(path, normalizer)
    print("Exp1 empirical normalizer audit passed.")


if __name__ == "__main__":
    main()
