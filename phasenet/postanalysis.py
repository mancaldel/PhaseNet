import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_config(path: Path | str) -> dict:
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Try to interpret lists/numbers; fall back to raw string
            try:
                config[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                config[key] = value
    return config


def extract_losses(loss_log: Path | str) -> dict[str : list[float]]:
    with open(loss_log, "r") as file:
        lines = file.readlines()

    losses = {"train": [], "valid": []}
    for line in lines:
        if line.lower().startswith("epoch"):
            losses["train"].append(float(line.split(": ")[-1]))
        elif line.lower().startswith("valid"):
            losses["valid"].append(float(line.split(": ")[-1]))
        elif len(line) > 0:
            print(f"Unknown line: {line}")
    return losses


def plot_losses(
    model_loss_log_path: Path | str, ax: plt.Axes = None, ls=None
) -> plt.Figure:
    # Get relevant information and initialize figure
    model_station = model_loss_log_path.parent.name.split("_")[0]
    model_losses = extract_losses(model_loss_log_path)
    best_train_step = np.argmin(model_losses["train"])
    best_valid_step = np.argmin(model_losses["valid"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Plot train loss evolution and minimum
    ax.plot(model_losses["train"], label=f"Train ({best_train_step})", ls=ls)
    c_train = ax.get_lines()[-1].get_color()
    ax.plot(best_train_step, min(model_losses["train"]), "o", color=c_train)
    # Plot valid loss evolution and minimum
    ax.plot(model_losses["valid"], label=f"Valid ({best_valid_step})", ls=ls)
    c_valid = ax.get_lines()[-1].get_color()
    ax.plot(best_valid_step, min(model_losses["valid"]), "o", color=c_valid)
    # Set labels, titles, and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{model_station}")
    ax.legend()

    return ax.get_figure()


# ====================================================


def find_matches(
    idx_true: list, idx_pred: list, tolerance: float | int, unique: bool = False
) -> list[tuple[int, int]]:
    """Find matches between two list of numbers within a tolerance window"""

    def remove_duplicates(matches: list[tuple]) -> list[tuple]:
        """Remove tuples with duplicated indices"""
        matches_ = [(i, a, b, abs(a - b)) for i, (a, b) in enumerate(matches)]
        matches_.sort(key=lambda x: x[-1])
        chosen_idxs, v1_used, v2_used = [], set(), set()
        for i, v1, v2, gap in matches_:
            if v1 in v1_used or v2 in v2_used:
                continue
            chosen_idxs.append(i)
            v1_used.add(v1), v2_used.add(v2)
        unique_matches = [matches[i] for i in sorted(chosen_idxs)]
        return unique_matches

    m_true = np.array(idx_true).reshape((-1, 1)).repeat(len(idx_pred), axis=1)
    m_pred = np.array(idx_pred).reshape((1, -1)).repeat(len(idx_true), axis=0)
    m = m_true - m_pred
    idx_matches = np.argwhere(np.abs(m) <= tolerance)
    matches = [(idx_true[i], idx_pred[j]) for i, j in idx_matches]
    if unique:
        matches = remove_duplicates(matches)

    return matches


def match_picks(
    name: str,
    itp: list[int],
    its: list[int],
    itp_pred: list[int],
    its_pred: list[int],
    tolerance: float,
    dt: float = 0.01,
    unique: bool = False,
) -> pd.Series:
    """Match and classify predicted picks according to a ground truth

    Args:
        name (str): Name of the time series.
        itp (list[int]): List of indexes where the event P phase happens (ground truth).
        its (list[int]): List of indexes where the event S phase happens (ground truth).
        itp_pred (list[int]): List of indexes where an event P phase is detected (prediction).
        its_pred (list[int]): List of indexes where an event S phase is detected (prediction).
        tolerance (float): Tolerance window for phase detection in seconds.
        dt (float, optional): Time step for two consecutive indexes. Defaults to 0.01.
        unique (bool, optional): Whether to ????????????????????????????????????????????????????????????????????????????. Defaults to False.

    Returns:
        pd.Series: pick classification per phase
    """

    # Get P and S phase matches
    PasP = find_matches(itp, itp_pred, tolerance / dt, unique)
    SasS = find_matches(its, its_pred, tolerance / dt, unique)

    # Classify picks into true (TP), false (FP), and miss (FN)
    P_true = [_[1] for _ in PasP]  # Similar to PasP
    S_true = [_[1] for _ in SasS]  # SasS
    P_false = list(set(itp_pred) - set(P_true))  # SasP and NasP
    S_false = list(set(its_pred) - set(S_true))  # PasS and NasS
    P_miss = list(set(itp) - set([_[0] for _ in PasP]))  # PasS and PasN
    S_miss = list(set(its) - set([_[0] for _ in SasS]))  # SasP and SasN

    # Extra data to know the type of wrong detection (do not use for CM)
    SasP = find_matches(its, P_false, tolerance / dt, unique)
    PasS = find_matches(itp, S_false, tolerance / dt, unique)
    NasP = list(set(itp_pred) - set(_[1] for _ in PasP) - set(_[1] for _ in SasP))
    NasS = list(set(its_pred) - set(_[1] for _ in SasS) - set(_[1] for _ in PasS))
    PasN = list(set(itp) - set(_[0] for _ in PasP) - set(_[0] for _ in PasS))
    SasN = list(set(its) - set(_[0] for _ in SasS) - set(_[0] for _ in SasP))
    # NasN = []

    matched_picks = {
        "P_true": P_true, "P_false": P_false, "P_miss": P_miss,
        "S_true": S_true, "S_false": S_false, "S_miss": S_miss,
        "PasP": PasP, "SasP": SasP, "NasP": NasP,
        "PasS": PasS, "SasS": SasS, "NasS": NasS,
        "PasN": PasN, "SasN": SasN, "NasN": np.NAN,
    }

    return pd.Series(matched_picks, name=name)


def calc_phase_metrics(df_matches: pd.DataFrame) -> dict[str, dict[str, float]]:
    # Get number of correct detections, total events, and total detections for each phase
    p_correct = df_matches["P_true"].apply(len).sum()
    s_correct = df_matches["S_true"].apply(len).sum()
    p_events = df_matches[["P_true", "P_miss"]].map(len).sum().sum()
    s_events = df_matches[["S_true", "S_miss"]].map(len).sum().sum()
    p_detections = df_matches[["P_true", "P_false"]].map(len).sum().sum()
    s_detections = df_matches[["S_true", "S_false"]].map(len).sum().sum()

    # Calculate metrics for P and S phases
    p_pr = p_correct / p_detections if p_detections > 0 else 0 if p_events>0 else 1
    p_re = p_correct / p_events if p_events > 0 else 1
    p_f1 = 2 * p_pr * p_re / (p_pr + p_re) if p_pr + p_re > 0 else 0
    s_pr = s_correct / s_detections if s_detections > 0 else 0 if s_events>0 else 1
    s_re = s_correct / s_events if s_events > 0 else 1
    s_f1 = 2 * s_pr * s_re / (s_pr + s_re) if s_pr + s_re > 0 else 0

    # Summarize metrics
    metrics = {
        "P": {"precision": p_pr, "recall": p_re, "f1": p_f1},
        "S": {"precision": s_pr, "recall": s_re, "f1": s_f1},
    }

    return metrics


def aggregate_picks(df_picks: pd.DataFrame) -> pd.DataFrame:
    picks = df_picks.to_dict(orient="records")
    # named_picks = name_picks(picks)
    named_picks = {_["file_name"]: {"P": [], "S": []} for _ in picks}
    for p in picks:
        fname = p["file_name"]
        ptype = p["phase_type"]
        pindex = p["phase_index"]
        named_picks[fname][ptype].append(pindex)
    return pd.DataFrame(named_picks).T


def analyze_results(
    test_list_file: Path | str,
    prediction_dir: Path | str,
    tolerance: float,
    dt: float = 0.01,
    unique: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    # Load dataset and pick predictions
    df_dataset = pd.read_csv(test_list_file, sep="\t", index_col=0)
    df_picks = pd.read_csv(prediction_dir / "picks.csv")
    df_aggr_picks = aggregate_picks(df_picks)

    # Define auxiliary stuff
    mpicks_list = []
    total_events = {"P": 0, "S": 0}
    total_picks = {"P": 0, "S": 0}

    # Iterate over events
    for fname in df_dataset.fname.values:  # named_picks.keys():
        # Get and count total number of events per wave type
        itp = df_dataset.query("fname == @fname")["p_idx"].to_list()
        its = df_dataset.query("fname == @fname")["s_idx"].to_list()
        total_events["P"] += len(itp)
        total_events["S"] += len(its)

        if fname in df_aggr_picks.index:
            # Get and count total number of picks per wave type
            itp_pred = df_aggr_picks.loc[fname, "P"]
            its_pred = df_aggr_picks.loc[fname, "S"]
            total_picks["P"] += len(itp_pred)
            total_picks["S"] += len(its_pred)
        else:
            itp_pred = []
            its_pred = []

        # Match events with picks and classify results
        mpicks = match_picks(fname, itp, its, itp_pred, its_pred, tolerance, dt, unique)
        mpicks_list.append(pd.Series(mpicks, name=fname))

    # Create DataFrame with all classified matches
    df_matches = pd.concat(mpicks_list, axis=1).T
    df_results = pd.merge(
        df_dataset.set_index("fname"),
        df_matches,
        how="outer",
        left_index=True,
        right_index=True,
    )

    # Calculate metrics
    metrics = calc_phase_metrics(df_matches)
    cm_p = np.array(df_matches[["P_true", "P_false", "P_miss"]].map(len).sum().to_list()+[np.NAN]).reshape((2,2))
    cm_s = np.array(df_matches[["S_true", "S_false", "S_miss"]].map(len).sum().to_list()+[np.NAN]).reshape((2,2))

    # Validate results
    assert total_events["P"] == df_matches[["P_true", "P_miss"]].map(len).sum().sum()
    assert total_events["S"] == df_matches[["S_true", "S_miss"]].map(len).sum().sum()
    assert total_picks["P"] == df_matches[["P_true", "P_false"]].map(len).sum().sum()
    assert total_picks["S"] == df_matches[["S_true", "S_false"]].map(len).sum().sum()

    # Report metrics, confusion matrix, and classified matches
    if verbose:
        print("Total events:  P ={:4d}; S ={:4d}".format(*total_events.values()))
        print("Total picks:   P ={:4d}; S ={:4d}".format(*total_picks.values()))
        print()
        print("               precision  recall    f-1")
        print("P-wave metrics:    {:5.3f}   {:5.3f}  {:5.3f}".format(*metrics["P"].values()))
        print("S-wave metrics:    {:5.3f}   {:5.3f}  {:5.3f}".format(*metrics["S"].values()))
        print()
        print("Confusion matrix (P phase)")
        print(cm_p)
        print()
        print("Confusion matrix (S phase)")
        print(cm_s)
        print()
        print(df_results.head())

    return df_results, metrics


def group_results(df_results: pd.DataFrame, field: str) -> pd.DataFrame:
    gruped_metrics = {name: {} for name in df_results[field]}
    for name in df_results[field].values:
        # cm = calc_confidence_matrix(df_results.query(f"{field} == @name"))
        # gruped_metrics[name] = calc_metrics(cm)
        gruped_metrics[name] = calc_phase_metrics(df_results.query(f"{field} == @name"))

    return pd.concat({k: pd.DataFrame(v) for k, v in gruped_metrics.items()}, axis=1).T
