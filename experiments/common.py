"""Shared helpers. The model is only ever called through blackbox.predict_batch."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blackbox import FEATURES, predict_batch  # noqa: E402

pd.set_option("display.width", 200)


def load_data():
    return pd.read_csv(os.path.join(ROOT, "data", "validation_applications.csv"))


def predict(df):
    """Score every row. Returns (risk_score, rejected) as two Series."""
    out = predict_batch(df[FEATURES].to_dict("records"))
    risk = pd.Series([o["risk_score"] for o in out], index=df.index)
    rejected = pd.Series([o["decision"] == "reject" for o in out], index=df.index)
    return risk, rejected


def what_if(df, **changes):
    """Same applicants with some fields overwritten, e.g. what_if(df, age=25)."""
    return predict(df.assign(**changes))


def sweep(df, column, values, probe_file=None):
    """Set `column` to each value in turn (everything else unchanged) and
    return the reject rate and average risk at each value."""
    probes = []
    for v in values:
        # drop the original label: it describes the real applicant, not this probe
        p = df.drop(columns="decision", errors="ignore").assign(**{column: v})
        p["risk_score"], p["rejected"] = predict(p)
        probes.append(p)
    probes = pd.concat(probes)
    if probe_file:
        try:
            probes.to_csv(os.path.join(ROOT, "probes", probe_file), index=False)
        except PermissionError:
            print(f"WARNING: could not save probes/{probe_file} (is it open in another program?). "
                  "Results below are unaffected.")
    return (probes.groupby(column)
            .agg(reject_rate=("rejected", "mean"), avg_risk=("risk_score", "mean"))
            .round(3).reset_index())


BLUE, ORANGE = "#2a78d6", "#eb6834"


def finish_chart(ax, title, filename):
    """Common look for every chart: y-axis is '% rejected'. Also saves a PNG to results/."""
    from matplotlib.ticker import PercentFormatter
    ax.set_title(title, loc="left")
    ax.set_ylabel("rejected")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color="#e4e3df")
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(ROOT, "results", filename), dpi=130)
