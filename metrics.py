"""Metrics, statistics and plots for the experiment (PRD section 9, build step 8).

Runs on the HOST (needs numpy, scipy, matplotlib), not in the containers:

    python metrics.py --results results/full --out results/full/analysis

Reads results/<...>/<case>/runs.csv and runs.jsonl and writes to --out:
  summary.csv        one row per case x setting x agent
  stats.csv          Q-learning vs each baseline: Mann-Whitney U + A12
  operator_usage.csv share of each operator per case x setting x agent
  greedy_policy.csv  Q-learning greedy action over visited states (RQ3)
  report.md          the tables above in Markdown, plus notes
  coverage_A.png, coverage_B.png, operator_usage.png

Analysis choices (all marked # NOTE: below and repeated in report.md):
- AUC: the episode stops at the first crash, so a crashed run's coverage
  curve is extended flat to T with its last value. AUC is reported as the
  mean coverage over executions 1..T (area / T), in features.
- Executions to crash in the test: runs with no verified crash are ranked
  as T + 1 (worse than any crash), so every run takes part in the test. The
  median in summary.csv is over successful runs only, as the PRD says.
- A12 is reported as "probability Q-learning is better than the baseline":
  higher coverage/AUC is better, fewer executions to crash is better.
  0.5 = no difference; Vargha-Delaney thresholds 0.56 / 0.64 / 0.71
  (small / medium / large) apply to |A12 - 0.5| in either direction.

The helper functions at the top are pure Python so the unit tests run on
any interpreter; numpy/scipy/matplotlib are imported only where needed.
"""

import argparse
import csv
import json
import os
import statistics
import sys

AGENTS = ["random", "ucb1", "qlearning"]
AGENT_LABELS = {"random": "Random", "ucb1": "UCB1", "qlearning": "Q-learning"}
# Validated categorical palette, fixed order (blue, orange, aqua).
AGENT_COLORS = {"random": "#2a78d6", "ucb1": "#eb6834", "qlearning": "#1baf7a"}
CASE_ORDER = ["12950", "13222", "3848", "10304", "3522", "16091"]
SETTINGS = ["A", "B"]
ALPHA = 0.05


# ---------------------------------------------------------------- pure helpers

def pad_curve(curve, T):
    """Extend a coverage curve flat to length T (episode stopped at a crash)."""
    curve = list(curve)
    if not curve:
        return [0] * T
    return curve[:T] + [curve[-1]] * (T - len(curve))


def coverage_auc(curve, T):
    """Mean coverage over executions 1..T (area under the padded curve / T)."""
    return sum(pad_curve(curve, T)) / float(T)


def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 * P(X == Y)."""
    if not x or not y:
        return float("nan")
    gt = eq = 0
    for a in x:
        for b in y:
            if a > b:
                gt += 1
            elif a == b:
                eq += 1
    return (gt + 0.5 * eq) / float(len(x) * len(y))


def a12_magnitude(a):
    d = abs(a - 0.5)
    if d != d:            # nan
        return ""
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def crash_exec_or_censored(row, T):
    """Executions to first VERIFIED crash; T + 1 when there is none."""
    if row["verified"] == "True" and row["first_crash_exec"] != "":
        return int(row["first_crash_exec"])
    return T + 1


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


# ---------------------------------------------------------------- loading

def load_runs(results_dir):
    csv.field_size_limit(sys.maxsize)
    rows = []
    for case in CASE_ORDER:
        d = os.path.join(results_dir, case)
        p = os.path.join(d, "runs.csv")
        if not os.path.exists(p):
            continue
        extra = {}
        jp = os.path.join(d, "runs.jsonl")
        if os.path.exists(jp):
            with open(jp) as f:
                for line in f:
                    j = json.loads(line)
                    extra[(j["agent"], j["setting"], str(j["seed"]), str(j["T"]))] = j
        with open(p) as f:
            for r in csv.DictReader(f):
                r["T"] = int(r["T"])
                r["curve"] = json.loads(r["coverage_curve"])
                r["ops"] = json.loads(r["operator_counts"])
                j = extra.get((r["agent"], r["setting"], r["seed"], str(r["T"])), {})
                r["q_greedy_policy"] = j.get("q_greedy_policy")
                rows.append(r)
    return rows


def group(rows, case, setting, agent):
    return [r for r in rows
            if r["case"] == case and r["setting"] == setting and r["agent"] == agent]


# ---------------------------------------------------------------- tables

def summarise(rows):
    out = []
    for case in CASE_ORDER:
        for setting in SETTINGS:
            for agent in AGENTS:
                g = group(rows, case, setting, agent)
                if not g:
                    continue
                T = g[0]["T"]
                ok = [r for r in g if r["verified"] == "True"]
                found = [r for r in g if r["first_crash_exec"] != ""]
                execs = [int(r["first_crash_exec"]) for r in ok]
                cov_m, cov_s = mean_std([int(r["final_coverage"]) for r in g])
                auc_m, auc_s = mean_std([coverage_auc(r["curve"], T) for r in g])
                out.append({
                    "case": case, "setting": setting, "agent": agent,
                    "runs": len(g), "T": T,
                    "crashes_found": len(found),
                    "verified_crashes": len(ok),
                    "success_rate": len(ok) / float(len(g)),
                    "median_exec_to_crash": statistics.median(execs) if execs else "",
                    "final_cov_mean": round(cov_m, 1), "final_cov_std": round(cov_s, 1),
                    "auc_mean": round(auc_m, 1), "auc_std": round(auc_s, 1),
                })
    return out


def compare(rows):
    from scipy.stats import mannwhitneyu
    out = []
    metrics = [
        # name, value function, higher_is_better
        ("final_coverage", lambda r: int(r["final_coverage"]), True),
        ("coverage_auc", lambda r: coverage_auc(r["curve"], r["T"]), True),
        ("exec_to_crash", lambda r: crash_exec_or_censored(r, r["T"]), False),
    ]
    for case in CASE_ORDER:
        for setting in SETTINGS:
            q = group(rows, case, setting, "qlearning")
            if not q:
                continue
            for base in ["random", "ucb1"]:
                b = group(rows, case, setting, base)
                for name, fn, higher in metrics:
                    x = [fn(r) for r in q]
                    y = [fn(r) for r in b]
                    if len(set(x + y)) == 1:
                        # All values identical (e.g. no crash in any run):
                        # nothing to test.
                        u, p = "", ""
                        note = "all values equal"
                    else:
                        res = mannwhitneyu(x, y, alternative="two-sided")
                        u, p = float(res.statistic), float(res.pvalue)
                        note = ""
                    a = a12(x, y) if higher else a12(y, x)
                    out.append({
                        "case": case, "setting": setting,
                        "comparison": "qlearning vs %s" % base, "metric": name,
                        "n_q": len(x), "n_base": len(y),
                        "U": u if u == "" else round(u, 1),
                        "p_value": p if p == "" else round(p, 4),
                        "significant": (p != "" and p < ALPHA),
                        "A12_q_better": round(a, 3),
                        "effect": a12_magnitude(a), "note": note,
                    })
    return out


def operator_usage(rows, names):
    out = []
    for case in CASE_ORDER:
        for setting in SETTINGS + ["all"]:
            for agent in AGENTS:
                g = [r for r in rows if r["case"] == case and r["agent"] == agent
                     and (setting == "all" or r["setting"] == setting)]
                if not g:
                    continue
                tot = [0] * len(names)
                for r in g:
                    for i, c in enumerate(r["ops"]):
                        tot[i] += c
                n = float(sum(tot)) or 1.0
                row = {"case": case, "setting": setting, "agent": agent,
                       "mutations": int(sum(tot))}
                for i, nm in enumerate(names):
                    row[nm] = round(tot[i] / n, 4)
                out.append(row)
    return out


def greedy_policy_table(rows, names):
    """For each case: over every Q-learning run, how often each operator is
    the greedy action in a VISITED state (policy value -1 = unvisited)."""
    out = []
    for case in CASE_ORDER:
        tally = [0] * len(names)
        visited = 0
        for r in rows:
            if r["case"] != case or r["agent"] != "qlearning" or not r["q_greedy_policy"]:
                continue
            for a in r["q_greedy_policy"]:
                if a >= 0:
                    tally[a] += 1
                    visited += 1
        if not visited:
            continue
        row = {"case": case, "visited_states": visited}
        for i, nm in enumerate(names):
            row[nm] = round(tally[i] / float(visited), 4)
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- plots

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#5f5e5a", labelsize=8)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_coverage(rows, setting, path, case_meta):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, case in zip(axes.flat, CASE_ORDER):
        _style(ax)
        for agent in AGENTS:
            g = group(rows, case, setting, agent)
            if not g:
                continue
            T = g[0]["T"]
            m = np.array([pad_curve(r["curve"], T) for r in g], dtype=float)
            x = np.arange(1, T + 1)
            ax.fill_between(x, m.min(axis=0), m.max(axis=0),
                            color=AGENT_COLORS[agent], alpha=0.14, linewidth=0)
            ax.plot(x, m.mean(axis=0), color=AGENT_COLORS[agent], linewidth=2,
                    label=AGENT_LABELS[agent])
        meta = case_meta.get(case, {})
        ax.set_title("%s  %s (%s)" % (case, meta.get("project", ""),
                                      meta.get("input_format", "")),
                     fontsize=10, color="#1f1e1c", loc="left")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_xlabel("executions", fontsize=8, color="#5f5e5a")
        ax.set_ylabel("coverage features", fontsize=8, color="#5f5e5a")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Coverage vs executions, Setting %s (mean of 5 seeds; band = min-max)"
                 % setting, x=0.01, ha="left", fontsize=12, color="#1f1e1c")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_operator_usage(usage, names, path, case_meta):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    y = np.arange(len(names))
    h = 0.26
    for ax, case in zip(axes.flat, CASE_ORDER):
        _style(ax)
        ax.grid(axis="x", color="#e6e5df", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        for k, agent in enumerate(AGENTS):
            r = [u for u in usage if u["case"] == case and u["setting"] == "all"
                 and u["agent"] == agent]
            if not r:
                continue
            shares = [100 * r[0][nm] for nm in names]
            # Random on top, matching the legend order (axis is inverted).
            ax.barh(y + (k - 1) * h, shares, height=h * 0.85,
                    color=AGENT_COLORS[agent], label=AGENT_LABELS[agent])
        ax.axvline(100.0 / len(names), color="#8a8983", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        meta = case_meta.get(case, {})
        ax.set_title("%s  %s (%s)" % (case, meta.get("project", ""),
                                      meta.get("input_format", "")),
                     fontsize=10, color="#1f1e1c", loc="left")
        ax.set_xlabel("% of mutations", fontsize=8, color="#5f5e5a")
    # sharey=True: invert once (inverting every panel would toggle it back).
    axes.flat[0].invert_yaxis()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Operator usage, both settings pooled (dashed line = uniform 12.5%)",
                 x=0.01, ha="left", fontsize=12, color="#1f1e1c")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- report

def md_table(rows, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


NOTES = """## Notes on the analysis

- **AUC**: the episode ends at the first crash, so a crashed run's coverage
  curve is held flat at its last value up to T. AUC = mean coverage over
  executions 1..T (features).
- **Setting B coverage is confounded with success**: a run that crashes early
  stops, so it has low final coverage and AUC. In Setting B, lower coverage
  can mean the agent *succeeded* sooner. Read coverage there together with
  the crash columns.
- **Executions-to-crash test**: runs without a verified crash count as T + 1
  (ranked worse than any crash). "all values equal" means no agent crashed
  in that cell, so there is nothing to test.
- **A12** is the probability that Q-learning does better than the baseline
  (higher coverage, fewer executions). 0.5 = no difference; below 0.5 means
  the baseline did better. Effect: negligible < 0.06 <= small < 0.14 <=
  medium < 0.21 <= large (distance from 0.5).
- **n = 5 per group**: the smallest two-sided Mann-Whitney p-value possible
  with 5 vs 5 is about 0.008, so only near-complete separation is significant.
  No correction for multiple comparisons is applied; with this many tests,
  some p < 0.05 results are expected by chance.
- **Operator usage** pools mutation counts over both settings and all seeds,
  so longer runs weigh more. Per-setting shares are in operator_usage.csv.
- **Greedy policy**: per Q-learning run, the argmax action of each state the
  run visited; unvisited states are left out.
"""


def write_report(path, summary, stats, usage, greedy, names, results_dir):
    out = ["# Results: Q-learning vs UCB1 vs Random on six ARVO cases", "",
           "Source: `%s` (generated by `metrics.py`)." % results_dir, ""]
    out.append("## Summary per case, setting and agent\n")
    s2 = []
    for r in summary:
        s2.append(dict(r,
                       crash="%d/%d" % (r["verified_crashes"], r["runs"]),
                       cov="%.0f ± %.0f" % (r["final_cov_mean"], r["final_cov_std"]),
                       auc="%.0f ± %.0f" % (r["auc_mean"], r["auc_std"]),
                       med=r["median_exec_to_crash"] if r["median_exec_to_crash"] != "" else "-"))
    out.append(md_table(s2, ["case", "setting", "agent", "crash", "med", "cov", "auc"],
                        ["case", "setting", "agent", "verified crashes",
                         "median execs to crash", "final coverage", "coverage AUC"]))
    out.append("\nEvery crash found was checked against the -fix build; "
               "crashes found vs verified: %d vs %d.\n"
               % (sum(r["crashes_found"] for r in summary),
                  sum(r["verified_crashes"] for r in summary)))

    out.append("## Q-learning vs baselines (two-sided Mann-Whitney U, alpha = 0.05)\n")
    st2 = [dict(r, p=r["p_value"] if r["p_value"] != "" else "-",
                sig="**yes**" if r["significant"] else "no",
                note=r["note"] or "") for r in stats]
    out.append(md_table(st2, ["case", "setting", "comparison", "metric", "p", "sig",
                              "A12_q_better", "effect", "note"],
                        ["case", "setting", "comparison", "metric", "p", "p < 0.05",
                         "A12 (Q better)", "effect", "note"]))
    sig = [r for r in stats if r["significant"]]
    testable = [r for r in stats if r["p_value"] != ""]
    out.append("\n%d of %d testable comparisons have p < 0.05 (%d more had "
               "identical values, i.e. no crash in either group). By chance "
               "alone about %.1f would be expected at alpha = %.2f.\n"
               % (len(sig), len(testable), len(stats) - len(testable),
                  ALPHA * len(testable), ALPHA))

    out.append("## Operator usage, both settings pooled (% of mutations)\n")
    u2 = []
    for r in usage:
        if r["setting"] != "all":
            continue
        d = {"case": r["case"], "agent": r["agent"]}
        for nm in names:
            d[nm] = "%.1f" % (100 * r[nm])
        u2.append(d)
    out.append(md_table(u2, ["case", "agent"] + names))
    out.append("")
    if greedy:
        out.append("## Q-learning greedy action over visited states (% of states)\n")
        g2 = []
        for r in greedy:
            d = {"case": r["case"], "visited_states": r["visited_states"]}
            for nm in names:
                d[nm] = "%.1f" % (100 * r[nm])
            g2.append(d)
        out.append(md_table(g2, ["case", "visited_states"] + names))
        out.append("")
    out.append("## Figures\n")
    out.append("- `coverage_A.png`, `coverage_B.png`: coverage vs executions")
    out.append("- `operator_usage.png`: operator usage per case and agent\n")
    out.append(NOTES)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "analysis")
    if not os.path.isdir(out):
        os.makedirs(out)

    import config
    import mutators
    names = list(mutators.OPERATOR_NAMES)

    rows = load_runs(args.results)
    if not rows:
        sys.exit("no runs.csv found under %s" % args.results)
    summary = summarise(rows)
    stats = compare(rows)
    usage = operator_usage(rows, names)
    greedy = greedy_policy_table(rows, names)

    write_csv(os.path.join(out, "summary.csv"), summary)
    write_csv(os.path.join(out, "stats.csv"), stats)
    write_csv(os.path.join(out, "operator_usage.csv"), usage)
    write_csv(os.path.join(out, "greedy_policy.csv"), greedy)
    for s in SETTINGS:
        plot_coverage(rows, s, os.path.join(out, "coverage_%s.png" % s), config.CASES)
    plot_operator_usage(usage, names, os.path.join(out, "operator_usage.png"),
                        config.CASES)
    write_report(os.path.join(out, "report.md"), summary, stats, usage, greedy,
                 names, args.results)
    print("%d runs -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
