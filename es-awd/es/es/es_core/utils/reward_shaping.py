import numpy as np

def softmax_rank_utilities(seeds_perf, tau=2.0, eps=1e-12):
    """
    Convert rewards -> centered ranks s_i in [-1,1] -> utilities u_i.
    u_i = softmax(tau * s_i) - 1/n  (mean-zero). Optionally L1-normalize.

    Args:
        seeds_perf: dict[key]->{"avg_reward": float}
        tau: temperature; higher = more elite pressure (try 1.5–3.0)
        l1_norm: if True, scale so sum(|u|)=1 to keep alpha comparable.

    Returns:
        seeds_perf with seeds_perf[k]["norm_reward"] = utility
    """
    keys = list(seeds_perf.keys())
    vals = np.array([seeds_perf[k]["avg_reward"] for k in keys], dtype=np.float64)
    n = len(vals)
    if n == 0:
        return seeds_perf
    if n == 1:
        seeds_perf[keys[0]]["norm_reward"] = 0.0
        return seeds_perf

    # ---- centered ranks s_i in [-1, 1] (ties averaged)
    idx = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1..n
        ranks[idx[i:j + 1]] = avg_rank
        i = j + 1
    den = (n - 1) / 2.0 if n > 1 else 1.0
    s = (ranks - (n + 1) / 2.0) / den  # centered ranks in [-1, 1]

    # ---- utilities: softmax(tau * s) - 1/n (mean zero by construction)
    logits = tau * s
    logits -= logits.max()  # stabilize
    p = np.exp(logits) / (np.exp(logits).sum() + eps)
    u = p - (1.0 / n)
    u = (u - u.mean()) / (u.std() + 1e-8)

    print("u stats:", u.min(), u.max(), u.mean(), u.std())

    for k, ui in zip(keys, u):
        seeds_perf[k]["norm_reward"] = float(ui)

    return seeds_perf

def centered_ranks(seeds_perf):
        # collect rewards
        keys = list(seeds_perf.keys())
        vals = [seeds_perf[k]["avg_reward"] for k in keys]
        n = len(vals)

        # ranks with ties averaged (same block as above) → `ranks` in 1..n
        idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[idx[j+1]] == vals[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[idx[t]] = avg_rank
            i = j + 1

        # centered ranks in [-1, 1]
        den = (n - 1) / 2.0 if n > 1 else 1.0
        scores = [ (r - (n + 1)/2.0) / den for r in ranks ]

        for k, s in zip(keys, scores):
            seeds_perf[k]["norm_reward"] = s

        return seeds_perf
    
def z_score(seeds_perf, mean_reward, std_reward):
    for k in seeds_perf:
        seeds_perf[k]["norm_reward"] = (
            seeds_perf[k]["avg_reward"] - mean_reward
        ) / (std_reward + 1e-8)
        print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

    return seeds_perf