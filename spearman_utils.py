#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spearman_utils.py

Export loss–KL relations to CSV for TikZ/PGFPlots.
Also computes Spearman and RMSE for reporting.
"""

import os
import csv
import numpy as np
from scipy.stats import spearmanr


# ======================================================
# Helpers
# ======================================================

def extract_metric(results, key):
    return np.array([r[key] for r in results])


def compute_rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


# ======================================================
# CSV export
# ======================================================

def export_loss_kl_csv(
    results,
    kl_key,
    csv_path,
):
    """
    Export CSV with columns:
      loss, kl
    """

    loss = extract_metric(results, "loss")
    kl = extract_metric(results, kl_key)

    rho, p = spearmanr(loss, kl)
    rmse = compute_rmse(loss, kl)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "kl"])
        for l, k in zip(loss, kl):
            writer.writerow([l, k])

    print(f"[CSV] Saved: {csv_path}")
    print(f"      Spearman ρ = {rho:.4f}, p = {p:.3e}")
    print(f"      RMSE = {rmse:.4f}")

    return {
        "rho": rho,
        "p": p,
        "rmse": rmse,
        "path": csv_path,
    }


def export_all_loss_kl(results, save_dir="files"):
    """
    Export all three loss–KL relations.
    """

    outputs = {}

    outputs["KL_no_noise"] = export_loss_kl_csv(
        results,
        kl_key="KL_no_noise",
        csv_path=os.path.join(save_dir, "loss_kl_no_noise.csv"),
    )

    outputs["KL_HS"] = export_loss_kl_csv(
        results,
        kl_key="KL_HS",
        csv_path=os.path.join(save_dir, "loss_kl_hs.csv"),
    )

    outputs["KL_Uhlmann"] = export_loss_kl_csv(
        results,
        kl_key="KL_Uhlmann",
        csv_path=os.path.join(save_dir, "loss_kl_uhlmann.csv"),
    )

    return outputs
