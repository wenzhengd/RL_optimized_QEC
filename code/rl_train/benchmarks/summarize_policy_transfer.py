"""Summarize Expr1->Expr2 transfer evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize transfer-eval JSON files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/policy_transfer",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    reports = []
    for path in sorted(input_dir.glob("expr1_to_expr2_transfer_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_args = payload["target_args"]
        transfer = payload["transfer_eval_metrics"]["learned"]
        fixed = payload["transfer_eval_metrics"]["fixed_zero"]
        target = payload.get("reference_target_learned_metrics")
        reports.append(
            {
                "report_path": str(path),
                "target_label": Path(payload["target_run_json"]).parent.name,
                "scale": float(target_args["steane_channel_regime_a"]),
                "f_hz": float(target_args["steane_channel_corr_f"]),
                "g": float(target_args["steane_channel_corr_g"]),
                "transfer_success_rate": float(transfer["success_rate"]),
                "transfer_ler_proxy": float(transfer["ler_proxy"]),
                "fixed_zero_success_rate": float(fixed["success_rate"]),
                "fixed_zero_ler_proxy": float(fixed["ler_proxy"]),
                "transfer_improve_vs_fixed_zero": float(payload["transfer_improvement_vs_fixed_zero"]["ler_proxy"]),
                "target_learned_success_rate": float(target["success_rate"]) if target else float("nan"),
                "target_learned_ler_proxy": float(target["ler_proxy"]) if target else float("nan"),
                "delta_success_vs_target": float(payload["transfer_vs_target_learned"]["delta_success_rate"])
                if "transfer_vs_target_learned" in payload
                else float("nan"),
                "delta_ler_vs_target": float(payload["transfer_vs_target_learned"]["delta_ler_proxy"])
                if "transfer_vs_target_learned" in payload
                else float("nan"),
            }
        )

    grouped = {}
    for row in reports:
        grouped.setdefault(row["target_label"], []).append(row)

    summary = {"reports": reports, "targets": {}}
    md_lines = ["# Transfer Summary", ""]
    for target_label, rows in sorted(grouped.items()):
        arr_imp = np.asarray([r["transfer_improve_vs_fixed_zero"] for r in rows], dtype=float)
        arr_delta = np.asarray([r["delta_success_vs_target"] for r in rows], dtype=float)
        sample = rows[0]
        summary["targets"][target_label] = {
            "scale": sample["scale"],
            "f_hz": sample["f_hz"],
            "g": sample["g"],
            "n_reports": len(rows),
            "transfer_improve_vs_fixed_zero_mean": float(np.mean(arr_imp)),
            "transfer_improve_vs_fixed_zero_std": float(np.std(arr_imp)),
            "delta_success_vs_target_mean": float(np.mean(arr_delta)),
            "delta_success_vs_target_std": float(np.std(arr_delta)),
            "target_learned_success_rate": sample["target_learned_success_rate"],
            "fixed_zero_success_rate": sample["fixed_zero_success_rate"],
        }
        md_lines.append(f"## {target_label}")
        md_lines.append("")
        md_lines.append(
            f"- condition: `scale={sample['scale']:g}, f={sample['f_hz']:.0f}, g={sample['g']:.1f}`"
        )
        md_lines.append(
            f"- transfer improve vs fixed_zero: `{np.mean(arr_imp):+.4f} +- {np.std(arr_imp):.4f}`"
        )
        md_lines.append(
            f"- delta success vs Expr2-trained: `{np.mean(arr_delta):+.6f} +- {np.std(arr_delta):.6f}`"
        )
        md_lines.append(
            f"- reference target learned success: `{sample['target_learned_success_rate']:.6f}`"
        )
        md_lines.append(
            f"- reference fixed_zero success: `{sample['fixed_zero_success_rate']:.6f}`"
        )
        md_lines.append("")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved summary JSON: {out_json}")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"Saved summary markdown: {out_md}")


if __name__ == "__main__":
    main()
