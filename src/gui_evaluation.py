#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
from pathlib import Path

from metrics_evaluation import (
    evaluate_model_metrics_with_thresholds,
    save_thresholded_metrics,
    aggregate_model_metrics
)
from config import IOU_THRESHOLD

MODEL_METRICS_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")

MODELS = ["YOLO", "Mask-RCNN", "YOLO_DeepLabV3"]
# Mapping from GUI-friendly names to internal identifiers
MODEL_NAME_MAP = {
    "YOLO": "yolo",
    "Mask-RCNN": "maskrcnn",
    "YOLO_DeepLabV3": "yolo_deeplab"
}
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_INSTANCES = 5


class MetricsEvaluationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Automatic Metric Calibration GUI")
        self.geometry("1100x650")

        self.selected_model = tk.StringVar(value=MODELS[0])
        self.conf_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.max_instances = tk.IntVar(value=DEFAULT_MAX_INSTANCES)
        self.best_mAP = None
        self.calibrated_metrics = None
        self.last_eval_mode = None  # or "auto", "manual"

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_combo = ttk.Combobox(self, textvariable=self.selected_model, values=MODELS, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self, text="Confidence Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.conf_slider = ttk.Scale(self, from_=0.2, to=0.8, variable=self.conf_threshold, orient="horizontal")
        self.conf_slider.grid(row=1, column=1, padx=10, pady=5, sticky="we")
        self.conf_label = ttk.Label(self, textvariable=self.conf_threshold)
        self.conf_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        ttk.Label(self, text="Max Instances:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.max_inst_combo = ttk.Combobox(self, textvariable=self.max_instances, values=[3, 5, 7, 10], state="readonly")
        self.max_inst_combo.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ttk.Button(self, text="Run Auto Calibration", command=self.run_greedy_calibration).grid(
            row=3, column=0, columnspan=2, padx=10, pady=10)

        ttk.Button(self, text="Run Manual Evaluation", command=self.run_manual_evaluation).grid(
            row=4, column=0, columnspan=2, padx=10, pady=10)

        self.metrics_table = ttk.Treeview(self, columns=("Category", "mAP", "F1", "ProcTime"), show="headings", height=8)
        for col in ("Category", "mAP", "F1", "ProcTime"):
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=120)
        self.metrics_table.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        self.details_text = tk.Text(self, width=50, height=30)
        self.details_text.grid(row=0, column=3, rowspan=6, padx=10, pady=10, sticky="nsew")

        ttk.Button(self, text="Save Metrics", command=self.save_calibrated_metrics).grid(
            row=6, column=0, columnspan=3, padx=10, pady=10)

    def run_greedy_calibration(self):
        model_display = self.selected_model.get()
        model = MODEL_NAME_MAP.get(model_display)
        if model is None:
            messagebox.showerror("Model Error", f"Unknown model name: {model_display}")
            return
        self.details_text.insert(tk.END, f"\nRunning auto calibration for model: {model}\n")

        def calibration():
            import numpy as np
            import pandas as pd

            best_score = -1
            best_conf = None
            best_max_inst = None
            best_metrics = None
            all_results = []

            confidence_values = [round(v, 2) for v in np.arange(0.2, 0.81, 0.1)]
            max_instances_values = [3, 5, 7, 10]

            for conf in confidence_values:
                for max_inst in max_instances_values:
                    metrics = evaluate_model_metrics_with_thresholds(model, conf, max_inst)
                    aggregated = aggregate_model_metrics(metrics)

                    mAP = aggregated["all_objects"].get("mAP", 0.0)
                    f1 = aggregated["all_objects"].get("detection", {}).get("f1", {}).get("mean", 0.0)
                    combined_score = 0.5 * mAP + 0.5 * f1

                    self.details_text.insert(
                        tk.END,
                        f"Trying conf={conf}, max_inst={max_inst}... "
                        f"mAP={mAP * 100:.1f}%, F1={f1 * 100:.1f}%, Score={combined_score * 100:.1f}%\n"
                    )

                    all_results.append({
                        "confidence": conf,
                        "max_instances": max_inst,
                        "mAP": mAP,
                        "F1": f1,
                        "combined_score": combined_score
                    })

                    if combined_score > best_score:
                        best_score = combined_score
                        best_conf = conf
                        best_max_inst = max_inst
                        best_metrics = aggregated

            # Save best combo
            self.conf_threshold.set(best_conf)
            self.max_instances.set(best_max_inst)
            self.calibrated_metrics = best_metrics
            self.best_mAP = best_metrics["all_objects"].get("mAP", None)
            self.last_eval_mode = "auto"

            self.details_text.insert(tk.END, f"\n✅ Best combo: conf={best_conf}, max_inst={best_max_inst} "
                                             f"→ mAP={self.best_mAP * 100:.1f}%, Score={best_score * 100:.1f}%\n")
            self.update_metrics_table(best_metrics)

            self.last_eval_mode = "auto"
            # Save best metrics JSON
            self.save_calibrated_metrics(auto=True)

            # Save all combinations to CSV
            csv_df = pd.DataFrame(all_results)
            safe_model_name = model.lower().replace("-", "").replace("_", "")
            out_dir = MODEL_METRICS_DIR / model / "thresholded_metrics" / "calibrated"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"{safe_model_name}_thresholding_combinations_metrics.csv"
            csv_df.to_csv(csv_path, index=False)

            self.details_text.insert(tk.END, f"📄 Saved all thresholding combinations to {csv_path.name}\n")

        threading.Thread(target=calibration, daemon=True).start()

    def run_manual_evaluation(self):
        model_display = self.selected_model.get()
        model = MODEL_NAME_MAP.get(model_display)
        if model is None:
            messagebox.showerror("Model Error", f"Unknown model name: {model_display}")
            return
        conf = self.conf_threshold.get()
        max_inst = self.max_instances.get()

        self.details_text.insert(tk.END, f"Running manual evaluation for {model}...\n")

        def evaluation():
            metrics = evaluate_model_metrics_with_thresholds(model, conf, max_inst)
            self.calibrated_metrics = aggregate_model_metrics(metrics)
            self.best_mAP = self.calibrated_metrics["all_objects"].get("mAP", None)
            self.update_metrics_table(self.calibrated_metrics)
            self.details_text.insert(tk.END, f"Manual mAP: {self.best_mAP:.3f}\n")
            self.last_eval_mode = "manual"
            self.save_calibrated_metrics(auto=False)

        threading.Thread(target=evaluation, daemon=True).start()

    def update_metrics_table(self, metrics):
        def to_percent(val):
            return f"{val * 100:.1f}%" if isinstance(val, float) else val

        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)

        for category in ["single_object", "multi_object", "all_objects"]:
            cat_metrics = metrics.get(category, {})
            f1 = cat_metrics.get("detection", {}).get("f1", {}).get("mean", 0.0)
            mAP = cat_metrics.get("mAP", 0.0)
            proc = cat_metrics.get("processing_time", {}).get("mean", 0.0)

            self.metrics_table.insert(
                "", "end",
                values=(category, to_percent(mAP), to_percent(f1), f"{proc:.3f}")
            )

    def save_calibrated_metrics(self, auto=False):
        if not self.calibrated_metrics:
            messagebox.showwarning("Nothing to save", "No metrics available. Run calibration or evaluation first.")
            return

        model_display = self.selected_model.get()
        model = MODEL_NAME_MAP.get(model_display)
        if model is None:
            messagebox.showerror("Model Error", f"Unknown model name: {model_display}")
            return
        conf = self.conf_threshold.get()
        max_inst = self.max_instances.get()

        subdir = "calibrated" if self.last_eval_mode == "auto" else "manual"
        out_dir = MODEL_METRICS_DIR / model / "thresholded_metrics" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{model}_conf{conf:.2f}_max{int(max_inst)}.json"

        results = {
            "model": model,
            "optimized_confidence": conf,
            "optimized_max_instances": max_inst,
            "best_mAP": self.best_mAP,
            "thresholded_metrics": self.calibrated_metrics
        }

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        self.details_text.insert(tk.END, f"Saved metrics to {out_file}\n")
        messagebox.showinfo("Saved", f"Metrics saved to {out_file}")


if __name__ == "__main__":
    app = MetricsEvaluationGUI()
    app.mainloop()