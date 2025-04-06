#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import optuna
import json
from pathlib import Path

from metrics_evaluation import (
    evaluate_model_metrics_with_thresholds,
    save_thresholded_metrics,
    aggregate_model_metrics
)
from config import IOU_THRESHOLD

MODEL_METRICS_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")

MODELS = ["yolo", "maskrcnn", "yolo_deeplab"]
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_INSTANCES = 5


class MetricsEvaluationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Automatic Calibration GUI")
        self.geometry("1100x650")

        self.selected_model = tk.StringVar(value=MODELS[0])
        self.conf_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.max_instances = tk.IntVar(value=DEFAULT_MAX_INSTANCES)
        self.best_mAP = None
        self.optimization_result = None
        self.calibrated_metrics = None

        self.create_widgets()

    def create_widgets(self):
        # Model selection
        ttk.Label(self, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_combo = ttk.Combobox(self, textvariable=self.selected_model, values=MODELS, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Confidence threshold slider
        ttk.Label(self, text="Confidence Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.conf_slider = ttk.Scale(self, from_=0.0, to=1.0, variable=self.conf_threshold, orient="horizontal")
        self.conf_slider.grid(row=1, column=1, padx=10, pady=5, sticky="we")
        self.conf_label = ttk.Label(self, textvariable=self.conf_threshold)
        self.conf_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # Max instances dropdown
        ttk.Label(self, text="Max Instances:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.max_inst_combo = ttk.Combobox(self, textvariable=self.max_instances, values=[3, 5, 10], state="readonly")
        self.max_inst_combo.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Buttons
        ttk.Button(self, text="Run Auto Calibration", command=self.run_auto_calibration).grid(
            row=3, column=0, columnspan=2, padx=10, pady=10)

        ttk.Button(self, text="Run Manual Evaluation", command=self.run_manual_evaluation).grid(
            row=4, column=0, columnspan=2, padx=10, pady=10)

        # Metrics table
        self.metrics_table = ttk.Treeview(self, columns=("Category", "mAP", "F1", "ProcTime"), show="headings", height=8)
        for col in ("Category", "mAP", "F1", "ProcTime"):
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=120)
        self.metrics_table.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Output area
        self.details_text = tk.Text(self, width=50, height=30)
        self.details_text.grid(row=0, column=3, rowspan=6, padx=10, pady=10, sticky="nsew")

        # Save button
        ttk.Button(self, text="Save Metrics", command=self.save_calibrated_metrics).grid(
            row=6, column=0, columnspan=3, padx=10, pady=10)

    def run_auto_calibration(self):
        model = self.selected_model.get()
        self.details_text.insert(tk.END, f"Running Optuna calibration for model: {model}\n")

        def objective(trial):
            conf = trial.suggest_float("confidence", 0.1, 0.99)
            max_inst = trial.suggest_int("max_instances", 1, 10)

            metrics = evaluate_model_metrics_with_thresholds(model, conf, max_inst)
            aggregated = aggregate_model_metrics(metrics)
            return aggregated["all_objects"].get("mAP", 0.0)

        def optimization():
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            self.optimization_result = study.best_trial
            best_conf = study.best_trial.params["confidence"]
            best_inst = study.best_trial.params["max_instances"]

            self.conf_threshold.set(best_conf)
            self.max_instances.set(best_inst)

            self.details_text.insert(tk.END, f"Best thresholds - Confidence: {best_conf:.3f}, Max Instances: {best_inst}\n")

            metrics = evaluate_model_metrics_with_thresholds(model, best_conf, best_inst)
            self.calibrated_metrics = aggregate_model_metrics(metrics)
            self.best_mAP = self.calibrated_metrics["all_objects"].get("mAP", None)
            self.update_metrics_table(self.calibrated_metrics)

            # Save to calibrated subdirectory
            self.save_calibrated_metrics(auto=True)

        threading.Thread(target=optimization, daemon=True).start()

    def run_manual_evaluation(self):
        model = self.selected_model.get()
        conf = self.conf_threshold.get()
        max_inst = self.max_instances.get()

        self.details_text.insert(tk.END, f"Running manual thresholded evaluation for {model}...\n")

        def evaluation():
            metrics = evaluate_model_metrics_with_thresholds(model, conf, max_inst)
            self.calibrated_metrics = aggregate_model_metrics(metrics)
            self.best_mAP = self.calibrated_metrics["all_objects"].get("mAP", None)
            self.update_metrics_table(self.calibrated_metrics)
            self.details_text.insert(tk.END, f"Evaluation complete. mAP: {self.best_mAP:.3f}\n")

            # Save manually
            self.save_calibrated_metrics(auto=False)

        threading.Thread(target=evaluation, daemon=True).start()

    def update_metrics_table(self, metrics):
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)

        for category in ["single_object", "multi_object", "all_objects"]:
            cat_metrics = metrics.get(category, {})
            f1 = cat_metrics.get("detection", {}).get("f1", {}).get("mean", 0.0)
            mAP = cat_metrics.get("mAP", 0.0)
            proc = cat_metrics.get("processing_time", {}).get("mean", 0.0)
            self.metrics_table.insert("", "end", values=(category, f"{mAP:.3f}", f"{f1:.3f}", f"{proc:.3f}"))

    def save_calibrated_metrics(self, auto=False):
        if not self.calibrated_metrics:
            messagebox.showwarning("Nothing to save", "No metrics available. Run calibration or evaluation first.")
            return

        model = self.selected_model.get()
        conf = self.conf_threshold.get()
        max_inst = self.max_instances.get()

        subdir = "calibrated" if auto else "manual"
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