#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import optuna
import json
from pathlib import Path

# Import functions from your metrics_evaluation module.
from metrics_evaluation import (
    get_cached_aggregated_metrics,  # Loads cached aggregated raw metrics
    evaluate_with_thresholds  # Re-evaluates metrics with given thresholds
)

# Configuration constants (you can also import these from your config module)
MODEL_METRICS_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")
THRESHOLDED_SUBDIR = "thresholded_metrics"
MODELS = ["yolo", "maskrcnn", "yolo_deeplab"]

# Default threshold values (from your config)
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_INSTANCES = 5


class MetricsEvaluationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metrics Evaluation & Automatic Calibration")
        self.geometry("1100x650")

        # Variables
        self.best_mAP = None
        self.selected_model = tk.StringVar(value=MODELS[0])
        self.conf_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.max_instances = tk.IntVar(value=DEFAULT_MAX_INSTANCES)
        self.optimization_result = None  # Will store best trial from Optuna
        self.thresholded_metrics = None  # Will store re-evaluated metrics with chosen thresholds

        # Create GUI widgets
        self.create_widgets()
        # Load cached aggregated raw metrics (computed using default thresholds)
        self.load_cached_metrics()

    def create_widgets(self):
        # Row 0: Model selection
        ttk.Label(self, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_combo = ttk.Combobox(self, textvariable=self.selected_model, values=MODELS, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.load_cached_metrics())

        # Row 1: Confidence threshold (displayed as a slider and label)
        ttk.Label(self, text="Confidence Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.conf_scale = ttk.Scale(self, from_=0.0, to=1.0, variable=self.conf_threshold, orient="horizontal")
        self.conf_scale.grid(row=1, column=1, padx=10, pady=5, sticky="we")
        self.conf_label = ttk.Label(self, textvariable=self.conf_threshold)
        self.conf_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # Row 2: Max Instances selection
        ttk.Label(self, text="Max Instances:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.max_inst_combo = ttk.Combobox(self, textvariable=self.max_instances, values=[3, 5, 10], state="readonly")
        self.max_inst_combo.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Row 3: Automatic Calibration Button
        self.auto_calib_button = ttk.Button(self, text="Run Automatic Calibration", command=self.run_auto_calibration)
        self.auto_calib_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Row 4: Manual Evaluation Button (in case you want to run evaluation with current slider settings)
        self.manual_eval_button = ttk.Button(self, text="Run Evaluation (Manual Thresholds)",
                                             command=self.run_thresholded_evaluation)
        self.manual_eval_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        # Row 5: Table for Aggregated Metrics Output
        self.metrics_table = ttk.Treeview(self, columns=("Category", "mAP", "F1", "Processing Time"), show="headings",
                                          height=8)
        for col in ("Category", "mAP", "F1", "Processing Time"):
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=120)
        self.metrics_table.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Row 0-5, Column 3: Text widget for additional output details
        self.details_text = tk.Text(self, width=50, height=30)
        self.details_text.grid(row=0, column=3, rowspan=6, padx=10, pady=10, sticky="nsew")

        # Row 6: Save Results Button
        self.save_button = ttk.Button(self, text="Save Thresholded Metrics", command=self.save_thresholded_metrics)
        self.save_button.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

    def load_cached_metrics(self):
        model = self.selected_model.get()
        try:
            cached = get_cached_aggregated_metrics(model)
            self.details_text.insert(tk.END, f"Loaded cached aggregated raw metrics for model: {model}\n")
        except Exception as e:
            self.details_text.insert(tk.END, f"Error loading cached metrics for {model}: {str(e)}\n")

    def run_auto_calibration(self):
        model = self.selected_model.get()

        def objective(trial):
            conf = trial.suggest_float("confidence", 0.0, 1.0)
            max_inst = trial.suggest_int("max_instances", 1, 10)
            mAP, _ = evaluate_with_thresholds(model, conf, max_inst)
            return mAP

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best = study.best_trial
        self.optimization_result = best  # Important line to keep!

        # Update GUI sliders
        self.conf_threshold.set(best.params["confidence"])
        self.max_instances.set(best.params["max_instances"])

        # Re-evaluate with best thresholds
        overall_mAP, aggregated_metrics = evaluate_with_thresholds(model, best.params["confidence"],
                                                                   best.params["max_instances"])
        self.thresholded_metrics = aggregated_metrics
        self.update_metrics_table(aggregated_metrics)

        # Set best mAP explicitly for saving
        self.best_mAP = best.value

        # Optionally, you can also save these thresholded metrics automatically.

    def run_thresholded_evaluation(self):
        model = self.selected_model.get()
        conf = self.conf_threshold.get()
        max_inst = self.max_instances.get()
        self.details_text.insert(tk.END,
                                 f"Running thresholded evaluation for {model} with confidence={conf:.3f} and max_instances={max_inst}\n")

        def eval_thread():
            overall_mAP, metrics = evaluate_with_thresholds(model, conf, max_inst)
            self.thresholded_metrics = metrics
            self.update_metrics_table(metrics)
            self.details_text.insert(tk.END, f"Evaluation complete. Overall mAP: {overall_mAP:.3f}\n")

        threading.Thread(target=eval_thread, daemon=True).start()

    def update_metrics_table(self, metrics):
        # Clear existing table entries
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)

        # Iterate through each category
        for cat in ["single_object", "multi_object", "all_objects"]:
            cat_data = metrics.get(cat, {})
            det = cat_data.get("detection", {})

            # Correctly retrieve mAP directly from the category dictionary
            mAP_val = cat_data.get("mAP", 0)

            f1_mean = det.get("f1", {}).get("mean", 0)
            proc_time = cat_data.get("processing_time", {}).get("mean", 0)

            # Insert correctly retrieved values into the table
            self.metrics_table.insert("", "end",
                                      values=(cat, f"{mAP_val:.3f}", f"{f1_mean:.3f}", f"{proc_time:.3f}"))
    def save_thresholded_metrics(self):
        if not self.thresholded_metrics:
            messagebox.showwarning("No Data", "No thresholded metrics available. Run calibration first.")
            return

        model = self.selected_model.get()
        out_dir = MODEL_METRICS_DIR / model / THRESHOLDED_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{model}_thresholded_metrics.json"

        # Define overall_mAP explicitly from "all_objects" category
        overall_mAP = self.thresholded_metrics.get("all_objects", {}).get("mAP", None)

        results = {
            "model": model,
            "optimized_confidence": self.conf_threshold.get(),
            "optimized_max_instances": self.max_instances.get(),
            "best_mAP": self.best_mAP if self.best_mAP is not None else overall_mAP,
            "thresholded_metrics": self.thresholded_metrics
        }

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        messagebox.showinfo("Saved", f"Thresholded metrics saved to {out_file}")
        self.details_text.insert(tk.END, f"Saved thresholded metrics to {out_file}\n")


if __name__ == "__main__":
    app = MetricsEvaluationGUI()
    app.mainloop()