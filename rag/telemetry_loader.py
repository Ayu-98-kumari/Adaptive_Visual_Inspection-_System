"""
Telemetry Loader for Ingesting Step 1 (Vision) and Step 2 (RL Scheduler) Outputs
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from rag.config import RAGConfig

class TelemetryLoader:
    def __init__(self, vision_dir: Path = None, scheduler_dir: Path = None):
        self.vision_dir = vision_dir or RAGConfig.get_vision_outputs_dir()
        self.scheduler_dir = scheduler_dir or RAGConfig.get_scheduler_outputs_dir()

    def load_inference_telemetry(self) -> pd.DataFrame:
        """
        Loads the combined operational inference dataset.
        Merges Vision anomaly scores with RL scheduler decisions.
        """
        sched_file = self.scheduler_dir / "scheduler_inference_ready.csv"
        vision_file = self.vision_dir / "current_anomaly_score.csv"

        if sched_file.exists():
            sched_df = pd.read_csv(sched_file)
            if vision_file.exists():
                vision_df = pd.read_csv(vision_file)
                # Keep mahalanobis method rows from vision
                vision_mah = vision_df[vision_df["method"].str.lower() == "mahalanobis"].copy()
                merged = pd.merge(
                    sched_df,
                    vision_mah[["image_path", "anomaly_score", "normalized_score", "predicted_defect_type"]],
                    on="image_path",
                    how="left"
                )
                return merged
            return sched_df

        # If files are not present locally (e.g. running outside Google Drive), generate mock telemetry
        return self._generate_mock_inference_telemetry()

    def load_adaptation_telemetry(self) -> Dict[str, Any]:
        """Loads the RL adaptation experiment results and summaries."""
        results_file = self.scheduler_dir / "adaptation_demo_results.csv"
        summary_file = self.scheduler_dir / "adaptation_demo_summary.csv"

        if results_file.exists() and summary_file.exists():
            return {
                "results_df": pd.read_csv(results_file),
                "summary_df": pd.read_csv(summary_file),
                "source": "live_file"
            }

        return self._generate_mock_adaptation_telemetry()

    def _generate_mock_inference_telemetry(self, n_samples: int = 120) -> pd.DataFrame:
        """Generates realistic synthetic telemetry matching the exact schema of Step 1 and Step 2."""
        rng = np.random.default_rng(42)
        categories = ["grid", "zipper", "hazelnut", "leather", "toothbrush"]
        defect_map = {
            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
            "zipper": ["broken_teeth", "split_teeth", "rough", "fabric_border", "fabric_interior"],
            "hazelnut": ["crack", "cut", "hole", "print"],
            "leather": ["color", "cut", "fold", "glue", "poke"],
            "toothbrush": ["defective"]
        }

        records = []
        for i in range(n_samples):
            cat = rng.choice(categories)
            is_anomaly = rng.random() < 0.25
            norm_score = float(rng.uniform(0.92, 0.99) if is_anomaly else rng.uniform(0.40, 0.94))
            threshold = 0.95
            margin = norm_score - threshold

            if margin < -0.10:
                risk = "low"
                action = "skip"
            elif -0.10 <= margin <= 0.10:
                risk = "medium"
                action = rng.choice(["prediction", "expert"], p=[0.7, 0.3])
            else:
                risk = "high"
                action = "expert"

            pred_defect = rng.choice(defect_map[cat]) if is_anomaly else "good"
            img_path = f"{cat}/test/{pred_defect}/{i:03d}.png"

            records.append({
                "image_path": img_path,
                "category": cat,
                "risk_level": risk,
                "selected_action": action,
                "mahalanobis_prediction": 1 if norm_score >= threshold else 0,
                "recommended_next_step": (
                    "Route image to an expert for review" if action == "expert"
                    else "Use the Mahalanobis model prediction" if action == "prediction"
                    else "No additional inspection selected by the scheduler"
                ),
                "final_decision_status": (
                    "Pending expert review" if action == "expert"
                    else "Use model prediction" if action == "prediction"
                    else "No additional inspection selected"
                ),
                "anomaly_score": float(norm_score * 12.5),
                "normalized_score": norm_score,
                "predicted_defect_type": pred_defect
            })

        return pd.DataFrame(records)

    def _generate_mock_adaptation_telemetry(self) -> Dict[str, Any]:
        """Generates realistic adaptation demo metrics matching Step 2 Cell 70-74."""
        summary_data = [
            {"period": "Pre-shift (Items 1-3000)", "defect_rate": 0.032, "expert_review_rate": 0.041, "missed_defects": 2, "avg_cost": 0.38},
            {"period": "Post-shift (Items 3001-6000)", "defect_rate": 0.185, "expert_review_rate": 0.231, "missed_defects": 3, "avg_cost": 1.42}
        ]
        return {
            "summary_df": pd.DataFrame(summary_data),
            "results_df": pd.DataFrame(),
            "source": "synthetic_mock"
        }
