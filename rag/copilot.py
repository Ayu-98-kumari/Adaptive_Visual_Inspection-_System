"""
Operator Copilot - Interactive RAG Assistant for Quality Technicians & Line Supervisors
"""

from typing import Dict, Any, Optional
import pandas as pd
from rag.retriever import RAGRetriever
from rag.telemetry_loader import TelemetryLoader

class OperatorCopilot:
    def __init__(self, retriever: RAGRetriever, telemetry_loader: TelemetryLoader = None):
        self.retriever = retriever
        self.loader = telemetry_loader or TelemetryLoader()
        self._telemetry_df: Optional[pd.DataFrame] = None

    @property
    def telemetry_df(self) -> pd.DataFrame:
        if self._telemetry_df is None:
            self._telemetry_df = self.loader.load_inference_telemetry()
        return self._telemetry_df

    def diagnose_image(self, image_path: str) -> Dict[str, Any]:
        """
        Retrieves telemetry for a specific image, queries the SOP knowledge base,
        and generates an end-to-end diagnostic explanation.
        """
        matches = self.telemetry_df[self.telemetry_df["image_path"].str.contains(image_path, regex=False)]
        if matches.empty:
            # Try fuzzy or filename match
            filename = image_path.split("/")[-1]
            matches = self.telemetry_df[self.telemetry_df["image_path"].str.endswith(filename)]

        if matches.empty:
            return {
                "status": "error",
                "message": f"Image path '{image_path}' not found in telemetry records."
            }

        row = matches.iloc[0].to_dict()
        category = str(row.get("category", "general")).lower()
        defect_type = str(row.get("predicted_defect_type", "general")).lower()
        risk_level = str(row.get("risk_level", "medium")).lower()
        selected_action = str(row.get("selected_action", "prediction")).lower()
        norm_score = float(row.get("normalized_score", 0.0))

        # Retrieve relevant SOP documentation
        query = f"rework root cause severity disposal for {defect_type} in {category}"
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            category=category,
            defect_code=defect_type if defect_type != "good" else None,
            top_k=2
        )

        formatted_sop = self.retriever.format_context_for_prompt(retrieved_chunks)

        explanation = self._build_operator_explanation(
            row=row,
            category=category,
            defect_type=defect_type,
            risk_level=risk_level,
            selected_action=selected_action,
            norm_score=norm_score,
            formatted_sop=formatted_sop
        )

        return {
            "status": "success",
            "image_path": row.get("image_path"),
            "category": category,
            "predicted_defect_type": defect_type,
            "normalized_score": norm_score,
            "risk_level": risk_level,
            "selected_action": selected_action,
            "recommended_next_step": row.get("recommended_next_step"),
            "retrieved_sop": formatted_sop,
            "operator_guidance": explanation
        }

    def ask(self, question: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        General operator inquiry answered with grounded citations from the manufacturing SOPs.
        """
        chunks = self.retriever.retrieve(
            query=question,
            category=category,
            top_k=3
        )
        context = self.retriever.format_context_for_prompt(chunks)

        answer = (
            f"### Grounded Engineering Response\n\n"
            f"Based on the official Standard Operating Procedures retrieved for your query:\n\n"
            f"{context}\n\n"
            f"**Operational Note**: All procedural actions must be logged in the Shift Quality Ledger."
        )

        return {
            "question": question,
            "category": category,
            "retrieved_chunks_count": len(chunks),
            "response": answer
        }

    def _build_operator_explanation(
        self,
        row: dict,
        category: str,
        defect_type: str,
        risk_level: str,
        selected_action: str,
        norm_score: float,
        formatted_sop: str
    ) -> str:
        return f"""
### Diagnostic Inspection Summary
- **Part Identifier**: `{row.get('image_path')}`
- **Product Category**: `{category.upper()}`
- **Vision Evaluation**: Normalized Anomaly Score = `{norm_score:.3f}` (Threshold ~ 0.95)
- **Identified Defect Type**: `{defect_type}`
- **RL Risk Tier**: `{risk_level.upper()}`
- **RL Action Executed**: `{selected_action.upper()}` ({row.get('recommended_next_step')})

### Actionable Remediation Guidelines (Ref: Official Manufacturing SOP)
{formatted_sop}
"""
