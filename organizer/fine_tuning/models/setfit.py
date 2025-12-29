from typing import Dict, List, Optional, Tuple

import numpy as np
from setfit import SetFitModel
from storage.training_models import TrainingSample

from ..taxonomy import get_labels
from .base import BaseClassifier


class SetFitClassifier(BaseClassifier):
    """Wrapper for SetFit model (fine-tuned or baseline)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        taxonomy: str = "legacy",
        use_baseline: bool = False,
    ):
        """Initialize SetFit classifier.

        Args:
            model_path: Path to saved fine-tuned SetFit model (required unless use_baseline=True)
            taxonomy: Which label set to use ('v1', 'v2', or 'legacy')
            use_baseline: Use baseline pre-trained model without fine-tuning
        """
        if use_baseline:
            # Use baseline pre-trained sentence transformer
            print("Loading baseline model: sentence-transformers/all-MiniLM-L6-v2")
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            if not model_path:
                raise ValueError("model_path required when use_baseline=False")
            self.model = SetFitModel.from_pretrained(model_path)

        self.taxonomy = taxonomy
        self.use_baseline = use_baseline

        # Get label set for taxonomy
        self.labels = get_labels(taxonomy)

    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels for multiple samples.

        Args:
            samples: List of TrainingSample objects

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        # Extract text features
        texts = [sample.text for sample in samples]

        # Get predictions
        predictions = self.model.predict(texts)

        # Get probabilities
        try:
            probs = self.model.predict_proba(texts)
            confidences = np.max(probs, axis=1).tolist()

            # Convert to list of dicts
            label_list = sorted(self.labels)
            all_probabilities = []
            for prob_row in probs:
                prob_dict = {
                    label: float(prob) for label, prob in zip(label_list, prob_row)
                }
                all_probabilities.append(prob_dict)
        except Exception:
            # Fallback if predict_proba not available
            confidences = [1.0] * len(predictions)
            all_probabilities = [
                {label: (1.0 if label == pred else 0.0) for label in self.labels}
                for pred in predictions
            ]

        return predictions.tolist(), confidences, all_probabilities
