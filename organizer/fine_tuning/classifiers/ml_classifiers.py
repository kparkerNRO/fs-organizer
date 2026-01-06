"""
Classifier models for fine-tuning pipeline.

- SetFitClassifier: Wrapper for fine-tuned or baseline SetFit models.
- ZeroShotClassifier: Zero-shot classification using sentence embeddings.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from storage.training_models import TrainingSample
from fine_tuning.taxonomy import get_labels

from setfit import SetFitModel  # type: ignore
from sentence_transformers import SentenceTransformer


class ZeroShotClassifier:
    """Zero-shot classifier using embedding similarity (no training needed)."""

    def __init__(self, taxonomy: str = "v2"):
        """Initialize zero-shot classifier.

        Args:
            taxonomy: Which label set to use ('v1', 'v2', or 'legacy')
        """
        self.taxonomy = taxonomy
        self.labels = get_labels(taxonomy)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.label_descriptions = self._get_label_descriptions(taxonomy)
        self.label_names = list(self.label_descriptions.keys())
        self.label_embeddings = self.model.encode(
            list(self.label_descriptions.values()), show_progress_bar=False
        )

    def _get_label_descriptions(self, taxonomy: str) -> Dict[str, str]:
        """Get semantic descriptions for each label in the taxonomy."""
        if taxonomy == "v1":
            return {
                "person_or_group": "creator publisher studio artist cartographer author person group organization company collaboration presents by patreon team",
                "content": "location place setting scene environment world region city town village dungeon tavern temple shop castle building encounter battleground proper noun",
                "media_bucket": "asset pack tokens maps music illustrations portraits items cards stat blocks handouts paper minis tiles backgrounds adventure module media",
                "descriptor": "variant version style theme genre setting gridded gridless vtt print day night seasonal weather interior exterior furnished unfurnished clean phased empty looping",
                "other": "organizational administrative bonus rewards tiers instructions guide readme license archive polls wip work in progress",
                "unknown": "unclear ambiguous uncertain unclassifiable miscellaneous general uncategorized",
            }
        elif taxonomy == "v2":
            return {
                "creator_or_studio": "creator publisher studio artist cartographer author collaboration presents by patreon team",
                "content_subject": "location place setting scene environment world region city town village dungeon tavern temple shop castle building encounter battleground proper noun",
                "asset_type": "asset pack tokens maps music illustrations portraits items cards stat blocks handouts paper minis tiles backgrounds adventure module media",
                "descriptor": "variant version style theme genre setting gridded gridless vtt print day night seasonal weather interior exterior furnished unfurnished clean phased empty looping",
                "other": "organizational administrative bonus rewards tiers instructions guide readme license archive polls wip work in progress",
                "unknown": "unclear ambiguous uncertain unclassifiable miscellaneous general uncategorized",
            }
        else:  # legacy
            return {
                "primary_author": "main creator primary author original artist cartographer publisher",
                "secondary_author": "collaborator featured creator secondary author co-creator collaboration",
                "collection": "collection series set group pack bundle compilation",
                "subject": "subject matter topic theme location specific content castle tavern dungeon",
                "media_format": "file format extension type jpg png pdf mp3 wav webm vtt",
                "media_type": "media category asset type maps tokens music audio images videos handouts",
                "variant": "variant alternate version style modifier day night gridded season weather",
                "other": "organizational administrative rewards tiers bonus year month instructions notes",
            }

    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels using zero-shot embedding similarity."""
        texts = [sample.text for sample in samples]
        text_embeddings = self.model.encode(texts, show_progress_bar=True)
        similarities = cosine_similarity(text_embeddings, self.label_embeddings)

        predictions = []
        confidences = []
        all_probabilities = []

        for sim_row in similarities:
            best_idx = int(np.argmax(sim_row))
            predicted_label = self.label_names[best_idx]
            confidence = float(sim_row[best_idx])

            predictions.append(predicted_label)
            confidences.append(confidence)

            temperature = 0.5
            exp_sim = np.exp(sim_row / temperature)
            probs = exp_sim / np.sum(exp_sim)
            prob_dict = {
                label: float(prob) for label, prob in zip(self.label_names, probs)
            }
            all_probabilities.append(prob_dict)

        return predictions, confidences, all_probabilities


class SetFitClassifier:
    """Wrapper for SetFit model (fine-tuned or baseline)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        taxonomy: str = "legacy",
        use_baseline: bool = False,
    ):
        """Initialize SetFit classifier."""
        if use_baseline:
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            if not model_path:
                raise ValueError("model_path required when use_baseline=False")
            self.model = SetFitModel.from_pretrained(model_path)

        self.taxonomy = taxonomy
        self.use_baseline = use_baseline
        self.labels = get_labels(taxonomy)

    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels for multiple samples."""
        texts = [sample.text for sample in samples]
        predictions = self.model.predict(texts)
        try:
            predictions_list = predictions.tolist()  # type: ignore[union-attr]
        except AttributeError:
            predictions_list = list(predictions)  # type: ignore[arg-type]

        # Convert integer predictions to string labels
        # SetFit models return integer indices, need to map to string labels
        # Check if we have integer predictions
        try:
            # Try to convert first prediction to int - if it works, they're integers
            int(predictions_list[0])
            is_integer = True
        except (ValueError, TypeError):
            is_integer = False

        if predictions_list and is_integer:
            # Get the label mapping from the model
            # Check if model.classes_ contains actual string labels or just indices
            has_string_classes = False
            if hasattr(self.model, "model_head") and hasattr(
                self.model.model_head, "classes_"
            ):
                # Check if classes are string labels (not integers)
                try:
                    first_class = self.model.model_head.classes_[0]  # type: ignore[index]
                    # If it's an integer or can be converted to int, use taxonomy labels
                    int(first_class)
                    has_string_classes = False
                except (ValueError, TypeError):
                    has_string_classes = True

                if has_string_classes:
                    label_mapping = {
                        i: str(label)
                        for i, label in enumerate(self.model.model_head.classes_)  # type: ignore[arg-type]
                    }
                else:
                    # Model has integer classes, use sorted taxonomy labels
                    label_list = sorted(self.labels)
                    label_mapping = {i: label for i, label in enumerate(label_list)}

                predictions_list = [
                    label_mapping[int(pred)] for pred in predictions_list
                ]
            else:
                # Fallback: use sorted labels if no mapping available
                label_list = sorted(self.labels)
                predictions_list = [label_list[int(pred)] for pred in predictions_list]

        try:
            probs = self.model.predict_proba(texts)
            confidences = np.max(probs, axis=1).tolist()
            label_list = sorted(self.labels)
            all_probabilities = [
                {label: float(prob) for label, prob in zip(label_list, prob_row)}
                for prob_row in probs
            ]
        except Exception:
            confidences = [1.0] * len(predictions_list)
            all_probabilities = [
                {label: (1.0 if label == pred else 0.0) for label in self.labels}
                for pred in predictions_list
            ]

        return predictions_list, confidences, all_probabilities
