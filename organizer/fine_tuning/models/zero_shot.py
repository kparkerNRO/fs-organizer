from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from storage.training_models import TrainingSample

from ..taxonomy import get_labels
from .base import BaseClassifier


class ZeroShotClassifier(BaseClassifier):
    """Zero-shot classifier using embedding similarity (no training needed)."""

    def __init__(self, taxonomy: str = "v2"):
        """Initialize zero-shot classifier.

        Args:
            taxonomy: Which label set to use ('v1', 'v2', or 'legacy')
        """
        self.taxonomy = taxonomy
        self.labels = get_labels(taxonomy)

        # Load sentence transformer model
        print("Loading sentence transformer: sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Define label descriptions for semantic matching
        self.label_descriptions = self._get_label_descriptions(taxonomy)

        # Pre-compute label embeddings
        print(f"Computing embeddings for {len(self.label_descriptions)} labels...")
        self.label_names = list(self.label_descriptions.keys())
        self.label_embeddings = self.model.encode(
            list(self.label_descriptions.values()), show_progress_bar=False
        )

    def _get_label_descriptions(self, taxonomy: str) -> Dict[str, str]:
        """Get semantic descriptions for each label in the taxonomy.

        Args:
            taxonomy: Taxonomy name ('v1', 'v2', or 'legacy')

        Returns:
            Dict mapping label names to semantic descriptions
        """
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
        """Predict labels using zero-shot embedding similarity.

        Args:
            samples: List of TrainingSample objects

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        # Extract text features from samples
        texts = [sample.text for sample in samples]

        # Encode sample texts
        text_embeddings = self.model.encode(texts, show_progress_bar=True)

        # Compute cosine similarity between each sample and each label
        similarities = cosine_similarity(text_embeddings, self.label_embeddings)

        # Get predictions and confidences
        predictions = []
        confidences = []
        all_probabilities = []

        for sim_row in similarities:
            # Find best matching label
            best_idx = int(np.argmax(sim_row))
            predicted_label = self.label_names[best_idx]
            confidence = float(sim_row[best_idx])

            predictions.append(predicted_label)
            confidences.append(confidence)

            # Convert similarities to probability-like scores (softmax)
            # Use temperature scaling to make the distribution sharper
            temperature = 0.5
            exp_sim = np.exp(sim_row / temperature)
            probs = exp_sim / np.sum(exp_sim)

            prob_dict = {label: float(prob) for label, prob in zip(self.label_names, probs)}
            all_probabilities.append(prob_dict)

        return predictions, confidences, all_probabilities
