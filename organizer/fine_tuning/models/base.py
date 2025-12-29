from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from storage.training_models import TrainingSample


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""

    @abstractmethod
    def predict(
        self, samples: List[TrainingSample]
    ) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """Predict labels for multiple samples.

        Args:
            samples: List of TrainingSample objects

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        pass
