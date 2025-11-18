from abc import ABC, abstractmethod

class AnalysisBase(ABC):
    @abstractmethod
    def setup(self, parent_layout):
        """Set up analysis-specific UI components."""
        pass

    @abstractmethod
    def process(self, annotated_frame, results):
        """Process each frame for the analysis."""
        pass

    @abstractmethod
    def reset(self):
        """Reset analysis-specific variables."""
        pass
