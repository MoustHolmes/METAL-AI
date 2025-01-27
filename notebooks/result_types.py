from dataclasses import dataclass
import numpy as np
import ASF

@dataclass
class CalculationResult:
    runtime: float = np.inf
    initial_asf: ASF = None

@dataclass
class ConvergedResult(CalculationResult):
    eigenvalues: np.ndarray = None
    eigenvectors: np.ndarray = None
    slices: np.ndarray = None

@dataclass
class NonConvergedResult(CalculationResult):
    error_csf: str = None
    error: str = None

@dataclass
class CrashedResult(CalculationResult):
    pass