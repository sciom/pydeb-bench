"""DEB parameter containers and default AmP-derived prior specifications."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Mapping


@dataclass
class DEBParams:
    """Simplified DEB growth parameters, AmP-compatible.

    Attributes
    ----------
    Linf : float
        Ultimate structural length (mm).
    rB : float
        Von Bertalanffy growth rate constant (d^-1).
    L0 : float
        Structural length at birth (mm).
    T_A : float
        Arrhenius temperature (K).
    T_ref_C : float
        Reference temperature at which ``rB`` is expressed (degrees C).
    sigma : float
        Observation noise standard deviation (mm). Used for Bayesian and
        likelihood-based workflows; ignored by the deterministic forward
        model.
    """

    Linf: float = 4.8
    rB: float = 0.15
    L0: float = 0.8
    T_A: float = 8000.0
    T_ref_C: float = 20.0
    sigma: float = 0.12

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def daphnia_magna(cls) -> "DEBParams":
        """Return AmP-derived defaults for *Daphnia magna*.

        Values taken from the AmP entry for *Daphnia magna* as used in the
        benchmark of Hackenberger & Djerdj (2026); ``sigma`` is a conservative
        observation-noise default, not an AmP parameter.
        """
        return cls(Linf=4.8, rB=0.15, L0=0.8, T_A=8000.0, T_ref_C=20.0, sigma=0.12)


@dataclass(frozen=True)
class LogNormalPrior:
    """Lognormal prior on a strictly positive parameter."""

    mu_log: float
    sigma_log: float

    def summary(self) -> str:
        return f"LogNormal(mu_log={self.mu_log:.3f}, sigma_log={self.sigma_log:.3f})"


@dataclass(frozen=True)
class UniformPrior:
    """Uniform prior for weakly-informative cases (e.g. observation noise)."""

    lower: float
    upper: float

    def summary(self) -> str:
        return f"Uniform({self.lower:g}, {self.upper:g})"


Prior = LogNormalPrior | UniformPrior


@dataclass(frozen=True)
class PriorSpec:
    """Collection of priors matching the R benchmark (Daphnia magna / Daphniidae)."""

    Linf: Prior = LogNormalPrior(mu_log=1.5686159, sigma_log=0.2)  # log(4.8)
    rB: Prior = LogNormalPrior(mu_log=-1.8971200, sigma_log=0.3)   # log(0.15)
    L0: Prior = LogNormalPrior(mu_log=-0.2231436, sigma_log=0.2)   # log(0.8)
    sigma: Prior = UniformPrior(lower=0.01, upper=1.0)

    def describe(self) -> Mapping[str, str]:
        return {
            "Linf": self.Linf.summary(),
            "rB": self.rB.summary(),
            "L0": self.L0.summary(),
            "sigma": self.sigma.summary(),
        }


# Module-level default priors, matching the R benchmark informative priors.
AMP_PRIORS: PriorSpec = PriorSpec()
