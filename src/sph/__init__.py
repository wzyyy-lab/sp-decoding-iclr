"""Current PARC-16 research package."""

from .parc import (
    BLOCK_LENGTH,
    CANDIDATES,
    EXPECTED_PARAMETER_COUNT,
    PARC16Head,
    PARCLossOutput,
    PARCOutput,
    assert_frozen_architecture,
    nonshift_full16_prediction_hidden,
    parc_fixed_reference_loss,
)

__all__ = [
    "BLOCK_LENGTH",
    "CANDIDATES",
    "EXPECTED_PARAMETER_COUNT",
    "PARC16Head",
    "PARCLossOutput",
    "PARCOutput",
    "assert_frozen_architecture",
    "nonshift_full16_prediction_hidden",
    "parc_fixed_reference_loss",
]
