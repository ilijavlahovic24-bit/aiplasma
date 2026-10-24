"""
normalization.py
Shared implementation of the unit-normalization protocol (ADR-018).

Lives in: core/normalization.py

A domain problem defines scales() and delegates to_normalized_units() /
from_normalized_units() to the two functions below.

scales() keys:
    length: divides every spatial coordinate (all columns except the last)
    time:   divides the last coordinate column
    value:  divides the field values. A float, or one scale per channel.

The scales used are stored in tensor.metadata, so from_normalized_units()
inverts the exact normalization that was applied.

Like to_bohm_units(), the Domain of the tensor is left in its original
units.
"""

import torch
from torch import Tensor

from data.sources.physical_tensor import PhysicalTensor

_STORED_KEYS = ("normalized", "scales", "original_units")


def _value_tensor(value, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def normalize_tensor(tensor: PhysicalTensor, scales: dict) -> PhysicalTensor:
    """Divides coordinates and values by the given reference scales."""
    if tensor.metadata.get("normalized", False):
        raise ValueError("normalize_tensor(): tensor is already normalized.")

    length  = float(scales.get("length", 1.0))
    t_scale = float(scales.get("time", 1.0))
    value   = scales.get("value", 1.0)

    coords = tensor.coordinates.clone()
    coords[:, :-1] = coords[:, :-1] / length     # spatial columns
    coords[:, -1]  = coords[:, -1] / t_scale     # time is always last

    values = tensor.values / _value_tensor(value, tensor.values)

    return PhysicalTensor(
        values=values,
        coordinates=coords,
        units=_normalized_units(tensor),
        coord_system=tensor.coord_system,
        domain=tensor.domain,
        metadata={
            **tensor.metadata,
            "normalized":     True,
            "scales":         {"length": length, "time": t_scale, "value": value},
            "original_units": tensor.units,
        },
    )


def denormalize_tensor(tensor: PhysicalTensor) -> PhysicalTensor:
    """Inverse of normalize_tensor(), using the scales stored in metadata."""
    if not tensor.metadata.get("normalized", False):
        raise ValueError(
            "denormalize_tensor(): tensor is not normalized. "
            "Call to_normalized_units() first."
        )

    s = tensor.metadata["scales"]

    coords = tensor.coordinates.clone()
    coords[:, :-1] = coords[:, :-1] * s["length"]
    coords[:, -1]  = coords[:, -1] * s["time"]

    values = tensor.values * _value_tensor(s["value"], tensor.values)

    return PhysicalTensor(
        values=values,
        coordinates=coords,
        units=tensor.metadata["original_units"],
        coord_system=tensor.coord_system,
        domain=tensor.domain,
        metadata={k: v for k, v in tensor.metadata.items() if k not in _STORED_KEYS},
    )


def _normalized_units(tensor: PhysicalTensor):
    """Same UnitSystem type as the input, with every unit set to 'normalized'."""
    return type(tensor.units)(spatial="normalized", temporal="normalized", field="normalized")