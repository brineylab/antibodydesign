# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


def boltz1(
    fasta_path: str,
    output_path: str,
    constraints: str | None = None,
    glycans: str | None = None,
    numbering_reference: str | None = None,
    device: str | None = None,
) -> None:
    """
    Run inference with `Boltz-1`_.

    .. _Boltz-1: https://github.com/jwohlwend/boltz

    """
