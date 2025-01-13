# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


def protenix(
    fasta_path: str, output_path: str, model_path: str, device: str | None = None
) -> None:
    """
    Run inference with `Protenix`_.

    .. _Protenix: https://github.com/bytedance/Protenix

    """
