# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import queue
import subprocess as sp
from typing import Iterable

import torch


def gpu_worker(
    cmd: str,
    gpu_queue: queue.Queue,
) -> None:
    """
    A worker function for running commands on a GPU.

    Parameters
    ----------
    cmd : str
        The command to run, as a string. Should not include the ``CUDA_VISIBLE_DEVICES``
        environment variable, as this will be added by the `gpu_worker` function.

    gpu_queue : queue.Queue
        A queue of GPU IDs which will be used to determine which GPU to use for the command.

    Returns
    -------
    result : subprocess.CompletedProcess
        The result of the command.

    """
    gpu_id = gpu_queue.get()
    try:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        result = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    finally:
        gpu_queue.put(gpu_id)
    return result


def get_gpu_queue(gpus: int | Iterable[int | str] | str | None = None) -> queue.Queue:
    """
    Get a queue of GPU IDs.

    Parameters
    ----------
    gpus : int | Iterable[int] | str | None, optional
        A single GPU ID, a list of GPU IDs, or a string of GPU IDs separated by commas.
        If ``None``, all GPUs will be used.

    Returns
    -------
    gpu_queue : queue.Queue
        A queue of GPU IDs.

    Examples
    --------
    >>> get_gpu_queue(0)
    >>> get_gpu_queue([0, 1])
    >>> get_gpu_queue("0,1")
    >>> get_gpu_queue(None)

    """
    # parse gpu IDs
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    elif isinstance(gpus, str):
        gpus = [int(gpu) for gpu in gpus.split(",")]
    elif isinstance(gpus, int):
        gpus = [gpus]
    else:
        gpus = [int(gpu) for gpu in gpus]
    if not gpus:
        raise ValueError("No GPUs specified.")

    # create queue
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    return gpu_queue
