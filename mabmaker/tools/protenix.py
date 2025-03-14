# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import concurrent.futures as cf
import os
from typing import Iterable

import abutils
from tqdm.auto import tqdm

from ..utils.inputs import StructurePredictionRun, setup_structure_prediction_run
from ..utils.jobs import get_gpu_queue, gpu_worker


def protenix(
    json_path: str,
    output_path: str,
    gpus: int | Iterable[int] | None = None,
    use_msa_server: bool = True,
) -> None:
    """
    Structure prediction with `Protenix`_.

    Parameters
    ----------
    json_path : str
        The path to the JSON file containing the input parameters, or a folder containing
        one or more JSON files. Each JSON file should follow the schema of the
        `AlphaFold3 input JSON file`_, which allows for multiple runs to be specified in
        a single file.

    output_path : str
        The path to the output directory. If it does not exist, it will be created.

    numbering_reference : str | None, optional
        The path to a PDB file containing the numbering reference. If provided, the
        residues in the input JSON files will be renumbered based on the numbering in
        the PDB file.

    gpus : Union[int, Iterable, None], optional, default=None
        GPU(s) to use. Can be provided as:
            - a single integer: ``0``
            - a comma-separated string of integers: ``"0,1"``
            - a list or tuple of integers: ``[0, 1]``
        If not provided, all available GPUs will be used.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server.

    .. _Protenix: https://github.com/bytedance/Protenix
    .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold/tree/main/server

    """
    # setup runs
    runs = setup_structure_prediction_run(json_path, output_path)

    # get GPU queue
    gpu_queue = get_gpu_queue(gpus)
    num_gpus = gpu_queue.qsize()

    # run predictions
    futures = []
    with cf.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for run in runs:
            run_output_dir = os.path.join(output_path, run.name)
            abutils.io.make_dir(run_output_dir)
            cmd = _build_protenix_command(run, run_output_dir, use_msa_server)
            futures.append(executor.submit(gpu_worker, cmd, gpu_queue))

        # monitor progress
        with tqdm(
            total=len(futures),
            desc="Protenix",
            bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        ) as pbar:
            for _ in cf.as_completed(futures):
                pbar.update(1)

    # write prediction logs (stdout and stderr)
    for run, future in zip(runs, futures):
        result = future.result()
        with open(os.path.join(output_path, run.name, "stdout.log"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(output_path, run.name, "stderr.log"), "w") as f:
            f.write(result.stderr)


def _build_protenix_command(
    run: StructurePredictionRun,
    out_dir: str,
    use_msa_server: bool = True,
) -> str:
    """
    Build a command for running `Protenix`_.

    Parameters
    ----------
    run : StructurePredictionRun
        The run to build the command for.

    output_path : str
        The path to the output directory.

    use_msa_server : bool, optional, default=True
        Whether to use the MSA server.

    Returns
    -------
    command : str
        The command to run.

    """
    json_path = os.path.join(out_dir, f"{run.name}.json")
    seeds = ",".join(run.seeds)

    # build command
    cmd = "protenix predict"
    cmd += f" --input '{json_path}'"
    cmd += f" --out_dir '{out_dir}'"
    cmd += f" --seeds {seeds}"
    if use_msa_server:
        cmd += " --use_msa_server"
    return cmd
