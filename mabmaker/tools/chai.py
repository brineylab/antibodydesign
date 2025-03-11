# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import json
import os
from pathlib import Path
from typing import Iterable

import abutils
import magika
import pandas as pd
from chai_lab.chai1 import run_inference
from natsort import natsorted


def chai1(
    fasta_path: str,
    output_path: str,
    use_esm_embeddings: bool = False,
    # use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: str | None = None,
    num_trunk_recycles: int = 3,
    num_diffusion_timesteps: int = 200,
    num_diffusion_samples: int = 5,
    constraints: str | None = None,
    glycans: str | None = None,
    numbering_reference: str | None = None,
    seed: int | Iterable[int] = 42,
    device: str | None = None,
    low_memory: bool = False,
    verbose: bool = False,
    debug: bool = False,
    started_from_cli: bool = False,
) -> None:
    """
    Run inference with `Chai-1`_.

    Parameters
    ----------
    fasta_path : str, required
        Path to the input FASTA file or a directory of FASTA files. If a directory,
        each FASTA file in the directory will be processed separately.

    output_path : str, required
        Path to the output directory. If it does not exist, it will be created.

    use_esm_embeddings : bool, optional, default=False
        Whether to use ESM embeddings instead of MSAs for the input sequence.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        URL of the MSA server to use.

    msa_directory : str, optional
        Path to the directory containing MSAs. If provided, the MSAs in the directory
        will be used instead of the MSA server. If `use_esm_embeddings` is ``True``,
        the MSAs in the directory will be ignored.

    num_trunk_recycles : int, optional, default=3
        Number of trunk recycles to perform.

    num_diffusion_timesteps : int, optional, default=200
        Number of diffusion timesteps to use.

    num_diffusion_samples : int, optional, default=5
        Number of diffusion samples to generate.

    constraints : str, optional
        Path to a CSV-formatted constraints file. If both `constraints` and `glycans`
        are provided, the constraints required for glycans will be appended to the
        provided constraints file and the combined constraints file will be used.

    glycans : str, optional
        Path to a JSON-formatted file containing glycosylation sites.
        If provided as a dictionary mapping chains to positions::

        ``` json
        {
            "A": {160: "NAG(4-1 NAG(6-1 MAN(6-1(MAN(6-1 MAN))))))", 332: "NAG(4-1 NAG)"},
            "B": {157: "NAG(4-1 NAG)", 334: "NAG(4-1 NAG(6-1 MAN(6-1(MAN(6-1 MAN))))))"},
        }
        ```

        The glycan sites will be applied to all FASTA files (if a directory is provided).
        If provided as a dictionary mapping FASTA file names to nested dictionaries mapping chains to positions and glycan type::

        ``` json
        {
            "fasta_file.fasta": {
                "A": {160: "NAG(4-1 NAG(6-1 MAN(6-1(MAN(6-1 MAN))))))", 334: "NAG(4-1 NAG)"},
                "B": {157: "NAG(4-1 NAG)", 334: "NAG(4-1 NAG(6-1 MAN(6-1(MAN(6-1 MAN))))))"},
            }
        }
        ```

        The glycan sites will be applied to the specified FASTA file. If any of the input
        FASTA files or chains are not found, no glycans will be applied.

    numbering_reference : str, optional
        Path to a JSON file containing sequences to be used as a reference for
        calculating glycan site numbering. Should map chains to a reference sequence (as a string)::

        ``` json
        {
            "A": "METFLISDLKIHITER",
            "B": "METFLISDLKIHITER",
        }
        ```

        Or a JSON file mapping FASTA file names to chains and reference sequences::

        ``` json
        {
            "fasta_file.fasta": {
                "A": "METFLISDLKIHITER",
                "B": "METFLISDLKIHITER",
            }
        }
        ```

    seed : int, optional, default=42
        Random seed.

    device : str, optional
        Device to use. If `device` is not provided and `fasta_path` is the path to a
        single FASTA file, the device will be set to ``"cuda"``. If `device` is not
        provided and `fasta_path` is the path to a directory of FASTA files, jobs will
        be distributed across available GPUs. If `device` is not provided and CUDA is
        not available, the device will be set to ``"cpu"``.

    low_memory : bool, optional, default=False
        Whether to use low memory mode.

    verbose : bool, optional, default=False
        Whether to print verbose output.

    debug : bool, optional, default=False
        Whether to print debug output.

    started_from_cli : bool, optional, default=False
        Whether the function was called from the CLI. This changes logging behavior.

    Returns
    -------
    None

    .. _Chai-1: https://github.com/chaidiscovery/chai-lab/tree/main?tab=readme-ov-file

    """
    # TODO:
    #  INPUT/OUTPUT
    #  - check whether fasta_path is a file or a directory
    #      - if a directory, make a list of all FASTA files; if not, make a single-element list of the FASTA file
    #  - check whether output_path exists; if not, create it

    #  GLYCANS AND OTHER CONSTRAINTS
    #  - if constraints is provided, make sure it exists
    #  - if glycans is provided, parse it
    #  - if numbering_reference is provided, parse it
    #      - if numbering_reference is provided but glycans is not, raise an error
    #  - use glycans and numbering_reference to calculate glycan positions
    #  - if glycans and constraints are both provided, append the glycan constraints to the constraints file

    #  SEED AND DEVICE
    #  - verify that seed is an integer
    #  - check whether device has been provided
    #      - if not, set to:
    #          - "cuda" if CUDA is available and `fasta_path` is the path to a single FASTA file
    #          - use all CUDA devices (queue and thread pool) if CUDA is available and `fasta_path` is the path to a directory of FASTA files
    #          - "cpu" if CUDA is not available

    # output directory
    abutils.io.make_dir(output_path)

    # setup logging
    global logger
    if started_from_cli:
        abutils.log.setup_logging(
            logfile=os.path.join(output_path, "chai1.log"),
            add_stream_handler=verbose,
            single_line_handler=False,
            print_log_location=False,
            debug=debug,
        )
        logger = abutils.log.get_logger(
            name="chai1",
            add_stream_handler=verbose,
            single_line_handler=False,
        )
    elif verbose:
        logger = abutils.log.NotebookLogger(verbose=verbose, end="")
    else:
        logger = abutils.log.null_logger()

    # FASTA path(s)
    if isinstance(fasta_path, str):
        if os.path.isdir(fasta_path):
            fastas = [
                fasta
                for fasta in abutils.io.list_files(fasta_path)
                if abutils.io.determine_fastx_format(fasta) == "fasta"
            ]
        else:
            fastas = [fasta_path]
    else:
        fastas = natsorted(fasta_path)
    fastas = [os.path.abspath(fasta) for fasta in fastas if os.path.isfile(fasta)]
    if len(fastas) == 0:
        raise FileNotFoundError(f"No FASTA files found in {fasta_path}")
    fasta_names = [
        ".".join(os.path.basename(fasta).split(".")[:-1]) for fasta in fastas
    ]

    # log FASTA file info (only if started from CLI)
    if started_from_cli:
        log_fasta_file_info(fastas=fastas)

    # check constraints file
    if constraints is not None:
        check_file_exists_and_is_correct_format(constraints, "constraints" "csv")
        base_constraints_df = pd.read_csv(constraints)
    else:
        base_constraints_df = None

    # check glycans file
    if glycans is not None:
        check_file_exists_and_is_correct_format(glycans, "glycans" "json")
        with open(glycans, "r") as f:
            glycans_dict = json.load(f)
    else:
        glycans_dict = {}

    # check numbering_reference file
    if numbering_reference is not None:
        check_file_exists_and_is_correct_format(
            numbering_reference, "numbering reference", "json"
        )
        with open(numbering_reference, "r") as f:
            numbering_reference_dict = json.load(f)
    else:
        numbering_reference_dict = {}


def log_fasta_file_info(fastas: Iterable[str]) -> None:
    num_files = len(fastas)
    plural = "s" if num_files > 1 else ""
    logger.info("")
    logger.info("INPUT FILES")
    logger.info("===========")
    logger.info(f"found {num_files} input FASTA file{plural}:")
    if num_files < 6:
        for fasta in fastas:
            logger.info(f"  {os.path.basename(fasta)}")
    else:
        for fasta in fastas[:5]:
            logger.info(f"  {os.path.basename(fasta)}")
        logger.info(f"  ... and {num_files - 5} more")
    logger.info("")


def check_file_exists_and_is_correct_format(
    file: str, file_kind: str, filetype: str | None
) -> None:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"{file_kind} file not found: {file}")
    if filetype is not None:
        if magika.identify_path(Path(file)).output.label != filetype:
            raise ValueError(f"{file_kind} file must be a {filetype} file: {file}")
