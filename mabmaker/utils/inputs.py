# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import json
from functools import cached_property
from pathlib import Path
from typing import Iterable

import magika
import yaml
from natsort import natsorted


class StructurePredictionInput:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        seed: int | Iterable[int] = 42,
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
        msa_directory: str | None = None,
        num_recycles: int = 3,
        num_diffusion_timesteps: int = 200,
        num_diffusion_samples: int = 5,
        device: str | None = None,
        low_memory: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to a JSON or YAML file containing the structure prediction parameters.
            The data for each run should be in the format of the `AlphaFold3 input JSON file`_::

            ``` json
            {
                "name": "Job name goes here",
                "modelSeeds": [1, 2],  # At least one seed required.
                "sequences": [
                    {"protein": {...}},
                    {"rna": {...}},
                    {"dna": {...}},
                    {"ligand": {...}}
                ],
                "bondedAtomPairs": [...],  # Optional
                "userCCD": "...",  # Optional
                "options": {"opt": "...", "opt2": "..."}  # Optional
            }
            ```

            For inputs containing multiple runs, the JSON file should contain a list of
            dictionaries, each with the same structure. Note that the `name` field is
            used to name the output directory, so it should be unique for each run.

            The `options` field is used to pass additional options to the structure
            prediction tool that should be separately applied to each run, such as
            the number of recycles or diffusion timesteps.

            To use a YAML file, the file should contain a list of dictionaries, the input
            data should follow the same general structure::

            ``` yaml
            - name: "Job name goes here"
              modelSeeds:  # At least one seed required.
                - 1
                - 2
              sequences:
                - protein:
                    ...
                - rna:
                    ...
                - dna:
                    ...
                - ligand:
                    ...
              bondedAtomPairs:  # Optional
                - ...
                - ...
              userCCD:  # Optional
                ...
              options:  # Optional
                opt: ...
                opt2: ...
            ```

        output_path : str
            Path to the output directory.

        seed : int | Iterable[int], optional
            Random seed.

        use_msa_server : bool, optional
            Whether to use the MSA server.

        msa_server_url : str, optional
            URL of the MSA server.

        msa_directory : str | None, optional
            Path to the directory containing the MSA files.

        num_recycles : int, optional
            Number of recycles.

        num_diffusion_timesteps : int, optional
            Number of diffusion timesteps.

        num_diffusion_samples : int, optional
            Number of diffusion samples.

        device : str | None, optional
            Device to use.

        low_memory : bool, optional
            Whether to use low memory mode.

        verbose : bool, optional
            Whether to print verbose output.

        debug : bool, optional
            Whether to print debug output.


        .. _AlphaFold3 input JSON file: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

        """
        self.input_path = input_path
        self.output_path = output_path
        self.seed = seed
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url
        self.msa_directory = msa_directory
        self.num_recycles = num_recycles
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.num_diffusion_samples = num_diffusion_samples
        self.device = device
        self.low_memory = low_memory
        self.verbose = verbose
        self.debug = debug

        # input parsing
        if magika.identify_path(Path(input_path)).output.label == "yaml":
            with open(input_path, "r") as f:
                self.input_data = yaml.load_all(f)
        elif magika.identify_path(Path(input_path)).output.label == "json":
            with open(input_path, "r") as f:
                self.input_data = json.load(f)
        else:
            raise ValueError(f"Input file must be a YAML or JSON file: {input_path}")

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.jobs[index]
        elif isinstance(index, slice):
            return self.jobs[index]
        elif isinstance(index, str) and index in self.job_names():
            return next(job for job in self.jobs if job["name"] == index)
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    @cached_property
    def jobs(self):
        return natsorted(self.input_data, key=lambda x: x["name"])

    def job_names(self):
        return [job["name"] for job in self.jobs]


# The actual input should probably mimic that of AlphaFold3,
# which can be found here: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

# Briefly, it's a JSON file with the following structure:

# {
#   "name": "Job name goes here",
#   "modelSeeds": [1, 2],  # At least one seed required.
#   "sequences": [
#     {"protein": {...}},
#     {"rna": {...}},
#     {"dna": {...}},
#     {"ligand": {...}}
#   ],
#   "bondedAtomPairs": [...],  # Optional
#   "userCCD": "...",  # Optional
#   "dialect": "alphafold3",  # Required
#   "version": 2  # Required
# }



class StructurePredictionJob:
    def __init__(self, job_data: dict):
        self.job_data = job_data
