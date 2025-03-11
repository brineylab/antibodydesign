# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


def build_glycan(
    glycan_type: str,
    model: str,
) -> str:
    """
    Build a glycan from a glycan type and a model.

    Parameters
    ----------
    glycan_type : str
        The type of glycan to build.
    model : str
        The model to use to build the glycan.

    Returns
    -------
    str
        The glycan string, formatted for the model.
    """
    pass



def _build_glycan_for_chai(
    glycan_type: str,
) -> str:
    """
    Build a glycan for Chai.

    Parameters
    ----------
    glycan_type : str
        The type of glycan to build. Should be one of the following:
          - "man3"
          - "man6"
          - "man8"
          - "man9"
          - a CCD string, like "NAG(4-1 NAG(6-1 MAN(6-1(MAN(6-1 MAN))))))"

    Returns
    -------
    str
        The glycan string, formatted for Chai-1.
    """
    if 



def _build_glycan_for_protenix(
    glycan_type: str,
) -> str:
    """
    Build a glycan for Protenix.

    https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md#ligand

    Protenix can take multi-CCD strings, but all need to be prefixed with "CCD_", so man3 would be: "CCD_NAG_NAG_MAN_MAN_MAN"
    """
    pass
