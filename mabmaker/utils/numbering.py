# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import abutils


class NumberingReference:
    def __init__(self, query_sequence: str, reference_sequence: str):
        self.query_sequence = query_sequence
        self.reference_sequence = reference_sequence
        self.aln = abutils.alignment.global_alignment(
            query_sequence,
            reference_sequence,
            gap_open_penalty=10,
            gap_extend_penalty=0.5,
        )
        self._cache = {}

    def __getitem__(self, reference_position: int) -> int:
        return self.get(reference_position)

    def __setitem__(self, reference_position: int, query_position: int) -> None:
        self._cache[reference_position] = query_position

    def get(self, reference_position: int) -> int:
        """
        Get the query position for a reference position.
        """
        if reference_position in self._cache:
            return self._cache[reference_position]
        query_pos = 0
        ref_pos = 0
        for q, r in zip(self.aln.aligned_query, self.aln.aligned_target):
            if q == "-":
                ref_pos += 1
            elif r == "-":
                query_pos += 1
            else:
                query_pos += 1
                ref_pos += 1
            if ref_pos == reference_position:
                self._cache[reference_position] = query_pos
                return query_pos


def build_numbering_reference(
    query_sequence: str,
    reference_sequence: str,
) -> NumberingReference:
    """
    Build a numbering reference for a query sequence.

    The numbering reference is a mapping from the reference sequence to the query,
    and can be accessed like a dictionary::

    ``` python
    numbering_reference = build_numbering_reference(query_sequence, reference_sequence)
    numbering_reference[10]  # get the query position for reference position 10
    numbering_reference[10] = 20  # set the query position for reference position 10 to 20
    ```

    Parameters
    ----------
    query_sequence : str
        The query sequence.

    reference_sequence : str
        The reference sequence.

    Returns
    -------
    NumberingReference
        A numbering reference for the query sequence.

    """
    return NumberingReference(query_sequence, reference_sequence)
