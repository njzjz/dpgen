from typing import Optional

import ase
import ase.io
import dpdata
import numpy as np
from dpdata.periodic_table import ELEMENTS, Element


def compute_stats_from_redu(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the output statistics.

    Given the reduced output value and the number of atoms for each atom,
    compute the least-squares solution as the atomic output bais and std.

    Parameters
    ----------
    output_redu
        The reduced output value, shape is [nframes, *(odim0, odim1, ...)].
    natoms
        The number of atoms for each atom, shape is [nframes, ntypes].
    assigned_bias
        The assigned output bias, shape is [ntypes, *(odim0, odim1, ...)].
        Set to a tensor of shape (odim0, odim1, ...) filled with nan if the bias
        of the type is not assigned.
    rcond
        Cut-off ratio for small singular values of a.

    Returns
    -------
    np.ndarray
        The computed output bias, shape is [ntypes, *(odim0, odim1, ...)].
    np.ndarray
        The computed output std, shape is [*(odim0, odim1, ...)].
    """
    natoms = np.array(natoms)
    nf, _ = natoms.shape
    output_redu = np.array(output_redu)
    var_shape = list(output_redu.shape[1:])
    output_redu = output_redu.reshape(nf, -1)
    # check shape
    assert output_redu.ndim == 2
    assert natoms.ndim == 2
    assert output_redu.shape[0] == natoms.shape[0]  # nframes
    if assigned_bias is not None:
        assigned_bias = np.array(assigned_bias).reshape(
            natoms.shape[1], output_redu.shape[1]
        )
    # compute output bias
    if assigned_bias is not None:
        # Atomic energies stats are incorrect if atomic energies are assigned.
        # In this situation, we directly use these assigned energies instead of computing stats.
        # This will make the loss decrease quickly
        assigned_bias_atom_mask = ~np.isnan(assigned_bias).any(axis=1)
        # assigned_bias_masked: nmask, ndim
        assigned_bias_masked = assigned_bias[assigned_bias_atom_mask]
        # assigned_bias_natoms: nframes, nmask
        assigned_bias_natoms = natoms[:, assigned_bias_atom_mask]
        # output_redu: nframes, ndim
        output_redu -= np.einsum(
            "ij,jk->ik", assigned_bias_natoms, assigned_bias_masked
        )
        # remove assigned atom
        natoms[:, assigned_bias_atom_mask] = 0

    # computed_output_bias: ntypes, ndim
    computed_output_bias, _, _, _ = np.linalg.lstsq(natoms, output_redu, rcond=rcond)
    if assigned_bias is not None:
        # add back assigned atom; this might not be required
        computed_output_bias[assigned_bias_atom_mask] = assigned_bias_masked
    # rest_redu: nframes, ndim
    rest_redu = output_redu - np.einsum("ij,jk->ik", natoms, computed_output_bias)
    output_std = rest_redu.std(axis=0)
    computed_output_bias = computed_output_bias.reshape([natoms.shape[1]] + var_shape)  # noqa: RUF005
    output_std = output_std.reshape(var_shape)
    return computed_output_bias, output_std


def _convert_atom_names(name: str) -> str:
    if name == "OW":
        return ELEMENTS[48]
    elif name == "HW":
        return ELEMENTS[49]
    elif name.startswith("m"):
        return ELEMENTS[Element(name[1:]).Z + 49]
    else:
        return name


def convert_atom_names_to_Z(name: str) -> int:
    return Element(_convert_atom_names(name)).Z


def convert_training_data_to_mace(
    type_map: list[str], systems: list[str], mace_xyz_file: str
) -> np.ndarray:
    """Convert training data to mace format.

    Parameters
    ----------
    type_map : list of str
        type map
    systems : list of str
        system
    mace_xyz_file : str
        mace xyz file name

    Returns
    -------
    np.ndarray
        computed output bias
    """
    append = False
    ener = []
    natoms = []
    for ii in systems:
        ss = dpdata.LabeledSystem(
            ii, fmt="deepmd/hdf5" if "#" in ii else "deepmd/npy", type_map=type_map
        )
        ss.data["atom_names"] = [_convert_atom_names(a) for a in ss.get_atom_names()]
        atoms = ss.to_ase_structure()

        ase.io.write(mace_xyz_file, atoms, append=append, format="extxyz")
        append = True

        ener.append(ss["energies"])
        for _ in range(ss.get_nframes()):
            natoms.append(ss["atom_numbs"])

    ener = np.concatenate(ener)
    natoms = np.array(natoms, dtype=int)
    assigned_bias = [
        0.0 if tt.startswith("m") or tt in {"OW", "HW"} else float("nan")
        for tt in type_map
    ]
    computed_output_bias, _ = compute_stats_from_redu(ener, natoms, assigned_bias)
    return computed_output_bias
