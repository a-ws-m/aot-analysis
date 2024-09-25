"""Tools for analysing vesicles."""

from MDAnalysis.analysis.base import AtomGroup


class Vesicle:
    """Class for analysing vesicles."""

    def __init__(self, atoms: AtomGroup, headgroup: str = "PO4") -> None:
        self.atoms = atoms
        self.residues = atoms.residues
        self.headgroups = self.residues.select_atoms(f"name {headgroup}")

        self.com = self.atoms.center_of_mass()
        self.cog = self.atoms.center_of_geometry()

        # Partition into inner and outer layer
        self.inner, self.outer = self._partition()

    def _partition(self) -> "tuple[AtomGroup, AtomGroup]":
        """Split into an inner and outer layer of atoms."""
        molecular_cogs = self.residues.center_of_geometry(compound="residues")
        headgroup_locs = self.headgroups.positions

        # Find the vector from a molecule's headroup to its center of geometry
        mol_alignments = molecular_cogs - headgroup_locs

        # Find the vector from the vesicle's center of geometry to the molecule's center of geometry
        mol_displacement = molecular_cogs - self.cog

        # Calculate the dot product of the two vectors
        dot_products = (mol_alignments * mol_displacement).sum(axis=1)

        # Partition into inner and outer layer
        inner = self.residues[dot_products > 0]
        outer = self.residues[dot_products <= 0]
        return inner, outer
