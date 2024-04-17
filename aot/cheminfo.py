"""Quickly accessible cheminformatics for Aerosol-OT."""
from rdkit import Chem
from rdkit.Chem import Crippen

AOT_SMILES = "CCCCC(CC)COC(=O)CC(C(=O)OCC(CC)CCCC)S(=O)(=O)[O-]"
AOT_NA_SMILES = "CCCCC(CC)COC(=O)CC(C(=O)OCC(CC)CCCC)S(=O)(=O)[O-].[Na+]"


AOT_MOL = Chem.MolFromSmiles(AOT_SMILES)
AOT_NA_MOL = Chem.MolFromSmiles(AOT_NA_SMILES)

def predict_pow() -> "tuple[float, float]":
    """Predict the log P of AOT using Crippen method."""
    return Crippen.MolLogP(AOT_MOL), Crippen.MolLogP(AOT_NA_MOL)

if __name__ == "__main__":
    print(predict_pow())
