import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle as pkl


def smiles_to_AtonBondDescriptor_PCAembedings(cpd_smiles):
    # 1. clean smiles
    cpd_smiles = cpd_smiles[~cpd_smiles['smiles'].isnull()]
    cpd_smiles = cpd_smiles.drop_duplicates(subset='smiles', keep='first')
    
    cpd_smiles = (
        cpd_smiles[(
            ~cpd_smiles['smiles'].isin(
                ['[Li].Cl', ## Cannot properly define graph
                'C[N+]1(C)C2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c3ccccc3.[O-][N+](=O)[O-]',  #'C[N+]1(C)C2CC[C@H]1C[C@@H](OC(=O)C(CO)C1=CC=CC=C1)C2.O=[N+]([O-])[O-]', ## dfs() function breaks
                'C=CCNC1=C2C[C@@H](C)C[C@H](OC)[C@H](O)[C@@H](C)/C=C(\C)[C@H](OC(N)=O)[C@@H](OC)/C=C\C=C(/C)C(=O)NC(=CC1=O)C2=O.NC1=NC(=O)N([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)C=C1', ## dfs() function breaks
                'C=CCNC1=C2C[C@@H](C)C[C@H](OC)[C@H](O)[C@@H](C)/C=C(\C)[C@H](OC(N)=O)[C@@H](OC)/C=C\C=C(/C)C(=O)NC(=CC1=O)C2=O.CC(C)C[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C1=CN=CC=N1)B(O)O', ## dfs() function breaks
                'N.N.[Cl-].[Cl-].[Pt+2]' ## TODO: hoping to include cisplatin, see below.
                ]))]
    )

    # 2. get atom and bond feature PCA embeddings
    res = mol2local(cpd_smiles['smiles'].values.tolist(), onehot=True, pca = True, ids = cpd_smiles.index.values)
    
    f_atoms_pca = pd.DataFrame(res.f_atoms_pca)
    f_bonds_pca = pd.DataFrame(res.f_bonds_pca)
    
    # 3. get RDKit Descriptors
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    
    rdkit_descriptors = cpd_smiles['smiles'].apply(smi_to_descriptors)
    rdkit_descriptors_df = pd.DataFrame(rdkit_descriptors.to_list(), columns=descriptor_names)
    
    cpd_smiles_reset = cpd_smiles.reset_index(drop=True)
    
    # 4. get PCA embeddings from atom, bond, and RDKit Descriptors
    ## embedings:
    
    cpd_features_PCA_embedding, _, _, _ = getDataEmbedding(
        f_atoms_pca,
        f_bonds_pca,
        rdkit_descriptors_df
    )

    cpd_features_PCA_embedding = cpd_features_PCA_embedding.set_index(cpd_smiles.index.values)
    
    return cpd_features_PCA_embedding



def getDataEmbedding(
      atom_features: pd.DataFrame, 
      bond_features: pd.DataFrame, 
      smiles_features: pd.DataFrame,
      n_pcs: int = 50,
      rdkit_scaler: StandardScaler = None,
      var_thresholder: VarianceThreshold = None, 
      ):

      sheader = []
      smiles_features = smiles_features.fillna(0)

      sheader = list(smiles_features.columns.values)

      rdkit_scaler = StandardScaler()
      scaled = rdkit_scaler.fit_transform(smiles_features)
      
      scaled_df = pd.DataFrame(scaled, columns = sheader)

      atom_features = atom_features.add_prefix('A_')
      bond_features = bond_features.add_prefix('B_')

      for i in range (0,bond_features.shape[1]):
        i = str(i)
        column = 'B_'+i
        b1 = bond_features[column]
        
        name = column
        i = int(i)
        atom_features.insert(i,name,b1)

      pca_res = PCA(n_components = n_pcs) 
      data_specific_pca_embed = pca_res.fit_transform(atom_features)
      
      pcaNames = []
      for p in range(0, n_pcs):
        pc = str(p)
        pca = 'PCA'+pc
        pcaNames.append(pca)


      data_specific_pca_embed = pd.DataFrame(data=data_specific_pca_embed, columns=pcaNames)

      j = 0
      for col in pcaNames:
        col_data = data_specific_pca_embed[col]
        scaled_df.insert(j,col,col_data)
        
        j = j+1

      if var_thresholder is None:
          var_thresholder = VarianceThreshold(0)
          cleaned = var_thresholder.fit_transform(scaled_df)
      else:
          cleaned = var_thresholder.transform(scaled_df)

      return pd.DataFrame(cleaned), pca_res, rdkit_scaler, var_thresholder






def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices))
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

ATOM_FEATURES = {
    'atomic_num': list(range(118)), # type of atom (ex. C,N,O), by atomic number, size = 118
    'degree': [0, 1, 2, 3, 4, 5], # number of bonds the atom is involved in, size = 6
    'formal_charge': [-1, -2, 1, 2, 0], # integer electronic charge assigned to atom, size = 5
    'chiral_tag': [0, 1, 2, 3], # chirality: unspecified, tetrahedral CW/CCW, or other, size = 4
    'num_Hs': [0, 1, 2, 3, 4], # number of bonded hydrogen atoms, size = 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], # size = 5
}

def atom_features_raw(atom):
    features = [atom.GetAtomicNum()] + \
               [atom.GetTotalDegree()] + \
               [atom.GetFormalCharge()] + \
               [int(atom.GetChiralTag())] + \
               [int(atom.GetTotalNumHs())] + \
               [int(atom.GetHybridization())] + \
               [atom.GetIsAromatic()] + \
               [atom.GetMass()]
    return features

def atom_features_onehot(atom): # size: 151
    features = one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features

def bond_features_raw(bond):
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: btt = 0
    elif bt == Chem.rdchem.BondType.DOUBLE: btt = 1
    elif bt == Chem.rdchem.BondType.TRIPLE: btt = 2
    elif bt == Chem.rdchem.BondType.AROMATIC: btt = 3
    fbond = [
        btt,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0),
        int(bond.GetStereo())]
    return fbond

def bond_features_onehot(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0),
    ]
    fbond += one_hot_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond

class LocalFeatures:
    def __init__(self, mol, onehot = False, pca = False, ids = None):
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.mol = mol
        self.onehot = onehot
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.f_atoms_pca = []
        self.f_bonds_pca = []
        self.mol_id_atoms = []
        self.mol_id_bonds = []

        if onehot:
            self.f_atoms = [atom_features_onehot(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_onehot(bond) for bond in mol.GetBonds()]
        else:
            self.f_atoms = [atom_features_raw(atom) for atom in mol.GetAtoms()]
            self.f_bonds = [bond_features_raw(bond) for bond in mol.GetBonds()]

        self.n_atoms = len(self.f_atoms)
        self.n_bonds = len(self.f_bonds)
        self.f_atoms_dim = np.shape(self.f_atoms)[1]
        self.f_bonds_dim = np.shape(self.f_bonds)[1]

        if pca:
            fa = np.array(self.f_atoms).T
            fb = np.array(self.f_bonds).T
            pca = PCA(n_components=1)
            pc_atoms = pca.fit_transform(fa)
            pc_bonds = pca.fit_transform(fb)

            self.f_atoms_pca = pc_atoms.T
            self.f_bonds_pca = pc_bonds.T

        if ids is not None:
            self.mol_id_atoms = [ids for i in range(self.n_atoms)]
            self.mol_id_bonds = [ids for i in range(self.n_bonds)]
            self.mol_ids = ids

class BatchLocalFeatures:
    def __init__(self, mol_graphs):
        self.mol_graphs = mol_graphs
        self.n_atoms = 0
        self.n_bonds = 0
        self.a_scope = []
        self.b_scope = []
        f_atoms, f_bonds = [], []
        f_atoms_pca, f_bonds_pca= [], []
        f_atoms_id, f_bonds_id= [], []
        mol_ids = []

        for mol_graph in self.mol_graphs: # for each molecule graph
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_atoms_pca.extend(mol_graph.f_atoms_pca)
            f_bonds_pca.extend(mol_graph.f_bonds_pca)

            f_atoms_id.extend(mol_graph.mol_id_atoms)
            f_bonds_id.extend(mol_graph.mol_id_bonds)
            mol_ids.append(mol_graph.mol_ids)
            

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.f_atoms_pca = f_atoms_pca
        self.f_bonds_pca = f_bonds_pca
        self.f_atoms_id = f_atoms_id
        self.f_bonds_id = f_bonds_id
        self.mol_ids = mol_ids

def mol2local(mols, onehot = True, pca = True, ids = None):
    if ids is not None:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca, iid) for mol,iid in zip(mols,ids)])
    else:
        return BatchLocalFeatures([LocalFeatures(mol, onehot, pca, ids) for mol in mols])

def smi_to_descriptors(smile):
    mol = Chem.MolFromSmiles(smile)
    descriptors = []
    if mol:
        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        descriptors = np.array(get_descriptors.ComputeProperties(mol))
    return descriptors