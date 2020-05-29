import parmed
import os
import parmed.periodic_table as pt
import numpy as np
from tqdm import tqdm, trange
import dpdata
from collections import Counter
from multiprocessing import Pool

def GetSelectedAtomIndices(param, maskstr):
    sele = []
    if len(maskstr) > 0:
        newmaskstr = maskstr.replace("@0", "!@*")
        sele = [param.atoms[i].idx for i in parmed.amber.mask.AmberMask(
            param, newmaskstr).Selected()]
    return sele


class DataContainer:
    def __init__(self, parmfile, target, type_map, important_atoms=None, interactwith=None):
        self.parm = parmed.load_file(parmfile)
        self.parmfile = parmfile
        self.target = target
        self.interactwith = interactwith
        self.type_map = type_map
        if important_atoms:
            self.important_atoms=important_atoms
        else:
            self.important_atoms=[]

    def ReadFiles(self, nc, ll, hl):
        s_ll = dpdata.LabeledSystem(ll, fmt='amber/md', parm7_file=self.parmfile, nc_file=nc)
        s_hl = dpdata.LabeledSystem(hl, fmt='amber/md', parm7_file=self.parmfile, nc_file=nc)

        self.frcs = s_hl.data['forces'] - s_ll.data['forces']
        self.enes = s_hl.data['energies'] - s_ll.data['energies']
        self.crds = s_hl.data['coords']
        self.cells= s_hl.data['cells']

    def get_data(self, iframe):
        self.parm.initialize_topology(xyz=self.crds[iframe, :, :])
        idxt = GetSelectedAtomIndices(self.parm, self.target)
        namet = [pt.Element[self.parm.atoms[a].atomic_number] for a in idxt]
        idxw = None
        namew = None
        if self.interactwith is not None:
            idxw = GetSelectedAtomIndices(self.parm, self.interactwith)
            idxw = [a for a in idxw if a not in idxt]
            namew = [self.parm.atoms[a].type for a in idxw]

        idxs = idxt + idxw
        names = namet + namew


        crds = self.crds[iframe:iframe+1, idxs, :]
        frcs = self.frcs[iframe:iframe+1, idxs, :]
        enes = self.enes[iframe:iframe+1]
        
        type_map = dict(zip(self.type_map, range(len(self.type_map))))
        types = [type_map[n] for n in names]
        cou = Counter(types)

        numbs = [cou[n] for n in range(len(type_map))]

        shape1, shape2, _ = crds.shape
        atom_pref = np.ones((shape1, shape2))
        atom_pref[:, self.important_atoms] = 5

        s = dpdata.LabeledSystem.from_dict({'data': {'atom_names': self.type_map, 'atom_types': types, 'coords': crds, 'forces': frcs,
                                                    'energies': enes, 'cells': self.cells[iframe:iframe+1], 'atom_numbs': numbs, 'orig': [0, 0, 0], 'nopbc': True, "atom_pref": atom_pref}})
        return s



def get_amber_fp(cutoff, parmfile, nc, ll, hl, type_map, important_atoms=None):
    target = ":1"
    # interactwith is set to all residues within 5 A
    # of the target, excluding the M-site particles
    # and the target, itself
    interactwith = "(%s)<:%f &! (%s|(%s))" % (target, cutoff, r"@%EP", target)
    data = DataContainer(parmfile, ":1", 
        interactwith=interactwith,
        type_map=type_map,
        important_atoms=important_atoms
    )
    ms = dpdata.MultiSystems()
    try:
        data.ReadFiles(nc, ll, hl)
    except Exception as e:
        print("skip", nc, ll, hl)
        print(e)
        return ms
    nframes = data.enes.shape[0]
    with Pool() as p:
        for s in p.imap_unordered(data.get_data, trange(nframes)):
            ms.append(s)
    return ms
    #ms.to_deepmd_npy(args.output, set_size=100000)
