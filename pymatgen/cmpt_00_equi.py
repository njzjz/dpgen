#!/usr/bin/env python3

import os, glob, argparse, json, warnings, re
import numpy as np
import lib.lammps as lammps
import lib.vasp as vasp

global_equi_name = '00.equi'

def comput_e_shift(poscar, task_name) :
    a_types = vasp.get_poscar_types(poscar)
    a_natoms = vasp.get_poscar_natoms(poscar)
    ener_shift = 0
    if len(a_types) > 1 :
        if not os.path.isdir('stables') :
            raise RuntimeError('no dir "stable". Stable energy and volume of components should be computed before calculating formation energy of an alloy')
        for ii in range(len(a_types)) :
            ref_e_file = a_types[ii] + '.' + task_name + '.e'
            ref_e_file = os.path.join('stables', ref_e_file)
            ener = float(open(ref_e_file).read())
            ener_shift += a_natoms[ii] * ener
    return ener_shift

def comput_lmp_nev(conf_dir, task_name) :
    conf_path = re.sub('confs', global_equi_name, conf_dir)
    conf_path = os.path.abspath(conf_path)
    poscar = os.path.join(conf_path, 'POSCAR')
    ener_shift = comput_e_shift(poscar, task_name)

    lmp_path = os.path.join(conf_path, task_name)
    log_lammps = os.path.join(lmp_path, 'log.lammps')
    if os.path.isfile(log_lammps):
        natoms, epa, vpa = lammps.get_nev(log_lammps)
        epa = (epa * natoms - ener_shift) / natoms
        return natoms, epa, vpa
    else :
        return None, None, None

def comput_vasp_nev(jdata, conf_dir) :
    kspacing = jdata['vasp_params']['kspacing']
    conf_path = re.sub('confs', global_equi_name, conf_dir)
    conf_path = os.path.abspath(conf_path)
    poscar = os.path.join(conf_path, 'POSCAR')
    ener_shift = comput_e_shift(poscar, 'vasp-k%.2f' % kspacing)

    vasp_path = os.path.join(conf_path, 'vasp-k%.2f' % kspacing)
    outcar = os.path.join(vasp_path, 'OUTCAR')
    # tag_fin = os.path.join(vasp_path, 'tag_finished')
    if not os.path.isfile(outcar) :
        warnings.warn("cannot find OUTCAR in "+vasp_path+" skip")
    elif not vasp.check_finished(outcar):
        warnings.warn("incomplete job "+vasp_path+" use the last frame")
    if os.path.isfile(outcar):
        natoms, epa, vpa = vasp.get_nev(outcar)
        epa = (epa * natoms - ener_shift) / natoms
        return natoms, epa, vpa
    else :
        return None, None, None

def _main():
    parser = argparse.ArgumentParser(
        description="cmpt 00.equi")
    parser.add_argument('PARAM', type=str,
                        help='the json param')
    parser.add_argument('CONF', type=str,
                        help='the dir of conf')
    args = parser.parse_args()
    with open (args.PARAM, 'r') as fp :
        jdata = json.load (fp)

    ln, le, lv = comput_lmp_nev(args.CONF, 'deepmd')
    mn, me, mv = comput_lmp_nev(args.CONF, 'meam')
    vn, ve, vv = comput_vasp_nev(jdata, args.CONF)
    if le == None or ve == None or lv == None or vv == None:
        print("%s" % args.CONF)
    else :
        print("%s\t %8.4f %8.4f %8.4f  %7.3f %7.3f %7.3f  %8.4f %7.3f" % (args.CONF, ve, le, (me), vv, lv, (mv), (le-ve), (lv-vv)))

if __name__ == '__main__' :
    _main()
    
