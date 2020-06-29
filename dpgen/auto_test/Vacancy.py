import glob
import json
import os

from monty.serialization import loadfn, dumpfn
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.core.structure import Structure

from dpgen import dlog
from dpgen.auto_test.Property import Property
from dpgen.auto_test.refine import make_refine
from dpgen.auto_test.reproduce import make_repro
from dpgen.auto_test.reproduce import post_repro


class Vacancy(Property):
    def __init__(self,
                 parameter):
        parameter['reprod-opt'] = parameter.get('reprod-opt', False)
        self.reprod = parameter['reprod-opt']
        if not self.reprod:
            default_supercell = [1, 1, 1]
            self.supercell = parameter.get('supercell', default_supercell)
            parameter['cal_type'] = parameter.get('cal_type', 'relaxation')
            self.cal_type = parameter['cal_type']
            default_cal_setting = {"relax_pos": True,
                                   "relax_shape": True,
                                   "relax_vol": True}
            if 'cal_setting' not in parameter:
                parameter['cal_setting'] = default_cal_setting
            elif "relax_pos" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_pos'] = default_cal_setting['relax_pos']
            elif "relax_shape" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_shape'] = default_cal_setting['relax_shape']
            elif "relax_vol" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_vol'] = default_cal_setting['relax_vol']
            self.cal_setting = parameter['cal_setting']
        else:
            parameter['cal_type'] = 'static'
            self.cal_type = parameter['cal_type']
            default_cal_setting = {"relax_pos": False,
                                   "relax_shape": False,
                                   "relax_vol": False}
            if 'cal_setting' not in parameter:
                parameter['cal_setting'] = default_cal_setting
            elif "relax_pos" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_pos'] = default_cal_setting['relax_pos']
            elif "relax_shape" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_shape'] = default_cal_setting['relax_shape']
            elif "relax_vol" not in parameter['cal_setting']:
                parameter['cal_setting']['relax_vol'] = default_cal_setting['relax_vol']
            self.cal_setting = parameter['cal_setting']
            parameter['init_from_suffix'] = parameter.get('init_from_suffix', '00')
            self.init_from_suffix = parameter['init_from_suffix']
        self.parameter = parameter

    def make_confs(self,
                   path_to_work,
                   path_to_equi,
                   refine=False):
        path_to_work = os.path.abspath(path_to_work)
        if os.path.exists(path_to_work):
            dlog.warning('%s already exists' % path_to_work)
        else:
            os.makedirs(path_to_work)
        path_to_equi = os.path.abspath(path_to_equi)

        if 'start_confs_path' in self.parameter and os.path.exists(self.parameter['start_confs_path']):
            init_path_list = glob.glob(os.path.join(self.parameter['start_confs_path'], '*'))
            struct_init_name_list = []
            for ii in init_path_list:
                struct_init_name_list.append(ii.split('/')[-1])
            struct_output_name = path_to_work.split('/')[-2]
            assert struct_output_name in struct_init_name_list
            path_to_equi = os.path.abspath(os.path.join(self.parameter['start_confs_path'],
                                                        struct_output_name, 'relaxation'))

        task_list = []
        cwd = os.getcwd()

        if self.reprod:
            print('vacancy reproduce starts')
            if 'vasp_lmp_path' not in self.parameter:
                raise RuntimeError("please provide the vasp_lmp_path for reproduction")
            vasp_lmp_path = os.path.abspath(self.parameter['vasp_lmp_path'])
            task_list = make_repro(vasp_lmp_path, self.init_from_suffix, path_to_work)
            os.chdir(cwd)

        else:
            equi_contcar = os.path.join(path_to_equi, 'CONTCAR')
            if not os.path.exists(equi_contcar):
                raise RuntimeError("please do relaxation first")

            ss = Structure.from_file(equi_contcar)
            vds = VacancyGenerator(ss)
            dss = []
            for jj in vds:
                dss.append(jj.generate_defect_structure(self.supercell))

            if refine:
                print('vacancy refine starts')
                task_list = make_refine(self.parameter['init_from_suffix'],
                                        self.parameter['output_suffix'],
                                        path_to_work)
                for ii in task_list:
                    os.chdir(ii)
                    # np.savetxt('supercell.out', self.supercell, fmt='%d')
                    dumpfn(self.supercell, 'supercell.json')
                os.chdir(cwd)

            else:
                print('gen vacancy with supercell ' + str(self.supercell))
                os.chdir(path_to_work)
                if os.path.isfile('POSCAR'):
                    os.remove('POSCAR')
                os.symlink(os.path.relpath(equi_contcar), 'POSCAR')
                #           task_poscar = os.path.join(output, 'POSCAR')

                for ii in range(len(dss)):
                    output_task = os.path.join(path_to_work, 'task.%06d' % ii)
                    os.makedirs(output_task, exist_ok=True)
                    os.chdir(output_task)
                    for jj in ['INCAR', 'POTCAR', 'POSCAR', 'conf.lmp', 'in.lammps']:
                        if os.path.exists(jj):
                            os.remove(jj)
                    task_list.append(output_task)
                    dss[ii].to('POSCAR', 'POSCAR')
                    # np.savetxt('supercell.out', self.supercell, fmt='%d')
                    dumpfn(self.supercell, 'supercell.json')
                os.chdir(cwd)
        return task_list

    def post_process(self, task_list):
        pass

    def task_type(self):
        return self.parameter['type']

    def task_param(self):
        return self.parameter

    def _compute_lower(self,
                       output_file,
                       all_tasks,
                       all_res):
        output_file = os.path.abspath(output_file)
        res_data = {}
        ptr_data = os.path.dirname(output_file) + '\n'

        if not self.reprod:
            ptr_data += "Structure: \tVac_E(eV)  E(eV) equi_E(eV)\n"
            idid = -1
            for ii in all_tasks:
                idid += 1
                structure_dir = os.path.basename(ii)
                task_result = loadfn(all_res[idid])
                natoms = task_result['atom_numbs'][0]
                equi_path = os.path.abspath(os.path.join(os.path.dirname(output_file), '../relaxation'))
                equi_result = loadfn(os.path.join(equi_path, 'result.json'))
                equi_epa = equi_result['energies'][-1] / equi_result['atom_numbs'][0]
                evac = task_result['energies'][-1] - equi_epa * natoms

                supercell_index = loadfn(os.path.join(ii, 'supercell.json'))
                ptr_data += "%s: %7.3f  %7.3f %7.3f \n" % (str(supercell_index) + '-' + structure_dir,
                                                           evac, task_result['energies'][-1], equi_epa * natoms)
                res_data[str(supercell_index) + '-' + structure_dir] = [evac, task_result['energies'][-1],
                                                                        equi_epa * natoms]

        else:
            if 'vasp_lmp_path' not in self.parameter:
                raise RuntimeError("please provide the vasp_lmp_path for reproduction")
            vasp_lmp_path = os.path.abspath(self.parameter['vasp_lmp_path'])
            res_data, ptr_data = post_repro(vasp_lmp_path, self.parameter['init_from_suffix'], all_tasks, ptr_data)

        with open(output_file, 'w') as fp:
            json.dump(res_data, fp, indent=4)

        return res_data, ptr_data
