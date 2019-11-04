#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from collections import defaultdict

import dpdata


def stat_iter(target_folder, param_file="param.json", verbose=True, mute=False):
    jdata = {}
    with open(f"{target_folder}/{param_file}") as param_file:
        jdata = json.load(param_file)
    iter_dict = defaultdict(lambda: defaultdict(int))
    output = subprocess.run(
        [f"wc -l {target_folder}/iter.??????/02.fp/*out"],
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout
    data = output.decode()  # split(b'\n')
    for line in data.split("\n"):
        if "out" in line:
            num, relative_path_doc = line.strip().split(" ")
            path_doc = os.path.abspath(relative_path_doc)
            num = int(num)
            prefix, iter_dirname, stage, out_filename = path_doc.rsplit("/", 3)
            pk_id, out_filename = path_doc.rsplit("/", 1)
            iter = int(iter_dirname.split(".")[-1])
            out_id = int(out_filename.strip().split(".")[-2])
            out_type = out_filename.strip().split(".")[0]
            iter_dict[pk_id][out_type] += num
    # for ii in
    output2 = subprocess.run(
        [f"ls -d -1 {target_folder}/iter.??????/02.fp/task.*/OUTCAR"],
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout
    data2 = output2.decode()
    if verbose:
        # print('find OUTCAR', data2)
        print("use param_jsonfile jdata['type_map']", jdata["type_map"])
    for line in data2.split("\n"):
        if line:
            # [/home/felix/workplace/SiC/iter.000002/02.fp/task.018.000040/OUTCAR]
            path_doc = os.path.abspath(line)
            pk_id, task_dirname, OUTCAR_filename = path_doc.rsplit("/", 2)
            try:
                _sys = dpdata.LabeledSystem(path_doc, type_map=jdata["type_map"])
            except:
                try:
                    _sys = dpdata.LabeledSystem(
                        path_doc.replace("OUTCAR", "vasprun.xml"),
                        type_map=jdata["type_map"],
                    )
                except:
                    _sys = dpdata.LabeledSystem()
            if len(_sys) == 1:
                pass
            else:
                if verbose:
                    print(
                        "OUTCAR not label by dpdata, not convergence or unfinshed",
                        path_doc,
                    )
                iter_dict[pk_id]["OUTCAR_not_convergence"] += 1
            iter_dict[pk_id]["OUTCAR_total_count"] += 1
    for pk_id in {**iter_dict}:
        if iter_dict[pk_id]["OUTCAR_total_count"]:
            iter_dict[pk_id]["reff"] = round(
                iter_dict[pk_id]["OUTCAR_not_convergence"]
                / iter_dict[pk_id]["OUTCAR_total_count"],
                5,
            )
    for pk_id, value in iter_dict.items():
        print(
            f"{pk_id}:candidate:{value['candidate']}"
            f":rest_failed:{value['rest_failed']}"
            f":rest_accurate:{value['rest_accurate']}"
            f":OUTCAR_total_count:{value['OUTCAR_total_count']}"
            f":OUTCAR_not_convergence:{value['OUTCAR_not_convergence']}"
            f":reff:{value['reff']}"
        )


def stat_time(target_folder, param_file="param.json", verbose=True, mute=False):
    script = os.path.join(os.path.dirname(__file__), "update_time.sh")
    output = subprocess.run(
        [f"bash {script} {target_folder}"], shell=True, stdout=subprocess.PIPE
    ).stdout
    data = output.decode()
    print(data)
