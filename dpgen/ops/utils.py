import os,re
from pathlib import Path
from typing import Iterable

def create_path (path) :
    """Create path. If path exists, it will moved to path.bk%03d.    
    """
    if path.is_dir() : 
        dirname = path.name
        counter = 0
        while True :
            bk_dirname = Path(dirname + ".bk%03d" % counter)
            if not bk_dirname.is_dir():
                path.replace(path.parent / bk_dirname)
                break
            counter += 1
        (path.parent / dirname).mkdir(parents=True)
    path.mkdir(parents=True)


def os_path_split(path):
    """split a path, for example: 'a/b/c' -> ['a', 'b', 'c']    
    """
    parts = []
    while True:
        newpath, tail = os.path.split(path)
        if newpath == path:
            assert not tail
            if path: parts.append(path)
            break
        parts.append(tail)
        path = newpath
    parts.reverse()
    return parts


def link_dp_data(
        source_path : Path,
        target_path : Path,
        link_abspath : bool = False, 
        data_path_pattern : str = '.*'
)->None:
    """Link deepmd-kit training data. It creates the same tree to the
    training data (directories has 'type.raw') in the target_path as
    the source_path, then symbol links the training data. 
    
    For example we have source_path == 'source', target_path == 'target'
        source/
        ├── data0
        │   ├── subdata0
        │   └── subdata1
        └── data1
    Then the output is
        target/
        ├── data0
        │   ├── subdata0 -> ../../source/data0/subdata0
        │   └── subdata1 -> ../../source/data0/subdata1
        └── data1 -> ../source/data1


    Parameters
    ----------
    source_path
                source path containing training data
    target_path
                target path
    link_abspath
                linking to absolute path or relative path
    data_path_pattern
                the data dir should match this regular expression pattern 
    """
    if source_path is None:
        return
    source_path = Path(source_path)
    target_path = Path(target_path)
    all_data_sys = source_path.rglob('type.raw')
    pattern = re.compile(data_path_pattern)
    for ii in all_data_sys:
        source_data_dir = ii.parent
        if not pattern.search(str(source_data_dir)):
            continue
        rel_data_dir = os.path.relpath(source_data_dir, source_path)
        target_data_dir = target_path / rel_data_dir
        target_data_dir.parent.mkdir(parents=True, exist_ok=True)
        if link_abspath : 
            link_source = source_data_dir.resolve()
        else:
            link_source = os.path.relpath(source_data_dir, target_data_dir.parent)
        target_data_dir.symlink_to(link_source)



def link_dirs(
        source_path : Iterable[Path],
        target_path : Path,
        link_abspath : bool = False,
        data_path_pattern : str = '.*'
)->None:
    """Link all Path object in `source_path` to `target_path`

    For example we set `source_path` to ['source/data0/subdata0', 'source/data0/subdata1', 'source/data1'], then all dirs in the `source_path` will be linked to `target_path` like    
        target/
        └── source
            ├── data0
            │   ├── subdata0 -> ../../../source/data0/subdata0
            │   └── subdata1 -> ../../../source/data0/subdata1
            └── data1 -> ../../source/data1


    Parameters
    ----------
    source_path
                collection of the source paths
    target_path
                target path
    link_abspath
                linking to absolute path or relative path
    data_path_pattern
                the data dir should match this regular expression pattern 
    """
    if source_path is None:
        return
    target_path = Path(target_path)
    all_data_sys = [Path(ii) for ii in source_path]
    pattern = re.compile(data_path_pattern)
    for source_data_dir in all_data_sys:
        if not pattern.search(str(source_data_dir)):
            continue
        target_data_dir = target_path / source_data_dir
        target_data_dir.parent.mkdir(parents=True, exist_ok=True)
        if link_abspath : 
            link_source = source_data_dir.resolve()
        else:
            link_source = os.path.relpath(source_data_dir, target_data_dir.parent)
        target_data_dir.symlink_to(link_source)
