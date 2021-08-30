import os

iteration_format = 'iter.%06d'

class Context(object):
    r"""The context of DP-GEN events.

    Parameters
    ----------
    local_path: str
                The path on local machine where the event happens
    remote_path: str
                The path on remote machine where the event happens
    """
    def __init__(self,
                 dpgen_path : str,
                 iteration : int = 0
    ) -> None:
        # for example /path/to/dpgen
        self.dpgen_path = os.path.abspath(dpgen_path)
        # for example /path/to/dpgen/iter.000002
        self.iter_path = os.path.join(self.dpgen_path, iteration_format%iteration)
        # for example /path/to/dpgen/iter.000001
        if iteration > 0 :
            self.prev_iter_path = os.path.join(self.dpgen_path, iteration_format%(iteration-1))
        else:
            self.prev_iter_path = None
        # for example /path/to/dpgen/iter.000003
        self.next_iter_path = os.path.join(self.dpgen_path, iteration_format%(iteration+1))
        # for example [/path/to/dpgen/iter.000000, /path/to/dpgen/iter.000001]
        self.all_prev_iter = [os.path.join(self.dpgen_path, (iteration_format % ii)) for ii in range(iteration)]
