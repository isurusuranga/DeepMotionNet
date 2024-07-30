import argparse


class BaseOptions(object):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # the first two arguments are mandatory i.e. ref_mesh_path and dataroot
        self.parser.add_argument('--ref_mesh_path', required=True, help='Reference mesh path')
        self.parser.add_argument('--dataroot', required=True, help='root folder to deformed meshes and corresponding '
                                                                   'synthetic kV images')
        self.parser.add_argument('--img_res', type=int, default=256,
                                 help='Rescale images to size [img_res, img_res] before feeding to the network')
        self.parser.add_argument('--fpm', type=int, default=5, help='Features per FPN')
        self.parser.add_argument('--vpm', type=int, default=785, help='Vertices per mesh')
        self.parser.add_argument('--aps', type=int, default=7, help='average pool size')
        self.parser.add_argument('--nf', type=int, default=16, help='number of filters for the first conv layer')
        self.parser.add_argument('--nc', type=int, default=64, help='number of channels for the GNN')

