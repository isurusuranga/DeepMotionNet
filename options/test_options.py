import os
import json
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super(TestOptions, self).__init__()
        # pass dataroot argument to the root of test images and meshes folder path
        self.parser.add_argument('--test_results_dir', required=True, help='Directory to store test results')

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()

        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')

        self.args.test_gt_dir = os.path.join(self.args.test_results_dir, 'groundtruth')
        if not os.path.exists(self.args.test_gt_dir):
            os.makedirs(self.args.test_gt_dir)

        self.args.test_pred_dir = os.path.join(self.args.test_results_dir, 'predictions')
        if not os.path.exists(self.args.test_pred_dir):
            os.makedirs(self.args.test_pred_dir)

        self.save_dump()

        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "test_config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return


