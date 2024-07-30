import os
import torch
import meshio
import torchvision.transforms.functional as TF
from PIL import Image
from utils import *
from models import DeepMotionNet
from transforms import TetraToEdge
import glob


class Evaluator(object):
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        self.deformed_test_mesh_paths = glob.glob(os.path.join(self.options.dataroot, 'meshes/test/*.vtk'))
        self.deformed_test_img_paths = glob.glob(os.path.join(self.options.dataroot, 'images/test/*.png'))

        # load DMM model
        self.motion_model = DeepMotionNet(options).to(self.device)
        self.best_checkpoint = self.saver.load_best_check_point()
        self.motion_model.load_state_dict(self.best_checkpoint['model_state_dict'])

        self.deformed_test_img_dict = get_drr_image_dict(self.deformed_test_img_paths)

    def evaluate(self):
        self.motion_model.eval()
        t2e = TetraToEdge(remove_tetras=False)

        reference_mesh = meshio.read(self.options.ref_mesh_path)
        reference_data = from_meshio(reference_mesh)
        reference_data = t2e(reference_data)

        for i, path in enumerate(self.deformed_test_mesh_paths):
            deformed_mesh = meshio.read(path)
            deformed_mesh_name = os.path.basename(path).rsplit('.', 1)[0]
            orig_deformed_mesh_path = os.path.join(self.options.test_gt_dir, deformed_mesh_name + ".vtk")
            meshio.write(orig_deformed_mesh_path, deformed_mesh, binary=False)

            deformed_data = from_meshio(deformed_mesh)

            img_key = os.path.basename(path).rsplit('.', 1)[0]

            test_drr_img_path_list = self.deformed_test_img_dict[img_key]

            for drr_img_path in test_drr_img_path_list:
                ref_data_duplicate = reference_data.clone()
                img_name_str = os.path.basename(drr_img_path).rsplit('.', 1)[0]
                # projection angle in degrees
                gantry_angle = float(os.path.basename(img_name_str).rsplit('_', 1)[1])

                img = Image.open(drr_img_path)
                img = img.resize((self.options.img_res, self.options.img_res))
                img = TF.to_tensor(img)
                # need to provide the batch dimension at dim0 (e.g. torch.Size([1, 1, 256, 256]))
                img.unsqueeze_(0)

                ref_data_duplicate.x = ref_data_duplicate.pos
                ref_data_duplicate.img = img
                ref_data_duplicate.gantry_angle = torch.tensor([gantry_angle / 360])

                with torch.no_grad():
                    predicted_coords = self.motion_model(ref_data_duplicate.to(self.device))

                print('original deformed coords:')
                print(deformed_data.pos)

                print('predicted_ deformed coords:')
                print(predicted_coords.cpu())

                ref_data = reference_data.clone()

                ref_data.pos = predicted_coords.cpu()

                predicted_deformed_mesh = to_meshio(ref_data)
                predicted_mesh_name = deformed_mesh_name + '_' + str(int(gantry_angle))
                predicted_deformed_mesh_path = os.path.join(self.options.test_pred_dir, predicted_mesh_name + ".vtk")
                meshio.write(predicted_deformed_mesh_path, predicted_deformed_mesh, binary=False)



