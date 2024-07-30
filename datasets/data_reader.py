import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import meshio
from transforms import TetraToEdge
from utils import *


def dataset_reader(ref_mesh_path, deformed_mesh_paths, img_dict, img_res):
    t2e = TetraToEdge(remove_tetras=False)
    data_list = []

    reference_mesh = meshio.read(ref_mesh_path)
    reference_data = from_meshio(reference_mesh)
    reference_data = t2e(reference_data)

    for i, path in enumerate(deformed_mesh_paths):
        # meshio package to read .msh files
        deformed_mesh = meshio.read(path)
        # we need deformed_data to calculate and attach the position displacements as response variable y to ref_data
        deformed_data = from_meshio(deformed_mesh)

        img_key = os.path.basename(path).rsplit('.', 1)[0]

        # hold a list of DRR images for a given deformed mesh
        drr_img_path_list = img_dict[img_key]

        # generate training samples for all the drr images in the list by extracting features
        for drr_img_path in drr_img_path_list:
            ref_data = reference_data.clone()
            # target variable
            ref_data.y = deformed_data.pos
            # extract the drr image name with extension from the absolute path (e.g. volume-10_0.png)
            img_name_str = os.path.basename(drr_img_path).rsplit('.', 1)[0]
            # projection angle in degrees
            gantry_angle = float(os.path.basename(img_name_str).rsplit('_', 1)[1])

            # need to get projection angle as well to pass to the DRR projector
            # Path should be come with row data for a given deformed graph
            img = Image.open(drr_img_path)
            img = img.resize((img_res, img_res))
            img = TF.to_tensor(img)
            img = TF.normalize(img, (0.5,), (0.5,))
            # need to provide the batch dimension at dim0 (e.g. torch.Size([1, 3, 224, 224]))
            img.unsqueeze_(0)

            ref_data.x = ref_data.pos

            ref_data.img = img
            ref_data.gantry_angle = torch.tensor([gantry_angle / 360])

            data_list.append(ref_data)

    return data_list
