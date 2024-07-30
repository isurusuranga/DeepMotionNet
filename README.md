# Deep-Motion-Net

This is the code repository (model architecture implementation) for the paper: [Deep-Motion-Net: GNN-based volumetric organ shape reconstruction from single-view 2D projections](https://arxiv.org/abs/2407.06692). The whole code is implemented in [PyTorch](https://pytorch.org/) and [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). 

<img src="/imgs/Graphical_abstract_DMM.png" width="900"/>

## Getting Started

### Installation Instructions

We recommend that you create a new virtual environment for a clean installation of all relevant dependencies.

```
virtualenv dmn
source dmn/bin/activate
pip3 install --no-cache-dir torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip3 install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip3 install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip3 install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip3 install --no-cache-dir torch-geometric
pip3 install --no-cache-dir numpy
pip3 install --no-cache-dir meshio
pip3 install --no-cache-dir Pillow
```

### Dataset Preparation

Paired sets of organ motion instances (i.e. deformed volumetric meshes in.vtk unstructured grid format) and corresponding kilo-voltage (kV) X-ray images are required for model training/validation. To train and evaluate the model, we use [synthetically generated data](https://github.com/isurusuranga/SyntheticMotionDataGenerator) toolkit (Surrogate Parameterized Respiratory Motion Model). Plausible patient-specific motion patterns are extracted from 4D-CT images, and new synthetic instances are produced by interpolating and, within reasonable bounds, extrapolating from these with SuPReMo. 

The process is as follows:

1) The SuPReMo toolkit is used to analyse 4D-CT images, which produces a model of the motion present in the images linked to appropriate surrogate signals. 
2) Plausible motion instances generate by randomly perturbing the surrogate signal.
3) The motion fields are then used to deform the reference (phase 0) 3D-CT volume.
4) Extract reference volumetric mesh configuration from the motion reconstructed 3D-CT which output when fitting the SuPReMo (use [iso2mesh](https://iso2mesh.sourceforge.net/cgi-bin/index.cgi) MATLAB package).
5) Apply corresponding motion fields to the reference volumetric mesh to generate deformed volumetric mesh instances (use [SimpleITK](https://simpleitk.org/) python package).
6) Generate Digitally Reconstructed Radiographs (DRRs) from deformed 3D-CTs for all required projection angles using [RTK](http://www.openrtk.org/Doxygen/DocGeo3D.html) tool-kit.
7) Style transfer to DRRs to match the kV image intensity and noise distributions using [DRR2kVTranslation](https://github.com/isurusuranga/DRR2kVTranslation).

Once you have created relevant dataset, you can then divide the dataset into train, validation and test sets.

### Train

```
python train.py --ref_mesh_path *** --dataroot ***
```
```--ref_mesh_path``` => Reference volumetric mesh (in .vtk format) path

```--dataroot``` => Root folder to train/validation deformed meshes and corresponding synthetic kV images

The hyper-parameters can be changed from command.

### Test

To evaluate the model on one example, please use the following command

```
python eval.py --ref_mesh_path *** --dataroot *** --test_results_dir ***
```
```--ref_mesh_path``` => Reference volumetric mesh (in .vtk format) path

```--dataroot``` => Root folder to test deformed meshes and corresponding synthetic kV images

```--test_results_dir``` => Root folder to store predicted results from the model

## Some Examples

Qualitative results for one liver patient data is depicted below. High prediction accuracy was achieved in almost all regions of the organ models.

Recovered motion from projection angle 80.849°, viewed in coronal plane – a liver patient. Rendered liver mesh overlaid on the ground truth CT data.

![Alt Text](/imgs/animation_coronal_slice_breathing_cropped_IDAP775756_11.gif)

Recovered motion from projection angle 80.849°, viewed in sagittal plane – a liver patient. Rendered liver mesh overlaid on the ground truth CT data.

![Alt Text](/imgs/animation_saggital_slice_breathing_cropped_IDAP775756_11.gif)

## Citing

If you find this code useful for your research, please consider citing the following paper:

```
@article{wijesinghe2024deep,
  title={Deep-Motion-Net: GNN-based volumetric organ shape reconstruction from single-view 2D projections},
  author={Wijesinghe, Isuru and Nix, Michael and Zakeri, Arezoo and Hokmabadi, Alireza and Al-Qaisieh, Bashar and Gooya, Ali and Taylor, Zeike A},
  journal={arXiv preprint arXiv:2407.06692},
  year={2024}
}
```