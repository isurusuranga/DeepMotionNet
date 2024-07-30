import torch
from torch_geometric.data import Data
import meshio


def to_meshio(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`.msh format`.

    Args:
        data (torch_geometric.data.Data): The data object.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    points = data.pos.detach().cpu().numpy()
    tetra = data.tetra.detach().t().cpu().numpy()

    cells = [("tetra", tetra)]

    return meshio.Mesh(points, cells)


def from_meshio(mesh):
    r"""Converts a :.msh file to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (meshio.read): A :obj:`meshio` mesh.
    """

    if meshio is None:
        raise ImportError('Package `meshio` could not be found.')

    pos = torch.from_numpy(mesh.points).to(torch.float)
    tetra = torch.from_numpy(mesh.cells_dict['tetra']).to(torch.long).t().contiguous()

    return Data(pos=pos, tetra=tetra)