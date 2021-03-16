import time, math
import torch
import numpy as np


class TicToc:
    """
    TicToc class for time pieces of code.
    """

    def __init__(self):
        self._TIC_TIME = {}
        self._TOC_TIME = {}

    def tic(self, tag=None):
        """
        Timer start function
        :param tag: Label to save time
        :return: current time
        """
        if tag is None:
            tag = 'default'
        self._TIC_TIME[tag] = time.time()
        return self._TIC_TIME[tag]

    def toc(self, tag=None):
        """
        Timer ending function
        :param tag: Label to the saved time
        :param fmt: if True, formats time in H:M:S, if False just seconds.
        :return: elapsed time
        """
        if tag is None:
            tag = 'default'
        self._TOC_TIME[tag] = time.time()

        if tag in self._TIC_TIME:
            d = (self._TOC_TIME[tag] - self._TIC_TIME[tag])
            return d
        else:
            print("No tic() start time available for tag {}.".format(tag))

    # Timer as python context manager
    def __enter__(self):
        self.tic('CONTEXT')

    def __exit__(self, type, value, traceback):
        self.toc('CONTEXT')


def save_checkpoint(model, optimizer, ckp_file):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                ckp_file)


def load_checkpoint(ckp_file, model=None, optimizer=None):
    chk_dict = torch.load(ckp_file)
    
    if model is not None:
        model.load_state_dict(chk_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(chk_dict['optimizer_state_dict'])
    

def batchify(arr, batch_size):
    num_batches = math.ceil(len(arr) / batch_size)
    return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]


def batch_meshes(list_trimesh):    
    batch_vertices, batch_faces, batch_lenghts, vertices_cumsum = [], [], [], 0.0
    for mesh in list_trimesh:
        batch_vertices.append(mesh.vertices)
        batch_faces.append(mesh.faces +  vertices_cumsum)
        batch_lenghts.append([mesh.vertices.shape[0], mesh.faces.shape[0]])
        vertices_cumsum += mesh.vertices.shape[0]
    batch_vertices, batch_faces, batch_lenghts = np.concatenate(batch_vertices, axis=0), np.concatenate(batch_faces, axis=0), np.array(batch_lenghts)
    return batch_vertices, batch_faces, batch_lenghts