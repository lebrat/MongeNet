from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
import trimesh
import numpy as np
import torch
from src.utils import batch_meshes, load_checkpoint, TicToc
from src.models import MongeNet
from src.mesh_sampler import MeshSampler


# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name='sample')
def sample_app(cfg):

    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Mesh sampling with MongeNet\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))
    os.makedirs(cfg.sampler.output_dir, exist_ok=True)
    timer = TicToc();timer.tic()

    # loading meshes
    timer.tic('loading')
    logger.info('loading... {}'.format(cfg.sampler.input_meshes))
    meshes = []
    for m_path in cfg.sampler.input_meshes:
        mesh = trimesh.load(m_path)
        mesh.remove_duplicate_faces()
        meshes.append(mesh)
    vertices, faces, lenghts = batch_meshes(meshes)
    vertices, faces = torch.from_numpy(vertices).float().to(cfg.mongenet.device), torch.from_numpy(faces).long().to(cfg.mongenet.device)
    lenghts = torch.from_numpy(lenghts).long().to(cfg.mongenet.device)
    logger.info("Batched vertices: {}".format(vertices.shape))
    logger.info("Batched faces: {}".format(faces.shape))
    logger.info("Batched lenghts: {}".format(lenghts))
    logger.info("...data loaded in {} secs".format(timer.toc('loading')))

    # loading MongeNet model 
    timer.tic('model')
    logger.info('Setting up MongeNet model...')
    mongenet = MongeNet(cfg).to(cfg.mongenet.device)
    logger.info("MongeNet model:\n{}".format(mongenet))
    load_checkpoint(cfg.mongenet.checkpoint, model=mongenet)
    mongenet.train()
    logger.info("weights loaded from {}".format(cfg.mongenet.checkpoint))
    logger.info("...MongeNet setup in {} secs".format(timer.toc('model')))

    # running mesh sampler
    timer.tic('sampling')
    logger.info('Sampling meshes...')
    mesh_sampler = MeshSampler(mongenet, cfg.sampler.num_sampled_points, cfg.sampler.compute_normals, cfg.sampler.network_batch_size).to(cfg.mongenet.device)    
    points, face_ids, normals = mesh_sampler(vertices, faces, lenghts)
    logger.info("... meshes sampled in {} secs".format(timer.toc('sampling')))
    
    # save results    
    for i, m_path in enumerate(cfg.sampler.input_meshes):
        file_name = os.path.basename(m_path).split('.')[0]
        out_path = os.path.join(cfg.sampler.output_dir, "{}_{}.ply".format(file_name, cfg.sampler.output_file_suffix))
        trimesh.Trimesh(vertices=points[i].cpu().detach().numpy(), normals=normals[i].cpu().detach().numpy(), process=False).export(out_path)
        logger.info("Point cloud for mesh {} saved to {}".format(m_path, out_path))

    logger.info('Mesh Sampler finished in {} secs'.format(timer.toc()))
    

if __name__ == "__main__":
    sample_app()
