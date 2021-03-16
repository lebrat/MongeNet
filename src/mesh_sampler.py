import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import numpy as np
from src.utils import batchify


class MeshSampler(nn.Module):   
    def __init__(self, tri_sampler_model, sample_size, return_normals=False, network_batch_size=25000):
        super(MeshSampler, self).__init__()
        self.model = tri_sampler_model      
        self.device = self.model.device        
        self.sample_size = sample_size
        self.max_samples_per_face = self.model.max_num_points       
        self.return_normals = return_normals
        self.network_batch_size = network_batch_size

        
    def forward(self, vertices, faces, lenghts):
    # vertices V x 3: Concatenation of the vertices of B meshes in R^3.
    # faces F x 3: Concatenation of faces (idexes of vertices) of B meshes.
    # lenghts Bx2: arrays holding the number of vertices (col 0) and faces (col 1) for B meshes.
    # return: sampled points (BxPx3) and their respective faces ids (BxP)

        vertices, faces, lenghts = vertices.float(), faces.long(), lenghts.long()               
        # read in triangles
        (V, D), (F, T), (B, L) = vertices.shape, faces.shape, lenghts.shape
        assert (V, F) == tuple(lenghts.sum(0)) 
        assert T == 3 and D == 3, "faces are not triangles in R^3"
        assert L == 2, "We need the combined lenght of vertices and faces in the meshes"        
        triangles = vertices[faces]
        
        # computing the number of points to sample per face and subdivide faces that require more than self.max_samples_per_face        
        triangles, face_ids, lenghts, sampling_counts = self.sampling_counts_with_edge_split(triangles, lenghts)
        F = triangles.shape[0]
        if self.return_normals:
            triangles_normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 1], dim=-1)
            triangles_normals = triangles_normals / torch.norm(triangles_normals, p=2, dim=-1, keepdim=True)

        # mapping triangles to default triangle (#Fx3x3 and #BxFmax4x4)     
        # permute according to largest side
        pre_comp_order = torch.Tensor([[0, 1, 2], [0, 2, 1], [2, 1, 0]]).view(3, 1, 3).long().to(self.device)       
        inv_transforms = torch.zeros((F, 4, 4)).float().to(self.device); inv_transforms[:, -1, -1] = 1.0
        sides = torch.norm(triangles[:, [0, 0, 1], :] - triangles[:, [1, 2, 2], :], p=2, dim=-1) 
        order = sides.argmax(dim=-1, keepdim=True).long()
        assert order.min() >= 0 and order.max() <= 2, "Inconsistent permutation computation"        
        order_seq = (order == 0) * pre_comp_order[0] + (order==1) * pre_comp_order[1] + (order == 2) * pre_comp_order[2]
        triangles = torch.gather(triangles, dim=1, index=order_seq.view(F, T, 1).repeat(1, 1, 3))
        # compute translations
        T = -triangles[:, [0], :]
        triangles = triangles + T
        inv_transforms[:, :3, [-1]] = -T.transpose(1,2)
        # compute rotations (since the cross product will be computed compute also the areas)
        U = triangles[:, 1] / (torch.norm(triangles[:, 1], p=2, dim=-1, keepdim=True) + 1e-12)
        W = torch.cross(triangles[:, 1], triangles[:, 2], dim=-1)
        W = W / (torch.norm(W, p=2, dim=-1, keepdim=True) + 1e-12).clamp(min=1e-12)     
        V = torch.cross(U, W, dim=-1)
        R = torch.stack([V, U, W], dim=-1)
        triangles = torch.matmul(triangles, R).float()      
        # compute scaling
        S = (0.998 / triangles[:, 1, 1]).view(-1, 1, 1)     
        triangles = S * triangles
        inv_transforms[:, :3, :3] = R / S
        # reflection if necessary
        mask = (triangles[:, -1, 0] < 0.0); RF = mask * (-1.0) + (~mask) * (1.0);
        triangles[:, -1, 0] = RF * triangles[:, -1, 0]
        inv_transforms[:, :3, 0] = inv_transforms[:, :3, 0] * RF.view(-1, 1)                

        # sample points using a neural network  
        # batched due to memory problems            
        sampled_sets_of_points = torch.cat([self.model(triangles[:, :, :-1][triangles_block])[0] for triangles_block in batchify(list(range(triangles.shape[0])), self.network_batch_size)], dim=0)       
        samples_sets_len_idxs = 1.0 * self.model.output_index
        # sampled_sets_of_points, samples_sets_len_idxs = self.model(triangles[:, :, :-1])      
        points_mask = (sampling_counts.view(-1, 1) ==  samples_sets_len_idxs.view(1, -1))
        sampled_points = sampled_sets_of_points[points_mask]                                            
        sampled_points = torch.cat([sampled_points, torch.zeros((sampled_points.shape[0], 1), device=self.device), torch.ones((sampled_points.shape[0], 1), device=self.device)], dim=-1)       
        points_ids = points_mask.nonzero()[:, 0]
        sampled_points = torch.matmul(inv_transforms[points_ids], sampled_points.unsqueeze(-1))[:, :3, 0]

        # return points as BxSx3 and face ids as BxS
        sampled_points = sampled_points.view(B, self.sample_size, 3)
        face_ids = face_ids[points_ids].view(B, self.sample_size)

        if triangles_normals is None:
            return sampled_points, face_ids         
        else:
            triangles_normals = triangles_normals[points_ids].view(B, self.sample_size, 3)
            return sampled_points, face_ids, triangles_normals


    def sampling_counts_with_edge_split(self, triangles, lenghts):
        # this function performs a sequence of multinomial sampling and faces subdivision until reach the total number of points without exceeding 
        # the faces maximum. triangles is the concatenation of the triangles in B meshes, while <lengths> is a Bx2 array with the
        # the number of vertices and faces of the B meshes.
        # returns a (F + new faces) array with the number of points that should be sampled from each face in the concatenated F vector and the updated length vector
        
        # compute sampling quantities
        B, F = lenghts.shape[0], triangles.shape[0] 
        areas = 0.5 * torch.norm(torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1), p=2, dim=-1, keepdim=True)      
        sampling_counts = torch.zeros((B, F)).float().to(self.device)
        batched_index = torch.arange(B).long().to(self.device).repeat_interleave(lenghts[:, 1]).view(1, -1)     
        sampling_counts = sampling_counts.scatter(0, batched_index, areas.view(1, -1)) # in place operation 
        sampling_counts = sampling_counts / (sampling_counts.sum(dim=-1, keepdim=True) + 1e-12)
        sampling_counts = torch.distributions.multinomial.Multinomial(total_count=self.sample_size, probs=sampling_counts).sample().long()
        assert sampling_counts.shape[0] == 1 or torch.prod(sampling_counts, dim=0).sum() == 0, "faces from different mesh have been mixed"
        sampling_counts, batched_index, tri_index = sampling_counts.sum(0), batched_index.view(-1), torch.arange(F).long().to(self.device)
        # iterate until all faces can be sampled with only maximum_number_of_points per face
        while True:
            # select triangles exceeding the sampling limit
            exceeded_mask = (sampling_counts > self.max_samples_per_face)
            if torch.any(exceeded_mask):
                old_triangles, old_tri_index = triangles[exceeded_mask], tri_index[exceeded_mask]
                old_counts, old_batch_index =  sampling_counts[exceeded_mask], batched_index[exceeded_mask]         
                # compute mid points and combine them to get the new triangles                          
                longest_side = torch.norm(old_triangles[:, [0, 1, 2], :] - old_triangles[:, [1, 2, 0], :], p=2, dim=-1).argmax(dim=1).long()
                aux_index = torch.arange(longest_side.numel())
                mid_points = ((old_triangles[:, [0, 1, 2] ,:] + old_triangles[:, [1, 2, 0] ,:]) / 2.0)[aux_index, longest_side]
                start_points, end_points, oposite_points = old_triangles[aux_index, longest_side], old_triangles[aux_index, (longest_side + 1) % 3], old_triangles[aux_index, (longest_side + 2) % 3]
                new_triangles = torch.stack([start_points, mid_points,  oposite_points, mid_points, end_points, oposite_points], dim=1).view(-1, 2, 3, 3)               
                # compute new triangles areas and num samples (it is unecessasry to compute the area of those triangles since they are the same)                
                new_counts = torch.zeros(new_triangles.shape[:2]).long().to(self.device)
                for i in range(old_counts.shape[0]):
                    new_counts[i] = torch.distributions.multinomial.Multinomial(total_count=int(old_counts[i]), probs=torch.tensor([ 1., 1.])).sample().long()              
                # join new faces and sampling counts
                triangles = torch.cat([triangles[~exceeded_mask], new_triangles.view(-1, 3, 3)], dim=0)
                sampling_counts = torch.cat([sampling_counts[~exceeded_mask], new_counts.view(-1)], dim=0)
                batched_index = torch.cat([batched_index[~exceeded_mask], old_batch_index.repeat_interleave(2)], dim=0)
                tri_index = torch.cat([tri_index[~exceeded_mask], old_tri_index.repeat_interleave(2)], dim=0)
            else:
                # check for correctness
                assert torch.all(sampling_counts <= self.max_samples_per_face), "sampler can not sample more than {} per face".format(self.max_samples_per_face) 
                assert sampling_counts.sum() == B*self.sample_size, "number of samples per mesh is not reached"             
                break
        # reorder faces and counts according to batches and recompute lengths
        unique_index, lenghts[:, 1] = torch.unique(batched_index, return_counts=True)
        batched_index = (unique_index.view(-1 ,1) == batched_index.view(1, -1)).nonzero()[:,1]      
        triangles, tri_index, sampling_counts = triangles[batched_index], tri_index[batched_index], sampling_counts[batched_index]
        return triangles, tri_index, lenghts, sampling_counts
