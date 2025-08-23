import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.optim import rmsprop  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 4:
                aggregated_tokens, patch_start_idx, past_key_values, attn_maps_global_layers = aggregator_output
                # attn_maps_global_layers is a list with len=#layers. for each layer we have attention map averaged on heads between current frame tokens and
                # tokens from previous frames + current frame
                if i==0:
                    cum_attn_maps = [map.clone() for map in attn_maps_global_layers]

                elif i>0:
                    for j, attn_map in enumerate(attn_maps_global_layers):
                        temp = cum_attn_maps[j]
                        cum_attn_maps[j] = attn_map
                        cum_attn_maps[j][:, :temp.size(1)] + temp
                    

                    kv_remove_indices = self.get_remove_indices(cum_attn_maps, S=i, rm_percentage=10)
                    past_key_values, keep_idx_list, cum_attn_maps = self.compact_kv_cache_per_layer(past_key_values, kv_remove_indices, cum_attn_maps)
                
                del attn_maps_global_layers
                kv_cache_mem = 0
                for kv_cache in past_key_values:
                    for cashe in kv_cache:
                        kv_cache_mem = kv_cache_mem + (cashe.numel() * cashe.element_size())
                print(f"GPU RAM usage of kv cashe in iteration {i}: {kv_cache_mem} bytes")
                print(f"k cashe shape for each layer in iteration {i}: {past_key_values[0][0].shape} bytes")
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output

    def get_remove_indices(self, attn_maps_global_layers, S,  rm_percentage=5, rand_rm= False):
        '''returns a list where item i contains a tensor of indices to evict in layer i'''
        # dont remove special tokens
        # dont remove tokens from first frame
        
        # attn_maps = torch.stack(attn_maps_global_layers)
        # attn_maps = attn_maps.sum(dim=1) # shape: [num layer, num cashed tokens]
        # del attn_maps_global_layers

        rm_indices_per_layer=[]
        for attn_map in attn_maps_global_layers:
            token_per_frame_num, all_token_num = attn_map.shape
            device = attn_map.device

            #rm percentage is per layer
            k = int(all_token_num * (rm_percentage / 100))
            if rand_rm:
                #select random indices
                # flat_indices = torch.randperm(y - x, device=device)[:k] + x
                flat_indices = torch.randperm(all_token_num - 5, device=device)[:k] + 5

            else:
                
            
                # avg on Q dim
                scores = attn_map.mean(dim=0).clone()
                # avg on iterations per frame
                scores = scores / (S+1) #maybe changing it to EMA
                # set scores of first frame to infinity
                scores[:token_per_frame_num] = float('inf')

                # columns of specials: [s, s+1, ..., s+n_special-1] for every s in frame_starts
                # specials = (start_frame_indices[:, None] + torch.arange(patch_start_idx, device=device)).flatten()
                # specials = specials[(specials >= 0) & (specials < y)].unique()

                # boolean mask over columns
                # col_mask = torch.zeros(y, dtype=torch.bool, device=device)
                # col_mask[specials] = True

                # set those columns to +inf for all rows
                # attn_maps[:, col_mask] = float('inf')

                
            
                # 2. Flatten the tensor and find the k smallest values and their 1D indices
                # Using largest=False makes topk find the smallest values.
                # We only need the indices, so we can ignore the values with _.
                _, flat_indices = torch.topk(scores, k, largest=False)

            
            
            rm_indices_per_layer.append(flat_indices)

        return rm_indices_per_layer


    @torch.no_grad()
    def compact_kv_cache_per_layer(self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        per_layer_remove: List[torch.Tensor],
        cum_attn_maps
    ):
        """
        Evict token columns from a KV cache *layer by layer*.

        Args
        ----
        kv_cache : list of length L
            Each item is a tuple (K, V, pos)
            - K, V: [B, H, T, D]  (remove along dim=2)
            - pos : [B, T, 2]     (remove along dim=1)
        per_layer_remove : list of length L
            Each item i is a 1D LongTensor of column indices to remove for layer i.
            (Same indices applied to K, V, and pos of that layer.)

        Returns
        -------
        new_kv_cache : list of length L
            Layer-wise compacted (K, V, pos).
        keep_idx_list : list of length L
            For each layer i, a 1D LongTensor mapping new -> old column indices.
        """
        # assert len(kv_cache) == len(per_layer_remove), "one removal tensor per layer"

        new_kv_cache = []
        keep_idx_list = []
        new_cum_maps = []
        # can we paralelise this using hreads?
        for (K, V, pos), rem, attn_map in zip(kv_cache, per_layer_remove, cum_attn_maps):
            # Shapes & device
            # assert K.ndim == 4 and V.ndim == 4 and pos.ndim == 3, "bad KV/pos shapes"
            B, H, T, D = K.shape
            # assert V.shape == (B, H, T, D)
            # assert pos.shape == (B, T, 2)
            # attn_amp -> (T1, T)
            device = K.device

            # --- build keep indices (complement of rem), preserving original order ---
            if rem is None or rem.numel() == 0:
                keep_idx = torch.arange(T, device=device)
            else:
                rem = rem.to(device).long()
                # if T > 0:
                # #     # clamp to valid range, unique-sort
                #     rem = rem[(rem >= 0) & (rem < T)].unique()
                if rem.numel() == 0:
                    keep_idx = torch.arange(T, device=device)
                elif rem.numel() == T:
                    # everything removed -> keep nothing
                    keep_idx = torch.empty(0, dtype=torch.long, device=device)
                else:
                    keep_mask = torch.ones(T, dtype=torch.bool, device=device)
                    keep_mask[rem] = False
                    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # sorted ascending

            # --- slice tensors once (fast, contiguous result) ---
            K_new = K.index_select(2, keep_idx)
            V_new = V.index_select(2, keep_idx)
            pos_new = pos.index_select(1, keep_idx)
            attn_map_new = attn_map.index_select(1, keep_idx)

            new_kv_cache.append((K_new, V_new, pos_new))
            keep_idx_list.append(keep_idx)
            new_cum_maps.append(attn_map_new)

        return new_kv_cache, keep_idx_list, new_cum_maps

