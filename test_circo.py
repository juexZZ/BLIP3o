import json
from pathlib import Path
from typing import Union, List, Dict, Literal, Tuple
import PIL
import PIL.Image
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import os
from tqdm import tqdm
import shortuuid
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import re, random

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]

def img_process(images, processor, image_aspect_ratio):
    if image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images

class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'], mode: Literal['relative', 'classic'],
                 preprocess: callable=None):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def get_semantic_aspects(self, index):
        """ Returns the semantic aspects for a given query"""
        return self.annotations[index].get('semantic_aspects', [])

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            # reference_img = self.preprocess(PIL.Image.open(reference_img_path).convert('RGB'))
            reference_img = str(reference_img_path)

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                #target_img = self.preprocess(PIL.Image.open(target_img_path).convert('RGB'))
                target_img = str(target_img_path)

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_img_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = Image.open(img_path).convert("RGB")
            img = img_process(img, self.preprocess, image_aspect_ratio="square")
            img = img.squeeze(0) # [1, 3, 448, 448] -> [3, 448, 448]
            return {
                'img': img,
                'img_path': str(img_path),
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")



class RetrievalPipeline:
    """
    Circo retrieval pipeline
    """
    def __init__(self, blip3o_model, tokenizer, data_path):
        self.blip3o_inference_model = blip3o_model
        self.dtype = next(self.blip3o_inference_model.parameters()).dtype
        self.device = next(self.blip3o_inference_model.parameters()).device
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.tokenizer = tokenizer

        # self.dataset = CIRCODataset(
        #     data_path=base_path, 
        #     split='val', 
        #     mode='classic', 
        #     preprocess=self.blip3o_inference_model.get_gen_vision_tower().image_processor,
        # )
        # print(len(self.dataset))
        # self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=512, shuffle=False, num_workers=8)

    def encode_images(self, images):
        images_feature = self.blip3o_inference_model.encode_image(images) # [bsz, dim(1792), 8, 8] 8 is the latent size
        num_img, c = images_feature.size()[:2]
        images_feature = images_feature.view(num_img, c, -1).permute(0,2,1).contiguous()
        images_emb = images_feature.mean(dim=1)
        # print("image embedding size", images_emb.size())
        return images_emb

    def encode_images_prompt(self, image_paths, prompt="Please summarize this image in a few words"):
        """This function encodes Image + Text as Vision Langugae input to the Qwen 2.5 VL model """
        batch_images_emb = []
        for p in image_paths:
            image = Image.open(p).convert('RGB')
            multimodal_inputs = []
            text_prompt = add_template([f"<image> {prompt}"])

            # ! NOTE: because of its original code logic, have to put text as the first element
            multimodal_inputs.extend(text_prompt)
            multimodal_inputs.append(image)
            text_input, pixel_values, image_grid_thw = self.blip3o_inference_model.prepare_vl_input_for_generation(
                inputs = multimodal_inputs,
                processor = self.processor
            )
            pixel_values = pixel_values.to(self.device)
            image_grid_thw = image_grid_thw.to(self.device)

            gen_clip_features = self.blip3o_inference_model.generate_image(
                text = [text_input],
                tokenizer = self.tokenizer,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw
            )
            image_prompt_emb = gen_clip_features.mean(dim=1)
            batch_images_emb.append(image_prompt_emb)
        return torch.cat(batch_images_emb, dim=0)

    # ! Stage 1: classic
    def embed_candi(self, out_dir):
        self.dataset = CIRCODataset(
            data_path=base_path, 
            split='val', 
            mode='classic', 
            preprocess=self.blip3o_inference_model.get_gen_vision_tower().image_processor,
        )
        print(len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=512, shuffle=False, num_workers=8)
        image_embeds = []
        image_paths = []
        for batch in tqdm(self.dataloader):
            img_paths = batch["img_path"]
            # print(len(img_paths), img_paths[0])
            images = batch["img"].to(self.device)
            print("image size", images.size())
            candi_embs = self.encode_images(images=images)
            print(candi_embs.size())
            #pdb.set_trace()
            image_embeddings = candi_embs
            
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            image_embeds.append(image_embeddings.cpu())
            image_paths.extend(img_paths)
        
        image_embeds = torch.cat(image_embeds, 0).cpu()
        output = {"img_path": image_paths, "img_embeds": image_embeds}
        torch.save(output, f'{out_dir}/megapairs_clip_L.pt')

    #! Stage 2: relative
    def embed_relative(self, out_dir):
        # reconfigure the dataset
        self.dataset = CIRCODataset(
            data_path=base_path, 
            split='val', 
            mode='relative', 
            preprocess=self.blip3o_inference_model.get_gen_vision_tower().image_processor,
        )
        print("relative dataset", len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        cand = torch.load(f'{out_dir}/megapairs_clip_L.pt')
        embeddings = cand['img_embeds'].cuda()
        # print("embeddings", embeddings.dtype, embeddings.size(), embeddings.device)
        cand_paths = cand['img_path']
        def extract_id(s):
            return int(s.split("/")[-1].split(".")[0])

        reference_img_list = []
        caption_list = []
        gt_img_list = []
        result_list = []
        mod_caption_list = []

        for idx, batch in tqdm(enumerate(self.dataloader)):
            image = batch['reference_img'][0] # because the batch size is one
            text = batch['relative_caption'][0]
            # print("image", image)
            # print("text", text)

            ret_embed = self.encode_images_prompt(
                prompt=text, 
                image_paths=[image],
            ).float()
            # print("ret embed", ret_embed.dtype, ret_embed.size(), ret_embed.device)
            ret_embed /= ret_embed.norm(dim=-1, keepdim=True)
            cosine_sim = (embeddings@ret_embed.T).squeeze().cpu().float()
            topk_indices = torch.argsort(cosine_sim, descending=True)[:50]
            topk_image_paths = [cand_paths[i] for i in topk_indices]
            result_img_ids = [extract_id(img_path) for img_path in topk_image_paths]
            reference_image = batch['reference_img']
            caption = batch['relative_caption']
            gt_img_ids = batch['gt_img_ids']
            gt_img_ids_set = set()
            for ids in gt_img_ids:
                if ids[0]!="":
                    gt_img_ids_set.add(int(ids[0]))
            gt_img_list.append(gt_img_ids_set)
            caption_list.append(caption)
            reference_img_list.append(reference_image)
            result_list.append(result_img_ids)

        with open("circo_val_result_megapair_clipL.pkl", "wb") as f:
            results = {
                "reference_img": reference_img_list,
                "caption": caption_list,
                "gt_img": gt_img_list,
                "result": result_list,
            }
            pickle.dump(results, f)


def compute_metrics(data_path: Path, predictions_dict: Dict[int, List[int]], ranks: List[int]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[str, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]: Dictionaries with the AP and Recall for each rank,
            and the semantic mAP@10 for each semantic aspect
    """

    relative_val_dataset = CIRCODataset(data_path, split='val', mode='relative', preprocess=None)

    semantic_aspects_list = ['cardinality', 'addition', 'negation', 'direct_addressing', 'compare_change',
                              'comparative_statement', 'statement_with_conjunction', 'spatial_relations_background',
                              'viewpoint']

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)
    semantic_aps_at10 = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = relative_val_dataset.get_target_img_ids(int(query_id))
        semantic_aspects = relative_val_dataset.get_semantic_aspects(int(query_id))
        gt_img_ids = target['gt_img_ids']
        target_img_id = target['target_img_id']

        # Check if the predictions are unique
        if len(set(predictions)) != len(predictions):
            raise ValueError(f"Query {query_id} has duplicate predictions. Please ensure to provide unique predictions"
                             f"for each query.")

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

        # Compute the AP@10 for each semantic aspect
        for aspect in semantic_aspects:
            semantic_aps_at10[aspect].append(float(np.sum(precisions[:10]) / min(len(gt_img_ids), 10)))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    map_atk = {}
    recall_atk = {}
    semantic_map_at10 = {}
    for rank in ranks:
        map_atk[rank] = float(np.mean(aps_atk[rank]))
        recall_atk[rank] = float(np.mean(recalls_atk[rank]))

    # Compute the mean AP@10 for each semantic aspect and store them in a dictionary
    for aspect in semantic_aspects_list:
        semantic_map_at10[aspect] = float(np.mean(semantic_aps_at10[aspect]))

    return map_atk, recall_atk, semantic_map_at10


def extract_id(s):
    return int(s.split("/")[-1].split(".")[0])

def main_eval():
    # Parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default="/group-volume/visualnuggets/dat/retrieval/CIRCO")
    args.add_argument('--ranks', type=int, nargs='+', default=[5, 10, 25, 50])
    args.add_argument('--predictions_file_path', '-p', type=str,
                      default="circo_val_result.pkl")
    args = args.parse_args()

    # Load the predictions from the given file
    # try:
    #     with open(args.predictions_file_path, 'r') as f:
    #         predictions_dict = json.load(f)
    # except FileNotFoundError as e:
    #     raise Exception("predictions_file_path must be a valid path to a json file")
    with open(args.predictions_file_path, "rb") as f:
        results = pickle.load(f)

    reference_img_list = results['reference_img']
    result_list = results['result']
    predictions_dict = {}
    for i, (ref_img, res) in enumerate(zip(reference_img_list, result_list)):
        print(i, extract_id(ref_img[0]), res)
        predictions_dict[i] = res

    # Ensure that the query ids are consecutive and start from zero
    assert np.all(np.sort(np.array(list(predictions_dict.keys()), dtype=int)) == np.arange(
        len(predictions_dict.keys()))), "The keys of the predictions dictionary must be all the query ids"

    # Compute the metrics and print them
    map_atk, recall_atk, semantic_map_at10 = compute_metrics(args.data_path, predictions_dict, args.ranks)

    print("\nWe remind that the mAP@k metrics are computed considering all the ground truth images for each query, the "
          "Recall@k metrics are computed considering only the target image for each query (the one we used to write "
          "the relative caption)")

    print("\nmAP@k metrics")
    for rank in args.ranks:
        print(f"mAP@{rank}: {map_atk[rank] * 100:.2f}")

    print("\nRecall@k metrics")
    for rank in args.ranks:
        print(f"Recall@{rank}: {recall_atk[rank] * 100:.2f}")

    print("\nSemantic mAP@10 metrics")
    for aspect, map_at10 in semantic_map_at10.items():
        print(f"Semantic mAP@10 for aspect '{aspect}': {map_at10 * 100:.2f}")

if __name__ == "__main__":

    # device_1 = 0
    # set_global_seed(seed=42)
    # model_path = "/group-volume/juexiaozhang/hf_cache/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/3c307c309d94a594efea23afc54ecebe82798b6a/"

    # disable_torch_init()
    # model_path = os.path.expanduser(model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)

    # base_path = "/group-volume/visualnuggets/dat/retrieval/CIRCO"
    # pipeline = RetrievalPipeline(
    #     blip3o_model = multi_model,
    #     tokenizer = tokenizer,
    #     data_path = base_path
    # )

    # out_dir = './circo_results'
    # # ! Stage 1
    # pipeline.embed_candi(out_dir = out_dir)
    # # ! Stage 2
    # pipeline.embed_relative(out_dir = out_dir)

    main_eval()
