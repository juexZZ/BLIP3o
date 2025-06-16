""" 
retrieval test

directly modified from inference.py
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import os
import json
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

model_path = "/group-volume/juexiaozhang/hf_cache/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/3c307c309d94a594efea23afc54ecebe82798b6a/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        help="specify the dataset for testing"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./vl_embeddings",
        help="directory to store the embeddings"
    )
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    parser.add_argument("--seed", type=int, default=42, help="Control all seeds")
    opt = parser.parse_args()
    return opt


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


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


# device_1 = 0

# disable_torch_init()
# model_path = os.path.expanduser(model_path)
# model_name = get_model_name_from_path(model_path)
# print("model path", model_path)
# print("model name", model_name)
# tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)

FLICKR30k_PATH = "/group-volume/deenmohan/Datasets/flickr30k/"
COCO_PATH = "/group-volume/visualnuggets/dat/visual_haystacks/coco/"


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

def stats_metaquery(x):
    """Compute metaquery stats on all data"""
    D, N, C = x.shape  # N = 64, C = 3584
    x_norm = F.normalize(x, p=2, dim=2)
    # cosine similarity -> [D, 64, 64]
    sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))
    # remove diagonal
    triu_mask = torch.triu(torch.ones(N, N, dtype=bool, device=x.device), diagonal=1)  # [64, 64]
    pairwise_sim = sim_matrix[:, triu_mask]  # shape: [D, 2016], 64*63/2 = 2016
    # mean and std over all metaquery for each datapoint in the data D
    mean_sim = pairwise_sim.mean(dim=1)   # [D]
    std_sim = pairwise_sim.std(dim=1)     # [D]

    print("Mean of mean similarities across dataset:", mean_sim.mean().item())
    print("Std of mean similarities across dataset:", mean_sim.std().item())
    print("Mean of std similarities across dataset:", std_sim.mean().item())
    return mean_sim, std_sim


# * -------------------------- code (BELOW) from LAION AI CLIP Benchmark --------------------------- * #

def dataloader_with_indices(dataloader):
    start = 0
    for data in dataloader:
        bsz = len(data["images"])
        end = start + bsz
        inds = torch.arange(start, end)
        yield data, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

# * -------------------------- code (ABOVE) from LAION AI CLIP Benchmark --------------------------- * #
# * -------------------------- EVA CLIP and OpenCLIP uses this --------------------------- * #

class Flickr30kT2IRetrieval(Dataset):
    """dataset for flick3r T2I retrieval"""
    def __init__(self, dataset_dir):
        self.image_dir = os.path.join(dataset_dir, "Images")
        self.all_image_ids = [os.path.basename(f).split(".jpg")[0] for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.caption_file = os.path.join(dataset_dir, "captions.txt")

        self.image_caption_map = dict()
        with open(self.caption_file, 'r', encoding='utf-8') as capf:
            lines = capf.readlines()[1:]
            for line in lines:
                image_id, caption = line.strip().split(".jpg,")
                if image_id not in self.image_caption_map:
                    self.image_caption_map[image_id] = list()
                self.image_caption_map[image_id].append(caption)
        self.image_caption_list = [(image_id, captions) for image_id, captions in self.image_caption_map.items()]
        
    
    def __len__(self,):
        return len(self.image_caption_list)

    def __getitem__(self, idx):
        image_id, captions = self.image_caption_list[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(image_path).convert("RGB")
        img = img_process(img, self.image_processor, image_aspect_ratio="square")
        img = img.squeeze(0) # [1, 3, 448, 448] -> [3, 448, 448]

        return (img, caption)

class COCORetrieval(Dataset):
    """Dataset for COCO retrieval"""
    def __init__(self, dataset_path, split, image_processor, chunk_id=0, n_chunks=1):
        self.dataset_path = dataset_path
        self.split = split
        self.image_dir = os.path.join(self.dataset_path, f"{self.split}2017")
        self.image_ids = [fname.split(".jpg")[0] for fname in os.listdir(self.image_dir) if fname.endswith(".jpg")]
        self.annotation_path = os.path.join(self.dataset_path, "annotations", f"captions_{self.split}2017.json")
        self.image_processor = image_processor

        self.chunk_id = chunk_id
        self.n_chunks = n_chunks

        # extract captions
        self.image_caption_map = dict()
        with open(self.annotation_path, 'r', encoding='utf-8') as af:
            annotations = json.load(af)["annotations"]
            for sample in annotations:
                image_id = '%012d' % sample['image_id']
                if image_id not in self.image_caption_map:
                    self.image_caption_map[image_id] = list()
                self.image_caption_map[image_id].append(sample["caption"])
        self.num_captions = torch.tensor([len(self.image_caption_map[image_id]) for image_id in self.image_caption_map]).sum().item()
        image_caption_list = [(image_id, captions) for image_id, captions in self.image_caption_map.items()]
        if n_chunks > 1: 
            # only run a portion of the entire set
            chunk_size = math.ceil(len(image_caption_list) / self.n_chunks)
            chunk_start = self.chunk_id * chunk_size
            total_length = len(image_caption_list)
            chunk_end = min(total_length, chunk_start + chunk_size)
            self.image_caption_list = image_caption_list[chunk_start: chunk_end]
        else:
            self.image_caption_list = image_caption_list
        # print("number of captions", self.num_captions)
    
    def __len__(self,):
        return len(self.image_caption_list)

    def __getitem__(self, idx):
        # image preprocessing
        image_id, captions = self.image_caption_list[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(image_path).convert("RGB")
        img = img_process(img, self.image_processor, image_aspect_ratio="square")
        img = img.squeeze(0) # [1, 3, 448, 448] -> [3, 448, 448]
        # print("in get item, image size", img.size())
        # text
        # a list of texts that maps to the same image


        return (img, captions, image_path)


def list_string_collate_fn(batch):
    images, caption_lists, image_paths = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = list(caption_lists)
    return {
        "images": images,
        "captions": captions,
        "image_paths": image_paths
    }

class BLIP3oT2IRetrievalPipeline:
    """
    This class is modified from the diffusion image generation pipeline 
    (blip3o inference, pipeline_llava_gen)
    """
    def __init__(self, blip3o_model, tokenizer, dataset_name, chunk_id=0, n_chunks=1):
        self.blip3o_inference_model = blip3o_model
        self.dtype = next(self.blip3o_inference_model.parameters()).dtype
        self.device = next(self.blip3o_inference_model.parameters()).device
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        # print("dtype", self.dtype, "device", self.device) # float 16
        self.tokenizer = tokenizer
        if "coco" in dataset_name:
            split = dataset_name.split('-')[-1]
            self.dataset = COCORetrieval(
                dataset_path = COCO_PATH,
                split = split,
                image_processor = self.blip3o_inference_model.get_gen_vision_tower().image_processor,
                chunk_id = chunk_id,
                n_chunks = n_chunks,
            )
            print("number of images", len(self.dataset))
        else:
            raise NotImplementedError
        
        self.dataloader = DataLoader(self.dataset, shuffle=False, batch_size=16, collate_fn=list_string_collate_fn)

        self.chunk_id = chunk_id
        self.n_chunks = n_chunks
        self.all_metaquery = []
        


    def encode_images(self, images):
        images_feature = self.blip3o_inference_model.encode_image(images) # [bsz, dim(1792), 8, 8] 8 is the latent size
        num_img, c = images_feature.size()[:2]
        images_feature = images_feature.view(num_img, c, -1).permute(0,2,1).contiguous()
        images_emb = images_feature.mean(dim=1)
        # print("image embedding size", images_emb.size())
        return images_emb

    def encode_texts(self, texts):
        """This function calls the blip3o inference model to encode text and diffuse CLIP features"""
        # ! the inference pipeline supports only 1 generation at a time, so need this loop over texts
        batch_texts_embed = []
        for cap in texts:
            prompt = add_template([f"Please generate image based on the following caption: {cap}"]) # a list
            # ! load metaquery for debug purpose
            # gen_clip_features, metaquery = self.blip3o_inference_model.generate_image(
            #     text = prompt, tokenizer = self.tokenizer
            # ) # [1 x 64 x 1792], 1792 is the feature dim, metaquerry [bsz(1) x 64 x 3584]
            gen_clip_features = self.blip3o_inference_model.generate_image(
                text = prompt, tokenizer = self.tokenizer
            ) # [1 x 64 x 1792], 1792 is the feature dim, metaquerry [bsz(1) x 64 x 3584]
            text_emb = gen_clip_features.mean(dim=1)
            batch_texts_embed.append(text_emb)
            # self.all_metaquery.append(metaquery)
        return torch.cat(batch_texts_embed, dim=0) # [bsz x feature_dim]
    
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

    def embed(self, amp=True, out_dir="./embeddings"):
        # list of batch of images embedding
        batch_images_emb_list = []
        # list of batch of text embedding
        batch_texts_emb_list = []
        # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
        texts_image_index = []
        dataloader = dataloader_with_indices(self.dataloader)
        for batch, inds in tqdm(dataloader):
            batch_images = batch["images"]
            batch_texts = batch["captions"]
            batch_image_paths = batch["image_paths"]
            batch_images = batch_images.to(self.device)
            
            batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
            batch_texts_flatten = [text for list_texts in batch_texts for text in list_texts]
            # # ! not the real text, but the image+prompt, treated as text emb
            # # ! do not send batch_image because it is preprocessed for eva_clip vision tower, not for qwen.
            # batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts)]

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=amp):
                with torch.no_grad():
                    batch_images_emb = F.normalize(self.encode_images(batch_images), dim=-1)
                    
                    batch_texts_emb = F.normalize(self.encode_texts(batch_texts_flatten), dim=-1)
                    # # ! not the real text, but the image+prompt, treated as text emb
                    # # ! do not send batch_image because it is preprocessed for eva_clip vision tower, not for qwen.
                    # batch_texts_emb = F.normalize(
                    #     self.encode_images_prompt(
                    #         batch_image_paths, 
                    #         prompt=""
                    #     ), 
                    #     dim=-1
                    # )

                
            # print(batch_images_emb.size(), batch_images_emb.dtype) # [bsz, 1792] torch.float32
            # print(batch_texts_emb.size(), batch_texts_emb.dtype) # [bsz, 1792] torch.float32
            batch_images_emb_list.append(batch_images_emb.to(dtype=torch.float32).cpu())
            batch_texts_emb_list.append(batch_texts_emb.to(dtype=torch.float32).cpu())
            texts_image_index.extend(batch_texts_image_index)
            
            # exit()
            # break
        
        batch_size = len(batch_images_emb_list[0])

        # concatenate all embeddings
        images_emb = torch.cat(batch_images_emb_list)
        texts_emb = torch.cat(batch_texts_emb_list)

        scores  = texts_emb @ images_emb.t()
        print("partial score size", scores.size())

        positive_pairs = torch.zeros_like(scores, dtype=bool) # T x I
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True

        # save embeddings and positive pairs, partially
        # os.makedirs(out_dir, exist_ok=True)
        torch.save(images_emb, os.path.join(out_dir, f"image_embed_chunk{self.chunk_id}_of_{self.n_chunks}.pt"))
        torch.save(texts_emb, os.path.join(out_dir, f"text_embed_chunk{self.chunk_id}_of_{self.n_chunks}.pt"))
        torch.save(positive_pairs, os.path.join(out_dir, f"positive_pairs_chunk{self.chunk_id}_of_{self.n_chunks}.pt"))
        print(f"finish embedding for chunk {self.chunk_id} out of {self.n_chunks}")

        #! DEBUG: check meta querry similarity
        all_metaquery = torch.cat(self.all_metaquery)
        mean_sim, std_sim = stats_metaquery(all_metaquery)
        torch.save(mean_sim, os.path.join(out_dir, f"metaquery_mean_chunk{self.chunk_id}_of_{self.n_chunks}.pt"))
        torch.save(std_sim, os.path.join(out_dir, f"metaquery_std_chunk{self.chunk_id}_of_{self.n_chunks}.pt"))
        print("saved metaquery stats")

    def evaluate_chunks(self, out_dir, recall_k_list=[5]):
        print("Skipping data encoding, directly read pre computed embedding results from", out_dir)
        total_img_emb_list = []
        total_text_emb_list = []
        total_pair_chunk_list = []
        # ! check if metaqueries in each data point is similar
        total_mq_mean = []
        total_mq_std = []
        for chunk_id in range(self.n_chunks):
            img_emb_file = os.path.join(out_dir, f"image_embed_chunk{chunk_id}_of_{self.n_chunks}.pt")
            text_emb_file = os.path.join(out_dir, f"text_embed_chunk{chunk_id}_of_{self.n_chunks}.pt")
            positive_pair_file = os.path.join(out_dir, f"positive_pairs_chunk{chunk_id}_of_{self.n_chunks}.pt")
            if not os.path.exists(img_emb_file):
                print("no file", img_emb_file, "skip chunk")
                continue
            total_img_emb_list.append(torch.load(img_emb_file))
            total_text_emb_list.append(torch.load(text_emb_file))
            total_pair_chunk_list.append(torch.load(positive_pair_file))

            # ! if there are mean and std
            mean_file = os.path.join(out_dir, f"metaquery_mean_chunk{chunk_id}_of_{self.n_chunks}.pt")
            std_file = os.path.join(out_dir, f"metaquery_std_chunk{chunk_id}_of_{self.n_chunks}.pt")
            if os.path.exists(mean_file):
                total_mq_mean.append(torch.load(mean_file))
            if os.path.exists(std_file):
                total_mq_std.append(torch.load(std_file))

        # ! if there are mean and std
        if len(total_mq_mean) != 0:
            mean_sim = torch.cat(total_mq_mean)
            plt.figure(1)
            plt.hist(mean_sim.cpu().numpy(), bins=30)
            plt.title("Distribution of Mean Pairwise Similarity per Sample")
            plt.xlabel(f"Mean Similarity total {mean_sim.size()}")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(out_dir, "mean_pair_hist.png"))
            plt.close()
        
        if len(total_mq_std) != 0:
            std_sim = torch.cat(total_mq_std)
            plt.figure(2)
            plt.hist(std_sim.cpu().numpy(), bins=30)
            plt.title("Distribution of Std of Pairwise Similarity per Sample")
            plt.xlabel(f"Std of Similarity total {std_sim.size()}")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(out_dir, "std_pair_hist.png"))
            plt.close()
        
        total_img_emb = torch.cat(total_img_emb_list, dim=0)
        total_text_emb = torch.cat(total_text_emb_list, dim=0)
        scores  = total_text_emb @ total_img_emb.t()
        print("scores matrix size, text x images = rol x col", scores.size())

        positive_pairs = torch.zeros_like(scores, dtype=bool)
        # populate using chunk positive pairs
        col = 0
        row = 0
        for chunk_pairs in total_pair_chunk_list:
            r, c = chunk_pairs.size()
            positive_pairs[row:row+r, col:col+c] = chunk_pairs
            col += c
            row += r
        
        batch_size =16
        metrics = {}
        for recall_k in recall_k_list:
            # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
            # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
            # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
            # for each image, that number will be greater than 1 for text retrieval.
            # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
            # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
            # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
            # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
            # it over the dataset.
            metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, self.device, k=recall_k)>0).float().mean().item()
            metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, self.device, k=recall_k)>0).float().mean().item()

        return metrics
        

        

    def evaluate(self, amp=True, recall_k_list=[5]):
        """
        main evaluate function
        refer to https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
        """
        # list of batch of images embedding
        batch_images_emb_list = []
        # list of batch of text embedding
        batch_texts_emb_list = []
        # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
        texts_image_index = []
        dataloader = dataloader_with_indices(self.dataloader)
        for batch, inds in tqdm(dataloader):
            batch_images = batch["images"]
            batch_texts = batch["captions"]
            batch_image_paths = batch["image_paths"]
            batch_images = batch_images.to(self.device)
            batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
            batch_texts_flatten = [text for list_texts in batch_texts for text in list_texts]

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=amp):
                with torch.no_grad():
                    batch_images_emb = F.normalize(self.encode_images(batch_images), dim=-1)
                    # batch_texts_emb = F.normalize(self.encode_texts(batch_texts_flatten), dim=-1) 
                    # ! not the real text, but the image+prompt, treated as text emb
                    # ! do not send batch_image because it is preprocessed for eva_clip vision tower, not for qwen.
                    batch_texts_emb = F.normalize(self.encode_images_prompt(batch_image_paths, prompt="Please summarize the provided image in a few words"), dim=-1) 
                
            # print(batch_images_emb.size(), batch_images_emb.dtype) # [bsz, 1792] torch.float32
            # print(batch_texts_emb.size(), batch_texts_emb.dtype) # [bsz, 1792] torch.float32
            batch_images_emb_list.append(batch_images_emb.to(dtype=torch.float32).cpu())
            batch_texts_emb_list.append(batch_texts_emb.to(dtype=torch.float32).cpu())
            texts_image_index.extend(batch_texts_image_index)
            
            # break
        
        batch_size = len(batch_images_emb_list[0])

        # concatenate all embeddings
        images_emb = torch.cat(batch_images_emb_list)
        texts_emb = torch.cat(batch_texts_emb_list)

        # get the score for each text and image pair
        scores  = texts_emb @ images_emb.t()
        print("scores matrix size, text x images = rol x col", scores.size())

        # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True
        metrics = {}
        for recall_k in recall_k_list:
            # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
            # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
            # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
            # for each image, that number will be greater than 1 for text retrieval.
            # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
            # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
            # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
            # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
            # it over the dataset.
            metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, self.device, k=recall_k)>0).float().mean().item()
            metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, self.device, k=recall_k)>0).float().mean().item()

        return metrics


def main(opt):
    device_1 = 0
    # set_global_seed(seed=42)
    set_global_seed(seed=int(opt.seed))
    model_path = "/group-volume/juexiaozhang/hf_cache/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/3c307c309d94a594efea23afc54ecebe82798b6a/"

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    # print("model path", model_path)
    # print("model name", model_name)
    tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)

    print(f"run eval for data {opt.data_name}, chunk id {opt.index} of total chunk {opt.n_chunks}")
    retrieval_pipeline = BLIP3oT2IRetrievalPipeline(
        blip3o_model = multi_model,
        tokenizer = tokenizer,
        dataset_name = opt.data_name,
        chunk_id = opt.index,
        n_chunks = opt.n_chunks,
    )
    seed_dir = os.path.join(opt.out_dir, str(opt.seed))
    os.makedirs(seed_dir, exist_ok=True)
    print("saving results to", seed_dir)
    # retrieval_pipeline.embed(out_dir=seed_dir)

    print(retrieval_pipeline.evaluate_chunks(out_dir=seed_dir, recall_k_list = [1, 5, 10]))
    
    # print(retrieval_pipeline.evaluate(recall_k_list = [1, 5, 10]))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
