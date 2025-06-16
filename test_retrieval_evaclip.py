""" 
retrieval test

directly modified from inference.py
"""


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
import torch.nn as nn

import re, random

from blip3o.model.multimodal_encoder.dev_eva_clip.eva_clip import create_model_and_transforms as evaclip_create_model_and_transforms
from blip3o.model.multimodal_encoder.dev_eva_clip.eva_clip import get_tokenizer as evaclip_get_tokenizer
# model_path = "/group-volume/juexiaozhang/hf_cache/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/3c307c309d94a594efea23afc54ecebe82798b6a/"


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



device_1 = 0

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

# * -------------------------- code (BELOW) from LAION AI CLIP Benchmark --------------------------- * #

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
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
    def __init__(self, dataset_dir, image_processor):
        self.image_dir = os.path.join(dataset_dir, "Images")
        self.all_image_ids = [os.path.basename(f).split(".jpg")[0] for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.caption_file = os.path.join(dataset_dir, "captions.txt")

        self.image_processor = image_processor

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
        # image preprocessing
        image_id, captions = self.image_caption_list[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = self.image_processor(Image.open(image_path))

        return (img, captions)

class COCORetrieval(Dataset):
    """Dataset for COCO retrieval"""
    def __init__(self, dataset_path, split, image_processor):
        self.dataset_path = dataset_path
        self.split = split
        self.image_dir = os.path.join(self.dataset_path, f"{self.split}2017")
        self.image_ids = [fname.split(".jpg")[0] for fname in os.listdir(self.image_dir) if fname.endswith(".jpg")]
        self.annotation_path = os.path.join(self.dataset_path, "annotations", f"captions_{self.split}2017.json")
        self.image_processor = image_processor

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
        self.image_caption_list = [(image_id, captions) for image_id, captions in self.image_caption_map.items()]
        print("number of captions", self.num_captions)
    
    def __len__(self,):
        return len(self.image_caption_list)

    def __getitem__(self, idx):
        # image preprocessing
        image_id, captions = self.image_caption_list[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = self.image_processor(Image.open(image_path))


        return (img, captions)


def list_string_collate_fn(batch):
    images, caption_lists = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = list(caption_lists)
    return images, captions

class T2IRetrievalPipeline:
    """
    This class is modified from the diffusion image generation pipeline 
    (blip3o inference, pipeline_llava_gen)
    """
    def __init__(self, model, tokenizer, image_processor, dataset_name, eva_clip_head=None):
        self.model = model
        self.dtype = next(self.model.parameters()).dtype
        self.device = next(self.model.parameters()).device
        print("dtype", self.dtype, "device", self.device) # float 16
        self.tokenizer = tokenizer
        if "coco" in dataset_name:
            split = dataset_name.split('-')[-1]
            self.dataset = COCORetrieval(
                dataset_path = COCO_PATH,
                split = split,
                image_processor = image_processor
            )
            print("number of images", len(self.dataset))
        elif "flickr" in dataset_name:
            self.dataset = Flickr30kT2IRetrieval(
                dataset_dir = FLICKR30k_PATH,
                image_processor = image_processor,
            )
            print("number of images", len(self.dataset))
        else:
            raise NotImplementedError
        
        self.dataloader = DataLoader(self.dataset, shuffle=False, batch_size=128, collate_fn=list_string_collate_fn)
        # TODO
        # self.text_pooling = "mean"
        # self.image_pooling = "mean"
        self.head = None
        if eva_clip_head is not None:
            print(f"load norm and head from {eva_clip_head}")
            self.head = nn.Sequential(
                nn.LayerNorm(1792),
                nn.Linear(1792, 1024, bias=True)
            )
            self.head[1].load_state_dict(
                torch.load(os.path.join(eva_clip_head, "eva_clip_head.pth")),
                strict = True,
            )
            self.head[0].load_state_dict(
                torch.load(os.path.join(eva_clip_head, "eva_clip_norm.pth")),
                strict = True,
            )
            self.head = self.head.to(self.device)
        


    def encode_images(self, images):
        """Original EVA CLIP uses CLS as image repr"""
        images_feature = self.model.encode_image(images, return_all_features=True)
        # print(images_feature.size())
        images_feature = images_feature[:, 1:].mean(dim=1) # * this makes it same compute graph as the blip3o's gen_vision_tower output
        # print("average all tokens except the cls", images_feature.size())
        return images_feature

    def encode_texts(self, texts):
        """Original EVA CLIP uses EOT as text repr"""
        texts = tokenizer(texts).to(self.device)
        # do not average padding tokens
        padding_mask = (texts != 0).unsqueeze(-1).float()
        actual_lengths = padding_mask.sum(dim=1).clamp(min=1)

        texts_feature = self.model.encode_text(texts, return_all_features=False)
        # print("text feature", texts_feature.size())
        # average non-pad tokens
        # texts_feature = (texts_feature * padding_mask).sum(dim=1) / actual_lengths
        return texts_feature

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
        for batch_images, batch_texts, inds in tqdm(dataloader):
            batch_images = batch_images.to(self.device)
            # print("batch image", batch_images.size(), batch_images.dtype)
            # print(len(batch_texts), batch_texts[0])
            batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
            batch_texts_flatten = [text for list_texts in batch_texts for text in list_texts]

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=amp):
                with torch.no_grad():
                    batch_images_emb = F.normalize(self.encode_images(batch_images), dim=-1)
                    batch_texts_emb = F.normalize(self.encode_texts(batch_texts_flatten), dim=-1) 
                
            # print(batch_images_emb.size(), batch_images_emb.dtype) # [bsz, 1792] torch.float32
            # print(batch_texts_emb.size(), batch_texts_emb.dtype) # [bsz, 1792] torch.float32
            batch_images_emb_list.append(batch_images_emb.to(dtype=torch.float32).cpu())
            batch_texts_emb_list.append(batch_texts_emb.to(dtype=torch.float32).cpu())
            texts_image_index.extend(batch_texts_image_index)

        
        batch_size = len(batch_images_emb_list[0])

        # concatenate all embeddings
        images_emb = torch.cat(batch_images_emb_list)
        texts_emb = torch.cat(batch_texts_emb_list)

        # # ! save the embdding
        # os.makedirs("eva_clip_embedding/", exist_ok=True)
        # torch.save(images_emb, "eva_clip_embedding/images_emb_norm_proj_nocls_mean.pt")
        # # exit()
        # torch.save(texts_emb, "eva_clip_embedding/texts_emb.pt")

        # get the score for each text and image pair
        scores  = texts_emb @ images_emb.t()
        print("scores matrix size, text x images = rol x col", scores.size())

        # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True
        # torch.save(positive_pairs, "eva_clip_embedding/positive_pairs.pt")

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


    def evaluate_emb(self, image_emb_path, text_emb_path, positive_pair_path=None, recall_k_list=[5], apply_head_on_text = False, apply_head_on_vision = False):
        print("Skipping data encoding, directly read pre computed embedding results")
        if image_emb_path.endswith(".pt"):
            total_img_emb = torch.load(image_emb_path)
        else:
            print(f"{image_emb_path} is a directory, read the chunks")
            total_img_emb_list = []
            n_chunks = 8
            for i in range(n_chunks):
                img_emb_file = os.path.join(image_emb_path, f"image_embed_chunk{i}_of_{n_chunks}.pt")
                img_emb = torch.load(img_emb_file)
                total_img_emb_list.append(img_emb)
            total_img_emb = torch.cat(total_img_emb_list, dim=0)
        
        if text_emb_path.endswith(".pt"):
            total_text_emb = torch.load(text_emb_path)
        else:
            print(f"{text_emb_path} is a directory, read the chunks")
            total_text_emb_list = []
            n_chunks=8
            for i in range(n_chunks):
                text_emb_file = os.path.join(text_emb_path, f"text_embed_chunk{i}_of_{n_chunks}.pt")
                text_emb = torch.load(text_emb_file)
                total_text_emb_list.append(text_emb)
            total_text_emb = torch.cat(total_text_emb_list, dim=0)

        if self.head is not None:
            # project the embedding to the head dimension
            if apply_head_on_text:
                print("before head", total_text_emb.size())
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
                    with torch.no_grad():
                        total_text_emb = self.head(total_text_emb.to(self.device)).cpu()
                print("after head", total_text_emb.size())
            if apply_head_on_vision:
                print("before head", total_img_emb.size())
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
                    with torch.no_grad():
                        total_img_emb = self.head(total_img_emb.to(self.device)).cpu()
                print("after head", total_img_emb.size())

        total_text_emb = F.normalize(total_text_emb, dim=-1)
        total_img_emb = F.normalize(total_img_emb, dim=-1)
        print(total_img_emb.size(), total_text_emb.size())

        scores  = total_text_emb @ total_img_emb.t()
        print("scores matrix size, text x images = rol x col", scores.size())


        if positive_pair_path is not None:
            if not positive_pair_path.endswith(".pt"):
                n_chunks = 8
                total_pair_chunk_list = []
                for i in range(n_chunks):
                    positive_pair_file = os.path.join(positive_pair_path, f"positive_pairs_chunk{i}_of_{n_chunks}.pt")
                    total_pair_chunk_list.append(torch.load(positive_pair_file))
                positive_pairs = torch.zeros_like(scores, dtype=bool)
                # populate using chunk positive pairs
                col = 0
                row = 0
                for chunk_pairs in total_pair_chunk_list:
                    r, c = chunk_pairs.size()
                    positive_pairs[row:row+r, col:col+c] = chunk_pairs
                    col += c
                    row += r
            else:
                positive_pairs = torch.load(positive_pair_path)
        else:
            positive_pairs = torch.eye(scores.size(0), dtype=bool)
            # compute euclidean distance
            euclidean_distance = torch.norm(total_img_emb - total_text_emb, dim=1)
            print("euc", euclidean_distance.size(), euclidean_distance.mean())
            print("euc", euclidean_distance.min(), euclidean_distance.max())

        
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


data_name = sys.argv[1]
# eva clip experiment
EVACLIP_model_name = "EVA02-CLIP-bigE-14-plus"
EVACLIP_pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_E_psz14_plus_s9B.pt"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = evaclip_create_model_and_transforms(EVACLIP_model_name, EVACLIP_pretrained, force_custom_clip=True)
tokenizer = evaclip_get_tokenizer(EVACLIP_model_name)
model = model.to(device)
print("model device", device)
# # print(model)
# print("******************************")
# print(model.visual.norm)
# print(model.visual.norm.state_dict())
# torch.save(model.visual.norm.state_dict(), "eva_clip_norm.pth")
# # print("******************************")
# # print(model.visual.fc_norm)
# # print(model.visual.fc_norm.state_dict())
# print("******************************")
# print(model.visual.head)
# print(model.visual.head.state_dict()["weight"].size(), model.visual.head.state_dict()["bias"].size())
# torch.save(model.visual.head.state_dict(), "eva_clip_head.pth")
# exit()


# image_path = "A photo of cute cat.png"
# caption = ["a diagram", "a dog", "a cat"]
# image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
# text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
# dtype = next(model.parameters()).dtype
# with torch.no_grad():
#     with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)


print("run eval for data", data_name)
retrieval_pipeline = T2IRetrievalPipeline(
    model = model,
    tokenizer = tokenizer,
    image_processor = preprocess,
    dataset_name = data_name,
    eva_clip_head = "./eva_clip_head/"
)

print(retrieval_pipeline.evaluate(recall_k_list = [1, 5, 10]))

# print(
#     retrieval_pipeline.evaluate_emb(
#         text_emb_path="embedding_coco_val_consqchunk/", 
#         image_emb_path="eva_clip_embedding/images_emb_nocls_mean.pt",
#         positive_pair_path = "eva_clip_embedding/positive_pairs.pt",
#         recall_k_list = [1, 5, 10],
#         apply_head_on_vision = False,
#         apply_head_on_text = False,
#     )
# )







    

    


    