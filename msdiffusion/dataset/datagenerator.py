import random
import re

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor, AutoImageProcessor, CLIPVisionModelWithProjection


# Dataset
def get_word_idx(tokenizer, word, prompt):
    # words = prompt.split()
    words = re.findall(r'\b[\w\'-]+\b', prompt)
    start = 1
    end = 0
    for w in words:
        if w == word:
            end = start + len(tokenizer.encode(w)) - 2
            break
        else:
            start += len(tokenizer.encode(w)) - 2

    if end == 0:
        return [0, 0]

    return [start, end]


def get_phrase_idx(tokenizer, phrase, prompt, get_last_word=False, num=0):
    def is_equal_words(pr_words, ph_words):
        if len(pr_words) != len(ph_words):
            return False
        for pr_word, ph_word in zip(pr_words, ph_words):
            if "-"+ph_word not in pr_word and ph_word != re.sub(r'[.!?,:]$', '', pr_word):
                return False
        return True

    phrase_words = phrase.split()
    if len(phrase_words) == 0:
        return [0, 0], None
    if get_last_word:
        phrase_words = phrase_words[-1:]
    # prompt_words = re.findall(r'\b[\w\'-]+\b', prompt)
    prompt_words = prompt.split()
    start = 1
    end = 0
    res_words = phrase_words
    for i in range(len(prompt_words)):
        if is_equal_words(prompt_words[i:i+len(phrase_words)], phrase_words):
            if num != 0:
                # skip this one
                num -= 1
                continue
            end = start
            res_words = prompt_words[i:i+len(phrase_words)]
            res_words = [re.sub(r'[.!?,:]$', '', w) for w in res_words]
            prompt_words[i+len(phrase_words)-1] = res_words[-1]  # remove the last punctuation
            for j in range(i, i+len(phrase_words)):
                end += len(tokenizer.encode(prompt_words[j])) - 2
            break
        else:
            start += len(tokenizer.encode(prompt_words[i])) - 2

    if end == 0:
        return [0, 0], None

    return [start, end], res_words


def get_eot_idx(tokenizer, prompt):
    words = prompt.split()
    start = 1
    for w in words:
        start += len(tokenizer.encode(w)) - 2
    return start


def mask_image(img, mask):
    img = np.array(img)
    mask = np.array(mask)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    img = img * mask
    return Image.fromarray(img)


class MyDataset(Dataset):

    def __init__(self, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05,
                 ti_drop_rate=0.05, g_drop_rate=0.1, image_encoder_type="clip", max_rn=4, device="cuda", resize_ref=True):

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.g_drop_rate = g_drop_rate
        self.max_rn = max_rn
        self.device = device
        self.resize_ref = resize_ref

        # read your data here
        example_image_path = "examples/example_dog.jpg"
        example_image_file = Image.open(example_image_path)
        example_height, example_width = example_image_file.height, example_image_file.width
        example = {
            "datas": {
                "text": "a dog",
                "image_file": example_image_file,
                "reference_image_files": [example_image_file],
                "image_boxes": [[0, 0, example_width, example_height]],
                "reference_image_boxes": [[0, 0, example_width, example_height]],
                "reference_image_masks": None,
                "image_phrases": ["dog"],
            }
        }
        self.data = [example] * 8

        # resize the short edge
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if image_encoder_type == "clip":
            self.image_processor = CLIPImageProcessor()
        else:
            raise ValueError(f"Unsupported image encoder type: {image_encoder_type}")

    def _crop_image(self, image_tensor, crop_coords_top_left=None):
        if crop_coords_top_left is not None:
            top, left = crop_coords_top_left[0].item(), crop_coords_top_left[1].item()
            image = transforms.functional.crop(
                image_tensor, top=top, left=left, height=self.size, width=self.size
            )
        else:
            delta_h = image_tensor.shape[1] - self.size
            delta_w = image_tensor.shape[2] - self.size
            assert not all([delta_h, delta_w])

            if self.center_crop:
                top = delta_h // 2
                left = delta_w // 2
            else:
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            image = transforms.functional.crop(
                image_tensor, top=top, left=left, height=self.size, width=self.size
            )
            crop_coords_top_left = torch.tensor([top, left])

        return image, crop_coords_top_left

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["datas"]["text"]
        image_file = item["datas"]["image_file"]
        reference_image_files = item["datas"]["reference_image_files"]
        image_boxes = item["datas"]["image_boxes"]
        reference_image_boxes = item["datas"]["reference_image_boxes"]
        reference_image_masks = item["datas"]["reference_image_masks"]
        image_phrases = item["datas"]["image_phrases"]

        # read image
        if isinstance(image_file, str):
            raw_image = Image.open(image_file)
            raw_reference_images = [Image.open(f) for f in reference_image_files]
        else:
            raw_image = image_file
            raw_reference_images = reference_image_files

        # process gt image
        # original size
        original_width, original_height = raw_image.size
        scale = self.size / original_width if original_width < original_height else self.size / original_height
        original_size = torch.tensor([original_height, original_width])

        # box out the reference images if needed
        for i, box in enumerate(reference_image_boxes):
            if reference_image_masks is not None:
                raw_reference_images[i] = mask_image(raw_reference_images[i], reference_image_masks[i]).crop(box)
            else:
                raw_reference_images[i] = raw_reference_images[i].crop(box)

        # crop
        image_tensor = self.transform(raw_image.convert("RGB"))
        image, crop_coords_top_left = self._crop_image(image_tensor)
        image_boxes = torch.tensor(image_boxes) * scale
        image_boxes = image_boxes - torch.tensor([crop_coords_top_left[1], crop_coords_top_left[0],
                                                  crop_coords_top_left[1], crop_coords_top_left[0]])
        image_boxes = torch.clamp(image_boxes, min=0, max=self.size)
        image_boxes = image_boxes / self.size

        if self.resize_ref:
            raw_reference_images = [raw_image] if len(raw_reference_images) == 0 else raw_reference_images
            raw_reference_images = [img.resize((self.size, self.size), Image.BILINEAR) for img in raw_reference_images]
        if len(raw_reference_images) == 0:
            processed_images = self.image_processor(images=[raw_image], return_tensors="pt").pixel_values
        else:
            processed_images = self.image_processor(images=raw_reference_images, return_tensors="pt").pixel_values

        # drop box and phrase
        drop_grounding_tokens = 0
        rand_num = random.random()
        if rand_num < self.g_drop_rate:
            drop_grounding_tokens = 1
        
        # get phrase idx
        phrase_idxes = []
        image_phrases_ = []
        for phrase in image_phrases:
            phrase_idx, phrase_words = get_phrase_idx(self.tokenizer, phrase, text, get_last_word=False)
            phrase_idxes.append(phrase_idx)
            if phrase_words is not None:
                image_phrases_.append(" ".join(phrase_words))
            else:
                image_phrases_.append(phrase)
        phrase_idxes = torch.tensor(phrase_idxes, dtype=torch.int)
        image_phrases = image_phrases_

        # get eot idx
        eot_idx = get_eot_idx(self.tokenizer, text)
        eot_idx = torch.tensor([eot_idx], dtype=torch.int)
        eot_idxes = eot_idx.repeat(processed_images.shape[0])

        # get tokenized phrases
        image_phrase_input_ids = self.tokenizer(
            image_phrases,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # padding to max_rn
        cur_rn = processed_images.shape[0]
        if cur_rn < self.max_rn:
            padding_images = torch.zeros((self.max_rn - cur_rn, processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]))
            processed_images = torch.cat([processed_images, padding_images], dim=0)
            padding_boxes = torch.zeros((self.max_rn - cur_rn, 4))
            image_boxes = torch.cat([image_boxes, padding_boxes], dim=0)
            padding_phrase_ids = torch.zeros((self.max_rn - cur_rn, image_phrase_input_ids.shape[1]), dtype=image_phrase_input_ids.dtype)
            image_phrase_input_ids = torch.cat([image_phrase_input_ids, padding_phrase_ids], dim=0)
            padding_phrase_idxes = torch.zeros((self.max_rn - cur_rn, 2), dtype=phrase_idxes.dtype)
            phrase_idxes = torch.cat([phrase_idxes, padding_phrase_idxes], dim=0)
            padding_eot_idxes = torch.zeros(self.max_rn - cur_rn, dtype=eot_idxes.dtype)
            eot_idxes = torch.cat([eot_idxes, padding_eot_idxes], dim=0)
        # generate attention mask: note that sd pipeline do not use attention mask
        attention_mask = torch.ones(self.max_rn, dtype=torch.uint8)
        attention_mask[cur_rn:] = 0

        # drop text and image
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
            phrase_idxes = torch.zeros_like(phrase_idxes)
            eot_idx = torch.zeros_like(eot_idx)
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            phrase_idxes = torch.zeros_like(phrase_idxes)
            eot_idx = torch.zeros_like(eot_idx)
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids if self.tokenizer_2 is not None else torch.zeros((1, 1))

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "processed_images": processed_images,
            "rf_attention_mask": attention_mask,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "drop_grounding_tokens": drop_grounding_tokens,
            "image_boxes": image_boxes,
            "image_phrase_input_ids": image_phrase_input_ids,
            "phrase_idxes": phrase_idxes,
            "eot_idxes": eot_idxes,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    processed_images = torch.stack([example["processed_images"] for example in data])
    rf_attention_mask = torch.stack([example["rf_attention_mask"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    drop_grounding_tokens = [example["drop_grounding_tokens"] for example in data]
    image_boxes = torch.stack([example["image_boxes"] for example in data])
    image_phrase_input_ids = torch.stack([example["image_phrase_input_ids"] for example in data])
    phrase_idxes = torch.stack([example["phrase_idxes"] for example in data])
    eot_idxes = torch.stack([example["eot_idxes"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "processed_images": processed_images,
        "rf_attention_mask": rf_attention_mask,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "drop_grounding_tokens": drop_grounding_tokens,
        "image_boxes": image_boxes,
        "image_phrase_input_ids": image_phrase_input_ids,
        "phrase_idxes": phrase_idxes,
        "eot_idxes": eot_idxes,
    }
