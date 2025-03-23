import os
from PIL import Image
import torch
import numpy as np
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

from llava_methods import *
from blip_methods import *
from utils import *
from info import *


def vicrop_qa(
    model_name, method_name, image_path, question, model, processor, short_question
):
    """
    Performs visual cropping and question answering using different attention methods.

    This function processes an image with a specified model (LLaVA or BLIP) and attention method,
    generates an attention map, crops the image based on the attention, and then performs
    question answering on both the original and cropped images.

    Args:
        model_name: String indicating which model to use ("llava" or "blip")
        method_name: String indicating which attention method to use (e.g., "grad_att", "rel_att", "pure_grad")
        image_path: Path to the input image file
        question: The full question to ask about the image
        model: The loaded model instance (LLaVA or BLIP)
        processor: The processor for the corresponding model
        short_question: A shortened version of the question for attention computation (only used in Vstar)

    Returns:
        tuple: (original_answer, cropped_answer, bounding_box)
            - original_answer: Model's answer using the full image
            - cropped_answer: Model's answer using the full image and the cropped image
            - bounding_box: The coordinates of the crop (left, top, right, bottom)
    """

    if model_name == "llava":
        bbox_size = 336
    elif model_name == "blip":
        bbox_size = 224

    image = Image.open(image_path).convert("RGB")
    model.eval()

    general_question = "Write a general description of the image."

    if model_name == "llava":
        short_prompt = f"<image>\nUSER: {short_question} Answer the question using a single word or phrase.\nASSISTANT:"
        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"

        inputs = processor(image, prompt, return_tensors="pt", padding=True).to(
            model.device, torch.bfloat16
        )
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = [
            i.split("ASSISTANT: ")[1]
            for i in processor.batch_decode(
                ori_generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ][0]

        del inputs
        torch.cuda.empty_cache()

        if method_name == "grad_att":
            att_map = gradient_attention_llava(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        elif method_name == "grad_att_high":
            att_maps = high_res(
                gradient_attention_llava,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)

        # ------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == "rel_att":
            att_map = rel_attention_llava(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        elif method_name == "rel_att_high":
            att_maps = high_res(
                rel_attention_llava,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)

        # ------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == "pure_grad":
            grad = pure_gradient_llava(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)

        elif method_name == "pure_grad_high":
            grads = high_res(
                pure_gradient_llava,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(grads, image.size, bbox_size)

        crop_image = image.crop(bbox)

        multi_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        multi_inputs = processor(
            multi_prompt, [image, crop_image], return_tensors="pt", padding=True
        ).to(model.device, torch.bfloat16)

        multi_generate_ids = model.generate(
            **multi_inputs, max_new_tokens=20, do_sample=False
        )
        multi_generation = [
            i.split("ASSISTANT: ")[1]
            for i in processor.batch_decode(
                multi_generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ][0]

        return ori_generation, multi_generation, bbox

    elif model_name == "blip":
        short_prompt = f"Question: {short_question} Short answer:"
        prompt = f"Question: {question} Short answer:"
        general_prompt = f"Question: {general_question} Short answer:"

        inputs = processor(
            images=image, text=prompt, return_tensors="pt", padding=True
        ).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = processor.batch_decode(
            ori_generate_ids, skip_special_tokens=True
        )[0]

        del inputs
        torch.cuda.empty_cache()

        if method_name == "grad_att":
            att_map = gradient_attention_blip(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        elif method_name == "grad_att_high":
            att_maps = high_res(
                gradient_attention_blip,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)

        # ------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == "rel_att":
            att_map = rel_attention_blip(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        elif method_name == "rel_att_high":
            att_map = high_res(
                rel_attention_blip,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        # ------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == "pure_grad":
            grad = pure_gradient_blip(
                image, short_prompt, general_prompt, model, processor
            )
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)

        elif method_name == "pure_grad_high":
            grad = high_res(
                pure_gradient_blip,
                image,
                short_prompt,
                general_prompt,
                model,
                processor,
            )
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)

        crop_image = image.crop(bbox)

        multi_inputs = processor(
            images=[image, crop_image], text=prompt, return_tensors="pt", padding=True
        ).to(model.device, torch.bfloat16)

        multi_generate_ids = model.generate(
            **multi_inputs, max_new_tokens=20, do_sample=False
        )
        multi_generation = processor.batch_decode(
            multi_generate_ids, skip_special_tokens=True
        )[0]

        return ori_generation, multi_generation, bbox


def reweight_qa(
    model_name, method_name, image_path, question, model, processor, short_question
):
    """
    Direct reweight based approach
    """

    image = Image.open(image_path).convert("RGB")
    model.eval()
    general_question = "Write a general description of the image."

    if model_name == "llava":
        short_prompt = f"<image>\nUSER: {short_question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"
        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"

        inputs = processor(image, prompt, return_tensors="pt", padding=True).to(
            model.device, torch.bfloat16
        )
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = [
            i.split("ASSISTANT: ")[1]
            for i in processor.batch_decode(
                ori_generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ][0]

        del inputs
        torch.cuda.empty_cache()

        if method_name == "simple_reweight":
            att_map = rel_attention_llava(
                image, short_prompt, general_prompt, model, processor
            )

            visual_token_weights = get_visual_token_weight(att_map, 0.6, "linear", 0.0)
            # print(multi_inputs.input_ids.shape)

            second_prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            second_inputs = processor(
                image, second_prompt, return_tensors="pt", padding=True
            ).to(model.device, torch.bfloat16)
            second_inputs_embeds = manual_embed_inputs(
                model,
                second_inputs.input_ids,
                second_inputs.pixel_values,
                visual_token_weights,
            )

            modified_inputs = {
                "inputs_embeds": second_inputs_embeds,
                "attention_mask": second_inputs.attention_mask,
            }
        elif method_name == "duplicate_reweight":
            att_map = rel_attention_llava(
                image, short_prompt, general_prompt, model, processor
            )
            visual_token_weights = get_visual_token_weight(att_map, 0.6, "linear", 0.0)

            # second_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            # second_inputs = processor(
            #     [image, image], second_prompt, return_tensors="pt", padding=True
            # ).to(model.device, torch.bfloat16)
            second_prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            second_inputs = processor(
                image, second_prompt, return_tensors="pt", padding=True
            ).to(model.device, torch.bfloat16)
            # print(second_inputs.input_ids.shape)
            # breakpoint()
            # modified_inputs = None
            second_inputs_embeds = manual_embed_inputs(
                model,
                second_inputs.input_ids,
                second_inputs.pixel_values,
                visual_token_weights,
            )
            modified_inputs = {
                "inputs_embeds": second_inputs_embeds,
                "attention_mask": second_inputs.attention_mask,
            }
        elif method_name == "contrastive_reweight":
            att_map = contrastive_attention_llava(
                image, general_prompt, model, processor
            )
            visual_token_weights = get_visual_token_weight(att_map, 0.6, "linear", 0.0)
            second_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
            second_inputs = processor(
                [image, image], second_prompt, return_tensors="pt", padding=True
            ).to(model.device, torch.bfloat16)
            # print(second_inputs.input_ids.shape)
            second_inputs_embeds = manual_embed_inputs(
                model,
                second_inputs.input_ids,
                second_inputs.pixel_values,
                visual_token_weights,
            )
            modified_inputs = {
                "inputs_embeds": second_inputs_embeds,
                "attention_mask": second_inputs.attention_mask,
            }
        else:
            raise ValueError(f"Method {method_name} not implemented")
        # print(multi_inputs_embeds.shape)

        modified_generate_ids = model.generate(
            **modified_inputs, max_new_tokens=20, do_sample=False
        )
        # print(processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        modified_generation = processor.batch_decode(
            modified_generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return ori_generation, modified_generation, None

    elif model_name == "blip":
        raise NotImplementedError("BLIP does not support reweighting")


def main(args):
    """
    Main function to run the visual cropping and question answering pipeline.

    This function loads the specified model and processor, processes the dataset,
    applies the visual cropping and question answering to each data point,
    and saves the results to a JSON file.

    Args:
        args: An argparse.Namespace object containing the following attributes:
            - model: String indicating which model to use ("llava" or "blip")
            - model_id: The model identifier for loading from HuggingFace
            - device: The device to run the model on ("cuda" or "cpu")
            - question_path: Path to the question dataset
            - image_path: Path to the directory containing images
            - task: The task identifier
            - method: The attention method to use
            - output_path: Path to save the results
            - total_chunks: Total number of chunks to split the dataset into
            - chunk_id: The ID of the current chunk to process

    Returns:
        None: Results are saved to the specified output file
    """

    if args.model == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id)
    elif args.model == "blip":
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(args.device)
        processor = InstructBlipProcessor.from_pretrained(args.model_id)

    if os.path.exists(args.question_path):
        with open(args.question_path, "r") as f:
            whole_data = json.load(f)
    else:
        whole_data = list(load_dataset(args.question_path)["test"])

    for data in whole_data:
        data["image_path"] = os.path.join(args.image_path, data["image_path"])

    splited_data = np.array_split(whole_data, args.total_chunks)

    data = splited_data[args.chunk_id]

    new_datas = []

    for d in tqdm(data, desc="Processing", ncols=100):
        question = d["question"]
        image_path = d["image_path"]
        if "short_question" in d:
            short_question = d["short_question"]
        else:
            short_question = d["question"]

        ori_generation, crop_generation, bbox = reweight_qa(
            args.model,
            args.method,
            image_path,
            question,
            model,
            processor,
            short_question,
        )

        d["original_answer"] = ori_generation
        d["crop_answer"] = crop_generation
        d["bbox"] = bbox

        new_datas.append(d)

    out_put_dir = os.path.dirname(args.output_path)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            old_datas = json.load(f)
        new_datas = old_datas + new_datas

    with open(args.output_path, "w") as f:
        json.dump(new_datas, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llava", choices=model_to_fullname.keys()
    )
    parser.add_argument(
        "--task", type=str, default="textvqa", choices=task_to_question_path.keys()
    )
    parser.add_argument(
        "--method",
        type=str,
        default="new",
        choices=[
            "rel_att",
            "pure_grad",
            "grad_att",
            "grad",
            "rel_att_high",
            "pure_grad_high",
            "grad_att_high",
            "grad_high",
            "simple_reweight",
            "duplicate_reweight",
            "contrastive_reweight",
        ],
    )
    parser.add_argument("--save_path", type=str, default="./playground/data/results")
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    output_name = f"{args.model}-{args.task}-{args.method}.json"

    args.model_id = model_to_fullname[args.model]

    args.output_path = os.path.join(args.save_path, output_name)

    args.image_path = task_to_image_path[args.task]

    args.question_path = task_to_question_path[args.task]

    main(args)
