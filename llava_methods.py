import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *
from typing import Literal

# hyperparameters
NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
IMAGE_TOKEN_INDEX = 32000
ATT_LAYER = 14

def get_visual_token_weight(
    att_map,
    threshold,
    weighting_type: Literal["linear", "exp", "uniform"] | str = "linear",
    lowest_weight=0.0,
):
    att_map = torch.from_numpy(att_map).flatten()
    sorted_indices = torch.argsort(att_map, descending=True)
    num_tokens_to_keep = int(len(att_map) * threshold)
    weight_vision_token = torch.zeros_like(att_map, dtype=torch.float)
    weight_vision_token[sorted_indices[:num_tokens_to_keep]] = 1.0
    if weighting_type == "linear":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.linspace(1.0, lowest_weight, len(att_map) - num_tokens_to_keep)
    elif weighting_type == "exp":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.exp(torch.linspace(0, lowest_weight, len(sorted_indices) - num_tokens_to_keep))
    elif weighting_type == "uniform":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = lowest_weight
    else:
        raise ValueError(f"Invalid weighting type: {weighting_type}")
    return weight_vision_token


def manual_embed_inputs(model, input_ids, pixel_values, vision_token_weights=None):
    # Get input embeddings from the model
    # print(input_ids.shape)
    # print(pixel_values.shape)
    # print(vision_token_weights.shape)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    # Process image features
    image_features = model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy
    )
    # Find image token positions
    special_image_mask = (input_ids == model.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    # print(special_image_mask.shape)
    # breakpoint()
    # Apply vision token weights if provided
    if vision_token_weights is not None:
        # Ensure weights are on the same device as image features
        vision_token_weights = vision_token_weights.to(image_features.device)
        # Apply weights to image features
        if image_features.shape[0] == 1:
            image_features = image_features * vision_token_weights.unsqueeze(0).unsqueeze(-1)
        else:
            image_features[1:] = image_features[1:] * vision_token_weights.unsqueeze(0).unsqueeze(-1)

    
    # Replace image tokens with image features
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    
    return inputs_embeds


def simple_attention_llava(image, prompt, model, processor):
    """
    Simplest attention map from LLaVA model
    """
    inputs = processor(image, prompt, return_tensors="pt", padding=True).to(
        model.device, torch.bfloat16
    )
    pos = inputs["input_ids"][0].tolist().index(IMAGE_TOKEN_INDEX)

    all_layer_attn = model(**inputs, output_attentions=True)["attentions"]
    att_map = (
        torch.cat(
            [
                layer_attn[0, :, -1, pos : pos + NUM_IMG_TOKENS]
                for layer_attn in all_layer_attn
            ],
            dim=0,
        )
        .mean(
            dim=(
                0,
                1,
            )
        )
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    att_map = att_map.reshape(NUM_PATCHES, NUM_PATCHES)
    return att_map

def gradient_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates an attention map using gradient-weighted attention from LLaVA model.
    
    This function computes attention maps from the LLaVA model and weights them by their
    gradients with respect to the loss. It focuses on the attention paid to image tokens
    in the final token prediction, highlighting regions relevant to the prompt.
    
    Args:
        image: Input image to analyze
        prompt: Text prompt for which to generate attention
        general_prompt: General text prompt (not directly used in this function)
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the gradient-weighted attention map
    """
    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    
    # Compute loss
    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)
    
    # Compute attention and gradients
    attention = outputs.attentions[ATT_LAYER]
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])
    
    # Compute the attention maps
    att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    
    return att_map

def rel_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a relative attention map by comparing specific prompt attention to general prompt attention.
    
    This function computes attention maps for both a specific prompt and a general prompt in the LLaVA model,
    then calculates their ratio to highlight regions that are uniquely relevant to the specific prompt.
    It focuses on the attention paid to image tokens in the final token prediction.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate attention
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the relative attention map (specific/general)
    """
    # Prepare inputs for the prompt
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute attention map for the 14th layer
    att_map = model(**inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    # Prepare inputs for the general prompt
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_pos = general_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute general attention map for the 14th layer
    general_att_map = model(**general_inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # Normalize attention map
    att_map = att_map / general_att_map

    return att_map

def pure_gradient_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a gradient-based attention map using direct image gradients in LLaVA.
    
    This function computes gradients of the loss with respect to the input image pixels
    for both specific and general prompts. It then calculates their ratio and applies
    a high-pass filter to highlight fine-grained details that are uniquely relevant to the specific prompt.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate gradients
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        grad: A 2D numpy array representing the processed gradient map highlighting
              regions relevant to the specific prompt
    """
    # Process inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    
    # Apply high pass filter
    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)
    
    # Enable gradients
    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass for inputs
    zero_logit = model(**inputs, output_hidden_states=False).logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]
    
    # Forward pass for general_inputs
    general_zero_logit = model(**general_inputs, output_hidden_states=False).logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)
    
    # Compute general gradients
    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]
    
    # Process gradients
    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Compute gradient norms
    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)
    
    # Normalize and apply high pass filter
    grad = grad / general_grad
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass
    
    # Reduce gradient block size
    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    return grad