# modified from https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train_sdxl.py
import os
import itertools
import time
import logging
import shutil
import math

import torch
import torch.nn.functional as F
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs as DDPK
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from config import parse_args
from msdiffusion.dataset.datagenerator import MyDataset, collate_fn
from msdiffusion.models.projection import ImageProjModel, Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.models.attention_processor import MaskedIPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor

logger = get_logger(__name__, log_level="INFO")
    

def main():
    args = parse_args()
    logging_dir = args.logging_dir

    # train id
    log_name = args.train_id
    report_dir = os.path.join(logging_dir, "runs", log_name)
    output_dir = os.path.join(args.output_dir, log_name)
    result_dir = os.path.join(args.result_dir, log_name)

    # clear tensorboard files
    if args.log_clear:
        shutil.rmtree(os.path.join(report_dir), ignore_errors=True)

    accelerator_project_config = ProjectConfiguration(project_dir=report_dir)
    accelerator_kwargs = DDPK(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_kwargs]
    )
    
    if accelerator.is_local_main_process:
        os.makedirs(report_dir, exist_ok=True)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logging_dir, log_name + ".log"),
        filemode="w" if args.log_clear else "a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir)
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", cache_dir=args.cache_dir)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir)

    # support other vae to be loaded
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(vae_path,
                                        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                                        cache_dir=args.cache_dir)

    if args.image_encoder_type == "clip":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path, cache_dir=args.cache_dir)
        image_encoder_projection_dim = image_encoder.config.projection_dim
    else:
        raise ValueError(f"Unsupported image encoder type: {args.image_encoder_type}")

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    num_tokens = args.num_tokens
    text_tokens = text_encoder.config.max_position_embeddings
    if args.image_proj_type == "linear":
        # use direct projection
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder_projection_dim,
            clip_extra_context_tokens=num_tokens,
        )
    elif args.image_proj_type == "resampler":
        # use resampler
        image_proj_model = Resampler(
            # dim=unet.config.cross_attention_dim,
            dim=1280,  # image_encoder_hidden_size
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
            latent_init_mode=args.latent_init_mode,
            phrase_embeddings_dim=text_encoder.config.projection_dim,
        )
    else:
        raise ValueError(f"Unsupported image projection type: {args.image_proj_type}")
    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                               num_tokens=num_tokens, text_tokens=text_tokens, scale=1.0)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ms_adapter = MSAdapter(unet, image_proj_model, adapter_modules=adapter_modules,
                           ckpt_path=args.pretrained_ms_adapter_path, num_tokens=num_tokens, text_tokens=text_tokens)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)  # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    accelerator.print("Finish model initialization!")

    # save model hook and load model hook for accelerator
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # only save the ip-adapter for now
            for model in models:
                if isinstance(model, MSAdapter):
                    model.save_to_checkpoint(output_dir)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, MSAdapter):
                model.load_from_checkpoint(input_dir)
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # scale lr according to train settings
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # optimizer
    params_to_opt = itertools.chain(ms_adapter.image_proj_model.parameters(),  ms_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution,
                              device=accelerator.device, image_encoder_type=args.image_encoder_type)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    accelerator.print("Finish dataset initialization!")
    
    # Prepare everything with our `accelerator`.
    ms_adapter, optimizer, train_dataloader = accelerator.prepare(ms_adapter, optimizer, train_dataloader)

    accelerator.print("Finish accelerator preparation!")

    # calculate the total steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("reference-based", config=vars(args))

    # args.push_logs_steps = args.push_logs_steps if args.push_logs_steps is not None else args.checkpointing_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # resume from hook
    if args.resume_from_checkpoint:
        checkpoint_dir = output_dir if args.checkpoint_dir is None else args.checkpoint_dir
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(checkpoint_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # init progress_bar
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    accelerator.print("Start training!")

    for epoch in range(first_epoch, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ms_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                # encode multiple images
                if args.image_proj_type == "resampler":
                    # use resampler
                    processed_images = []
                    for processed_image, drop_image_embed in zip(batch["processed_images"], batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            processed_images.append(torch.zeros_like(processed_image))
                        else:
                            processed_images.append(processed_image)
                    processed_images = torch.stack(processed_images, dim=0)  # (bsz, rn, ...)
                    with torch.no_grad():
                        processed_images = processed_images.view(-1, processed_images.shape[-3], processed_images.shape[-2], processed_images.shape[-1])  # (bsz*rn, ...)
                        image_embeds = image_encoder(processed_images.to(accelerator.device, dtype=weight_dtype),
                                                     output_hidden_states=True).hidden_states[-2]  # (bsz*rn, num_tokens, embedding_dim)
                else:
                    # default use direct projection
                    with torch.no_grad():
                        # image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                        processed_images = batch["processed_images"].view(-1, batch["processed_images"].shape[-3], batch["processed_images"].shape[-2], batch["processed_images"].shape[-1])  # (bsz*rn, ...)
                        image_embeds = image_encoder(processed_images.to(accelerator.device, dtype=weight_dtype)).image_embeds  # (bsz*rn, embedding_dim)
                    image_embeds_ = []
                    image_embeds = image_embeds.view(bsz, -1, image_embeds.shape[-1])  # (bsz, rn, embedding_dim)
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds = torch.stack(image_embeds_)  # (bsz, rn, embedding_dim)
                    image_embeds = image_embeds.view(-1, image_embeds.shape[-1])  # (bsz*rn, embedding_dim)
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                rf_attention_mask = batch["rf_attention_mask"].to(accelerator.device) if "rf_attention_mask" in batch else None
                boxes = batch["image_boxes"].to(accelerator.device, dtype=weight_dtype)
                phrase_idxes = batch["phrase_idxes"].to(accelerator.device)
                eot_idxes = batch["eot_idxes"].to(accelerator.device)
                cross_attention_kwargs = {"boxes": boxes, "phrase_idxes": phrase_idxes, "eot_idxes": eot_idxes}

                # get grounding tokens
                grounding_kwargs = None
                if args.latent_init_mode == "grounding" or args.latent_init_mode == "phrase":
                    phrase_input_ids = batch["image_phrase_input_ids"].view(-1, batch["image_phrase_input_ids"].shape[-1])
                    batch_boxes = boxes.view(-1, boxes.shape[-1])
                    phrase_embeds = text_encoder(phrase_input_ids.to(accelerator.device)).pooler_output
                    grounding_kwargs = {
                        "boxes": batch_boxes,
                        "phrase_embeds": phrase_embeds,
                        "drop_grounding_tokens": batch["drop_grounding_tokens"],
                    }

                noise_pred = ms_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds,
                                        rf_attention_mask=rf_attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                                        grounding_kwargs=grounding_kwargs)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                all_loss = accelerator.gather(loss.repeat(args.train_batch_size))
                avg_loss = all_loss.mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                params_to_opt = itertools.chain(ms_adapter.module.image_proj_model.parameters(),
                                                ms_adapter.module.adapter_modules.parameters())
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # save checkpoint
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            display_logs = {"epoch": epoch, "step": step, "loss": f"{avg_loss:.4f}"}
            record_logs = {"loss": avg_loss}
            progress_bar.set_postfix(**display_logs)
            accelerator.log(record_logs, step=global_step)

            accelerator.wait_for_everyone()
    
    accelerator.end_training()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
