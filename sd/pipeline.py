import torch
from tqdm import tqdm
from ddpm import DDPMSampler
import numpy as np

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(prompt: str, 
             uncond_prompt=None, 
             input_image=None, 
             strength=0.8,#how much noise to add: more the noise the more the image will be different from the input image
             do_cfg=True, 
             cfg_scale=7.5, 
             sampler_name='ddpm', 
             n_inference_steps=50, 
             models={}, 
             seed=None,
             device=None, 
             idle_device=None, 
             tokenizer=None):
    
    with torch.no_grad():# because we are inferencing the model

        if not 0 < strength <= 1:
            raise ValueError('Strength must be between 0 and 1')
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        if do_cfg:# classifier free guidance generation: in this case we combine the conditional and unconditional context
            # Convert the prompt into tokens using the tokenizer
            cond_tokenizer = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            #(Batch_size, seq_len)
            cond_tokens =  torch.tensor(cond_tokenizer, device=device, dtype=torch.long)
            # (Batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)

            # Convert the uncond_prompt into tokens using the tokenizer
            uncond_tokenizer = tokenizer.batch_encode_plus([uncond_prompt], max_length=77, padding='max_length').input_ids
            uncond_tokens = torch.tensor(uncond_tokenizer, device=device, dtype=torch.long)
            # (Batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)
            #(2,seq_len,dim) = (2,77,768)
            context = torch.cat((cond_context, uncond_context))# We take the conditional and unconditional context and concatenate them

        else:
            # Without combining the conditional and unconditional context. But in this case we can't decide the attention to the prompt
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], max_length=77, padding='max_length').input_ids
            tokens = torch.tensor(tokens, device=device, dtype=torch.long)
            #(1, 77, 768)
            context = clip(tokens)
        
        to_idle(clip)

        # sampler
        if sampler_name== 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown Sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)# (1, 4, 64, 64)

        # if the image is given
        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, 3)
            input_image_tensor = torch.tensor(input_image_tensor, device=device).float()

            # Rescale the image: -1 to 1 becasue the UNET wants every channel to be in the range of -1 to 1
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            # (Height, Width, channel) -> # (Batch, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (1, 3, 512, 512): Because the encoder expects the input as (Batch, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)
            
            # sample noise
            encoder_noise = torch.randn(latents_shape, generator=generator,device=device)

            # Run the image through the VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)

        else:
            # if we are doing text to image generation: start with random noise N(0,1)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # The timestep is the number of timesteps in the diffusion model: since the inference steps is 50,
        # the timesteps will be 100. With each timestep, we denoise the image
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #(1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_size, 4, latent_height, latent_width) -> (Batch_size, 4, 64, 64) because the input of encoder is 4 channels
            model_input = latents

            if do_cfg:# if we are using the classifier free guidance
                # (Batch_size, 4, 64, 64) -> (2*Batch_size, 4, 64, 64)
                model_input = model_input.repeat(2,1,1,1)# repeat the input twice: one to use with the prompt and one without the prompt

            # model output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Split the model output into two parts: one with the cond prompt and one without the uncond prompt
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Now we remove this noise predicted by the UNET from the latents
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        # We rescale again
        images = rescale(images, (-1,1), (0,255), clamp=True)
        # (Batch, Channel, Height, Width) -> (Batch, Height, Width, Channel)
        images = images.permute(0,2,3,1)
        images = images.to('cpu',torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    # x: tensor
    # old_range: tuple
    # new_range: tuple
    # clamp: bool
    # returns: tensor
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep: int):
    #(160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end = 160, dtype=torch.float32)/160)# THe formula for positional encoding
    #(1,160)
    x = torch.tensor([timestep],dtype=torch.float32)[:,None] * freqs[None]
    #(1,320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)