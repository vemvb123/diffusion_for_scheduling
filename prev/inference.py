import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import deepinv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.io as io
import scheduling.scheduling_utils as scheduling_utils
from torchvision import datasets, transforms

from tqdm import tqdm
import time



def adj_inference(proc_instance, model_path, n_samples, dim_x=8, dim_y=8, pad_x=16, pad_y=16):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=2, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()
    
    mask_value = 1
    x = None
    with torch.no_grad():
        
        # creating a matrix, everything out side of the submatrix dim_x,dim_y has the value 1, while the matrix dim_x,dim_y has a random value.
        # similair to how x was masked during training
        x = torch.randn(n_samples, 1, pad_x, pad_y).to(device)
        # Set rows outside dim_x to 1
        x[:, :, dim_x:, :] = 1
        # Set columns outside dim_y to 1 (for rows inside dim_x)
        x[:, :, :dim_x, dim_y:] = 1

        # må fore inn maske...
        proc_instance = proc_instance.unsqueeze(0).unsqueeze(0).to(device)

        model_input = torch.cat([
                x,
                proc_instance,
            ], dim=1)

        # start timer
        start_time = time.perf_counter()

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                proc_instance,
            ], dim=1)  # channels = 4

            predicted_noise = model(inputs, t_tensor, type_t="timestep")
                
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # ma ha maske??
                # noise = 0
        
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed









def apply_inference_444(proc_instance, model_path, n_samples, dim_x=16, dim_y=16):
    """
    all_losses = [] 
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch

        for batch_idx, (adj, procs) in enumerate(tqdm(train_loader,
                                                       desc=f"Epoch {epoch+1}/{num_epochs}",
                                                       leave=False)):

            adj = adj.to(device)
            procs = procs.to(device)
            adj = adj.unsqueeze(1)
            procs = procs.unsqueeze(1)

            batch_size_current = adj.shape[0]
            mask = torch.ones_like(adj)
            mask = mask * (adj != mask_value).float()

            known = adj * mask

            # Sample random timesteps
            t = torch.randint(0, timesteps, (adj.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(adj)

            # Apply forward diffusion process at timestep t
            noised_adj = known + ( sqrt_alphas_cumprod[t, None, None, None] * adj +
                                   sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise ) * (1.0 - mask)

            model_input = torch.cat([
                noised_adj,
                procs,
                mask,
                known
            ], dim=1)  # channels = 4


            optimizer.zero_grad()
            noised_input_xy = torch.cat((noised_adj, procs), 1)

            # Predict noise
            noise_pred = model(model_input, t, type_t="timestep")

            loss = mse(noise_pred * (1.0 - mask), noise * (1.0 - mask)).mean()

            loss.backward()
"""
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()
    
    mask_value = -1.0
    x = None
    with torch.no_grad():

        x = torch.randn(n_samples, 1, dim_x, dim_y).to(device) # adjacency matrix
        x[:,:,4:16,:] = -1.0

        # må fore inn maske...
        proc_instance = proc_instance.unsqueeze(0).unsqueeze(0).to(device)

        batch_size_current = x.shape[0]
        mask = torch.ones_like(x)
        mask = mask * (x != mask_value).float()
        known = x * mask

        model_input = torch.cat([
                x,
                proc_instance,
                mask,
                known
            ], dim=1)  # channels = 4


        
        # start timer
        start_time = time.perf_counter()


        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                proc_instance,
                mask,
                known
            ], dim=1)  # channels = 4

            predicted_noise = model(inputs, t_tensor, type_t="timestep")
                
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # ma ha maske??
                # noise = 0
        
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed










def apply_inference_4x2(proc_instance, model_path, n_samples, dim_x=16, dim_y=16):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=2, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    with torch.no_grad():
        

        x = torch.randn(n_samples, 1, dim_x, dim_y).to(device) # adjacency matrix
        proc_instance = proc_instance.unsqueeze(0).unsqueeze(0).to(device)

        inputs = torch.cat([
            x,
            proc_instance
        ], dim=1)
        
        # start timer
        start_time = time.perf_counter()


        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                proc_instance
            ], dim=1)

            predicted_noise = model(inputs, t_tensor, type_t="timestep")
                
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                # noise = 0
        
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            """
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"inference_imgs/4x2/sample_{idx+b}_{t}.png", img, cmap="gray"
                    )
            """
    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed



def apply_inference(dim_x, dim_y, model_path, n_samples, beta_start):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=1, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    logging.info("Making samples")
    with torch.no_grad():
        x = torch.randn(n_samples, 1, dim_x, dim_y).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise


            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/sample_{idx+b}_{t}_{model_path}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)


    logging.info(x.size())
    return x




def apply_inference(dim_x, dim_y, model_path, n_samples, beta_start):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=1, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    logging.info("Making samples")
    with torch.no_grad():
        x = torch.randn(n_samples, 1, dim_x, dim_y).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise


            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/sample_{idx+b}_{t}_{model_path}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)


    logging.info(x.size())
    return x






def apply_conditional_inference(dim_x, dim_y, model_path, n_samples, beta_start):
    """
    cond_images: Tensor, shape (n_samples, 1, dim_x, dim_y)
        These are the **thin** images you want to condition on.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load your trained model. Note: in_channels should be 2 now (noised + cond)
    model = deepinv.models.DiffUNet(
        in_channels=2,  # because you concatenated thin image during training
        out_channels=1,
        pretrained=None
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    device = "cuda"

    batch_size = 32
    image_size = 32
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ])

    # Load MNIST thin images (your conditioning)
    thin_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    thin_img, label = thin_dataset[42] 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond = thin_img.unsqueeze(0).to(device)  # (1, 1, 32, 32)
    logging.info("the label is: ")
    logging.info(label)


    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Prepare the conditioning images
    cond = cond.to(device)

    # Start from random noise
    x = torch.randn(n_samples, 1, dim_x, dim_y, device=device)

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

            # Build the input by concatenating the noisy x and the conditioning image
            model_input = torch.cat([x, cond], dim=1)  # channel dimension

            # Predict the noise
            predicted_noise = model(model_input, t_tensor, type_t="timestep")

            # Compute the usual DDPM reverse diffusion step
            alpha = alphas[t]
            alpha_cum = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            logging.info(noise.shape)
            logging.info(x.shape)
            logging.info(alpha.shape)
            logging.info(alpha_cum.shape)
            logging.info(beta.shape)
            logging.info(predicted_noise.shape)
            logging.info("exiting shaitlinga")
            exit()

            # This is the standard DDPM update rule (one variant)
            x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cum)) * predicted_noise
                ) + torch.sqrt(beta) * noise

            # Optional: save intermediate images
            if t % 100 == 0:
                for i in range(n_samples):
                    img = x[i, 0].cpu().detach().numpy()
                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    plt.imsave(f"mnist_cond/cond_sample_{i}_t{t}.png", img_norm, cmap="gray")

    x = torch.clamp(x, 0.0, 1.0)
    return x.cpu()








# mask = make_mask("4ma_2op_2j_0", 1, 64, 64, "cuda")
def existing_problem_to_instance(file_to_solve, n_samples, dim_x, dim_y, device, use_mask: bool = None):
    file_to_solve = file_to_solve + "_nodes.png"
    img = Image.open(file_to_solve).convert("L")  # “L” mode = grayscale (1-channel)
    # If you want color (3 channels) you could use convert("RGB") but your x has 1 channel.
    
    # 2) Resize (if needed) to match dim_x, dim_y
    transform = transforms.Compose([
        transforms.Resize((dim_x, dim_y)),
        transforms.ToTensor(),                       # converts to [C, H, W], floats in [0,1]
    ])
    img_tensor = transform(img)                    # shape [1, dim_x, dim_y]

    # 3) Expand/unsqueeze to batch of n_samples
    img_batch = img_tensor.unsqueeze(0).repeat(n_samples, 1, 1, 1)  # shape [n_samples, 1, dim_x, dim_y]

    # 4) Move to device
    img_batch = img_batch.to(device)
    
    mask = None
    if use_mask:
        file_to_solve.replace("_nodes.png","_all_edges.png")
        mask = scheduling_utils.make_mask(file_to_solve, 64, 64, "cuda")

    return img_batch, mask



def apply_inference_existing_problem(file_to_solve, dim_x, dim_y, model_path, n_samples, beta_start):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=1, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    logging.info("Making samples")
    with torch.no_grad():
        x = existing_problem_to_instance(file_to_solve, n_samples, dim_x, dim_y, device)
        # x = torch.randn(n_samples, 1, dim_x, dim_y).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise


            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/solved/sample_{idx+b}_{t}_{model_path}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)


    logging.info(x.size())
    return x


def apply_inference_existing_problem_test(file_to_solve, dim_x, dim_y, model_path, n_samples, beta_start):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=1, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    logging.info("Making samples")
    with torch.no_grad():
        orig = existing_problem_to_instance(file_to_solve, n_samples, dim_x, dim_y, device)
        x = orig.clone() 
        # x = torch.randn(n_samples, 1, dim_x, dim_y).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

            x = torch.where(orig != 1.0, orig, x) 

            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/solved/sample_{idx+b}_{t}_{model_path}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)


    logging.info(x.size())
    return x






def apply_inference_existing_problem_test_masks(file_to_solve, dim_x, dim_y, model_path, n_samples, beta_start):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=1, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()

    x = None
    logging.info("Making samples")
    with torch.no_grad():
        orig, mask = existing_problem_to_instance(file_to_solve, n_samples, dim_x, dim_y, device, True)
        x = orig.clone() 
        # x = torch.randn(n_samples, 1, dim_x, dim_y).to(device)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x_pred = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            mask = mask.to("cuda")
            orig = orig.to("cuda")
            x = mask * x_pred + (1 - mask) * orig           
            
            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                idx = 0  # Use this as a base index for the current batch
                for b in range(n_samples):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"solved_mask/solved_sample_{idx+b}_{t}_{model_path}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)


    logging.info(x.size())
    return x



"""
logging.info("running")
# model_path = "diffusionAsPlugAndPlay_lr_masks_0.0001.pth"
n_samples = 1
beta_start = 0.0001
# file_to_solve = "instances/4ma_2op_2j_80000"
model_path = "models/diffusion_mnist_guided.pth"
apply_conditional_inference(32, 32, model_path, n_samples, beta_start)
# apply_inference_existing_problem_test(file_to_solve, 64, 64, model_path, n_samples, beta_start)
logging.info("done")
"""
