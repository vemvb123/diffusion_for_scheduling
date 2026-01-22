import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import random
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data
from torchtyping import TensorType
from torchvision.transforms import Lambda
import torchvision.transforms.functional as F
import PIL
from torchvision.transforms import ToPILImage


from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
from deepinv.models.diffunet import DiffUNet
import os

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import deepinv
import torch.nn as nn
from typing import Tuple
import scheduling_utils
# import new_dataset

import sys




def mean_weight_value(model):
    total = 0.0
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.data.mean()
            count += 1
    return total / count if count > 0 else float('nan')



# GJOR MASKING SLIK:
# LEGGER INN EN MASKE, DER ALT SOM SKAL MASKES HAR VERDI 1
# LAG EN MASKE
# REGN UT LOSS MAP
# FRA LOSS MAPPET, NULL UT DE STEDENE DER MASKEN ER, SLIK AT MASKEREGIONENE IKKE BIDRAR I LOSS
# SÅ TA LOSS MEAN
def adj_diffusion(instance_size, pad_size, model_path: str, batch_size: int = 32, data_dim_x_y: int = 16, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):

    # VURD
    mask_value = 1
    # VURD
    root_dir = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/instances/adj/'

    dataset = scheduling_utils.AdjDataset(root_dir, instance_size, pad_size)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # x, x2 = dataset[10]

    model = deepinv.models.DiffUNet(in_channels=2, out_channels=1, pretrained=None).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    

    prev_loss = 0
    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch
        
        loop = tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False)

        for batch_idx, (adj, procs, adj_unpad_shape, procs_unpad_shape) in enumerate(loop):


            orig_h = adj_unpad_shape[0][0] # 8x8, oppgitt med en liste for hvert elem, en en loste med 8, sa enda en liste med 8
            orig_w = adj_unpad_shape[1][0] # 8
            #       logging.info("\n" + f"sizes: {orig_h}, {orig_w}")
            x_procs_undpad_shape = procs_unpad_shape[1][0] # 4
            y_procs_undpad_shape = procs_unpad_shape[1][0] # 4

            adj = adj.to(device)
            procs = procs.to(device)
            adj = adj.unsqueeze(1)
            procs = procs.unsqueeze(1)

            to_check = torch.cat([adj, procs], dim=1)
            if torch.isnan(to_check).any() or torch.isinf(to_check).any():
                # logging.info("adj or procs contains invalid values!")

                bad_mask = torch.isnan(to_check) | torch.isinf(to_check)
                bad_positions = bad_mask.nonzero()
                bad_batch_indices = bad_positions[:, 0].unique().tolist()

                # print("Bad batch sample indices:", bad_batch_indices)

                for bad_idx in bad_batch_indices:
                    # try random replacements until one is valid
                    while True:
                        random_idx = random.randrange(len(dataset))
                        new_adj, new_procs, new_adj_shape, new_proc_shape = dataset[random_idx]

                        # move to device + add batch dim
                        new_adj = new_adj.to(device).unsqueeze(0)
                        new_procs = new_procs.to(device).unsqueeze(0)

                        # check validity
                        candidate = torch.cat([new_adj, new_procs], dim=1)
                        if not torch.isnan(candidate).any() and not torch.isinf(candidate).any():
                            break

                    adj[bad_idx]   = new_adj
                    procs[bad_idx] = new_procs



            # Sample random timesteps
            t = torch.randint(0, timesteps, (adj.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(adj)






            # Apply forward diffusion process at timestep t
            noised_adj = (
                sqrt_alphas_cumprod[t, None, None, None] * adj
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )
            # CONDITION
            model_input = torch.cat([
                noised_adj,
                procs,
            ], dim=1)






            # LOSS
            optimizer.zero_grad()
            noise_pred = model(model_input, t, type_t="timestep")
            loss_map = (noise_pred - noise) ** 2  # shape = (32,1,16,16)
            # loss = mse(noise_pred, noise)
            
            if torch.isnan(noise_pred).any():
                logging.info("NaN in noise_pred!")
            if torch.isinf(noise_pred).any():
                logging.info("Inf in noise_pred!")
            if torch.isnan(noise).any() or torch.isinf(noise).any():
                logging.info("Noise contains invalid values!")
            if torch.isnan(model_input).any() or torch.isinf(model_input).any():
                logging.info("model input contains invalid values!")
                # boolean mask of invalid values
                bad_mask = torch.isnan(model_input) | torch.isinf(model_input)
                # find indices where bad_mask is True
                bad_positions = bad_mask.nonzero()
                # the first column is the batch index
                bad_batch_indices = bad_positions[:, 0].unique()
                logging.info("Bad batch sample indices:", bad_batch_indices.tolist())
                # optional: logging.info count of bad values per sample
                for idx in bad_batch_indices:
                    count = bad_mask[idx].sum().item()
                    logging.info(f"Sample {idx} has {count} invalid entries")
 
            #       logging.info("\n" + f"loss map {loss_map.mean()}")


            # MASK
            _, _, H, W = loss_map.shape
            mask = torch.zeros((1, 1, H, W), device=loss_map.device) # Make empty mask filled with 0s
            mask[..., :orig_h, :orig_w] = 1.0 # mark nonmasked areas as valid, using 1s
            mask = mask.expand(loss_map.size(0), -1, -1, -1) # expand over all batches

            # mask out padded areas
            loss_map = loss_map * mask
            # sum of valid pixels
            n_valid_pixels = mask.sum()
            #           logging.info("\n" + f"n valid pixels {n_valid_pixels}")
            # average only over valid pixels
            loss = loss_map.sum() / (n_valid_pixels + 1e-8)

            #       logging.info("\n" + f"loss mean {loss.mean()}")

            # BACKPROP
            # if loss.mean() < 0.001:
            #     torch.save(model.state_dict(),model_path,)
            #     logging.info("\n" + f"loss reached {loss} at epoch {epoch}. Exiting training. Saved model {model_path}")


            with torch.autograd.detect_anomaly():
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # STATS
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)
            
            # after optimizer.step()
            mean_w = mean_weight_value(model)
            loop.set_postfix(loss=loss_value, avg_loss=total_loss/(batch_idx+1), mean_weight=mean_w.item())


            if torch.isnan(loss):
                logging.error("\n" + "Encountered NaN loss — exiting.")
                exit()


        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info("\n" + f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


        torch.save(
            model.state_dict(),
            model_path,
        )

        # if abs(avg_loss - prev_loss) < 0.05:
        #    logging.info("improvment in loss was less than 2%. ending training")
        #    break
        # prev_loss = avg_loss




    logging.info("\n" + f"finished training. loss was {loss}. Exiting training. Saved model {model_path}")
    logging.info("\n" + "saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path




# ===


    # Function body
    device = "cuda"

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lr = 1e-3

    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch

        for batch_idx, (instances) in enumerate(train_loader):
            instances = instances.to(device)

            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(0, timesteps, (instances.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(instances)
            
            # Apply forward diffusion process at timestep t
            noised_images = (
                sqrt_alphas_cumprod[t, None, None, None] * instances
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )
            logging.info(noised_images.shape)
            # Predict noise
            noise_pred = model(noised_images, t, type_t="timestep")

            logging.info(f"noised images device: {noised_images.device}, pred noise: {noise_pred.device}" )

            v = get_makespan_intervall(dataset, noise_pred, noised_images, batch_size)
            # Calculate loss (the model predicts the noise that was added)
            # TODO legg til loss for makespan i MSE
            loss = nn.MSELoss()(noise_pred, noise) * v
            loss.backward()
            optimizer.step()
            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
 


    logging.info("saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path

















def get_makespan(dataset, data, batch_size):
    op_emb, ma_emb = dataset.decomposition_instance(data)
    logging.info('get makespan device')
    logging.info(op_emb.device)
    td = dataset.td
    env = dataset.env
    embed_dim = op_emb.shape[2]
    makespans = scheduling_utils.visulize_schedule(td, env, embed_dim, ma_emb, op_emb, batch_size)
    makespans_mean = makespans.mean()
    return makespans_mean


def normelize(b, a, v):
    return (a - v) / (a - b)

def get_makespan_intervall(dataset, pred, noised, batch_size):
    # TODO makespan noised
    noised_makespan = get_makespan(dataset, noised, batch_size)

    # TODO makespan noise pred
    pred_makespan = get_makespan(dataset, pred, batch_size)
    logging.info(noised_makespan)
    logging.info(pred_makespan)
    logging.info(f"noise makespan {noised_makespan} and pred makespan {pred_makespan}")

    # TODO sammenlign loss
    # TODO lag en verdi som kan brukes i loss
    a = 8
    b = 0
    v = noised_makespan - pred_makespan
    if v < 0: v = 0
    logging.info("v")
    logging.info(v)
    # normelize makespan intervall
    makespan_norm = normelize(b, a, v)
    logging.info("makespan norm")
    logging.info(makespan_norm)
    return makespan_norm



def apply_diffusion_makespan( batch_size: int, dataset, num_epochs: int, model_path: str ):
    # Function body
    device = "cuda"


    # størrelse pa data: (3, 1, 32, 32)
    # ====
    # ma gjore om data til noe data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logging.info(f"Instances in dataset: { len(train_loader.dataset) }")
    # ====
    # learningrate
    """
    trur beste er le 3 med betastart pa samma verdi
    tidligere har den hatt le 4 med beta start pa samma verdi
    """
   
    lr = 1e-3

    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch

        for batch_idx, (instances) in enumerate(train_loader):
            instances = instances.to(device)

            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(0, timesteps, (instances.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(instances)
            
            # Apply forward diffusion process at timestep t
            noised_images = (
                sqrt_alphas_cumprod[t, None, None, None] * instances
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )
            logging.info(noised_images.shape)
            # Predict noise
            noise_pred = model(noised_images, t, type_t="timestep")

            logging.info(f"noised images device: {noised_images.device}, pred noise: {noise_pred.device}" )

            v = get_makespan_intervall(dataset, noise_pred, noised_images, batch_size)
            # Calculate loss (the model predicts the noise that was added)
            # TODO legg til loss for makespan i MSE
            loss = nn.MSELoss()(noise_pred, noise) * v
            loss.backward()
            optimizer.step()
            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
 


    logging.info("saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path










    """
    trur beste er le 3 med betastart pa samma verdi
    tidligere har den hatt le 4 med beta start pa samma verdi
    """
# foreloig brukes ikke data_size ..
def apply_diffusion(model_path: str, batch_size: int = 32, image_size: int = 32, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    logging.info(f"Instances in dataset: { len(train_loader.dataset) }")
    logging.info(f"Instances in loader: { len(train_loader) }")
    logging.info(f'shape of dataset {train_loader.dataset[0][0].shape}')


    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(
        device
    )

    # if torch.cuda.device_count() > 1:
    #     logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
    #     model = nn.DataParallel(model)    # wrap
    # model = model.to(device)
    # logging.info("Using DataParallel on device ids:", model.device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    beta_start = lr # antar at skal vere det samme som lr
    # beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch

        for batch_idx, (instances, _) in enumerate(train_loader):
            instances = instances.to(device)
            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(0, timesteps, (instances.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(instances)

            # Apply forward diffusion process at timestep t
            noised_images = (
                sqrt_alphas_cumprod[t, None, None, None] * instances
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )

            # Predict noise
            noise_pred = model(noised_images, t, type_t="timestep")

            loss = nn.MSELoss()(noise_pred, noise)

            loss.backward()
            optimizer.step()
            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            model_path,
        )


    logging.info("saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path






    """
    trur beste er le 3 med betastart pa samma verdi
    tidligere har den hatt le 4 med beta start pa samma verdi
    """

# ======


def tensor_to_image(tensor: TensorType["channels, width, height"], save_path: str):
    # Method 1: Use PIL via ToPILImage
    to_pil = ToPILImage()  # converts C×H×W tensor -> PIL Image
    pil_tensor = to_pil(tensor)
    pil_tensor.save(save_path)




def apply_diffusion_guided_mnist_thicken(batch_size: int, num_epochs: int, model_path: str, lr=1e-3, cond_drop_prob=0.1):
    device = "cuda"

    batch_size = 32
    image_size = 32
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ])

    # Load MNIST thin images (your conditioning)
    thin_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Create “thick” versions of the digits: apply dilation (or some morphological op)
    def dilate(img_tensor):
        # img_tensor: [1, H, W], values ~ [0,1] maybe
        # Convert to PIL image, dilate, back to tensor
        img = transforms.ToPILImage()(img_tensor)
        # apply dilation (a simple approach)
        img = img.convert("L")
        # Use a kernel: e.g. 3x3
        kernel = PIL.ImageFilter.MaxFilter(3)
        dilated = img.filter(kernel)
        return transforms.ToTensor()(dilated)

    thick_tensors = []
    for img, _ in thin_dataset:
        thick = dilate(img)
        thick_tensors.append(thick)
    # Now build a paired dataset
    thin_images = [img for img, _ in thin_dataset]
    thick_images = thick_tensors

    pair_dataset = torch.utils.data.TensorDataset(
        torch.stack(thin_images),  # conditioning
        torch.stack(thick_images)  # target
    )

    train_loader = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"Dataset size (pairs): {len(pair_dataset)}")

    # Your model: modify UNet (DiffUNet) to accept conditioning
    # For simplicity: concatenate thin image as extra channel
    model = deepinv.models.DiffUNet(in_channels=2, out_channels=1, pretrained=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Diffusion hyperparams
    beta_start = lr  # as you set
    beta_end = 0.02
    timesteps = 1000
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (thin_imgs, thick_imgs) in enumerate(train_loader):
            thin_imgs = thin_imgs.to(device)   # conditioning
            thick_imgs = thick_imgs.to(device) # target
            
            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(0, timesteps, (thick_imgs.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(thick_imgs)

            # Forward diffusion on the **thick** image
            noised = (sqrt_alphas_cumprod[t, None, None, None] * thick_imgs +
                      sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise)

            # Classifier-free guidance: sometimes drop the conditioning
            # Build input for model: either concat thin image or zero
            if torch.rand(()) < cond_drop_prob:
                cond_input = torch.zeros_like(thin_imgs)
            else:
                cond_input = thin_imgs

            # Concatenate conditioning
            model_input = torch.cat([noised, cond_input], dim=1)  # channel dim

            # Predict noise
            noise_pred = model(model_input, t, type_t="timestep")

            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, loss = {avg_loss:.6f}")
        all_losses.append(avg_loss)

        # Save model + loss
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./losses", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        np.save(f"./losses/loss_epoch_{epoch}.npy", np.array(all_losses))

    return model






# FOR NYE DATASET
def new_apply_diffusion(model_path: str, batch_size: int = 32, data_dim_x_y: int = 16, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )

    mask_value = -1.0

    root = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/instances/'
    dataset = new_dataset.MyJSONMatrixDataset(root)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # adj, procs = dataset[1001]
    model = deepinv.models.DiffUNet(in_channels=4, out_channels=1, pretrained=None).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    # beta_start = lr # antar at skal vere det samme som lr
    beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

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
            optimizer.step()
            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)


        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            model_path,
        )


    logging.info("saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path





# logging.info("beginning to train", flush=True)

# batch_size = 32
# num_epochs = 100
# lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# for lr in lrs:
#     model_path = f"models/new_4x4_diffusion_{lr}.pth"
#     logging.info(f"training model {model_path}")
#     device="cuda"
#     image_size = 16
#     new_apply_diffusion(model_path, batch_size, image_size, num_epochs, lr, device)

# logging.info("done training all models", flush=True)
sys.stdout.flush()


