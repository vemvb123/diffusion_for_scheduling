import torch



def max_per_column_binary_matrix(tensor: torch.Tensor, valid_slots: torch.Tensor):
    mat = tensor[0]
    mask = valid_slots[0, 0].bool()

    masked = mat.masked_fill(~mask, float('-inf'))

    max_indices = torch.argmax(masked, dim=0)  # (W,)

    result = torch.zeros_like(mat)
    cols = torch.arange(mat.shape[1])

    valid_cols = masked[max_indices, cols] != float('-inf')
    result[max_indices[valid_cols], cols[valid_cols]] = 1

    return result.unsqueeze(0)





def topk_binary_matrix(tensor: torch.Tensor, k: int, valid_slots: torch.Tensor):
    """
    tensor: shape (1, H, W)
    valid_slots: shape (1, 1, H, W) with 1s where allowed
    k: number of top values to select
    """
    # Make valid_slots the same shape as tensor
    mask = valid_slots[0, 0].bool()  # shape (H, W)

    # Flatten tensor and mask
    flat = tensor.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Only consider valid positions
    # valid_values = flat.clone()
    valid_values = flat.clone().float()  
    valid_values[~mask_flat] = float('-inf')  # ignore invalid slots

    # Take top-k among valid positions
    topk_indices = torch.topk(valid_values, k).indices

    # Create binary result
    result = torch.zeros_like(flat)
    result[topk_indices] = 1

    return result.reshape_as(tensor)













