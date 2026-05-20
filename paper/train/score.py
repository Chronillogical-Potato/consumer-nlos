import torch.nn.functional as F
import torch


def dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    # input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)
    # input /= torch.sum(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.sum(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score

def normalized_dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score

def filtered_dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score


def tof_diff(input : torch.Tensor, pred : torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : difference in peak location along time axis (batch_size, )

    """
    # === Compute tof === #
    input_tof = torch.argmax(input, dim=-1) # (1, n_y, n_x)
    pred_tof = torch.argmax(pred, dim=-1) # (batch_size, n_y, n_x)

    # === Compute difference === #
    diff = torch.abs(input_tof - pred_tof).float().mean(dim=tuple(range(1, input_tof.dim()))) 
    score = input.shape[-1] - diff

    return score

def weighted_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : weighted score (batch_size, )

    """
    tof_score = mean_diff(input, pred)
    corr_score = dot_product_score(input, pred)
    score = 0.5 * (tof_score + corr_score)

    return score