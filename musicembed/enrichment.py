import torch

def calculate_mean(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean of the embedding vectors over time.
    """
    return torch.mean(embedding_tensor, dim=0)

def calculate_median(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the median of the embedding vectors over time.
    """
    return torch.median(embedding_tensor, dim=0).values

def calculate_mode(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mode of the embedding vectors over time.
    """
    return torch.mode(embedding_tensor, dim=0).values

def calculate_std_dev(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the standard deviation of the embedding vectors over time.
    """
    return torch.std(embedding_tensor, dim=0)

def calculate_variance(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the variance of the embedding vectors over time.
    """
    return torch.var(embedding_tensor, dim=0)

def calculate_min_max(embedding_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the min and max of the embedding vectors over time.
    """
    min_vals, _ = torch.min(embedding_tensor, dim=0)
    max_vals, _ = torch.max(embedding_tensor, dim=0)
    return min_vals, max_vals

def calculate_range(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the range (max - min) of the embedding vectors over time.
    """
    min_vals, max_vals = calculate_min_max(embedding_tensor)
    return max_vals - min_vals

def calculate_skewness(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the skewness of the embedding vectors over time.
    """
    n = embedding_tensor.shape[0]
    mean = torch.mean(embedding_tensor, dim=0)
    std_dev = torch.std(embedding_tensor, dim=0)
    # Avoid division by zero for dimensions with no variance
    std_dev[std_dev == 0] = 1
    
    diffs = embedding_tensor - mean
    skew = (torch.sum(diffs**3, dim=0) / n) / (std_dev**3)
    return skew

def calculate_kurtosis(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the kurtosis (Fisher's definition) of the embedding vectors over time.
    """
    n = embedding_tensor.shape[0]
    mean = torch.mean(embedding_tensor, dim=0)
    std_dev = torch.std(embedding_tensor, dim=0)
    # Avoid division by zero for dimensions with no variance
    std_dev[std_dev == 0] = 1

    diffs = embedding_tensor - mean
    kurt = (torch.sum(diffs**4, dim=0) / n) / (std_dev**4)
    return kurt - 3 # Fisher's definition (excess kurtosis)

def calculate_iqr(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Interquartile Range (IQR) of the embedding vectors over time.
    """
    q1 = torch.quantile(embedding_tensor, 0.25, dim=0)
    q3 = torch.quantile(embedding_tensor, 0.75, dim=0)
    return q3 - q1

def calculate_rms(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Root Mean Square (RMS) for each dimension.
    """
    return torch.sqrt(torch.mean(embedding_tensor**2, dim=0))

def calculate_zero_crossing_rate(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the zero-crossing rate for each dimension. The rate is normalized
    by the number of possible crossings (N-1).
    """
    # Count where the sign changes from one step to the next
    sign_changes = torch.sum(torch.ne(torch.sign(embedding_tensor[:-1]), torch.sign(embedding_tensor[1:])), dim=0)
    
    # Normalize by the number of time steps minus one
    return sign_changes.float() / (embedding_tensor.shape[0] - 1)

def calculate_entropy(embedding_tensor: torch.Tensor, bins: int = 250) -> torch.Tensor:
    """
    Estimates the entropy for each dimension by discretizing the data into bins.
    """
    entropy_vec = torch.zeros(embedding_tensor.shape[1], device=embedding_tensor.device)
    for i in range(embedding_tensor.shape[1]):
        # Create a histogram for the current dimension
        hist = torch.histc(embedding_tensor[:, i], bins=bins)
        # Normalize the histogram to get probabilities
        probs = hist / torch.sum(hist)
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        # Calculate Shannon entropy
        entropy_vec[i] = -torch.sum(probs * torch.log2(probs))
    return entropy_vec

def calculate_crest_factor(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the crest factor for each dimension.
    """
    peak_amplitude, _ = torch.max(torch.abs(embedding_tensor), dim=0)
    rms = calculate_rms(embedding_tensor)
    # Avoid division by zero
    rms[rms == 0] = 1e-9
    return peak_amplitude / rms

def enrich_and_concatenate(embedding_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a comprehensive suite of statistical heuristics to a temporal embedding
    and concatenates them into a single, fixed-size feature vector.
    """
    # 1. Calculate all individual heuristic vectors
    mean_vec = calculate_mean(embedding_tensor).cpu()
    median_vec = calculate_median(embedding_tensor).cpu()
    std_dev_vec = calculate_std_dev(embedding_tensor).cpu()
    variance_vec = calculate_variance(embedding_tensor).cpu()
    min_vec, max_vec = calculate_min_max(embedding_tensor)
    min_vec = min_vec.cpu()
    max_vec = max_vec.cpu()
    iqr_vec = calculate_iqr(embedding_tensor).cpu()
    rms_vec = calculate_rms(embedding_tensor).cpu()

    # 2. Create a list of all feature vectors
    feature_list = [
        mean_vec, median_vec, std_dev_vec, variance_vec, min_vec, max_vec, iqr_vec, rms_vec
    ]

    # 3. Concatenate all vectors into a single flat vector
    final_vector = torch.cat(feature_list, dim=0)

    return final_vector
