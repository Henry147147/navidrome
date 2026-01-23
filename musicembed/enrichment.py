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
    mode_vec = calculate_mode(embedding_tensor).cpu()
    std_dev_vec = calculate_std_dev(embedding_tensor).cpu()
    variance_vec = calculate_variance(embedding_tensor).cpu()
    min_vec, max_vec = calculate_min_max(embedding_tensor)
    range_vec = max_vec - min_vec
    range_vec = range_vec.cpu()
    min_vec = min_vec.cpu()
    max_vec = max_vec.cpu()
    skew_vec = calculate_skewness(embedding_tensor).cpu()
    kurtosis_vec = calculate_kurtosis(embedding_tensor).cpu()
    iqr_vec = calculate_iqr(embedding_tensor).cpu()
    rms_vec = calculate_rms(embedding_tensor).cpu()
    zcr_vec = calculate_zero_crossing_rate(embedding_tensor).cpu()
    entropy_vec = calculate_entropy(embedding_tensor).cpu()
    crest_factor_vec = calculate_crest_factor(embedding_tensor).cpu()

    # 2. Create a list of all feature vectors
    feature_list = [
        mean_vec, median_vec, mode_vec, std_dev_vec, variance_vec,
        min_vec, max_vec, range_vec, skew_vec, kurtosis_vec, iqr_vec,
        rms_vec, zcr_vec, entropy_vec, crest_factor_vec
    ]

    # 3. Concatenate all vectors into a single flat vector
    final_vector = torch.cat(feature_list, dim=0)

    return final_vector




#"AUTO ENCODER":
"""
Of course. This is the perfect application for an autoencoder. You have a very high-dimensional, structured input, and you want to learn a compressed, dense representation that captures its most important information.

Here is a complete, runnable example using PyTorch. I'll first define the autoencoder model and then show you how to train it on dummy data generated by the `enrich_and_concatenate` function from our previous discussion.

### The Concept: An Autoencoder for Dimensionality Reduction

An autoencoder consists of two parts:

1.  **Encoder:** A neural network that takes your large input vector (`15 * 3586 = 53,790` dimensions) and compresses it down into a much smaller "latent space" vector (`8192` dimensions). This is the part you will use for your database.
2.  **Decoder:** A neural network that takes the small latent vector and tries to reconstruct the original large vector as accurately as possible.

The whole model is trained by minimizing the **reconstruction error**—typically the Mean Squared Error (MSE) between the original input and the reconstructed output. By forcing the network to squeeze all the information through a small bottleneck (the latent space), it must learn to preserve only the most essential features.

---

### Complete Python Example

This single code block contains everything you need: the prerequisite functions, the autoencoder model, the training loop, and the final demonstration.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random

# ==============================================================================
#  Step 1: Prerequisite Enrichment Functions (from previous answers)
# ==============================================================================

# (I've collapsed these functions for brevity, the code is the same as before)
def calculate_mean(t): return torch.mean(t, dim=0)
def calculate_median(t): return torch.median(t, dim=0).values
def calculate_mode(t): return torch.mode(t, dim=0).values
def calculate_std_dev(t): return torch.std(t, dim=0)
def calculate_variance(t): return torch.var(t, dim=0)
def calculate_min_max(t): min_v, _ = torch.min(t, dim=0); max_v, _ = torch.max(t, dim=0); return min_v, max_v
def calculate_skewness(t): n=t.shape[0]; m=torch.mean(t,dim=0); s=torch.std(t,dim=0); s[s==0]=1e-9; d=t-m; return (torch.sum(d**3,dim=0)/n)/(s**3)
def calculate_kurtosis(t): n=t.shape[0]; m=torch.mean(t,dim=0); s=torch.std(t,dim=0); s[s==0]=1e-9; d=t-m; k=(torch.sum(d**4,dim=0)/n)/(s**4); return k-3
def calculate_iqr(t): return torch.quantile(t,0.75,dim=0) - torch.quantile(t,0.25,dim=0)
def calculate_rms(t): return torch.sqrt(torch.mean(t**2, dim=0))
def calculate_zero_crossing_rate(t):
    if t.shape[0]<2: return torch.zeros(t.shape[1]);
    sc=torch.sum(torch.ne(torch.sign(t[:-1]),torch.sign(t[1:])),dim=0); return sc.float()/(t.shape[0]-1)
def calculate_entropy(t, bins=100):
    e = torch.zeros(t.shape[1], device=t.device);
    for i in range(t.shape[1]):
        h=torch.histc(t[:,i],bins=bins,min=t[:,i].min(),max=t[:,i].max()); p=h/torch.sum(h); p=p[p>0]; e[i]=-torch.sum(p*torch.log2(p));
    return e
def calculate_crest_factor(t): peak,_=torch.max(torch.abs(t),dim=0); rms=calculate_rms(t); rms[rms==0]=1e-9; return peak/rms

def enrich_and_concatenate(embedding_tensor: torch.Tensor, entropy_bins: int = 100) -> torch.Tensor:
    if embedding_tensor.dim() != 2: raise ValueError(f"Input tensor must be 2D, but got {embedding_tensor.shape}")
    if embedding_tensor.shape[0] == 0: return torch.zeros(15 * embedding_tensor.shape[1])
    min_vec, max_vec = calculate_min_max(embedding_tensor)
    feature_list = [
        calculate_mean(embedding_tensor), calculate_median(embedding_tensor), calculate_mode(embedding_tensor),
        calculate_std_dev(embedding_tensor), calculate_variance(embedding_tensor), min_vec, max_vec, (max_vec - min_vec),
        calculate_skewness(embedding_tensor), calculate_kurtosis(embedding_tensor), calculate_iqr(embedding_tensor),
        calculate_rms(embedding_tensor), calculate_zero_crossing_rate(embedding_tensor),
        calculate_entropy(embedding_tensor, bins=entropy_bins), calculate_crest_factor(embedding_tensor)
    ]
    return torch.cat(feature_list, dim=0)


# ==============================================================================
#  Step 2: Define the Autoencoder Model
# ==============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # --- Encoder ---
        # Compresses the input from input_dim down to latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24576), # A middle step
            nn.ReLU(),
            nn.Linear(24576, 12288), # Another middle step
            nn.ReLU(),
            nn.Linear(12288, latent_dim) # The bottleneck
        )
        
        # --- Decoder ---
        # Expands from the latent_dim back up to the original input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12288),
            nn.ReLU(),
            nn.Linear(12288, 24576),
            nn.ReLU(),
            nn.Linear(24576, input_dim) # No ReLU on the final layer!
                                        # This allows it to reconstruct negative values.
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==============================================================================
#  Step 3: Training Setup and Loop
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    ORIGINAL_EMBEDDING_DIM = 3586
    NUM_HEURISTICS = 15
    INPUT_DIM = NUM_HEURISTICS * ORIGINAL_EMBEDDING_DIM # Should be 53790
    LATENT_DIM = 8192  # This is your target fixed-size vector for the database
    
    LEARNING_RATE = 1e-4
    EPOCHS = 20 # Use more epochs (e.g., 50-100) for real training
    BATCH_SIZE = 32
    NUM_SAMPLES = 1000 # Number of dummy "songs" to generate

    # --- Generate a Dummy Dataset ---
    print("Generating dummy dataset...")
    enriched_vectors = []
    for i in range(NUM_SAMPLES):
        # Simulate variable-length audio
        seq_len = random.randint(500, 2000) 
        dummy_temporal_embedding = torch.randn(seq_len, ORIGINAL_EMBEDDING_DIM)
        enriched_vector = enrich_and_concatenate(dummy_temporal_embedding)
        enriched_vectors.append(enriched_vector)
        if (i+1) % 100 == 0: print(f"  Generated {i+1}/{NUM_SAMPLES} samples...")

    # Stack into a single tensor for the dataset
    dataset_tensor = torch.stack(enriched_vectors)
    dataset = TensorDataset(dataset_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset created with shape: {dataset_tensor.shape}")

    # --- Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    criterion = nn.MSELoss() # Mean Squared Error is ideal for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training on {device}...")
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_data in dataloader:
            # The dataloader wraps the data in a list
            original_vectors = batch_data[0].to(device)
            
            # Forward pass
            reconstructed_vectors = model(original_vectors)
            loss = criterion(reconstructed_vectors, original_vectors)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Reconstruction Loss: {avg_loss:.6f}")

    print("\nTraining complete.")

# ==============================================================================
#  Step 4: Use the Trained Encoder for Inference
# ==============================================================================

    print("\n--- Demonstrating Inference ---")
    
    # Get a single sample from our dataset to compress
    sample_enriched_vector = dataset_tensor[0].to(device)
    
    # IMPORTANT: Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad(): # We don't need to calculate gradients for inference
        # Use ONLY the encoder part of the model
        final_database_vector = model.encoder(sample_enriched_vector)

    print(f"Original enriched vector shape: {sample_enriched_vector.shape}")
    print(f"Compressed database vector shape: {final_database_vector.shape}")

    if final_database_vector.shape[0] == LATENT_DIM:
        print("\nSuccess! The encoder produced a vector of the desired latent dimension.")

```
"""

# IDEA!!! train a model similar to auto encoder, but should work on the whole embedding dimentions.

