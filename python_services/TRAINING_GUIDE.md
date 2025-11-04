# Training Guide: Understanding Your Results

## Training Metrics Explained

### Loss Metrics

**InfoNCE Loss** (used in this training):
- **Range**: 0 to ∞ (lower is better)
- **Good values**: < 1.0 indicates strong alignment
- **What it means**: Measures how well text and audio embeddings align in the shared space
- **Typical progression**:
  - Epoch 1: ~3.5 (random initialization)
  - Epoch 5-10: ~1.5-2.0 (learning alignment)
  - Epoch 15-20: ~0.5-1.0 (good convergence)

### Retrieval Metrics

These measure how well the model can find the correct audio given text (text-to-audio) and vice versa (audio-to-text).

#### **R@K (Recall at K)**
- **R@1**: Percentage of queries where the correct match is rank #1
- **R@5**: Percentage where correct match is in top 5
- **R@10**: Percentage where correct match is in top 10

**Example interpretation:**
```
Val t2a R@1: 0.22%    ← Only 0.22% of text queries find exact match as #1 (very early training)
Val t2a R@5: 1.57%    ← 1.57% find correct match in top 5
Val t2a R@10: 2.47%   ← 2.47% find correct match in top 10
```

**Good target values (after full training):**
- **R@1**: > 30% (excellent: > 50%)
- **R@5**: > 60% (excellent: > 75%)
- **R@10**: > 70% (excellent: > 85%)

#### **MRR (Mean Reciprocal Rank)**
- **Range**: 0.0 to 1.0 (higher is better)
- **What it means**: Average of 1/rank across all queries
- **Formula**: If correct match is at rank 3, reciprocal = 1/3 = 0.333
- **Good values**: 
  - > 0.3: Decent performance
  - > 0.5: Good performance
  - > 0.7: Excellent performance

**Example:**
```
Val t2a MRR: 0.0127   ← Very early training (correct matches are ranked ~77th on average)
                        (1/0.0127 ≈ 78.7)
```

#### **Median Rank**
- **What it means**: The middle position where correct matches are found
- **Lower is better**: Median rank of 1 = perfect
- **Good values**:
  - < 5: Excellent
  - < 10: Good  
  - < 20: Acceptable
  - > 50: Needs more training

### Interpreting Your Training Progress

#### **Epoch 1 (Baseline)**
```
Train loss: 3.50
Val loss: 3.48
Val t2a R@1: 0.22%
Val t2a MRR: 0.01
```
✅ **This is NORMAL!** The model starts with random weights.

#### **Epoch 5-10 (Early Learning)**
```
Train loss: 1.8-2.2
Val loss: 1.9-2.3
Val t2a R@1: 5-15%
Val t2a MRR: 0.15-0.25
```
✅ Model is learning text-audio alignment.

#### **Epoch 15-20 (Convergence)**
```
Train loss: 0.5-1.0
Val loss: 0.6-1.2
Val t2a R@1: 30-50%
Val t2a MRR: 0.40-0.60
```
✅ Good convergence! Model can retrieve relevant audio for most queries.

#### **Signs of Good Training**
1. ✅ **Loss decreasing steadily** (both train and val)
2. ✅ **R@1, R@5, R@10 all increasing** over epochs
3. ✅ **MRR increasing** over epochs
4. ✅ **Val loss stays within 0.3 of train loss** (not overfitting)

#### **Warning Signs**
1. ⚠️ **Val loss >> train loss** (e.g., train=0.5, val=2.0) → Overfitting
2. ⚠️ **Loss not decreasing** after epoch 5 → Learning rate too high/low
3. ⚠️ **Metrics plateauing early** (before epoch 10) → May need more training data

## Comparing Audio Encoders

After training all three models, compare their final validation metrics:

### Example Results Table

| Model | Audio Dim | Val R@1 | Val R@5 | Val MRR | Best For |
|-------|-----------|---------|---------|---------|----------|
| **MuQ** | 1,536 | 45% | 72% | 0.55 | General music understanding |
| **MERT** | 76,800 | 52% | 78% | 0.62 | Musical features (tempo, key) |
| **Latent** | 576 | 38% | 65% | 0.48 | Efficient embeddings |

**Interpretation:**
- **MERT**: Highest performance (captures more audio detail with 76K dimensions)
- **MuQ**: Good balance of performance and efficiency
- **Latent**: Most compact, good for resource-constrained scenarios

## Checkpoints Explained

Three checkpoints are saved per model:

1. **`{model}_best_r1.pt`**
   - Weights from epoch with highest validation R@1
   - **Use this for**: Best retrieval accuracy

2. **`{model}_best_loss.pt`**
   - Weights from epoch with lowest validation loss
   - **Use this for**: Most generalizable model

3. **`{model}_last.pt`**
   - Weights from final epoch
   - **Use this for**: Resuming training if stopped early

## TensorBoard Visualization

Monitor training in real-time:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006

**What to look for:**
- **Scalars tab**: View loss curves, R@K metrics over time
- **Smooth curves**: Training is stable
- **Jagged curves**: May need to reduce learning rate

## Production Deployment

After training completes with good metrics:

1. **Select best checkpoint**: Usually `{model}_best_r1.pt`
2. **Test on your data**: Use test set metrics as final benchmark
3. **Deploy**: Load checkpoint in inference code

## Quick Reference

**Is my training working?**
- ✅ Loss going down after epoch 1? → YES
- ✅ R@1 > 1% by epoch 5? → YES
- ✅ R@1 > 10% by epoch 10? → YES
- ✅ R@1 > 30% by epoch 20? → EXCELLENT

**Should I train longer?**
- If R@1 still increasing at epoch 20 → YES, try 30-40 epochs
- If loss plateaued for 5+ epochs → NO, training complete
- If val loss increasing while train loss decreasing → NO, overfitting

## Example: Reading Your Log

```
Epoch 10/20:
  Train loss: 1.8234
  Val loss: 1.9012
  Val t2a R@1: 12.45%    ← 12.45% of text queries get exact match
  Val t2a R@5: 31.22%    ← 31.22% get correct match in top 5
  Val t2a R@10: 42.18%   ← 42.18% get correct match in top 10
  Val t2a MRR: 0.2156    ← Average rank of correct match ≈ 4.6
```

**Interpretation**: Model is learning well! At halfway through training:
- Already finding exact matches for 12% of queries
- Top-5 accuracy is 31% (good for epoch 10)
- On track for final R@1 > 30%
