# Recommender System Enhancement Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for 5 enhancement features for the Navidrome music recommender system. The enhancements focus on:
1. Text-based playlist generation
2. Multi-embedder support
3. Union bounds for improved recommendation quality
4. Negative prompting for playlist refinement
5. Batch re-embedding infrastructure

## Current Architecture Overview

### Components
- **Frontend**: React UI (`ui/src/explore/ExploreSuggestions.jsx`)
- **Backend API**: Go REST endpoints (`server/nativeapi/recommendations.go`)
- **Recommender Engine**: Python FastAPI service (`python_services/recommender_api.py`)
- **Vector Store**: Milvus with 3 collections (embedding, mert_embedding, latent_embedding)
- **Embedding Models**:
  - MuQEmbeddingModel (1536D) - supports text embedding
  - MertModel (76,800D) - no text support
  - MusicLatentSpaceModel (576D) - no text support

### Text Embedding Infrastructure
- **inference.py** provides `TextToAudioEmbedder` class for projecting text into audio space
- Trained models can project text queries into MuQ/MERT/Latent embedding spaces
- Models stored as checkpoints: `{encoder}_best_r1.pt`

---

## Idea 1: Text Embedding Playlist Generator

### Overview
Allow users to generate playlists by entering natural language descriptions (e.g., "upbeat rock with guitar solos", "relaxing jazz piano"). Text is embedded into the audio space and used for similarity search.

### Architecture

```
User Text Input
    ↓
Frontend (New TextPlaylistGenerator component)
    ↓
POST /recommendations/text
    ↓
Go Handler (buildTextSeeds)
    ↓
POST /embed_text (New Python endpoint)
    ↓
TextToAudioEmbedder
    ↓
Returns embedding vector
    ↓
Go Handler → Python Recommender Engine
    ↓
Milvus similarity search
    ↓
Playlist results
```

### Implementation Details

#### 1.1 Python Text Embedding Service

**New File**: `python_services/text_embedding_service.py`

```python
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch

from inference import TextToAudioEmbedder

class TextEmbeddingRequest(BaseModel):
    text: str
    model: str = "muq"  # muq, mert, or latent

class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dimension: int

class TextEmbeddingService:
    """Manages text-to-audio embedding models"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.embedders: Dict[str, TextToAudioEmbedder] = {}
        self._load_embedders()

    def _load_embedders(self):
        """Lazy load embedders on first use"""
        # Models will be loaded on demand
        pass

    def get_embedder(self, model_name: str) -> TextToAudioEmbedder:
        """Get or create embedder for specified model"""
        if model_name not in self.embedders:
            checkpoint_path = f"{self.checkpoint_dir}/{model_name}_best_r1.pt"
            try:
                self.embedders[model_name] = TextToAudioEmbedder(
                    checkpoint_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_name} not found or failed to load: {str(e)}"
                )
        return self.embedders[model_name]

    def embed_text(self, text: str, model: str = "muq") -> np.ndarray:
        """Embed text query into audio space"""
        embedder = self.get_embedder(model)
        return embedder.embed_text(text)

# FastAPI app
app = FastAPI()
text_service = TextEmbeddingService()

@app.post("/embed_text")
async def embed_text(request: TextEmbeddingRequest) -> TextEmbeddingResponse:
    """Embed text into audio embedding space"""
    try:
        embedding = text_service.embed_text(request.text, request.model)
        return TextEmbeddingResponse(
            embedding=embedding.tolist(),
            model=request.model,
            dimension=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available text embedding models"""
    return {
        "models": [
            {"name": "muq", "dimension": 1536, "status": "available"},
            {"name": "mert", "dimension": 76800, "status": "available"},
            {"name": "latent", "dimension": 576, "status": "available"}
        ]
    }
```

**Integration**: Run alongside recommender_api.py (can be same FastAPI app or separate service)

#### 1.2 Backend API Updates

**File**: `server/nativeapi/recommendations.go`

```go
// Add new endpoint
func (api *Router) textRecommendations(w http.ResponseWriter, r *http.Request) (*responses.Subsonic, error) {
    ctx := r.Context()
    user := request.UserFrom(ctx)

    var payload struct {
        Text               string   `json:"text"`
        Model              string   `json:"model"` // muq, mert, latent
        Limit              int      `json:"limit"`
        Diversity          *float64 `json:"diversity"`
        ExcludeTrackIDs    []string `json:"excludeTrackIds"`
        ExcludePlaylistIDs []string `json:"excludePlaylistIds"`
    }

    if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
        return nil, err
    }

    // Validate
    if payload.Text == "" {
        return nil, errors.New("text query is required")
    }

    // Get text embedding from Python service
    embedding, err := api.embedText(ctx, payload.Text, payload.Model)
    if err != nil {
        return nil, err
    }

    // Build seeds with text embedding
    seeds := api.buildTextSeeds(embedding, payload.Model)

    // Get recommendations
    settings := api.getRecommendationSettings(user.ID)
    recReq := buildRecommendationRequest(user, settings, "text", seeds, payload)

    result, err := api.recommender.Recommend(ctx, "text", recReq)
    if err != nil {
        return nil, err
    }

    // Return tracks
    tracks, warnings := api.processRecommendations(ctx, user, result, payload.ExcludePlaylistIDs)

    return &recommendations.RecommendationResponse{
        Name:     payload.Text,
        Mode:     "text",
        TrackIDs: extractTrackIDs(tracks),
        Tracks:   tracks,
        Warnings: warnings,
    }, nil
}

func (api *Router) embedText(ctx context.Context, text string, model string) ([]float64, error) {
    // Call Python text embedding service
    url := fmt.Sprintf("%s/embed_text", api.textEmbedServiceURL)

    reqBody, _ := json.Marshal(map[string]string{
        "text":  text,
        "model": model,
    })

    resp, err := http.Post(url, "application/json", bytes.NewBuffer(reqBody))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        Embedding []float64 `json:"embedding"`
        Model     string    `json:"model"`
        Dimension int       `json:"dimension"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return result.Embedding, nil
}

func (api *Router) buildTextSeeds(embedding []float64, model string) []recommendations.Seed {
    // Create a pseudo-seed with the text embedding
    return []recommendations.Seed{
        {
            TrackID:   "_text_query_", // Special marker
            Weight:    1.0,
            Source:    "text",
            Embedding: embedding, // Pass embedding directly
        },
    }
}
```

**Schema Changes**: Add `Embedding` field to `RecommendationSeed` in `python_services/schemas.py`:

```python
class RecommendationSeed(BaseModel):
    track_id: str
    weight: float = 1.0
    source: str
    played_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None  # NEW: Direct embedding override
```

#### 1.3 Python Recommender Engine Updates

**File**: `python_services/recommender_api.py`

Update `RecommendationEngine.recommend()` to handle direct embeddings:

```python
def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
    # ... existing validation ...

    # Separate seeds with embeddings vs. track_ids
    direct_embedding_seeds = [s for s in request.seeds if s.embedding is not None]
    track_id_seeds = [s for s in request.seeds if s.embedding is None]

    # Get embeddings for track_id seeds (existing logic)
    if track_id_seeds:
        track_names = resolver.ids_to_names([s.track_id for s in track_id_seeds])
        seed_embeddings = searcher.get_embeddings_by_name(track_names)
    else:
        seed_embeddings = {}

    # Process direct embedding seeds
    for seed in direct_embedding_seeds:
        if seed.track_id not in seed_embeddings:
            seed_embeddings[seed.track_id] = np.array(seed.embedding)

    # ... rest of similarity search logic remains the same ...
```

#### 1.4 Frontend Component

**New Component**: `ui/src/explore/TextPlaylistGenerator.jsx`

```jsx
import React, { useState } from 'react';
import { TextField, Button, Select, MenuItem } from '@material-ui/core';

export const TextPlaylistGenerator = () => {
    const [text, setText] = useState('');
    const [model, setModel] = useState('muq');
    const [loading, setLoading] = useState(false);
    const [playlist, setPlaylist] = useState(null);

    const handleGenerate = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/recommendations/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    model,
                    limit: 25,
                    diversity: 0.15
                })
            });
            const data = await response.json();
            setPlaylist(data);
        } catch (error) {
            console.error('Failed to generate playlist:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <TextField
                label="Describe the music you want"
                placeholder="e.g., upbeat rock with guitar solos"
                value={text}
                onChange={(e) => setText(e.target.value)}
                fullWidth
                multiline
                rows={2}
            />

            <Select value={model} onChange={(e) => setModel(e.target.value)}>
                <MenuItem value="muq">MuQ (Default)</MenuItem>
                <MenuItem value="mert">MERT</MenuItem>
                <MenuItem value="latent">Latent Space</MenuItem>
            </Select>

            <Button
                onClick={handleGenerate}
                disabled={!text || loading}
                variant="contained"
                color="primary"
            >
                Generate Playlist
            </Button>

            {/* Reuse existing playlist display component */}
            {playlist && <PlaylistPreview playlist={playlist} />}
        </div>
    );
};
```

**Integration**: Add tab or button in `ExploreSuggestions.jsx` to switch to text mode

#### 1.5 Testing Plan

1. Unit tests for text embedding service
2. Integration tests for Go → Python communication
3. E2E tests for frontend workflow
4. Manual testing with various text queries

---

## Idea 2: Multiple Embedder Support

### Overview
Allow users to select one or multiple embedding models. When multiple are selected, combine recommendations using union/intersection strategies.

### Architecture

```
User selects models: [muq, mert]
    ↓
For each selected model:
    - Run similarity search in respective collection
    - Get top-N candidates
    ↓
Merge Strategy:
    - Union: Combine all results, rank by max score across models
    - Intersection: Only tracks appearing in all model results
    - Priority: If no intersection, use highest priority model
```

### Implementation Details

#### 2.1 Schema Updates

**File**: `python_services/schemas.py`

```python
class RecommendationRequest(BaseModel):
    user_id: str
    user_name: str
    limit: int = 25
    mode: str
    seeds: List[RecommendationSeed] = []
    diversity: float = 0.0
    exclude_track_ids: List[str] = []

    # NEW: Multi-model support
    models: List[str] = ["muq"]  # Default to muq only
    merge_strategy: str = "union"  # union, intersection, priority
    model_priorities: Dict[str, int] = {"muq": 1, "mert": 2, "latent": 3}
```

#### 2.2 Database Query Updates

**File**: `python_services/database_query.py`

```python
class MultiModelSimilaritySearcher:
    """Searches across multiple embedding models"""

    def __init__(self, milvus_uri: str):
        self.client = MilvusClient(uri=milvus_uri)

        # Map model names to collections
        self.collection_map = {
            "muq": "embedding",
            "mert": "mert_embedding",
            "latent": "latent_embedding"
        }

    def search_multi_model(
        self,
        embeddings: Dict[str, np.ndarray],  # model -> embedding
        top_k: int = 25,
        exclude_names: List[str] = None,
        merge_strategy: str = "union"
    ) -> List[Dict]:
        """
        Search across multiple models and merge results

        Args:
            embeddings: Dict mapping model name to embedding vector
            top_k: Number of results per model
            exclude_names: Track names to exclude
            merge_strategy: How to combine results (union/intersection/priority)

        Returns:
            List of {name, track_id, distance, models} dicts
        """
        # Run searches in parallel
        all_results = {}
        for model, embedding in embeddings.items():
            collection = self.collection_map[model]
            results = self._search_single_model(
                collection, embedding, top_k * 2, exclude_names
            )
            all_results[model] = {r['name']: r for r in results}

        # Merge based on strategy
        if merge_strategy == "union":
            return self._merge_union(all_results, top_k)
        elif merge_strategy == "intersection":
            return self._merge_intersection(all_results, top_k)
        elif merge_strategy == "priority":
            return self._merge_priority(all_results, top_k, model_priorities)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    def _merge_union(self, all_results: Dict[str, Dict], top_k: int) -> List[Dict]:
        """
        Union: Combine all results, rank by maximum similarity across models
        """
        track_scores = {}  # track_name -> {max_score, models, data}

        for model, results in all_results.items():
            for track_name, data in results.items():
                if track_name not in track_scores:
                    track_scores[track_name] = {
                        'max_score': data['distance'],
                        'models': [model],
                        'data': data
                    }
                else:
                    # Update max score
                    track_scores[track_name]['max_score'] = max(
                        track_scores[track_name]['max_score'],
                        data['distance']
                    )
                    track_scores[track_name]['models'].append(model)

        # Sort by max score and return top-k
        sorted_tracks = sorted(
            track_scores.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True
        )

        return [
            {
                'name': name,
                'track_id': info['data']['track_id'],
                'distance': info['max_score'],
                'models': info['models']
            }
            for name, info in sorted_tracks[:top_k]
        ]

    def _merge_intersection(self, all_results: Dict[str, Dict], top_k: int) -> List[Dict]:
        """
        Intersection: Only tracks appearing in ALL model results
        """
        # Find tracks present in all models
        all_tracks = [set(results.keys()) for results in all_results.values()]
        common_tracks = set.intersection(*all_tracks)

        # Score by average similarity across models
        track_scores = []
        for track_name in common_tracks:
            avg_score = np.mean([
                all_results[model][track_name]['distance']
                for model in all_results.keys()
            ])
            track_scores.append((track_name, avg_score, all_results))

        # Sort and return
        track_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                'name': name,
                'track_id': all_results[next(iter(all_results))][name]['track_id'],
                'distance': score,
                'models': list(all_results.keys())
            }
            for name, score, _ in track_scores[:top_k]
        ]

    def _merge_priority(
        self,
        all_results: Dict[str, Dict],
        top_k: int,
        priorities: Dict[str, int]
    ) -> List[Dict]:
        """
        Priority: Try intersection first, fall back to highest priority model
        """
        # Try intersection
        intersection_results = self._merge_intersection(all_results, top_k)

        if len(intersection_results) >= top_k:
            return intersection_results

        # Fall back to highest priority model
        sorted_models = sorted(priorities.items(), key=lambda x: x[1])
        primary_model = sorted_models[0][0]

        primary_results = all_results[primary_model]
        return [
            {
                'name': name,
                'track_id': data['track_id'],
                'distance': data['distance'],
                'models': [primary_model]
            }
            for name, data in sorted(
                primary_results.items(),
                key=lambda x: x[1]['distance'],
                reverse=True
            )[:top_k]
        ]
```

#### 2.3 Recommender Engine Updates

**File**: `python_services/recommender_api.py`

```python
def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
    # Get seed embeddings for each requested model
    model_embeddings = {}

    for model in request.models:
        if model == "muq":
            # Use existing embedding collection
            model_embeddings[model] = self._get_embeddings_for_model(
                request.seeds, "muq"
            )
        elif model == "mert":
            model_embeddings[model] = self._get_embeddings_for_model(
                request.seeds, "mert"
            )
        elif model == "latent":
            model_embeddings[model] = self._get_embeddings_for_model(
                request.seeds, "latent"
            )

    # Multi-model search
    searcher = MultiModelSimilaritySearcher(self.milvus_uri)
    candidates = searcher.search_multi_model(
        embeddings=model_embeddings,
        top_k=request.limit * 3,
        exclude_names=request.exclude_track_ids,
        merge_strategy=request.merge_strategy
    )

    # ... rest of processing ...
```

#### 2.4 Frontend Updates

**File**: `ui/src/explore/ExploreSuggestions.jsx`

Add model selection UI:

```jsx
const [selectedModels, setSelectedModels] = useState(['muq']);
const [mergeStrategy, setMergeStrategy] = useState('union');

<FormControl>
    <InputLabel>Embedding Models</InputLabel>
    <Select
        multiple
        value={selectedModels}
        onChange={(e) => setSelectedModels(e.target.value)}
    >
        <MenuItem value="muq">MuQ (1536D)</MenuItem>
        <MenuItem value="mert">MERT (76,800D)</MenuItem>
        <MenuItem value="latent">Latent Space (576D)</MenuItem>
    </Select>
</FormControl>

{selectedModels.length > 1 && (
    <FormControl>
        <InputLabel>Merge Strategy</InputLabel>
        <Select
            value={mergeStrategy}
            onChange={(e) => setMergeStrategy(e.target.value)}
        >
            <MenuItem value="union">Union (combine all)</MenuItem>
            <MenuItem value="intersection">Intersection (common only)</MenuItem>
            <MenuItem value="priority">Priority (fallback)</MenuItem>
        </Select>
    </FormControl>
)}
```

---

## Idea 3: Union Bounds for False Positive Reduction

### Overview
Use statistical union bounds to reduce false positives by requiring tracks to appear in multiple model searches. This is a refinement of Idea 2's intersection strategy with configurable thresholds.

### Implementation Details

#### 3.1 Algorithm

**Union Bound Principle**: A track is only recommended if it appears in at least K out of N selected models, where K is user-configurable.

```python
def union_bound_filter(
    all_results: Dict[str, List[Dict]],
    min_models: int = 2,
    top_k: int = 25
) -> List[Dict]:
    """
    Filter recommendations using union bound principle

    Args:
        all_results: Dict mapping model name to list of results
        min_models: Minimum number of models that must agree
        top_k: Final number of results

    Returns:
        Filtered and ranked results
    """
    # Count how many models returned each track
    track_appearances = {}

    for model, results in all_results.items():
        for result in results:
            track_name = result['name']
            if track_name not in track_appearances:
                track_appearances[track_name] = {
                    'count': 0,
                    'scores': [],
                    'models': []
                }
            track_appearances[track_name]['count'] += 1
            track_appearances[track_name]['scores'].append(result['distance'])
            track_appearances[track_name]['models'].append(model)

    # Filter by minimum model count
    filtered = {
        name: info for name, info in track_appearances.items()
        if info['count'] >= min_models
    }

    # Rank by: (1) number of models, (2) average score
    ranked = sorted(
        filtered.items(),
        key=lambda x: (x[1]['count'], np.mean(x[1]['scores'])),
        reverse=True
    )

    return ranked[:top_k]
```

#### 3.2 Frontend Settings

Add new setting in user preferences:

```jsx
<TextField
    label="Minimum Model Agreement"
    type="number"
    value={minModelAgreement}
    onChange={(e) => setMinModelAgreement(parseInt(e.target.value))}
    helperText="Require tracks to appear in at least N models (reduces false positives)"
    inputProps={{ min: 1, max: selectedModels.length }}
/>
```

---

## Idea 4: Negative Prompting for Playlist Generation

### Overview
Allow users to specify negative prompts (text descriptions of music to avoid) when generating playlists. Negative prompts are treated similarly to 1-star rated songs.

### Architecture

```
Positive prompt: "upbeat rock music"
Negative prompts: ["slow ballads", "acoustic guitar"]
    ↓
Embed all prompts (positive + negative)
    ↓
Search with positive prompt
    ↓
For each candidate:
    - Compute similarity to negative prompts
    - Apply penalty if similar to negative prompts
    ↓
Return penalized rankings
```

### Implementation Details

#### 4.1 Schema Updates

```python
class RecommendationRequest(BaseModel):
    # ... existing fields ...

    # NEW: Negative prompts
    negative_prompts: List[str] = []
    negative_prompt_penalty: float = 0.85  # Similar to lowRatingPenalty
```

#### 4.2 Backend Implementation

**File**: `server/nativeapi/recommendations.go`

```go
func (api *Router) buildNegativeEmbeddings(ctx context.Context, prompts []string, model string) ([][]float64, error) {
    embeddings := make([][]float64, len(prompts))

    for i, prompt := range prompts {
        emb, err := api.embedText(ctx, prompt, model)
        if err != nil {
            return nil, err
        }
        embeddings[i] = emb
    }

    return embeddings, nil
}
```

#### 4.3 Python Recommender Updates

```python
def apply_negative_prompt_penalty(
    candidates: List[Dict],
    negative_embeddings: List[np.ndarray],
    track_embeddings: Dict[str, np.ndarray],
    penalty: float = 0.85
) -> List[Dict]:
    """
    Apply penalty to candidates similar to negative prompts

    Args:
        candidates: List of candidate tracks with scores
        negative_embeddings: List of negative prompt embeddings
        track_embeddings: Map of track names to their embeddings
        penalty: Penalty multiplier (like lowRatingPenalty)

    Returns:
        Candidates with adjusted scores
    """
    for candidate in candidates:
        track_name = candidate['name']
        track_emb = track_embeddings.get(track_name)

        if track_emb is None:
            continue

        # Compute max similarity to any negative prompt
        max_negative_sim = 0.0
        for neg_emb in negative_embeddings:
            sim = cosine_similarity(track_emb, neg_emb)
            max_negative_sim = max(max_negative_sim, sim)

        # Apply penalty proportional to negative similarity
        # If similarity is high, apply stronger penalty
        penalty_factor = 1.0 - (max_negative_sim * (1.0 - penalty))
        candidate['score'] *= penalty_factor
        candidate['negative_similarity'] = max_negative_sim

    return candidates
```

#### 4.4 Frontend Component

```jsx
const [negativePrompts, setNegativePrompts] = useState([]);
const [negativePenalty, setNegativePenalty] = useState(0.85);

<div>
    <Typography variant="h6">Negative Prompts (Optional)</Typography>
    <Typography variant="caption">
        Describe music styles to avoid
    </Typography>

    {negativePrompts.map((prompt, idx) => (
        <div key={idx}>
            <TextField
                value={prompt}
                onChange={(e) => {
                    const updated = [...negativePrompts];
                    updated[idx] = e.target.value;
                    setNegativePrompts(updated);
                }}
                placeholder="e.g., slow ballads"
            />
            <IconButton onClick={() => {
                setNegativePrompts(negativePrompts.filter((_, i) => i !== idx));
            }}>
                <DeleteIcon />
            </IconButton>
        </div>
    ))}

    <Button onClick={() => setNegativePrompts([...negativePrompts, ''])}>
        Add Negative Prompt
    </Button>

    <Slider
        value={negativePenalty}
        onChange={(e, val) => setNegativePenalty(val)}
        min={0.3}
        max={1.0}
        step={0.05}
        valueLabelDisplay="auto"
        marks={[
            { value: 0.5, label: 'Strong Penalty' },
            { value: 0.85, label: 'Default' },
            { value: 1.0, label: 'No Penalty' }
        ]}
    />
</div>
```

---

## Idea 5: Batch Re-embedding All Songs

### Overview
Provide a system administration feature to clear the Milvus database and re-embed all songs using all available embedding models. Useful for model updates or database maintenance.

### Architecture

```
User clicks "Re-embed All Songs" in settings
    ↓
Frontend shows progress modal
    ↓
Backend starts batch job
    ↓
For each song in database:
    1. Load audio file
    2. Generate embeddings with all models
    3. Store in respective Milvus collections
    4. Update progress
    ↓
Notify frontend of completion
```

### Implementation Details

#### 5.1 Backend Batch Job

**New File**: `python_services/batch_embedding_job.py`

```python
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import sqlite3
from tqdm import tqdm
import time

from embedding_models import MuQEmbeddingModel, MertModel, MusicLatentSpaceModel
from database_query import MilvusSimilaritySearcher
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)

@dataclass
class BatchJobProgress:
    total_tracks: int
    processed_tracks: int
    failed_tracks: int
    current_track: Optional[str]
    status: str  # running, completed, failed, cancelled
    started_at: float
    estimated_completion: Optional[float]

class BatchEmbeddingJob:
    """Manages batch re-embedding of entire music library"""

    def __init__(
        self,
        db_path: str,
        music_root: str,
        milvus_uri: str,
        checkpoint_interval: int = 100
    ):
        self.db_path = db_path
        self.music_root = Path(music_root)
        self.milvus_uri = milvus_uri
        self.checkpoint_interval = checkpoint_interval

        self.progress = BatchJobProgress(
            total_tracks=0,
            processed_tracks=0,
            failed_tracks=0,
            current_track=None,
            status="initialized",
            started_at=0,
            estimated_completion=None
        )

        # Initialize models (lazy loading)
        self.models = {
            "muq": MuQEmbeddingModel(),
            "mert": MertModel(),
            "latent": MusicLatentSpaceModel()
        }

        self.milvus_client = MilvusClient(uri=milvus_uri)
        self._cancelled = False

    def get_all_tracks(self) -> List[Dict]:
        """Query all tracks from Navidrome database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, path, artist, title, album
            FROM media_file
            WHERE path IS NOT NULL
            ORDER BY id
        """)

        tracks = [dict(row) for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Found {len(tracks)} tracks in database")
        return tracks

    def clear_embeddings(self, collections: List[str] = None):
        """Clear specified Milvus collections"""
        if collections is None:
            collections = ["embedding", "mert_embedding", "latent_embedding"]

        for collection in collections:
            try:
                self.milvus_client.drop_collection(collection)
                logger.info(f"Dropped collection: {collection}")
            except Exception as e:
                logger.warning(f"Failed to drop collection {collection}: {e}")

        # Recreate schemas
        for model_name, model in self.models.items():
            model.ensure_milvus_schemas(self.milvus_client)
            model.ensure_milvus_index(self.milvus_client)

    def run(self, models_to_use: List[str] = None, clear_existing: bool = True):
        """
        Run batch embedding job

        Args:
            models_to_use: List of model names (default: all)
            clear_existing: Whether to clear existing embeddings first
        """
        if models_to_use is None:
            models_to_use = ["muq", "mert", "latent"]

        logger.info(f"Starting batch embedding job with models: {models_to_use}")

        # Get all tracks
        tracks = self.get_all_tracks()
        self.progress.total_tracks = len(tracks)
        self.progress.status = "running"
        self.progress.started_at = time.time()

        # Clear existing embeddings
        if clear_existing:
            collections = [
                {"muq": "embedding", "mert": "mert_embedding", "latent": "latent_embedding"}[m]
                for m in models_to_use
            ]
            self.clear_embeddings(collections)

        # Process tracks
        failed_tracks = []

        for idx, track in enumerate(tqdm(tracks, desc="Embedding tracks")):
            if self._cancelled:
                self.progress.status = "cancelled"
                logger.info("Job cancelled by user")
                break

            self.progress.current_track = f"{track['artist']} - {track['title']}"

            try:
                self._process_track(track, models_to_use)
                self.progress.processed_tracks += 1
            except Exception as e:
                logger.error(f"Failed to process track {track['id']}: {e}")
                failed_tracks.append((track['id'], str(e)))
                self.progress.failed_tracks += 1

            # Update estimated completion
            if idx > 0:
                elapsed = time.time() - self.progress.started_at
                rate = elapsed / (idx + 1)
                remaining = self.progress.total_tracks - (idx + 1)
                self.progress.estimated_completion = time.time() + (rate * remaining)

        # Finalize
        if self._cancelled:
            self.progress.status = "cancelled"
        elif self.progress.failed_tracks > 0:
            self.progress.status = "completed_with_errors"
        else:
            self.progress.status = "completed"

        logger.info(f"Job completed: {self.progress.processed_tracks}/{self.progress.total_tracks} tracks")
        logger.info(f"Failed tracks: {self.progress.failed_tracks}")

        return {
            "progress": self.progress,
            "failed_tracks": failed_tracks
        }

    def _process_track(self, track: Dict, models: List[str]):
        """Process a single track with specified models"""
        # Resolve audio file path
        audio_path = self.music_root / track['path']
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        canonical_name = f"{track['artist']} - {track['title']}"

        # Generate embeddings with each model
        for model_name in models:
            model = self.models[model_name]

            # Generate embedding
            result = model.embed_music(str(audio_path), canonical_name)

            # Store in Milvus
            for segment in result['segments']:
                collection_map = {
                    "muq": "embedding",
                    "mert": "mert_embedding",
                    "latent": "latent_embedding"
                }
                collection = collection_map[model_name]

                self.milvus_client.insert(
                    collection_name=collection,
                    data=[{
                        "name": segment['title'],
                        "embedding": segment['embedding'],
                        "offset": segment['offset_seconds'],
                        "model_id": result['model_id']
                    }]
                )

    def cancel(self):
        """Cancel the running job"""
        self._cancelled = True
        logger.info("Job cancellation requested")

    def get_progress(self) -> BatchJobProgress:
        """Get current job progress"""
        return self.progress

# Global job instance (for API access)
_current_job: Optional[BatchEmbeddingJob] = None

def start_batch_job(*args, **kwargs) -> BatchEmbeddingJob:
    global _current_job
    _current_job = BatchEmbeddingJob(*args, **kwargs)
    return _current_job

def get_current_job() -> Optional[BatchEmbeddingJob]:
    return _current_job
```

#### 5.2 FastAPI Endpoints

**File**: `python_services/recommender_api.py` (add endpoints)

```python
from batch_embedding_job import BatchEmbeddingJob, start_batch_job, get_current_job
import threading

@app.post("/batch/start")
async def start_batch_embedding(
    models: List[str] = ["muq", "mert", "latent"],
    clear_existing: bool = True
):
    """Start batch re-embedding job"""
    current_job = get_current_job()
    if current_job and current_job.progress.status == "running":
        raise HTTPException(400, "A job is already running")

    job = start_batch_job(
        db_path=os.getenv("NAVIDROME_DB_PATH", "navidrome.db"),
        music_root=os.getenv("NAVIDROME_MUSIC_ROOT", "/music"),
        milvus_uri=os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")
    )

    # Run in background thread
    thread = threading.Thread(
        target=job.run,
        args=(models, clear_existing),
        daemon=True
    )
    thread.start()

    return {"status": "started", "job_id": id(job)}

@app.get("/batch/progress")
async def get_batch_progress():
    """Get current batch job progress"""
    job = get_current_job()
    if not job:
        return {"status": "no_job"}

    progress = job.get_progress()
    return {
        "total_tracks": progress.total_tracks,
        "processed_tracks": progress.processed_tracks,
        "failed_tracks": progress.failed_tracks,
        "current_track": progress.current_track,
        "status": progress.status,
        "progress_percent": (
            progress.processed_tracks / progress.total_tracks * 100
            if progress.total_tracks > 0 else 0
        ),
        "estimated_completion": progress.estimated_completion
    }

@app.post("/batch/cancel")
async def cancel_batch_job():
    """Cancel running batch job"""
    job = get_current_job()
    if not job:
        raise HTTPException(404, "No job running")

    job.cancel()
    return {"status": "cancelling"}
```

#### 5.3 Frontend Component

**New Component**: `ui/src/settings/BatchEmbeddingPanel.jsx`

```jsx
import React, { useState, useEffect } from 'react';
import {
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    LinearProgress,
    Typography,
    Checkbox,
    FormControlLabel
} from '@material-ui/core';

export const BatchEmbeddingPanel = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [progress, setProgress] = useState(null);
    const [selectedModels, setSelectedModels] = useState(['muq', 'mert', 'latent']);

    useEffect(() => {
        if (isRunning) {
            const interval = setInterval(async () => {
                const response = await fetch('/api/batch/progress');
                const data = await response.json();
                setProgress(data);

                if (data.status === 'completed' || data.status === 'cancelled') {
                    setIsRunning(false);
                    clearInterval(interval);
                }
            }, 1000);

            return () => clearInterval(interval);
        }
    }, [isRunning]);

    const handleStart = async () => {
        const response = await fetch('/api/batch/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                models: selectedModels,
                clear_existing: true
            })
        });

        if (response.ok) {
            setIsRunning(true);
        }
    };

    const handleCancel = async () => {
        await fetch('/api/batch/cancel', { method: 'POST' });
    };

    return (
        <div>
            <Typography variant="h6">Batch Re-embedding</Typography>
            <Typography variant="body2" color="textSecondary">
                Re-generate embeddings for all tracks in your library.
                This may take several hours for large libraries.
            </Typography>

            <Button
                variant="contained"
                color="secondary"
                onClick={() => setIsOpen(true)}
                disabled={isRunning}
            >
                {isRunning ? 'Job Running...' : 'Start Re-embedding'}
            </Button>

            <Dialog open={isOpen} onClose={() => !isRunning && setIsOpen(false)}>
                <DialogTitle>Batch Re-embedding</DialogTitle>
                <DialogContent>
                    {!isRunning ? (
                        <>
                            <Typography>Select models to use:</Typography>
                            {['muq', 'mert', 'latent'].map(model => (
                                <FormControlLabel
                                    key={model}
                                    control={
                                        <Checkbox
                                            checked={selectedModels.includes(model)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedModels([...selectedModels, model]);
                                                } else {
                                                    setSelectedModels(
                                                        selectedModels.filter(m => m !== model)
                                                    );
                                                }
                                            }}
                                        />
                                    }
                                    label={model.toUpperCase()}
                                />
                            ))}
                            <Typography color="error" style={{ marginTop: 16 }}>
                                Warning: This will clear all existing embeddings!
                            </Typography>
                        </>
                    ) : (
                        <>
                            <LinearProgress
                                variant="determinate"
                                value={progress?.progress_percent || 0}
                            />
                            <Typography>
                                Progress: {progress?.processed_tracks || 0} / {progress?.total_tracks || 0}
                            </Typography>
                            <Typography variant="caption">
                                Current: {progress?.current_track || 'Loading...'}
                            </Typography>
                            {progress?.estimated_completion && (
                                <Typography variant="caption">
                                    Estimated completion: {
                                        new Date(progress.estimated_completion * 1000)
                                            .toLocaleTimeString()
                                    }
                                </Typography>
                            )}
                        </>
                    )}
                </DialogContent>
                <DialogActions>
                    {isRunning ? (
                        <Button onClick={handleCancel} color="secondary">
                            Cancel
                        </Button>
                    ) : (
                        <>
                            <Button onClick={() => setIsOpen(false)}>
                                Close
                            </Button>
                            <Button
                                onClick={handleStart}
                                color="primary"
                                disabled={selectedModels.length === 0}
                            >
                                Start
                            </Button>
                        </>
                    )}
                </DialogActions>
            </Dialog>
        </div>
    );
};
```

---

## Stub Text Embedding Models

Since the text embedding models are still being trained, create stub implementations for testing:

**New File**: `python_services/stub_text_embedders.py`

```python
"""
Stub implementations of text embedding models for development/testing.
These will be replaced with trained models once available.
"""

import numpy as np
import hashlib
from typing import List

class StubTextEmbedder:
    """Base stub that generates deterministic random embeddings"""

    def __init__(self, dimension: int, model_name: str):
        self.dimension = dimension
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash"""
        # Use hash of text as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)

        # Generate random vector
        embedding = rng.randn(self.dimension).astype(np.float32)

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        return np.stack([self.embed_text(text) for text in texts])

class StubMuQTextEmbedder(StubTextEmbedder):
    """Stub for MuQ text embeddings (1536D)"""
    def __init__(self):
        super().__init__(dimension=1536, model_name="muq_stub")

class StubMERTTextEmbedder(StubTextEmbedder):
    """Stub for MERT text embeddings (76,800D)"""
    def __init__(self):
        super().__init__(dimension=76_800, model_name="mert_stub")

class StubLatentTextEmbedder(StubTextEmbedder):
    """Stub for Latent Space text embeddings (576D)"""
    def __init__(self):
        super().__init__(dimension=576, model_name="latent_stub")

def get_stub_embedder(model: str) -> StubTextEmbedder:
    """Factory function to get stub embedder by model name"""
    embedders = {
        "muq": StubMuQTextEmbedder,
        "mert": StubMERTTextEmbedder,
        "latent": StubLatentTextEmbedder
    }
    return embedders[model]()
```

Update `text_embedding_service.py` to use stubs when checkpoints unavailable:

```python
def get_embedder(self, model_name: str) -> TextToAudioEmbedder:
    """Get or create embedder for specified model"""
    if model_name not in self.embedders:
        checkpoint_path = f"{self.checkpoint_dir}/{model_name}_best_r1.pt"

        # Try loading real model first
        if os.path.exists(checkpoint_path):
            self.embedders[model_name] = TextToAudioEmbedder(
                checkpoint_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            # Fall back to stub
            logger.warning(f"Checkpoint not found for {model_name}, using stub")
            from stub_text_embedders import get_stub_embedder
            self.embedders[model_name] = get_stub_embedder(model_name)

    return self.embedders[model_name]
```

---

## Implementation Roadmap

### Phase 1: Core Text Embedding (Weeks 1-2)
- [ ] Implement text_embedding_service.py with stub embedders
- [ ] Add /embed_text endpoint to FastAPI
- [ ] Update Go backend with /recommendations/text endpoint
- [ ] Update Python recommender to handle direct embeddings
- [ ] Basic frontend for text input
- [ ] Integration tests

### Phase 2: Multi-Model Support (Weeks 3-4)
- [ ] Implement MultiModelSimilaritySearcher
- [ ] Add model selection to schemas
- [ ] Update recommender engine for multi-model search
- [ ] Frontend multi-model UI
- [ ] Merge strategy implementations (union, intersection, priority)
- [ ] Testing all merge strategies

### Phase 3: Union Bounds & Negative Prompts (Week 5)
- [ ] Implement union bound filtering algorithm
- [ ] Add negative prompt support to schemas
- [ ] Implement negative prompt penalty calculation
- [ ] Frontend negative prompt UI
- [ ] Integration testing

### Phase 4: Batch Re-embedding (Week 6)
- [ ] Implement BatchEmbeddingJob class
- [ ] Add batch API endpoints
- [ ] Frontend progress panel
- [ ] Admin UI integration
- [ ] Performance testing with large libraries

### Phase 5: Production Hardening (Week 7)
- [ ] Error handling and recovery
- [ ] Logging and monitoring
- [ ] Performance optimization
- [ ] Documentation
- [ ] User guide

### Phase 6: Real Model Integration (Week 8+)
- [ ] Replace stubs with trained text embedding models
- [ ] Benchmark and tune hyperparameters
- [ ] A/B testing with users
- [ ] Feedback collection and iteration

---

## Testing Strategy

### Unit Tests
- Text embedding service (with stubs)
- Multi-model search algorithms
- Negative prompt penalty calculation
- Union bound filtering
- Batch job progress tracking

### Integration Tests
- Frontend → Backend → Python service flow
- Milvus multi-collection queries
- Model selection and switching
- Progress reporting for batch jobs

### End-to-End Tests
- Complete text-to-playlist workflow
- Multi-model recommendation generation
- Negative prompt filtering
- Batch re-embedding of small dataset

### Performance Tests
- Text embedding latency
- Multi-model search scalability
- Batch job throughput
- Memory usage under load

---

## Deployment Considerations

### Environment Variables

```bash
# Text embedding service
TEXT_EMBEDDING_CHECKPOINT_DIR=/path/to/checkpoints
TEXT_EMBEDDING_SERVICE_PORT=9003

# Multi-model support
MILVUS_COLLECTIONS=embedding,mert_embedding,latent_embedding

# Batch processing
NAVIDROME_MUSIC_ROOT=/music
BATCH_JOB_WORKERS=4
```

### Resource Requirements

- **Text Embedding Service**:
  - GPU: 8GB+ VRAM (for full models)
  - CPU: 4+ cores
  - RAM: 16GB+

- **Batch Re-embedding**:
  - Storage: Temporary space for Milvus backup
  - Time: ~1-2 hours per 1000 tracks (depends on models)

### Monitoring

- Track text embedding request latency
- Monitor multi-model search performance
- Alert on batch job failures
- Log negative prompt usage patterns

---

## Future Enhancements

1. **Hybrid Search**: Combine text and audio-based seeds in single query
2. **Prompt Weighting**: Allow users to weight positive/negative prompts
3. **Prompt Templates**: Pre-defined mood/genre templates
4. **Collaborative Filtering**: Combine embedding-based with user behavior signals
5. **Incremental Re-embedding**: Only re-embed new/modified tracks
6. **Model A/B Testing**: Compare recommendation quality across models
7. **Explainability**: Show which models contributed to each recommendation

---

## References

- Current codebase architecture (see exploration report above)
- Milvus documentation: https://milvus.io/docs
- FastAPI documentation: https://fastapi.tiangolo.com
- React Material-UI: https://mui.com

---

## Conclusion

This implementation plan provides a comprehensive roadmap for all 5 enhancement ideas. The phased approach allows for incremental development and testing, with clear dependencies and milestones. The use of stub embedders enables development to proceed while real models are being trained.

**Key Success Metrics**:
- Text query response time < 2 seconds
- Multi-model search < 3 seconds
- Batch re-embedding throughput > 500 tracks/hour
- User satisfaction with recommendations > 80%
