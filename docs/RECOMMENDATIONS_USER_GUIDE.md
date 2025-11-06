# Navidrome Recommender System - User Guide

## Overview

Navidrome's recommender system uses advanced machine learning to help you discover music in your library. It analyzes your listening habits, favorites, and the audio characteristics of your music to generate personalized playlists.

## Features

### 1. Automatic Playlist Generation

The system offers several types of automatically generated playlists:

#### Recent Mix
- Based on tracks you've recently played
- Finds similar music to what you've been listening to
- Best for: Continuing your current music mood

#### Favorites Mix
- Based on your starred and highly-rated tracks
- Combines your favorites to create cohesive playlists
- Best for: Reliable playlists of music you love

#### All Metrics Mix
- Combines both recent plays and favorites
- Balanced approach for discovering new music
- Best for: General music discovery

#### Discovery Mix
- Higher diversity and exploration
- Finds music that's different from your usual choices
- Best for: Breaking out of your comfort zone

### 2. Text-Based Playlist Generation ⭐ NEW

Generate playlists by describing the music you want in natural language.

**How to use:**
1. Navigate to the Explore page
2. Select the "Text Generator" tab
3. Enter a description like:
   - "upbeat rock with guitar solos"
   - "chill jazz for studying"
   - "energetic dance music"
   - "melancholic acoustic songs"
4. Choose your embedding model (MuQ is recommended for most users)
5. Optionally add negative prompts to avoid certain styles
6. Click "Generate Playlist"

**Tips:**
- Be specific but not too narrow
- Use descriptive words about mood, genre, instruments
- Experiment with different models for varied results

### 3. Negative Prompting

Refine your text-based playlists by specifying what to avoid.

**Example:**
- Positive: "upbeat rock music"
- Negative: "slow ballads", "acoustic guitar"
- Result: Fast-paced rock without slow songs or acoustic elements

**Penalty Slider:**
- **Strong (0.3)**: Aggressively filters out negative matches
- **Default (0.85)**: Balanced filtering
- **None (1.0)**: Minimal filtering

### 4. Multiple Embedding Models

The system uses three different AI models to understand your music:

#### MuQ (Default)
- **Dimensions:** 1,536
- **Best for:** General use, balanced recommendations
- **Speed:** Fast
- **Description:** MuQ-MuLan model, good all-around performance

#### MERT
- **Dimensions:** 76,800
- **Best for:** Detailed feature matching, specific genres
- **Speed:** Slower, more detailed
- **Description:** High-dimensional analysis for nuanced recommendations

#### Latent Space
- **Dimensions:** 576
- **Best for:** Quick exploration, compact representation
- **Speed:** Very fast
- **Description:** Efficient model for rapid discovery

### 5. Multi-Model Recommendations

Combine multiple models for better results.

**Merge Strategies:**

#### Union (Default)
- Combines results from all selected models
- Ranks tracks that appear in multiple models higher
- Best for: Maximum variety with quality assurance

#### Intersection
- Only includes tracks that appear in ALL selected models
- More conservative, higher precision
- Best for: When you want high confidence recommendations

#### Priority
- Uses intersection first, falls back to highest priority model
- Balances quality and quantity
- Best for: Ensuring you get enough recommendations

#### Model Agreement
- Set minimum number of models that must agree (1-3)
- Higher values = more conservative recommendations
- Best for: Reducing false positives

### 6. Custom Playlists

Create playlists from specific songs:

1. Go to Explore page
2. Select "Custom" mode
3. Search and select seed tracks
4. Click "Generate" to find similar music

**Tips:**
- Use 2-5 seed tracks for best results
- Mix different styles for more variety
- Exclude playlists you've already heard

## Settings

### Customization Options

Access settings from the Explore page:

#### Mix Length
- Number of tracks in generated playlists
- Range: 10-100 tracks
- Default: 25 tracks

#### Base Diversity
- How much variety in recommendations
- Range: 0.0 (very similar) to 1.0 (very diverse)
- Default: 0.15
- **Tip:** Start low, increase if recommendations feel repetitive

#### Discovery Exploration
- Extra diversity boost for Discovery Mix
- Range: 0.0 to 1.0
- Default: 0.5

#### Seed Recency Window
- How far back to look for recent plays
- Range: 7-120 days
- Default: 30 days

#### Favorites Blend Weight
- Balance between recent plays and favorites in "All Metrics" mode
- Range: 0.0 (all recent) to 1.0 (all favorites)
- Default: 0.3 (30% favorites, 70% recent)

#### Low Rating Penalty
- How much to avoid tracks you've rated 1-2 stars
- Range: 0.3 (strong avoidance) to 1.0 (no penalty)
- Default: 0.7

## Advanced Features

### Playlist Exclusion

Exclude tracks from specific playlists to avoid repetition:

1. When generating a playlist, look for "Exclude Playlists"
2. Select playlists you've already listened to
3. Generate - these tracks won't appear in results

### Dislike Signals

Tracks you rate 1-2 stars are automatically de-prioritized in future recommendations.

## Troubleshooting

### "No recommendations found"
**Possible causes:**
- Not enough listening history (play at least 10-20 songs)
- No starred or rated tracks (for Favorites mode)
- Very restrictive filters
- All similar tracks already excluded

**Solutions:**
- Listen to more music to build history
- Try a different recommendation mode
- Reduce diversity setting
- Clear playlist exclusions

### "Text recommendations not working"
**Possible causes:**
- Text embedding service not running
- Using stub embedders (development mode)

**Solutions:**
- Check with your administrator
- Try a different model (MuQ/MERT/Latent)
- Simplify your text query

### Recommendations feel repetitive
**Solutions:**
- Increase diversity setting (try 0.25-0.35)
- Use Discovery Mix mode
- Try multi-model approach with union strategy
- Exclude playlists you've heard recently

### Recommendations are too random
**Solutions:**
- Decrease diversity setting (try 0.05-0.10)
- Use Favorites Mix instead of Discovery
- Use intersection merge strategy with multiple models
- Increase model agreement requirement

## Best Practices

### Getting Started
1. Start with "Recent Mix" using default settings
2. Star tracks you love during normal listening
3. Rate disliked tracks 1-2 stars to train the system
4. Explore other modes once you have 50+ plays

### For Best Results
- **Regularly star/rate tracks** - This improves recommendations
- **Listen to diverse music** - Helps the system understand your range
- **Use exclusions** - Avoid hearing the same tracks repeatedly
- **Experiment with models** - Different models work better for different genres
- **Adjust diversity per mood** - Low for focused listening, high for exploration

### Power User Tips
- **Multi-model union with min agreement = 2**: Great for discovering gems
- **Text mode + negative prompts**: Very precise playlist creation
- **Custom mode + high diversity**: Explore variations of a specific song
- **Intersection of 3 models**: Ultra-high-quality, but fewer results

## Privacy & Data

### What data is used?
- Play counts and timestamps
- Star ratings and ratings (1-5 stars)
- Track metadata (artist, title, album, genre)
- Audio analysis embeddings (locally computed)

### What is NOT used?
- Your queries are not shared externally
- No data leaves your Navidrome instance
- All processing happens on your server

### Can I reset recommendations?
- Clear your play history to start fresh
- Unstar/unrate tracks to change preferences
- Administrators can re-embed the entire library if needed

## FAQ

**Q: How long does it take for recommendations to appear?**
A: After 10-20 plays and a few starred tracks, you should get good results.

**Q: Can I use this on mobile?**
A: Yes! Access through your mobile browser or compatible apps.

**Q: Do recommendations work offline?**
A: The recommendation service requires server access, but once a playlist is generated, you can play it offline (if using a compatible app).

**Q: What's the difference between the three models?**
A: MuQ is balanced, MERT is detailed/precise, Latent is fast/broad. Try all three to see what works for your library and preferences.

**Q: Can I share generated playlists?**
A: Yes! Save a generated mix as a playlist, then share it like any other playlist.

**Q: How often are embeddings updated?**
A: New tracks are embedded when added to the library. Existing tracks don't need re-embedding unless the models are updated.

**Q: Why do some tracks never appear in recommendations?**
A: Tracks need audio embeddings to be recommended. Check with your admin if specific tracks are missing. Also, very dissimilar tracks might not rank highly for your listening patterns.

## Getting Help

If you encounter issues:
1. Check your settings - sometimes a simple reset helps
2. Try a different recommendation mode or model
3. Clear browser cache and reload
4. Contact your Navidrome administrator
5. Report issues on the Navidrome GitHub page

## Conclusion

The Navidrome recommender system is a powerful tool for music discovery. Start simple with default settings, then experiment with advanced features as you become comfortable. Happy listening!

---

**Version:** 2.0
**Last Updated:** 2025-11-05
**See Also:** [Admin Guide](RECOMMENDATIONS_ADMIN_GUIDE.md) | [API Documentation](API.md)
