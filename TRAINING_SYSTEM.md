# Training Data Collection System

## Overview

A system to collect training data by recording lip-reading attempts, capturing corrections, and storing them for model fine-tuning.

## Proposed Architecture

### 1. Data Collection Flow

```
User speaks → Lip-reading → Raw output → LLM correction → User review
                    ↓                                           ↓
              Video saved                              Correction saved
                    ↓                                           ↓
                    └─────────────→ Training Dataset ←─────────┘
```

### 2. What Gets Stored

**For each utterance:**
- ✅ Video file (.avi) - The actual lip-reading video
- ✅ Raw transcription - What the model initially predicted
- ✅ LLM correction - What the LLM thought it should be
- ✅ Ground truth - What the user confirms it actually was
- ✅ Metadata - Timestamp, speaker, context, etc.

### 3. Storage Format

**Option A: SQLite Database**
```sql
CREATE TABLE training_samples (
    id INTEGER PRIMARY KEY,
    video_path TEXT,
    raw_output TEXT,
    llm_correction TEXT,
    ground_truth TEXT,
    timestamp REAL,
    speaker_id TEXT,
    context TEXT,
    was_correct BOOLEAN
);
```

**Option B: JSON Files**
```json
{
  "id": "uuid-here",
  "video_path": "training_data/videos/sample_001.avi",
  "raw_output": "I HAVE READ MANY BOOKS",
  "llm_correction": "I have read many books.",
  "ground_truth": "I have read many books about this subject.",
  "timestamp": 1765076400.0,
  "was_correct": false,
  "metadata": {
    "speaker": "user_1",
    "context": "discussing reading habits"
  }
}
```

## Implementation Plan

### Phase 1: Basic Collection (Immediate)

Add a new hotkey (e.g., 'E' for Edit) that:
1. Shows the raw output and LLM correction
2. Allows user to type the correct transcription
3. Saves the video + all three versions to a training dataset
4. Marks whether the model was correct or not

### Phase 2: Review Interface (Short-term)

Create a separate script to:
1. Review collected samples
2. Edit corrections if needed
3. Export to training format
4. Generate statistics on model accuracy

### Phase 3: Model Fine-tuning (Long-term)

Use collected data to:
1. Fine-tune the VSR model
2. Improve LLM correction prompts
3. Build speaker-specific models
4. Analyze common error patterns

## Immediate Implementation

### New Hotkey: 'E' (Edit/Correct)

**When pressed after an utterance:**
1. Pause and show:
   ```
   RAW OUTPUT: I HAVE READ MANY BOOKS
   LLM CORRECTION: I have read many books.
   
   Was this correct? (y/n)
   ```

2. If 'n':
   ```
   Enter the correct transcription:
   > I have read many books about this subject.
   
   ✓ Training sample saved!
   ```

3. If 'y':
   ```
   ✓ Marked as correct - no training sample needed.
   ```

### Storage Structure

```
chaplin/
├── training_data/           # Training data directory (gitignored)
│   ├── videos/              # Copied video files with training_ prefix
│   │   ├── training_1765076400000.avi
│   │   └── training_1765076401000.avi
│   └── samples.jsonl        # Training data records
├── webcam1765076400000.avi  # Temporary files (auto-deleted after processing)
└── webcam1765076401000.avi  # Temporary files (auto-deleted after processing)
```

### File Naming Convention

**Temporary files** (auto-deleted):
- Pattern: `webcam*.avi`
- Location: Root directory
- Lifecycle: Created during recording → Processed → Deleted

**Training files** (preserved):
- Pattern: `training_*.avi`
- Location: `training_data/videos/`
- Lifecycle: Copied from temporary → Kept permanently

## Benefits

### For Model Improvement
- **Real-world data** - Actual usage patterns
- **Error analysis** - Identify common mistakes
- **Speaker adaptation** - Personalize to your speech
- **Context awareness** - Learn from corrections

### For User
- **Immediate feedback** - Correct mistakes right away
- **Quality control** - Ensure accuracy
- **Progress tracking** - See improvement over time
- **Custom vocabulary** - Train on your specific terms

## Advanced Features (Future)

### 1. Active Learning
- Model identifies low-confidence predictions
- Prioritizes those for user correction
- Focuses training on difficult cases

### 2. Batch Review
- Review multiple samples at once
- Bulk corrections
- Pattern identification

### 3. Model Versioning
- Track model performance over time
- A/B test different versions
- Roll back if needed

### 4. Collaborative Training
- Multiple users contribute data
- Shared model improvements
- Privacy-preserving federated learning

## Data Format for Fine-tuning

### For VSR Model (Video → Text)
```python
{
    "video": "path/to/video.avi",
    "text": "ground truth transcription",
    "duration": 2.5,
    "fps": 16
}
```

### For LLM Correction (Raw → Corrected)
```python
{
    "input": "I HAVE READ MANY BOOKS",
    "output": "I have read many books about this subject.",
    "context": "discussing reading habits"
}
```

## Privacy & Ethics

### Considerations
- ✅ All data stays local by default
- ✅ User controls what gets saved
- ✅ Can anonymize before sharing
- ✅ Easy to delete samples
- ⚠️ Video contains facial data - handle carefully

### Best Practices
- Get explicit consent before saving
- Allow easy data deletion
- Provide export functionality
- Document data usage clearly

## Getting Started

### Minimal Implementation (Quick Win)

Add to `chaplin.py`:
```python
def save_training_sample(self, video_path, raw_output, llm_correction, ground_truth):
    """Save a training sample for model improvement"""
    sample = {
        "id": str(uuid.uuid4()),
        "video_path": video_path,
        "raw_output": raw_output,
        "llm_correction": llm_correction,
        "ground_truth": ground_truth,
        "timestamp": time.time(),
        "was_correct": (llm_correction.strip() == ground_truth.strip())
    }
    
    # Append to JSONL file
    with open("training_data/samples.jsonl", "a") as f:
        f.write(json.dumps(sample) + "\n")
```

### Usage
1. Speak into camera
2. See output
3. Press 'E' if incorrect
4. Type correct version
5. Sample saved automatically

## Next Steps

1. **Implement basic collection** - Add 'E' hotkey and save functionality
2. **Test with real usage** - Collect 50-100 samples
3. **Analyze patterns** - What errors are most common?
4. **Fine-tune model** - Use collected data to improve
5. **Iterate** - Repeat the cycle

## Questions to Consider

1. **How often to save?** - Every utterance vs. only corrections?
2. **Storage limits?** - How much data to keep?
3. **Model updates?** - How often to retrain?
4. **Sharing?** - Contribute to community dataset?

## Resources Needed

### Storage
- ~10MB per video (2 seconds at 640x480)
- ~1KB per JSON sample
- For 1000 samples: ~10GB

### Compute
- Fine-tuning: GPU recommended
- Inference: Current setup works
- Analysis: CPU sufficient

### Time
- Collection: Automatic during use
- Review: ~30 seconds per sample
- Training: Hours to days (depending on data size)
