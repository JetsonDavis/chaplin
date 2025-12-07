# Using Context to Improve LLM Corrections

Chaplin now supports contextual information to help the LLM make better corrections to lip-reading output.

## Features

### 1. **Conversation History** (Automatic)
- Chaplin automatically tracks the last 5 corrected utterances
- This history is provided to the LLM for each new correction
- Helps the LLM understand the flow of conversation and make better word choices

### 2. **Meeting Context** (Optional)
- You can provide context about the meeting/conversation topic
- This helps the LLM choose more appropriate words when multiple interpretations are possible

## How to Use

### Setting Context via Config File

Edit `hydra_configs/default.yaml`:

```yaml
meeting_context: "discussing quarterly sales figures and revenue projections"
```

### Setting Context via Command Line

```bash
uv run --with-requirements requirements.txt --python 3.11 main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  camera_index=2 \
  meeting_context="technical presentation about machine learning models"
```

### Setting Context Programmatically

If you need to change context during runtime, you can modify the code to call:

```python
chaplin.set_meeting_context("now discussing budget allocations")
```

## Examples of Good Meeting Context

- **Business meetings**: "quarterly financial review with executive team"
- **Technical discussions**: "debugging Python code and discussing API design"
- **Medical consultations**: "discussing patient symptoms and treatment options"
- **Educational settings**: "teaching calculus concepts to undergraduate students"
- **Sales calls**: "presenting product features to potential enterprise clients"

## How It Works

When the LLM receives a lip-reading transcription, it also receives:

1. **Meeting Context** (if provided): Helps identify domain-specific vocabulary
2. **Recent Conversation** (last 3 utterances): Provides conversational flow

Example prompt sent to LLM:
```
Meeting context: discussing quarterly sales figures

Recent conversation:
- What were the Q3 results?
- Revenue increased by 15 percent.
- That's excellent news.

Transcription:
HOW ABOUT Q FOUR
```

The LLM can now better infer that "Q FOUR" should be "Q4" (fourth quarter) rather than "cue for" or other homophones.

## Benefits

- **Better word choice**: Context helps disambiguate homophones (e.g., "their" vs "there")
- **Domain vocabulary**: Technical terms are more likely to be recognized correctly
- **Coherent corrections**: Corrections maintain conversational flow
- **Reduced errors**: Fewer nonsensical corrections when context is clear

## Tips

1. **Be specific**: "discussing machine learning algorithms" is better than "tech meeting"
2. **Update as needed**: Change context when conversation topic shifts
3. **Keep it concise**: 1-2 sentences is usually sufficient
4. **Include key terms**: Mention important vocabulary that might appear (e.g., "using terms like ROI, KPIs, and conversion rates")
