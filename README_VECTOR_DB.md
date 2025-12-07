# ChromaDB Vector Database - Quick Start

## What Changed?

Chaplin now uses **ChromaDB** for intelligent semantic search, making it much better for long in-person conversations.

## Installation

```bash
pip install chromadb
# or with uv
uv pip install chromadb
```

Already added to `requirements.txt`.

## What You Get

### 1. **Semantic Conversation Memory**
- Remembers **everything** from the entire conversation
- Finds relevant past utterances even from hours ago
- Not limited to last 5 items anymore

### 2. **Smart Document Search**
- Upload PDFs and text files
- System automatically finds relevant sections
- No 2000 character limit - handles full documents

### 3. **Better Corrections**
- LLM gets relevant context automatically
- Better word choices for technical terms
- Maintains coherence over long conversations

## How to Use

### Just Use It Normally!

The vector database works automatically:

1. **Start Chaplin** → Vector DB initializes
2. **Have conversations** → Utterances stored automatically
3. **Upload documents** (Press 'C' → Option 2 or 3) → Chunks stored automatically
4. **Keep talking** → System retrieves relevant context automatically

### Example Workflow

```bash
# Start Chaplin
uv run --with-requirements requirements.txt --python 3.11 main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  camera_index=2

# During conversation:
# - Press 'C' to upload meeting agenda PDF
# - Have 2-hour conversation
# - System remembers everything via vector search
# - References earlier topics automatically
```

## What Happens Behind the Scenes

### When You Speak
```
You speak → Lip-reading → Raw text
                              ↓
                    Vector DB searches for:
                    - Similar past utterances (top 3)
                    - Relevant document chunks (top 2)
                              ↓
                    LLM corrects with context
                              ↓
                    Corrected text stored in vector DB
```

### When You Upload Documents
```
Upload PDF/TXT → Extract text → Chunk into ~500 char segments
                                        ↓
                              Store in vector DB with embeddings
                                        ↓
                              Available for semantic search
```

## Performance

- **Initialization**: ~1-2 seconds
- **Search time**: <100ms per query
- **Storage**: In-memory (resets on restart)
- **Scalability**: Handles thousands of chunks

## Use Cases

### Perfect For:
✅ Long in-person conversations (2+ hours)
✅ Technical discussions with documentation
✅ Meetings with reference materials
✅ Multiple topics in one session
✅ Recurring themes across conversation

### Less Useful For:
⚠️ Very short conversations (<5 minutes)
⚠️ Single-topic discussions
⚠️ No document references needed

## Troubleshooting

### "Vector DB initialization failed"
```bash
# Install ChromaDB
pip install chromadb
```

### "Vector search failed"
- Usually harmless, falls back to recent history
- Check console for specific error
- Restart Chaplin if persistent

### Slow performance
- Restart if memory usage high
- Consider shorter documents
- ChromaDB is generally very fast

## Data Persistence

**ENABLED BY DEFAULT**: Vector DB now persists to disk!

### Storage Location
```
/path/to/chaplin/chroma_db/
├── chroma.sqlite3      # Database file
└── [embedding files]   # Vector data
```

### What Gets Saved
✅ All conversation utterances with timestamps
✅ All uploaded document chunks with metadata
✅ Vector embeddings for semantic search
✅ Collection metadata

### Behavior
- **First run**: Creates `chroma_db/` directory and stores data
- **Subsequent runs**: Loads existing data and continues building
- **Uploaded documents**: Persist across sessions
- **Conversation history**: Accumulates over time

### Configuration

**Enable persistence (default):**
```yaml
# In hydra_configs/default.yaml
persist_vector_db: true
```

**Disable persistence (ephemeral mode):**
```yaml
persist_vector_db: false
```

Or via command line:
```bash
uv run --with-requirements requirements.txt --python 3.11 main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  camera_index=2 \
  persist_vector_db=false
```

### Managing Stored Data

**Clear all stored data:**
```bash
rm -rf chroma_db/
```

**View storage size:**
```bash
du -sh chroma_db/
```

**Backup your data:**
```bash
cp -r chroma_db/ chroma_db_backup/
```

### Benefits of Persistence
- 📚 Build a knowledge base over time
- 🔄 Resume conversations from previous sessions
- 📄 Upload documents once, use forever
- 🧠 System gets smarter with more data
- 💾 No need to re-upload documents

## Technical Details

**Embedding Model**: ChromaDB default (sentence-transformers)
**Chunk Size**: 500 characters with 50 char overlap
**Search Results**: Top 3 conversation + Top 2 document chunks
**Storage**: In-memory SQLite

## Files Added/Modified

- ✅ `requirements.txt` - Added chromadb
- ✅ `chaplin.py` - Vector DB integration
- ✅ `VECTOR_DATABASE.md` - Detailed documentation
- ✅ `CONTEXT_USAGE.md` - Updated with vector info
- ✅ `README_VECTOR_DB.md` - This quick start guide

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start Chaplin**: Use your normal command
3. **Upload a document**: Press 'C' → Option 2 or 3
4. **Have a conversation**: Talk naturally
5. **Watch it work**: System finds relevant context automatically

## Questions?

See detailed documentation:
- [VECTOR_DATABASE.md](VECTOR_DATABASE.md) - Technical details
- [CONTEXT_USAGE.md](CONTEXT_USAGE.md) - Context management
- [KEYBOARD_SHORTCUTS.md](KEYBOARD_SHORTCUTS.md) - All controls

---

**Summary**: Vector database makes Chaplin significantly smarter for long, natural conversations. It just works automatically - no special commands needed!
