# Vector Database Integration

Chaplin now uses ChromaDB for semantic search and intelligent context retrieval.

## What is it?

A vector database stores text as mathematical embeddings, allowing the system to find semantically similar content even when exact words don't match.

## Features

### 1. **Semantic Conversation History**
- Every corrected utterance is stored with embeddings
- When correcting new text, the system retrieves the 3 most semantically relevant past utterances
- Works even for long conversations (not limited to last 5 items)

**Example:**
```
Past conversation (30 minutes ago): "We need to increase our Q4 revenue targets."
Current lip-reading: "WHAT ABOUT THE TARGETS"
→ Vector DB retrieves the Q4 revenue context
→ LLM correctly interprets as "What about the targets?"
```

### 2. **Intelligent Document Search**
- Uploaded documents are chunked into ~500 character segments
- Each chunk is stored with embeddings
- When correcting, the system retrieves the 2 most relevant document chunks

**Example:**
```
Uploaded PDF: 50-page product specification
Current lip-reading: "HOW DOES AUTHENTICATION WORK"
→ Vector DB finds the authentication section (page 23)
→ LLM uses that context for correction
```

### 3. **No Token Limits**
- Unlike putting everything in the prompt, vector search is efficient
- Can handle multiple large documents
- Only relevant chunks are sent to the LLM

## How It Works

### Automatic Storage

**Conversation utterances:**
- Every corrected sentence is automatically stored
- Metadata includes timestamp and type
- Searchable by semantic similarity

**Uploaded documents:**
- Text files and PDFs are chunked intelligently
- Chunks break at sentence boundaries when possible
- Each chunk includes source file and position metadata

### Retrieval During Correction

When the LLM corrects lip-reading output:

1. **Query conversation history** - Find 3 semantically similar past utterances
2. **Query document chunks** - Find 2 relevant document excerpts
3. **Include recent context** - Last 3 utterances (chronological)
4. **Combine all context** - Send to LLM for correction

## Benefits for In-Person Conversations

### Long Conversations
- Remember topics from hours ago
- Reference earlier discussion points
- Maintain coherence over extended periods

### Multiple Topics
- Switch between topics naturally
- Retrieve relevant context for each topic
- No need to manually update context

### Document Reference
- Upload meeting agendas, notes, or reference materials
- System automatically finds relevant sections
- Better corrections for technical or domain-specific terms

## Technical Details

### Chunking Strategy
- **Chunk size**: 500 characters
- **Overlap**: 50 characters (prevents splitting important context)
- **Boundary detection**: Breaks at sentence endings when possible
- **Metadata**: Source file, chunk index, document type

### Vector Search
- **Conversation search**: Top 3 results
- **Document search**: Top 2 results
- **Embedding model**: ChromaDB default (sentence-transformers)
- **Storage**: In-memory (resets on restart)

### Performance
- **Search time**: <100ms for typical queries
- **Storage**: Minimal memory footprint
- **Scalability**: Handles thousands of chunks efficiently

## Usage Examples

### Example 1: Long Meeting
```
[Hour 1] "Let's discuss the new API design."
[Hour 2] "We should implement rate limiting."
[Hour 3] Lip-reading: "BACK TO THE API THING"
→ Vector DB retrieves "API design" from Hour 1
→ Correct interpretation: "Back to the API thing."
```

### Example 2: Technical Discussion
```
Uploaded: Python documentation (100 pages)
Lip-reading: "HOW DO WE USE ASYNC AWAIT"
→ Vector DB finds async/await section
→ LLM has technical context for correction
```

### Example 3: Multiple Documents
```
Uploaded: 
- Product spec (50 pages)
- Meeting agenda (2 pages)
- Previous meeting notes (10 pages)

Lip-reading: "WHAT DID WE DECIDE LAST TIME"
→ Vector DB searches all documents
→ Finds relevant decision from previous notes
→ Accurate correction with context
```

## Clearing Data

### Clear Conversation History
Press 'C' → Option 5 → Clears both:
- In-memory conversation list
- Vector database conversation collection

### Clear Documents
Currently documents persist until restart. To clear:
1. Restart Chaplin
2. Or manually clear via context management (future feature)

## Future Enhancements

Potential improvements:
- **Persistent storage** - Save vector DB to disk
- **Document management** - View/delete individual documents
- **Search UI** - Manually search conversation history
- **Export** - Save conversation history to file
- **Speaker identification** - Separate vector collections per speaker
- **Time-based weighting** - Prioritize recent utterances

## Troubleshooting

### "Vector DB initialization failed"
- ChromaDB may not be installed
- Run: `pip install chromadb`

### "Vector search failed"
- Usually harmless, falls back to recent history only
- Check console for specific error message

### Slow performance
- ChromaDB is fast, but very large documents may slow down
- Consider chunking large documents before upload
- Restart if memory usage becomes high

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Chaplin System                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Lip Reading → Raw Text → LLM Correction                │
│                              ↑                           │
│                              │                           │
│                         Context Layer                    │
│                              │                           │
│              ┌───────────────┴───────────────┐          │
│              │                                │          │
│         Vector Search                  Recent History   │
│              │                                │          │
│    ┌─────────┴─────────┐           ┌────────┴────────┐ │
│    │                   │           │                  │ │
│  Conversation      Document      Last 3           Meeting│
│  Collection        Collection    Utterances       Context│
│    │                   │           │                  │ │
│    └───────────────────┴───────────┴──────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Summary

Vector database integration makes Chaplin significantly more intelligent for:
- ✅ Long conversations (hours, not minutes)
- ✅ Multiple topics in one session
- ✅ Large document reference
- ✅ Technical/domain-specific vocabulary
- ✅ In-person conversations with natural flow

The system automatically handles storage and retrieval - you just upload documents and have conversations naturally!
