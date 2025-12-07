import cv2
import time
from ollama import AsyncClient
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
from pynput import keyboard
import asyncio
from elevenlabs import ElevenLabs, VoiceSettings
import tempfile
import pygame
import numpy as np
import PyPDF2
import chromadb
from chromadb.config import Settings
import uuid


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self, voice_sample_path=None, tts_speaker="Claribel Dervla", camera_index=0, meeting_context="", persist_vector_db=True):
        self.vsr_model = None
        self.camera_index = camera_index
        self.persist_vector_db = persist_vector_db

        # flag to toggle recording
        self.recording = False
        
        # flag for manual TTS input mode
        self.waiting_for_input = False
        
        # flag to track when we're typing (to prevent 't' key interference)
        self.is_typing = False
        self.last_typing_time = 0  # timestamp of last typing completion
        
        # conversation history for context
        self.conversation_history = []  # stores recent corrected utterances
        self.max_history_items = 5  # keep last 5 utterances for context
        
        # optional context that can be set by user
        self.meeting_context = meeting_context  # e.g., "discussing quarterly sales figures"
        if self.meeting_context:
            print(f"\033[96mMeeting context set: {self.meeting_context}\033[0m")
        
        # initialize ChromaDB for vector-based context retrieval
        print("\n\033[48;5;94m\033[97m\033[1m INITIALIZING VECTOR DATABASE... \033[0m")
        
        if self.persist_vector_db:
            # Persistent storage - saves to disk
            db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f"\033[96mUsing persistent storage: {db_path}\033[0m")
        else:
            # Ephemeral storage - in-memory only
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            print("\033[96mUsing ephemeral storage (in-memory only)\033[0m")
        
        # Create collections
        try:
            self.conversation_collection = self.chroma_client.get_or_create_collection(
                name="conversation_history",
                metadata={"description": "Stores conversation utterances for semantic search"}
            )
            self.document_collection = self.chroma_client.get_or_create_collection(
                name="uploaded_documents",
                metadata={"description": "Stores chunks from uploaded documents"}
            )
            print("\033[48;5;22m\033[97m\033[1m ✓ VECTOR DATABASE READY \033[0m\n")
        except Exception as e:
            print(f"\033[91mWarning: Vector DB initialization failed: {e}\033[0m")
            self.conversation_collection = None
            self.document_collection = None

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup text-to-speech with ElevenLabs
        print("\n\033[48;5;94m\033[97m\033[1m SETTING UP ELEVENLABS TTS... \033[0m")
        
        # initialize ElevenLabs client (requires ELEVENLABS_API_KEY env variable)
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set!")
        self.elevenlabs_client = ElevenLabs(api_key=api_key)
        self.tts_speaker = tts_speaker if tts_speaker else "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID
        
        # initialize pygame mixer for audio playback
        # Note: To route to Zoom, set your system default output to BlackHole or Multi-Output Device
        pygame.mixer.init()
        
        print(f"\033[48;5;22m\033[97m\033[1m TTS READY! \033[0m (voice: {self.tts_speaker})\n")

        # setup async ollama client
        self.ollama_client = AsyncClient()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording
        self.recording = not self.recording
        if self.recording:
            print("\n\033[48;5;196m\033[97m\033[1m ● RECORDING STARTED \033[0m")
        else:
            print("\n\033[48;5;22m\033[97m\033[1m ■ RECORDING STOPPED \033[0m")
    
    def set_meeting_context(self, context):
        """Update the meeting context for better LLM corrections"""
        self.meeting_context = context
        print(f"\n\033[96m📋 Meeting context updated: {context}\033[0m\n")
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        # Also clear vector database
        if self.conversation_collection:
            try:
                self.chroma_client.delete_collection("conversation_history")
                self.conversation_collection = self.chroma_client.create_collection(
                    name="conversation_history",
                    metadata={"description": "Stores conversation utterances for semantic search"}
                )
                print("\n\033[96m🗑️  Conversation history and vector DB cleared\033[0m\n")
            except Exception as e:
                print(f"\n\033[96m🗑️  Conversation history cleared (vector DB error: {e})\033[0m\n")
        else:
            print("\n\033[96m🗑️  Conversation history cleared\033[0m\n")
    
    def context_management_dialog(self):
        """Open context management dialog for live context updates and document uploads"""
        # Set flag to prevent interference
        self.waiting_for_input = True
        
        # Temporarily disable recording if it's active
        was_recording = self.recording
        if was_recording:
            self.recording = False
            print("\033[93mPausing lip-reading recording for context management...\033[0m")
        
        print("\n" + "="*70)
        print("\033[48;5;33m\033[97m\033[1m CONTEXT MANAGEMENT \033[0m")
        print("="*70)
        print("\nOptions:")
        print("  1. Update meeting context (type text)")
        print("  2. Upload text file (.txt)")
        print("  3. Upload PDF document (.pdf)")
        print("  4. View current context")
        print("  5. Clear context")
        print("  6. Cancel")
        print("="*70)
        
        context_window = 'Context Management'
        
        while True:
            img = self._create_context_menu_image()
            cv2.imshow(context_window, img)
            
            key = cv2.waitKey(0)
            
            if key == ord('1'):
                # Update context via text input
                cv2.destroyWindow(context_window)
                self._input_context_text()
                break
            elif key == ord('2'):
                # Upload text file
                cv2.destroyWindow(context_window)
                self._upload_text_file()
                break
            elif key == ord('3'):
                # Upload PDF
                cv2.destroyWindow(context_window)
                self._upload_pdf_file()
                break
            elif key == ord('4'):
                # View current context
                self._display_current_context()
            elif key == ord('5'):
                # Clear context
                self.meeting_context = ""
                print("\n\033[96m✓ Context cleared\033[0m\n")
            elif key == 27 or key == ord('6'):  # Esc or option 6
                print("\033[93mContext management cancelled.\033[0m\n")
                cv2.destroyWindow(context_window)
                break
        
        # Restore recording state if it was active
        if was_recording:
            self.recording = True
            print("\033[93mResuming lip-reading recording...\033[0m")
        
        # Clear the flag
        self.waiting_for_input = False
    
    def _create_context_menu_image(self):
        """Create menu image for context management"""
        img = 255 * np.ones((400, 800, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Context Management', (20, 40), font, 1.3, (0, 0, 0), 2)
        
        # Menu options
        y_pos = 100
        options = [
            "1. Update meeting context (type text)",
            "2. Upload text file (.txt)",
            "3. Upload PDF document (.pdf)",
            "4. View current context",
            "5. Clear context",
            "6. Cancel (or press Esc)"
        ]
        
        for option in options:
            cv2.putText(img, option, (40, y_pos), font, 0.7, (50, 50, 50), 1)
            y_pos += 45
        
        cv2.putText(img, 'Press the number key for your choice', (20, 370), font, 0.6, (100, 100, 100), 1)
        
        return img
    
    def _input_context_text(self):
        """Get context text input from user"""
        print("\n" + "="*70)
        print("\033[48;5;33m\033[97m\033[1m ENTER MEETING CONTEXT \033[0m")
        print("Type your context below (press Enter when done):")
        print("="*70)
        
        input_text = []
        input_window = 'Context Input'
        
        while True:
            img = self._create_context_input_image(''.join(input_text))
            cv2.imshow(input_window, img)
            
            key = cv2.waitKey(0)
            
            if key == 27:  # Esc
                print("\033[93mContext input cancelled.\033[0m\n")
                cv2.destroyWindow(input_window)
                break
            elif key == 13 or key == 10:  # Enter
                text = ''.join(input_text)
                cv2.destroyWindow(input_window)
                
                if text.strip():
                    self.set_meeting_context(text.strip())
                else:
                    print("\033[93mNo context entered.\033[0m\n")
                break
            elif key == 8 or key == 127:  # Backspace
                if input_text:
                    input_text.pop()
            elif 32 <= key <= 126:  # Printable ASCII
                input_text.append(chr(key))
    
    def _create_context_input_image(self, text):
        """Create input image for context text"""
        img = 255 * np.ones((300, 800, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Enter Meeting Context', (20, 40), font, 1.2, (0, 0, 0), 2)
        cv2.putText(img, 'Type your context below:', (20, 80), font, 0.6, (100, 100, 100), 1)
        cv2.putText(img, 'Press Enter to save | Esc to cancel', (20, 280), font, 0.5, (150, 150, 150), 1)
        
        # Draw input box
        cv2.rectangle(img, (20, 100), (780, 250), (200, 200, 200), 2)
        
        # Add text with word wrapping
        if text:
            max_width = 740
            y_pos = 135
            words = text.split(' ')
            current_line = ''
            
            for word in words:
                test_line = current_line + word + ' '
                text_size = cv2.getTextSize(test_line, font, 0.6, 1)[0]
                
                if text_size[0] > max_width:
                    if current_line:
                        cv2.putText(img, current_line.strip(), (30, y_pos), font, 0.6, (0, 0, 0), 1)
                        y_pos += 30
                        current_line = word + ' '
                else:
                    current_line = test_line
            
            if current_line:
                cv2.putText(img, current_line.strip(), (30, y_pos), font, 0.6, (0, 0, 0), 1)
        
        # Add cursor
        cursor_x = 30 + cv2.getTextSize(text, font, 0.6, 1)[0][0] if text else 30
        cv2.line(img, (cursor_x, 120), (cursor_x, 230), (0, 0, 255), 2)
        
        return img
    
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for vector storage"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclaim = chunk.rfind('!')
                last_sentence = max(last_period, last_question, last_exclaim)
                
                if last_sentence > chunk_size * 0.5:  # at least halfway through
                    chunk = chunk[:last_sentence + 1]
                    end = start + last_sentence + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def _upload_text_file(self):
        """Upload and read a text file for context"""
        print("\n" + "="*70)
        print("\033[48;5;33m\033[97m\033[1m UPLOAD TEXT FILE \033[0m")
        print("Enter the full path to your .txt file:")
        print("="*70)
        
        file_path = input("> ").strip()
        
        if not file_path:
            print("\033[93mNo file path provided.\033[0m\n")
            return
        
        # Remove quotes if present
        file_path = file_path.strip('"').strip("'")
        
        # Remove backslash escapes (from dragging files in Finder)
        file_path = file_path.replace('\\ ', ' ')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                # Chunk the document for vector storage
                chunks = self._chunk_text(content)
                
                # Store in vector database
                if self.document_collection:
                    try:
                        self.document_collection.add(
                            documents=chunks,
                            ids=[str(uuid.uuid4()) for _ in chunks],
                            metadatas=[{
                                "source": os.path.basename(file_path),
                                "chunk_index": i,
                                "type": "text_file"
                            } for i in range(len(chunks))]
                        )
                        print(f"\033[96m✓ Stored {len(chunks)} chunks in vector database\033[0m")
                    except Exception as e:
                        print(f"\033[93mWarning: Failed to store in vector DB: {e}\033[0m")
                
                # Also set a summary as meeting context
                summary = content[:500] + "..." if len(content) > 500 else content
                self.set_meeting_context(f"Document: {os.path.basename(file_path)}\n{summary}")
                print(f"\033[96m✓ Loaded {len(content)} characters from file ({len(chunks)} chunks)\033[0m\n")
            else:
                print("\033[93mFile is empty.\033[0m\n")
        except FileNotFoundError:
            print(f"\033[91m✗ File not found: {file_path}\033[0m\n")
        except Exception as e:
            print(f"\033[91m✗ Error reading file: {e}\033[0m\n")
    
    def _upload_pdf_file(self):
        """Upload and read a PDF file for context"""
        print("\n" + "="*70)
        print("\033[48;5;33m\033[97m\033[1m UPLOAD PDF FILE \033[0m")
        print("Enter the full path to your .pdf file:")
        print("="*70)
        
        file_path = input("> ").strip()
        
        if not file_path:
            print("\033[93mNo file path provided.\033[0m\n")
            return
        
        # Remove quotes if present
        file_path = file_path.strip('"').strip("'")
        
        # Remove backslash escapes (from dragging files in Finder)
        file_path = file_path.replace('\\ ', ' ')
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract text from all pages
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            
            if content.strip():
                # Chunk the document for vector storage
                chunks = self._chunk_text(content)
                
                # Store in vector database
                if self.document_collection:
                    try:
                        self.document_collection.add(
                            documents=chunks,
                            ids=[str(uuid.uuid4()) for _ in chunks],
                            metadatas=[{
                                "source": os.path.basename(file_path),
                                "chunk_index": i,
                                "type": "pdf",
                                "pages": len(pdf_reader.pages)
                            } for i in range(len(chunks))]
                        )
                        print(f"\033[96m✓ Stored {len(chunks)} chunks in vector database\033[0m")
                    except Exception as e:
                        print(f"\033[93mWarning: Failed to store in vector DB: {e}\033[0m")
                
                # Also set a summary as meeting context
                summary = content[:500] + "..." if len(content) > 500 else content
                self.set_meeting_context(f"PDF Document: {os.path.basename(file_path)} ({len(pdf_reader.pages)} pages)\n{summary}")
                print(f"\033[96m✓ Loaded {len(content)} characters from PDF ({len(pdf_reader.pages)} pages, {len(chunks)} chunks)\033[0m\n")
            else:
                print("\033[93mPDF appears to be empty or contains no extractable text.\033[0m\n")
        except FileNotFoundError:
            print(f"\033[91m✗ File not found: {file_path}\033[0m\n")
        except Exception as e:
            print(f"\033[91m✗ Error reading PDF: {e}\033[0m\n")
    
    def _display_current_context(self):
        """Display the current meeting context"""
        print("\n" + "="*70)
        print("\033[48;5;33m\033[97m\033[1m CURRENT CONTEXT \033[0m")
        print("="*70)
        
        if self.meeting_context:
            print(f"\n{self.meeting_context}\n")
            print(f"\033[96m({len(self.meeting_context)} characters)\033[0m")
        else:
            print("\n\033[93mNo context currently set.\033[0m")
        
        print("\n" + "="*70)
        print("Press any key to continue...")
        cv2.waitKey(0)

    async def correct_output_async(self, output, sequence_num):
        # perform inference on the raw output to get back a "correct" version
        print("\n\033[48;5;94m\033[97m\033[1m CORRECTING WITH LLM... \033[0m")
        
        # Build context from conversation history and vector search
        context_parts = []
        if self.meeting_context:
            context_parts.append(f"Meeting context: {self.meeting_context}")
        
        # Get semantically relevant past utterances from vector DB
        if self.conversation_collection:
            try:
                results = self.conversation_collection.query(
                    query_texts=[output],
                    n_results=3
                )
                if results['documents'] and results['documents'][0]:
                    relevant_history = "\n".join([f"- {doc}" for doc in results['documents'][0]])
                    context_parts.append(f"Relevant past conversation:\n{relevant_history}")
            except Exception as e:
                print(f"\033[93mWarning: Vector search failed: {e}\033[0m")
        
        # Also include recent conversation (last 3)
        if self.conversation_history:
            history_text = "\n".join([f"- {item}" for item in self.conversation_history[-3:]])  # last 3 items
            context_parts.append(f"Recent conversation:\n{history_text}")
        
        # Get relevant document chunks from vector DB (only if really relevant)
        if self.document_collection:
            try:
                doc_results = self.document_collection.query(
                    query_texts=[output],
                    n_results=1  # Reduced from 2 to 1 to avoid overwhelming
                )
                if doc_results['documents'] and doc_results['documents'][0]:
                    # Only include if it seems relevant (truncate heavily)
                    relevant_docs = "\n".join([f"- {doc[:150]}..." for doc in doc_results['documents'][0]])
                    context_parts.append(f"Relevant document excerpts (for reference only):\n{relevant_docs}")
            except Exception as e:
                print(f"\033[93mWarning: Document search failed: {e}\033[0m")
        
        context_section = "\n\n".join(context_parts) if context_parts else ""
        
        # Build system prompt with context
        system_prompt = f"""You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect. The input text will also be in all-caps, although your response should be capitalized correctly and should NOT be in all-caps.

CRITICAL: Your job is to CORRECT the transcription, NOT to replace it with something else entirely. The transcription represents what the person actually said - you are only fixing mistranscribed words.

IMPORTANT RULES:
1. PRESERVE ALL WORDS - Do not remove or delete any words from the transcription
2. Only REPLACE individual words that seem mistranscribed with similar-sounding alternatives
3. Do NOT add new words or content
4. Keep the same number of words and sentence structure
5. If a word seems unusual but could be correct (like "hamburger", "pizza", etc.), keep it
6. DO NOT replace the entire transcription with text from documents or context - only use context to help choose between similar-sounding words

If something seems unusual, assume it was mistranscribed and replace it with a similar-sounding word that makes more sense in context. For example:
- "THEIR" might be "THERE" or "THEY'RE"
- "TO" might be "TWO" or "TOO"
- "A CONVOLUTIONAL" stays "A CONVOLUTIONAL" (don't replace with document titles!)

Also, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'.

{context_section if context_section else "No prior context available."}

NOTE: The context above is for reference only to help disambiguate similar-sounding words. Do NOT replace the transcription with text from the context.

Return the corrected text in the format of 'list_of_changes' and 'corrected_text'."""
        
        response = await self.ollama_client.chat(
            model='qwen2.5:1.5b',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=ChaplinOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response['message']['content'])
        
        print(f"\033[48;5;22m\033[97m\033[1m ✓ LLM CORRECTION COMPLETE \033[0m")

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '
        
        # add to conversation history for future context
        corrected = chat_output.corrected_text.strip()
        self.conversation_history.append(corrected)
        if len(self.conversation_history) > self.max_history_items:
            self.conversation_history.pop(0)  # remove oldest item
        
        # add to vector database for semantic search
        if self.conversation_collection:
            try:
                self.conversation_collection.add(
                    documents=[corrected],
                    ids=[str(uuid.uuid4())],
                    metadatas=[{"timestamp": time.time(), "type": "utterance"}]
                )
            except Exception as e:
                print(f"\033[93mWarning: Failed to add to vector DB: {e}\033[0m")

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # this task's turn to type the corrected text
            # Only type if we're not in manual input mode
            if not self.waiting_for_input:
                print(f"\n\033[48;5;33m\033[97m\033[1m TYPING TEXT \033[0m: \"{chat_output.corrected_text.strip()}\"")
                self.is_typing = True  # Set flag before typing AND TTS
                self.kbd_controller.type(chat_output.corrected_text)
                print("\033[48;5;22m\033[97m\033[1m ✓ TYPING COMPLETE \033[0m")
                
                # speak the corrected text using TTS (still blocking keys)
                print("\033[48;5;94m\033[97m\033[1m STARTING TTS... \033[0m")
                self._speak_text(chat_output.corrected_text)
                
                # Clear flag and record time AFTER both typing and TTS complete
                self.is_typing = False
                self.last_typing_time = time.time()
            else:
                print("\033[93mSkipping typing/TTS - manual input mode active\033[0m")

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return chat_output.corrected_text

    def _speak_text(self, text):
        """Generate and play speech using ElevenLabs TTS"""
        try:
            print(f"\n\033[48;5;94m\033[97m\033[1m GENERATING SPEECH \033[0m: \"{text.strip()}\"")
            print(f"\033[93mUsing voice: {self.tts_speaker}\033[0m")
            
            # create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # generate speech with ElevenLabs (new API)
            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.tts_speaker,
                text=text,
                model_id="eleven_turbo_v2_5",  # fast model
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            # save audio to file
            with open(tmp_path, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            
            print("\033[48;5;22m\033[97m\033[1m 🔊 PLAYING AUDIO NOW \033[0m")
            
            # play the generated audio
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            
            # wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            print("\033[48;5;22m\033[97m\033[1m ✓ AUDIO PLAYBACK COMPLETE \033[0m\n")
            
            # cleanup temporary file
            os.remove(tmp_path)
        except Exception as e:
            print(f"\n\033[48;5;196m\033[97m\033[1m TTS ERROR \033[0m: {e}\n")

    def manual_tts_input(self):
        """Get text input using OpenCV window for manual TTS"""
        # Set flag to prevent lip-reading from interfering
        self.waiting_for_input = True
        
        # Temporarily disable recording if it's active
        was_recording = self.recording
        if was_recording:
            self.recording = False
            print("\033[93mPausing lip-reading recording for manual input...\033[0m")
        
        print("\n" + "="*60)
        print("\033[48;5;33m\033[97m\033[1m MANUAL TTS INPUT MODE \033[0m")
        print("Type in the 'TTS Input' window. Press Enter to speak, Esc to cancel.")
        print("="*60)
        
        # Text input state
        input_text = []
        input_window = 'TTS Input'
        
        # Create input window
        while True:
            img = self._create_input_image(''.join(input_text))
            cv2.imshow(input_window, img)
            
            key = cv2.waitKey(0)  # Wait indefinitely for key press
            
            if key == 27:  # Esc key
                print("\033[93mInput cancelled.\033[0m\n")
                cv2.destroyWindow(input_window)
                break
            elif key == 13 or key == 10:  # Enter key (13 on Windows, 10 on Mac/Linux)
                text = ''.join(input_text)
                cv2.destroyWindow(input_window)
                
                if text.strip():
                    print(f"\n\033[48;5;33m\033[97m\033[1m SPEAKING \033[0m: \"{text.strip()}\"")
                    # Speak the text directly without typing
                    self._speak_text(text.strip())
                else:
                    print("\033[93mNo text entered.\033[0m\n")
                break
            elif key == 8 or key == 127:  # Backspace (8 on Windows, 127 on Mac/Linux)
                if input_text:
                    input_text.pop()
            elif 32 <= key <= 126:  # Printable ASCII characters
                input_text.append(chr(key))
        
        # Restore recording state if it was active
        if was_recording:
            self.recording = True
            print("\033[93mResuming lip-reading recording...\033[0m")
        
        # Clear the flag
        self.waiting_for_input = False
    
    def _create_input_image(self, text):
        """Create an image showing the current input text"""
        img = 255 * np.ones((250, 700, 3), dtype=np.uint8)
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Manual TTS Input', (20, 40), font, 1.2, (0, 0, 0), 2)
        
        # Add instructions
        cv2.putText(img, 'Type your text below:', (20, 80), font, 0.6, (100, 100, 100), 1)
        cv2.putText(img, 'Press Enter to speak | Esc to cancel', (20, 230), font, 0.5, (150, 150, 150), 1)
        
        # Draw input box
        cv2.rectangle(img, (20, 100), (680, 200), (200, 200, 200), 2)
        
        # Add text with word wrapping
        if text:
            # Simple word wrapping
            max_width = 640
            y_pos = 135
            words = text.split(' ')
            current_line = ''
            
            for word in words:
                test_line = current_line + word + ' '
                text_size = cv2.getTextSize(test_line, font, 0.7, 1)[0]
                
                if text_size[0] > max_width:
                    # Draw current line and start new one
                    if current_line:
                        cv2.putText(img, current_line.strip(), (30, y_pos), font, 0.7, (0, 0, 0), 1)
                        y_pos += 30
                        current_line = word + ' '
                else:
                    current_line = test_line
            
            # Draw remaining text
            if current_line:
                cv2.putText(img, current_line.strip(), (30, y_pos), font, 0.7, (0, 0, 0), 1)
        
        # Add cursor
        cursor_x = 30 + cv2.getTextSize(text, font, 0.7, 1)[0][0] if text else 30
        cv2.line(img, (cursor_x, 120), (cursor_x, 180), (0, 0, 255), 2)
        
        return img

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # print the raw output to console
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        # assign sequence number for this task
        sequence_num = self.current_sequence
        self.current_sequence += 1

        # start the async LLM correction (non-blocking) with sequence number
        asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        )

        # return immediately without waiting for correction
        return {
            "output": output,
            "video_path": video_path
        }

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(self.camera_index)

        # set webcam resolution to 640x480 for better compatibility
        # iPhone camera defaults to 1920x1080 which causes issues with video encoding
        # We'll force resize in the frame processing loop
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Use fixed resolution for video writer (frames will be resized to match)
        frame_width = 640
        frame_height = 480
        print(f"\033[93mDEBUG: Video will be saved at {frame_width}x{frame_height}\033[0m")

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Check if the Chaplin window exists and is focused
            # This prevents keyboard controller typed characters from triggering dialogs
            try:
                window_focused = cv2.getWindowProperty('Chaplin', cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                # Window doesn't exist yet, assume not focused
                window_focused = False
            
            if key == ord('q'):
                # remove any remaining videos that were saved to disk
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and (file.endswith('.mp4') or file.endswith('.avi')):
                        os.remove(file)
                break
            elif key == ord('t'):
                # Only respond if NOT typing/TTS and NOT in input mode
                time_since_typing = time.time() - self.last_typing_time
                if not self.is_typing and not self.waiting_for_input and time_since_typing > 2.0:
                    # open manual TTS input dialog
                    print(f"\033[93mDEBUG: 't' key detected. Opening TTS input.\033[0m")
                    self.manual_tts_input()
                elif key == ord('t'):
                    # Debug: 't' key pressed but blocked
                    print(f"\033[93mDEBUG: 't' key blocked. is_typing={self.is_typing}, waiting_for_input={self.waiting_for_input}, time_since_typing={time_since_typing:.2f}s\033[0m")
            elif key == ord('c'):
                # Only respond if NOT typing/TTS and NOT in input mode
                time_since_typing = time.time() - self.last_typing_time
                if not self.is_typing and not self.waiting_for_input and time_since_typing > 2.0:
                    # open context management dialog
                    print(f"\033[93mDEBUG: 'c' key detected. Opening context management.\033[0m")
                    self.context_management_dialog()
                elif key == ord('c'):
                    # Debug: 'c' key pressed but blocked
                    print(f"\033[93mDEBUG: 'c' key blocked. is_typing={self.is_typing}, waiting_for_input={self.waiting_for_input}, time_since_typing={time_since_typing:.2f}s\033[0m")
            elif key == ord('r'):
                # Toggle recording with 'r' key (only when Chaplin window is focused)
                if not self.is_typing and not self.waiting_for_input:
                    self.toggle_recording()

            current_time = time.time()

            # conditional ensures that the video is recorded at the correct frame rate
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    # resize frame to 640x480 if it's larger (for iPhone camera compatibility)
                    if frame.shape[1] > 640 or frame.shape[0] > 480:
                        frame = cv2.resize(frame, (640, 480))
                    
                    # frame compression
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(
                        buffer, cv2.IMREAD_GRAYSCALE)

                    if self.recording:
                        if out is None:
                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.avi'
                            print(f"\033[93mDEBUG: Creating video writer for {output_path} with size {frame_width}x{frame_height}\033[0m")
                            # Use MJPEG codec in AVI container for maximum compatibility
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )
                            if not out.isOpened():
                                print(f"\033[91mERROR: Failed to open video writer!\033[0m")

                        out.write(compressed_frame)

                        last_frame_time = current_time

                        # circle to indicate recording, only appears in the window and is not present in video saved to disk
                        cv2.circle(compressed_frame, (frame_width -
                                                      20, 20), 10, (0, 0, 0), -1)

                        frame_count += 1
                    # check if not recording AND video is at least 2 seconds long
                    elif not self.recording and frame_count > 0:
                        if out is not None:
                            out.release()
                            out = None  # Important: set to None after release
                        
                        print(f"\033[93mDEBUG: Recording stopped. Recorded {frame_count} frames (need {self.fps * 2} for 2 seconds)\033[0m")
                        
                        # Give the file system a moment to finalize the file
                        time.sleep(0.1)

                        # only run inference if the video is at least 2 seconds long
                        if frame_count >= self.fps * 2:
                            print(f"\033[93mDEBUG: Starting inference on {output_path}\033[0m")
                            futures.append(self.executor.submit(
                                self.perform_inference, output_path))
                        else:
                            print(f"\033[93mDEBUG: Video too short ({frame_count} frames), deleting {output_path}\033[0m")
                            os.remove(output_path)

                        output_path = self.output_prefix + \
                            str(time.time_ns() // 1_000_000) + '.avi'
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            self.fps,
                            (frame_width, frame_height),
                            False  # isColor
                        )

                        frame_count = 0

                    # display the frame in the window
                    cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

            # ensures that videos are handled in the order they were recorded
            for fut in futures:
                if fut.done():
                    result = fut.result()
                    # once done processing, delete the video with the video path
                    os.remove(result["video_path"])
                    futures.remove(fut)
                else:
                    break

        # release everything
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # stop async event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.async_thread.shutdown(wait=True)

        # shutdown executor
        self.executor.shutdown(wait=True)
