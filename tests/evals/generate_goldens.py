import os
import sys
import glob
import json
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv
load_dotenv()

from tests.evals.custom_llm import CustomGroqLLM
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Max chunks to feed as context at once (keeps us under Groq's 12K TPM limit)
MAX_CHUNKS_PER_BATCH = 3
# Seconds to wait between batches to avoid rate-limiting (Groq free = 12K TPM)
RATE_LIMIT_DELAY = 65


def load_and_chunk(file_path: str) -> list[str]:
    """Load a .txt file and split it into text chunks (contexts)."""
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks if chunk.page_content.strip()]


def generate_golden_dataset():
    llm = CustomGroqLLM()
    all_goldens = []
    
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chroma_db'))
    txt_files = sorted(glob.glob(os.path.join(db_dir, "*.txt")))
    
    if not txt_files:
        print(f"No .txt files found in {db_dir}")
        return

    # Specific counts for specific files, default is 2
    custom_counts = {
        "dct-version.txt": 10,
        "Mango_analytics.txt": 20,
        "email_tracking_setup_detailed.txt": 5
    }
    
    for i, file_path in enumerate(txt_files):
        filename = os.path.basename(file_path)
        total_count = custom_counts.get(filename, 2)
        
        print(f"[{i+1}/{len(txt_files)}] Processing {filename} (target: {total_count} goldens)...")
        
        # Step 1: Load and chunk the document ourselves
        try:
            all_contexts = load_and_chunk(file_path)
        except Exception as e:
            print(f"  -> Error loading {filename}: {e}")
            continue
            
        if not all_contexts:
            print(f"  -> No text chunks from {filename}. Skipping.")
            continue
            
        print(f"  -> Extracted {len(all_contexts)} text chunks.")
        
        # Step 2: Split into batches to stay under token limits
        # For large files, we process batches of chunks and distribute the golden count
        context_batches = [
            all_contexts[j:j + MAX_CHUNKS_PER_BATCH] 
            for j in range(0, len(all_contexts), MAX_CHUNKS_PER_BATCH)
        ]
        
        # Distribute golden count across batches
        goldens_per_batch = max(1, total_count // len(context_batches))
        remainder = total_count % len(context_batches)
        
        file_goldens = []
        for batch_idx, batch in enumerate(context_batches):
            # Always generate exactly 1 golden per batch to minimize tokens per call
            batch_target = 1
            
            if len(file_goldens) >= total_count:
                break
                
            print(f"  -> Batch {batch_idx+1}/{len(context_batches)}: "
                  f"{len(batch)} chunks, generating {batch_target} golden...")
            
            synthesizer = Synthesizer(model=llm)
            try:
                goldens = synthesizer.generate_goldens_from_contexts(
                    contexts=[batch],
                    max_goldens_per_context=batch_target
                )
                # In DeepEval 4.x, generate_goldens_from_contexts returns the list directly
                if isinstance(goldens, list):
                    file_goldens.extend(goldens)
                elif hasattr(synthesizer, 'goldens'):
                    file_goldens.extend(synthesizer.goldens)
                    
                print(f"     ✓ Got {len(file_goldens)} goldens for this file so far.")
            except Exception as e:
                print(f"     ✗ Error: {e}")
            
            # Rate limit delay between batches
            if batch_idx < len(context_batches) - 1:
                print(f"     ⏳ Waiting {RATE_LIMIT_DELAY}s for rate limit...")
                time.sleep(RATE_LIMIT_DELAY)
        
        all_goldens.extend(file_goldens[:total_count])
        print(f"  => {len(file_goldens[:total_count])} goldens saved for {filename}.")
        print(f"  Total goldens so far: {len(all_goldens)}\n")
        
        # Rate limit delay between files
        if i < len(txt_files) - 1:
            print(f"⏳ Waiting {RATE_LIMIT_DELAY}s before next file...")
            time.sleep(RATE_LIMIT_DELAY)
        
    # Save results
    print("\nSaving dataset...")
    goldens_dict = []
    for g in all_goldens:
        try:
            goldens_dict.append(g.model_dump())
        except AttributeError:
            goldens_dict.append(g.dict())
            
    out_path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(goldens_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Saved {len(all_goldens)} golden test cases to {out_path}")


if __name__ == "__main__":
    generate_golden_dataset()
