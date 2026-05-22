import os
import re
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "segmented_v2"
OUTPUT_FILENAME = "segment_sum_concat.txt"
TEST_LIMIT = None  # Set to None to process all files

def process_segment_file(file_path):
    """
    Parses segment_sum.txt, groups summary lines by chunk, 
    and writes a concatenated version.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Parse the header (optional, for metadata)
    header_match = re.search(r'Summary uses chunk \[(.*?)\]', content)
    header_info = f"Summary uses chunk [{header_match.group(1)}]\n\n" if header_match else ""

    # 2. Split into Chunk blocks
    # We look for 'Chunk X:' followed by the text until the next 'Chunk X:'
    blocks = re.split(r'(Chunk \d+:)', content)
    
    # Store aggregated data: { chunk_id_string: { "text": "...", "sums": [] } }
    aggregated = {}

    # Iterate through blocks (split results in [garbage, 'Chunk 0:', content, 'Chunk 1:', content...])
    for i in range(1, len(blocks), 2):
        chunk_label = blocks[i].strip() # e.g., "Chunk 0:"
        chunk_content = blocks[i+1]
        
        # Extract parts of the chunk
        # Note: We take original/translation from the first time we see the chunk
        original_match = re.search(r'original:\s*(.*?)\s*translation:', chunk_content, re.DOTALL)
        trans_match = re.search(r'translation:\s*(.*?)\s*summary:', chunk_content, re.DOTALL)
        sum_line_match = re.search(r'summary line \d+:\s*(.*)', chunk_content, re.DOTALL)

        if chunk_label not in aggregated:
            aggregated[chunk_label] = {
                "original": original_match.group(1).strip() if original_match else "",
                "translation": trans_match.group(1).strip() if trans_match else "",
                "sums": []
            }
        
        if sum_line_match:
            aggregated[chunk_label]["sums"].append(sum_line_match.group(1).strip())

    # 3. Write the concatenated output
    output_path = file_path.parent / OUTPUT_FILENAME
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(header_info)
        for label, data in aggregated.items():
            out.write(f"{label} \n")
            out.write(f"original: \n{data['original']}\n")
            out.write(f"translation: \n{data['translation']}\n")
            out.write(f"summary:\n")
            # The Se3 Concat Logic: join all sentences belonging to this chunk
            full_summary = " ".join(data['sums'])
            out.write(f"{full_summary}\n\n")
            
    return output_path

def main():
    print(f"🚀 Starting Se3 Concatenation...")
    files = sorted(list(DATA_ROOT.rglob("segment_sum.txt")))
    
    if not files:
        print(f"❌ No segment_sum.txt files found in {DATA_ROOT}")
        return

    count = 0
    for f in files:
        if TEST_LIMIT and count >= TEST_LIMIT:
            print(f"--- Reached test limit of {TEST_LIMIT} files ---")
            break
            
        try:
            out_p = process_segment_file(f)
            print(f"✅ Processed: {f.relative_to(PROJECT_ROOT)} -> {OUTPUT_FILENAME}")
            count += 1
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")

    print(f"\n✨ Done! Processed {count} files.")

if __name__ == "__main__":
    main()
