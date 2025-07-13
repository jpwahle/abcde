import glob
import os
from collections import defaultdict

# Directory containing the chunk files
input_dir = 'outputs_google_ngrams/'
output_dir = 'outputs_google_ngrams/merged/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all tsv files
chunk_files = glob.glob(os.path.join(input_dir, '*.tsv'))

# Group files by prefix (e.g., 'do', 'dp', 'dr')
groups = defaultdict(list)
base_prefix = 'googlebooks-eng-fiction-all-5gram-20120701-'

for file_path in chunk_files:
    filename = os.path.basename(file_path)
    if '_chunk_' in filename:
        # Extract the prefix like 'do'
        part = filename.replace(base_prefix, '').split('_chunk_')[0]
        groups[part].append(file_path)

# For each group, merge the chunks
for prefix, files in groups.items():
    # Sort files by chunk number
    sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('_chunk_')[1].split('.')[0]))
    
    merged_file = os.path.join(output_dir, f'{base_prefix}{prefix}_merged.tsv')
    
    with open(merged_file, 'w', encoding='utf-8') as outfile:
        for i, infile_path in enumerate(sorted_files):
            with open(infile_path, 'r', encoding='utf-8') as infile:
                if i > 0:
                    next(infile)  # Skip header for all but the first file
                for line in infile:
                    outfile.write(line)
    print(f'Merged {len(sorted_files)} chunks for prefix "{prefix}" into {merged_file}')

# Now merge all prefix merged files into one big file
all_merged_file = os.path.join(output_dir, 'all_ngrams_merged.tsv')

merged_prefix_files = glob.glob(os.path.join(output_dir, '*_merged.tsv'))
merged_prefix_files = [f for f in merged_prefix_files if not os.path.basename(f) == 'all_ngrams_merged.tsv']
merged_prefix_files.sort()  # Sort alphabetically by filename

with open(all_merged_file, 'w', encoding='utf-8') as outfile:
    for i, infile_path in enumerate(merged_prefix_files):
        with open(infile_path, 'r', encoding='utf-8') as infile:
            if i > 0:
                next(infile)  # Skip header for all but the first
            for line in infile:
                outfile.write(line)

print(f'Merged all prefix files into {all_merged_file}')