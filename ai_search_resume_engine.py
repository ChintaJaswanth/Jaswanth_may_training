!pip install -q sentence-transformers faiss-cpu PyMuPDF

import os
import shutil
import zipfile
import fitz # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from google.colab import files

# Ensure the 'resumes' folder exists and is correctly set up as a directory
folder = "resumes"
if os.path.exists(folder):
    if not os.path.isdir(folder):
        # It exists but is not a directory (e.g., a file named 'resumes').
        # Remove the conflicting file and then create the directory.
        print(f"Warning: A non-directory item named '{folder}' was found. Removing it to create a proper directory.")
        os.remove(folder)
        os.makedirs(folder)
        print(f"Created directory: {folder}.")
    # If it exists and is a directory, no action needed.
else:
    # Folder does not exist, create it.
    os.makedirs(folder)
    print(f"Created directory: {folder}.")

uploaded = files.upload()

for name, data in uploaded.items():
    src_path = os.path.join('/content', name)

    # Handle double .zip extension if present
    if name.lower().endswith('.zip.zip'):
        print(f"Detected double .zip extension for file: {name}. Renaming...")
        new_name = name[:-4] # Remove the last .zip
        os.rename(src_path, os.path.join('/content', new_name))
        src_path = os.path.join('/content', new_name)
        name = new_name

    if name.lower().endswith('.zip'):
        print(f"Extracting ZIP file: {name}")
        try:
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                zip_ref.extractall('resumes')
            print(f"Successfully extracted {name} to 'resumes/'")
            # Optionally remove the zip file after extraction
            os.remove(src_path)
            print(f"Removed uploaded zip file: {name}")
        except zipfile.BadZipFile:
            print(f"Error: '{name}' is not a valid ZIP file.")
        except Exception as e:
            print(f"An unexpected error occurred during ZIP extraction: {e}")
    else:
        # For individual files, move them to the 'resumes' directory
        dest_path = os.path.join('resumes', name)
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved '{name}' to 'resumes/'")
        except Exception as e:
            print(f"Error moving file '{name}': {e}")

print("File upload and processing complete.")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

resume_texts = []
resume_names = []

# Process resumes from the 'resumes' folder and its subdirectories
for root, _, files_in_dir in os.walk(folder):
    for file_name in files_in_dir:
        if file_name.lower().endswith(".pdf"):
            path = os.path.join(root, file_name)
            try:
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                resume_names.append(os.path.relpath(path, folder))
                resume_texts.append(text)
                print("Indexed:", os.path.relpath(path, folder))
            except Exception as e:
                print("Error processing file:", os.path.relpath(path, folder), e)

if not resume_texts:
    print("No PDF resumes found in the 'resumes' folder or its subdirectories. Please upload some files and re-run this cell.")
else:
    # Generate embeddings
    embeddings = model.encode(resume_texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    # Save the index and metadata
    faiss.write_index(index, "faiss.index")
    np.save("metadata.npy", {"names": resume_names, "texts": resume_texts})

    print("Index and metadata saved successfully.")

EXPERIENCE_PATTERNS = [
    re.compile(r'experience:\s*(\d+)\s*years?'),
    re.compile(r'(\d+)\+?\s*years?\s*(?:of)?\s*experience'),
    re.compile(r'(\d+)\+?\s*yrs?\s*(?:of)?\s*experience'),
    re.compile(r'(\d+)\s*years?\s*(?:total|overall)\s*experience')
]

def extract_experience_from_text(text):
    text_lower = text.lower()
    for pattern in EXPERIENCE_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            return int(match.group(1))
    return None

def search_resumes(query, k=5, model_instance=None, index_instance=None, resume_names_list=None, resume_texts_list=None):
    if not resume_names_list:
        print("No resumes indexed. Cannot perform search.")
        return []

    if model_instance is None or index_instance is None:
        print("Error: Model or index not provided to search_resumes function.")
        return []

    query_embedding = model_instance.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index_instance.search(query_embedding.astype("float32"), k * 2)

    initial_results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        experience = extract_experience_from_text(resume_texts_list[idx])
        initial_results.append({
            "resume_name": resume_names_list[idx],
            "distance": distances[0][i],
            "text_snippet": resume_texts_list[idx][:500] + "...",
            "experience_years": experience
        })
    return initial_results

def main():
    try:
        index = faiss.read_index("faiss.index")
        metadata = np.load("metadata.npy", allow_pickle=True).item()
        resume_names = metadata["names"]
        resume_texts = metadata["texts"]
        print("FAISS index and metadata loaded successfully.")
    except FileNotFoundError:
        print("Error: 'faiss.index' or 'metadata.npy' not found. Please run the indexing cell first.")
        resume_names = []
        resume_texts = []
        return

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    query = input("Enter your search query (e.g., Python developer): ")
    experience_filter_str = input("Enter experience filter (optional, e.g., '10', '10+', '5-10'): ")

    search_results = search_resumes(query, k=10, model_instance=model, index_instance=index, resume_names_list=resume_names, resume_texts_list=resume_texts)

    final_results = []

    if experience_filter_str:
        match_exact = re.match(r'^(\d+)$', experience_filter_str.strip())
        match_at_least = re.match(r'^(\d+)\+$', experience_filter_str.strip())
        match_range = re.match(r'^(\d+)-(\d+)$', experience_filter_str.strip())

        for result in search_results:
            exp_years = result['experience_years']
            if exp_years is None:
                continue

            if match_exact:
                target_exp = int(match_exact.group(1))
                if exp_years == target_exp:
                    final_results.append(result)
            elif match_at_least:
                min_exp = int(match_at_least.group(1))
                if exp_years >= min_exp:
                    final_results.append(result)
            elif match_range:
                min_exp = int(match_range.group(1))
                max_exp = int(match_range.group(2))
                if min_exp <= exp_years <= max_exp:
                    final_results.append(result)
    else:
        final_results = search_results

    final_results = sorted(final_results, key=lambda x: x['distance'], reverse=True)[:3]

    if final_results:
        print(f"\nTop {len(final_results)} relevant resumes for query: '{query}'")
        if experience_filter_str:
            print(f"(Filtered by experience: {experience_filter_str})")
        for result in final_results:
            print("---------------------------------------------------")
            print(f"Resume: {result['resume_name']}")
            print(f"Similarity Score (IP): {result['distance']:.4f}")
            if result['experience_years'] is not None:
                print(f"Experience: {result['experience_years']} years")
            print(f"Snippet: {result['text_snippet']}")
    else:
        print("No relevant resumes found with the specified criteria.")

if __name__ == '__main__':
    main()

