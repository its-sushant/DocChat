import os
import gdown
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from utils import build_automerging_index

def download_pdf(url, save_dir):
    """
    Download a PDF file from the given URL and save it to the specified directory.

    Parameters:
        url (str): URL of the PDF file.
        save_dir (str): Directory to save the downloaded PDF file.

    Returns:
        str: Path to the downloaded PDF file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_id = url.split("/")[-2]
    file_url = f"https://drive.google.com/uc?&id={file_id}"
    output_path = f"{save_dir}/downloaded_file.pdf"

    # Download the PDF file
    gdown.download(file_url, output_path)
    print("PDF downloaded successfully! Saved as:", output_path)
    return output_path

def build_index(save_dir, file_url):
    """
    Build an automerging index from a PDF file.

    Parameters:
        save_dir (str): Directory to save the index.
        file_url (str): URL of the PDF file.

    Returns:
        None
    """
    file_path = download_pdf(file_url, save_dir)
    
    print("Loading PDF file...")
    documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()

    document = Document(text="\n\n".join([doc.text for doc in documents]))
    print("Loaded PDF file successfully!")

    print("Building index...")
    build_automerging_index(
        [document],
        save_dir="./merging_index",
    )

if __name__ == "__main__":
    save_dir = "pdfs"
    file_url = "https://drive.google.com/file/d/1PK7kkvKwCvTumbnoYsLReV68jFexJ06X/view?usp=sharing"
    build_index(save_dir, file_url)