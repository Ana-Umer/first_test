import os 
import shutil
from fastapi import FastAPI, UploadFile, File
import tempfile
upload_directory = "uploads_pdfs"
def save_uploaded_file(uploaded_file:list[UploadFile]):
    os.makedirs(upload_directory, exist_ok=True)
    file_paths = []
    for file in uploaded_file   :
        if file.filename is not None:
            temp_path = os.path.join(upload_directory, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(temp_path)    
        