# These are the tools (modules) we need to build our small web server.
# FastAPI helps us  recreate the backend (the brain behind the website).
# File and UploadFile let usceive uploaded files.
# Form lets us read text data (like the user’s question) from the form.
from fastapi import FastAPI, File, UploadFile, Form

# This helps us connect the backend with the frontend (your HTML page).
# CORS means “Cross-Origin Resource Sharing” — it allows your HTML (which runs in the browser)
# to talk to your FastAPI app (which runs locally or on a server).
from fastapi.middleware.cors import CORSMiddleware

# The 'os' module helps us work with files and folders on your computer.
import os

# Create the main FastAPI app — this is like turning the lights on for our backend.
app = FastAPI()

# Allow your frontend (the HTML site) to communicate with this backend.
# The stars ("*") mean “allow everything” — this is okay for local testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow requests from any website
    allow_credentials=True,
    allow_methods=["*"],          # Allow all types of requests (GET, POST, etc.)
    allow_headers=["*"],          # Allow all headers (technical info sent with requests)
)

# Name of the folder where we’ll save uploaded files and questions.
UPLOAD_FOLDER = "upload"

# If the folder doesn’t exist yet, create it.
# This is where your uploaded PDFs, CSVs, etc. will go.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# This tells FastAPI to listen for “POST” requests at the web address "/api/upload".
# A POST request means “send data” — like uploading a file or submitting a form.
@app.post("/api/upload")
async def upload_file_and_save_query(
    file: UploadFile = File(...),   # This means we expect a file (PDF, CSV, etc.)
    query: str = Form(...)          # This means we expect a text question from the user
):
    # --- Save the uploaded file ---
    # Create a path like "uploads/myfile.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Open the new file in "write binary" mode (wb) and copy the uploaded content into it.
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # --- Save the user’s question as a text file ---
    # Name it based on the uploaded file, like "myfile.pdf_query.txt"
    query_filename = f"{file.filename}_query.txt"
    query_path = os.path.join(UPLOAD_FOLDER, query_filename)

    # Open the text file and write the user’s question inside.
    with open(query_path, "w", encoding="utf-8") as f:
        f.write(query)

    # --- Tell the browser it worked! ---
    # This message is sent back to the frontend as a small JSON (dictionary) response.
    return {
        "message": "File and query received!",
        "file_saved_as": file_path,
        "query_saved_as": query_path
        }
