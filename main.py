from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from docx import Document
import mysql.connector
import json
import os
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume Extractor API", description="An API to extract resume information using LLMs.", version="1.1.0")

PROMPT = """I want you to extract following information from a resume.
Information required:
1. relevant_title - What is the main job title or role of the person?
2. years_of_experience - How many years of experience does this person have with different technologies or role, for example - [2 years experience in backend, 3 years of experience in frontend]?
3. techstack - List the technologies, tools, and frameworks this person is proficient in.
4. current_location - Where is the current location of the person?
5. certifications - List all certifications the person has obtained.
6. native_languages_known - What languages does this person speak natively?
7. computer_languages_known - What programming or computer languages does this person know?

Further instructions:
 - All details should be from the resume provided, if something is not present give NA, don't assume anything.
 - Output should always be in below specified json format.
output - {{"relevant_title": value, "years_of_experience": [value], "techstack": [value], "current_location": value, "certifications": [value], "native_languages_known": [value], "computer_languages_known": [value]}}
"""

# Database connection function
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",  # WampServer host
            user="root",       # Your MySQL username
            password="",       # Your MySQL password
            database="ses"     # Database name
        )
        return connection
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Function to extract text from different file types
def extract_text(file, content_type):
    formatted_document = []

    if content_type == "application/pdf":
        # Extract text from PDF
        reader = PdfReader(file)
        for page in reader.pages:
            formatted_document.append(page.extract_text())
    elif content_type == "text/plain":
        # Extract text from TXT
        for line in file.readlines():
            formatted_document.append(line.decode("utf-8"))
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Extract text from DOCX
        doc = Document(file)
        for paragraph in doc.paragraphs:
            formatted_document.append(paragraph.text)
    else:
        raise ValueError("Unsupported file type.")

    return formatted_document

# Function to process resume and extract information
def extract_resume_info(file, content_type):
    # Step 1: Extract text from the file
    formatted_document = extract_text(file, content_type)

    # Step 2: Use OpenAI's API for information extraction
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="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"
    )
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"This is the resume - \n{formatted_document}"}
        ],
        response_format={"type": "json_object"}
    )
    output = completion.choices[0].message.content
    return json.loads(output)

def save_to_db(data):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Insert data into the table
        query = """
        INSERT INTO cv_info 
        (relevant_title, years_of_experience, techstack, current_location, certifications, native_languages_known, computer_languages_known) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            data.get("relevant_title", "NA"),
            json.dumps(data.get("years_of_experience", [])),  # Store lists as JSON
            json.dumps(data.get("techstack", [])),  # Store lists as JSON
            data.get("current_location", "NA"),
            json.dumps(data.get("certifications", [])),  # Store lists as JSON
            json.dumps(data.get("native_languages_known", [])),  # Store lists as JSON
            json.dumps(data.get("computer_languages_known", []))  # Store lists as JSON
        )
        cursor.execute(query, values)
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")
    finally:
        cursor.close()
        connection.close()

@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    """
    Upload a resume (.pdf, .txt, .docx), extract structured information, and store it in the database.
    """
    supported_types = [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    if file.content_type not in supported_types:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files are supported.")

    try:
        # Extract resume information
        resume_info = extract_resume_info(file.file, file.content_type)

        # Save extracted information to the database
        save_to_db(resume_info)

        return {"status": "success", "message": "Resume information extracted and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

@app.get("/")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "Resume Extractor API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
