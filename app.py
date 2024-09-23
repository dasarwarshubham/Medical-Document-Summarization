import os
import json
import boto3
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import uuid  # For generating unique filenames

# Part 1 (Setup)
# Load environment variables (AWS credentials)
load_dotenv()

# Configuring Boto3 for retries
retry_config = Config(
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

# Create a boto3 session for accessing Bedrock and Textract
session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime',
                         aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                         aws_secret_access_key=os.environ.get(
                             "AWS_SECRET_ACCESS_KEY"),
                         config=retry_config)

textract = session.client('textract', config=retry_config)


# Function to upload the document
def upload_document(uploaded_file):
    """Save the uploaded document to the 'uploaded_files' folder and return its file path."""
    if uploaded_file is not None:
        try:
            # Create the folder if it doesn't exist
            upload_folder = "./uploaded_files"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Get the original file name and path
            file_path = os.path.join(upload_folder, uploaded_file.name)

            # If the file already exists, rename it with a unique ID
            if os.path.exists(file_path):
                file_name, file_extension = os.path.splitext(
                    uploaded_file.name)
                unique_id = str(uuid.uuid4())[:8]  # Generate a short unique ID
                file_path = os.path.join(
                    upload_folder, f"{file_name}_{unique_id}{file_extension}")

            # Write the file to the folder
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    return None


# Function to process the document (Extract text using Textract and split it)
def process_document(file_path, file_type):
    """Extract text from the document (TIFF, PDF, JPG, JPEG) and split it into manageable chunks."""

    # Extract text from TIFF or image (JPG, JPEG)
    def extract_text_from_image(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.detect_document_text(
                    Document={'Bytes': document.read()})

            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return None

    # Extract text from PDF using Textract's `analyze_document` API
    def extract_text_from_pdf(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.analyze_document(
                    Document={'Bytes': document.read()},
                    FeatureTypes=["TABLES", "FORMS"]
                )
            text = ""
            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    text += block["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None

    # Process the file based on its type
    if file_path and file_type:
        if file_type in ["jpg", "jpeg", "tiff"]:
            extracted_text = extract_text_from_image(file_path)
        elif file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            st.error("Unsupported file format.")
            return []

        # If text was successfully extracted, split it into chunks
        if extracted_text:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    chunk_overlap=0
                )
                texts = text_splitter.split_text(extracted_text)
                return texts
            except Exception as e:
                st.error(f"Error splitting text into chunks: {e}")
                return []
        else:
            st.error("Failed to extract text from the document.")
            return []
    else:
        st.error("Invalid file path or file type.")
        return []


# Function to generate summary using the Claude Model
def generate_summary(documents):
    """Generate summary from the provided document chunks using the Claude model."""
    try:
        summaries = []
        for chunk in documents:
            # Prepare the request body for the Claude model
            messages = [
                {"role": "user", "content": "I have a medical document that I'd like summarized. Output should have short description with the Patient's Complaint, History and Observations."},
                {"role": "assistant", "content": "Sure, I can help with that. Please provide the text of the medical document."},
                {"role": "user", "content": chunk}
            ]

            body = json.dumps({
                "max_tokens": 1000,
                "messages": messages,
                "anthropic_version": "bedrock-2023-05-31"
            })

            try:
                # Call the Claude model for summarization
                response = bedrock.invoke_model(
                    body=body, modelId=os.environ.get("BEDROCK_MODEL"))
                response_body = json.loads(response.get("body").read())
                output = response_body.get("content", "No summary generated")
                summaries.append(output[0]["text"])
            except ClientError as e:
                st.error(
                    f"Amazon Bedrock error: {e.response['Error']['Message']}")
                return None
            except Exception as e:
                st.error(f"Error invoking the Claude model: {e}")
                return None

        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


# Streamlit UI
st.title("Medical Document Summarization")

# Step 1: Upload the document
uploaded_file = st.file_uploader(
    "Upload a medical document (PDF, TIFF, JPG, JPEG)", type=["pdf", "tiff", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step 2: Process the document
    file_type = uploaded_file.name.split(".")[-1].lower()
    file_path = upload_document(uploaded_file)

    if file_path:
        st.write(f"Processing document: {uploaded_file.name}")
        texts = process_document(file_path, file_type)

        if texts:
            # Step 3: Generate summary
            st.write("Generating summary...")
            summary = generate_summary(texts)

            if summary:
                st.subheader("Generated Summary")
                st.write(summary)
            else:
                st.error("Failed to generate summary.")
    else:
        st.error("Failed to upload the document.")
else:
    st.info("Please upload a medical document to get started.")
