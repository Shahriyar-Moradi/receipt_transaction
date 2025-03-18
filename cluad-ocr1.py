import streamlit
import tempfile
from transformers import pipeline
import fitz
import os
import shutil
from openai import OpenAI
import base64
from datetime import datetime
import re
import anthropic
import cv2


import streamlit

import tempfile
from transformers import pipeline
import fitz
from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
# Load environment variables
load_dotenv() 




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize models
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model=model_name)
labels = ["Transaction receipt", "other"]

CONFIDENCE_THRESHOLD = 0.7

# Anthropic client

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


import os

BASE_DIR = os.path.dirname(__file__)  # Directory where main.py resides
IMAGE_DIR = os.path.join(BASE_DIR, "images")

EXAMPLE_IMAGE_1 = os.path.join(IMAGE_DIR, "img3.jpeg")
EXAMPLE_IMAGE_2 = os.path.join(IMAGE_DIR, "img4.jpg")
EXAMPLE_IMAGE_3 = os.path.join(IMAGE_DIR, "img2.jpeg")


example_receipt_1_base64 = encode_image_to_base64(EXAMPLE_IMAGE_1)
example_receipt_2_base64 = encode_image_to_base64(EXAMPLE_IMAGE_2)
example_receipt_3_base64 = encode_image_to_base64(EXAMPLE_IMAGE_3)


def generate_base_filename(transaction_data):
    # Try to extract date from transaction data
    try:
        date_match = re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', transaction_data)
        if date_match:
            date_str = date_match.group().replace('/', '-')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
    except (TypeError, AttributeError):
        # Fallback if transaction_data is not a string or has no pattern match
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Create timestamp for uniqueness
    timestamp = datetime.now().strftime('%H%M%S')
    
    # Create base filename without extension
    return f"receipt_{date_str}_{timestamp}"

def save_files(transaction_data, image_path, output_dir="receipt_data"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure transaction_data is a string
    if not isinstance(transaction_data, str):
        try:
            transaction_data = str(transaction_data)
        except Exception:
            transaction_data = "Error: Unable to convert transaction data to string"

    base_filename = generate_base_filename(transaction_data)

    text_filepath = os.path.join(output_dir, f"{base_filename}.txt")
    image_extension = os.path.splitext(image_path)[1]
    image_filepath = os.path.join(output_dir, f"{base_filename}{image_extension}")
    
    # Save the text data
    with open(text_filepath, 'w') as f:
        f.write(transaction_data)
    shutil.copy2(image_path, image_filepath)
    return text_filepath, image_filepath

def extract_transaction_data(image_path):
    base64_image = encode_image_to_base64(image_path)
    image_media_type = "image/jpeg"  # Adjust based on your image format
    
    response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[
            {
                "role": "assistant",
                "content": (
                    '''OCR and extract data from the image it accurately and read Persian text from images and extract it accurately. "
                    We will provide example images with the correct extracted text. Then you'll get a new image
                    and should provide the extracted text in Persian, Extract the following transaction details:
                    - Date and Time
                    - Amount
                    - Merchant/Recipient
                    - Transaction Type
                    - Reference Number (if any)
                    Return in a clear, structured format.'''    
                            
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 1**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال ساتنا و پایا\n"
                            "شماره شبا برداشت: IR۶۵۰۶۰۰۵۲۰۶۰۱۰۰۰۹۳۳۰۵۱۰۰۱\n"
                            "مبلغ: ۳,۱۲۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ: ۱۴۰۳/۱۱/۱\n"
                            "نام و نام خانوادگی: امین ساعتیان الکام توسعه اماد\n"
                            "بابت: پرداخت قرض و تأدیه دیون\n"
                            "نام بانک: بانک قرض الحسنه مهر ایران\n"
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_1_base64,
                        },
                    },
                ]
            },
            
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 2**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال پول اینترنت بانک\n"
                            "شماره شبا برداشت: IR۸۹۰۵۶۰۹۵۰۱۷۱۰۰۲۹۰۰۶۶۵۰۰\n"
                            "مبلغ: ۵۰۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ: ۱۴۰۳/۱۰/۲۷\n"
                            "نام و نام خانوادگی: الکام توسعه اماد\n"
                            "بابت: وثیقه\n"
                            "نام بانک: مشخص نیست\n"
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_2_base64,
                        },
                    },
                ]
            },
            
            
            
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is **Example 3**. I will provide the image plus the correct text:\n\n"
                            "**Image** (below)\n\n"
                            "Correct Extraction:\n"
                            "نوع انتقال: انتقال ساتنا\n"
                            "شماره شبا برداشت: IR۶۵۰۶۰۰۵۲۰۶۰۱۰۰۰۹۳۳۰۵۱۰۰۱\n"
                            "مبلغ : ۱,۱۰۰,۰۰۰,۰۰۰ ریال\n"
                            "تاریخ : مشخض نیست \n"
                            "نام و نام خانوادگی : الکام توسعه اماد\n"
                            "بابت: پرداخت قرض و تأدیه دیون\n"
                            "نام بانک: بانک قرض الحسنه مهر ایران\n"
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_receipt_3_base64,
                        },
                    },
                ]
            },
            
            
            
            
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": (
            #                 "Here is **Example 3**. I will provide the image plus the correct text:\n\n"
            #                 "**Image** (below)\n\n"
            #                 "Correct Extraction:\n"
            #                 "نوع انتقال: انتقال شبا\n"
            #                 "شماره شبا برداشت: IR۴۵۰۱۹۰۰۰۰۰۰۰۲۱۸۳۷۵۶۷۱۰۰۹\n"
            #                 "مبلغ : ۲,۰۰۰,۰۰۰,۰۰۰ ریال\n"
            #                 "تاریخ: ۱۴۰۳/۱۲/۲۷\n"
            #                 "نام و نام خانوادگی :پارسیان\n"
            #                 "بابت: پرداخت قرض و تأدیه دیون\n"
            #                 "نام بانک: بانک ملت \n"
            #             ),
            #         },
            #         {
            #             "type": "image",
            #             "source": {
            #                 "type": "base64",
            #                 "media_type": "image/jpeg",
            #                 "data": example_receipt_4_base64,
            #             },
            #         },
            #     ]
            # },
            
            
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Now process this **NEW receipt** image. "
                            "Please extract the Persian text in the same style."
                        )
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ]
            },
        ]
    )
    
    # Extract the text content from the response
    try:
        # Anthropic response structure has content that contains the actual text
        return response.content[0].text
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error extracting content from response: {e}")
        print(f"Response structure: {response}")
        # Return a fallback value
        return "Error extracting transaction data"

def process_image(file, output_folder="converted_images"):
    os.makedirs(output_folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name
        
    file_extension = file.name.lower().split(".")[-1]
    if file_extension in {"jpg", "jpeg", "png"}:
        return file_path
    elif file_extension == "pdf":
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        output_path = os.path.join(output_folder, f"converted_page_1.png")
        pix.save(output_path)
        doc.close()
        return output_path
    return None



# Streamlit UI
streamlit.set_page_config(page_title="Receipt Analyzer")
streamlit.write("# Receipt Analyzer")

if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []

uploaded_file = streamlit.file_uploader("Upload receipt", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    streamlit.image(uploaded_file)
    processed_file = process_image(uploaded_file)
    # processed_file = image_preprocess(processed_file_1)
    
    with streamlit.spinner("Validating receipt..."):
        scores = classifier(processed_file, candidate_labels=labels)
        is_receipt = scores[0]['label'] == "Transaction receipt" and scores[0]['score'] > CONFIDENCE_THRESHOLD
    
    if is_receipt:
        streamlit.success("✅ Valid transaction receipt")
        with streamlit.spinner("Extracting transaction details..."):
            try:
                transaction_data = extract_transaction_data(processed_file)
                
                # Save both the text data and image
                text_filepath, image_filepath = save_files(transaction_data, processed_file)
                
                # Display the data and file locations
                streamlit.write("## Transaction Details")
                streamlit.write(transaction_data)
                streamlit.success(f"Files saved:\n- Text data: {text_filepath}\n- Image: {image_filepath}")
            except Exception as e:
                streamlit.error(f"Error processing receipt: {str(e)}")
        
        # streamlit.write("## Ask Questions")
        # if prompt := streamlit.chat_input("Ask about specific details"):
        #     streamlit.chat_message("user").write(prompt)
        #     streamlit.session_state.messages.append({"role": "user", "content": prompt})
            
            # with streamlit.spinner("Analyzing..."):
            #     response = client.chat.completions.create(
            #         model='gpt-4.5-preview',
            #         messages=[
            #             {
            #                 "role": "user",
            #                 "content": [
            #                     {"type": "text", "text": prompt},
            #                     {
            #                         "type": "image_url",
            #                         "image_url": {
            #                             "url": f"data:image/jpeg;base64,{encode_image_to_base64(processed_file)}",
            #                             "detail": "high",
            #                         }
            #                     }
            #                 ]
            #             }
            #         ],
            #     ).choices[0].message.content
                
            # streamlit.chat_message("assistant").write(response)
            # streamlit.session_state.messages.append({"role": "assistant", "content": response})
    else:
        streamlit.error("❌ This document does not appear to be a valid transaction receipt")