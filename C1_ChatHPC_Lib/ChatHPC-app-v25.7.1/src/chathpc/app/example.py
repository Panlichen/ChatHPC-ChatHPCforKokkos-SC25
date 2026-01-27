#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document
import subprocess
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_interests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# LLM Configuration from environment variables
LLM_URL = os.getenv("LLM_URL", "https://api.siliconflow.cn/v1/chat/completions")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")


def call_llm(prompt, system_prompt="", max_tokens=500, temperature=0.0):
    """
    Call LLM API with the given prompt
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(LLM_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        return content
        
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return ""


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file
    """
    pdf_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
                    
    except Exception as e:
        logger.error(f"Error with pdfplumber on {pdf_path}: {e}")
        
        # Fallback to PyPDF2
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
        except Exception as e2:
            logger.error(f"Error with PyPDF2 on {pdf_path}: {e2}")
    
    return pdf_text


def extract_text_from_word(doc_path):
    """
    Extract text from a Word document (.doc/.docx)
    """
    doc_text = ""
    try:
        doc = Document(doc_path)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                doc_text += paragraph.text + "\n"
        
    except Exception as e:
        logger.error(f"Error extracting text from Word document {doc_path}: {e}")
    
    return doc_text


def get_file_hash(file_path):
    """
    Calculate SHA256 hash of a file for deduplication
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def convert_word_to_pdf(input_path, output_path):
    """
    Convert Word document to PDF using LibreOffice
    """
    try:
        # Use LibreOffice command line to convert Word to PDF
        # The command format is: libreoffice --headless --convert-to pdf input_file --outdir output_dir
        output_dir = os.path.dirname(output_path)
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                input_path,
                "--outdir",
                output_dir
            ],
            check=True,
            timeout=60
        )
        logger.info(f"Successfully converted {input_path} to PDF")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_path} to PDF: Command failed with exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Error converting {input_path} to PDF: Conversion timed out")
        return False
    except Exception as e:
        logger.error(f"Error converting {input_path} to PDF: {e}")
        return False


def is_libreoffice_available():
    """
    Check if LibreOffice is available on the system
    """
    try:
        subprocess.run(
            ["libreoffice", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_text(file_path):
    """
    Extract text from a file based on its extension
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.doc', '.docx']:
        return extract_text_from_word(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return ""


def analyze_content(content):
    """
    Analyze the content and generate research interests and keywords
    """
    # Generate research interests and keywords
    prompt = f"""Please analyze the following content and extract the research interests of the student.

Content: {content[:50000]}  # Limit content to first 5000 characters for better performance

Please return the result in the following format:
1. Chinese keywords: 3-5 research direction keywords in Chinese, separated by commas
2. English keywords: 3-5 research direction keywords in English, separated by commas
3. Chinese summary: A concise summary of the student's research interests in about 200 Chinese characters
4. English summary: A concise summary of the student's research interests in about 100 English words

Please use the exact format with labels as shown above.
"""
    
    system_prompt = "You are a helpful assistant that analyzes student research interests from academic documents."
    
    result = call_llm(prompt, system_prompt, max_tokens=1000, temperature=0.3)
    
    if not result:
        return None
    
    # Parse the result
    try:
        lines = result.split('\n')
        chinese_keywords = ""
        english_keywords = ""
        chinese_summary = ""
        english_summary = ""
        
        for line in lines:
            if line.startswith("1. Chinese keywords:"):
                chinese_keywords = line.split(":")[1].strip()
            elif line.startswith("2. English keywords:"):
                english_keywords = line.split(":")[1].strip()
            elif line.startswith("3. Chinese summary:"):
                chinese_summary = line.split(":")[1].strip()
            elif line.startswith("4. English summary:"):
                english_summary = line.split(":")[1].strip()
        
        return {
            "chinese_keywords": chinese_keywords,
            "english_keywords": english_keywords,
            "chinese_summary": chinese_summary,
            "english_summary": english_summary
        }
    
    except Exception as e:
        logger.error(f"Error parsing LLM result: {e}")
        return None


def generate_markdown_report(file_name, analysis_result):
    """
    Generate markdown report from analysis results
    """
    # Generate markdown content
    markdown_content = f"""# Research Interests Analysis

## File Name
{file_name}

## Research Direction Keywords

### Chinese
{analysis_result['chinese_keywords']}

### English
{analysis_result['english_keywords']}

## Research Interests Summary

### Chinese
{analysis_result['chinese_summary']}

### English
{analysis_result['english_summary']}
"""
    
    return markdown_content


def process_file(file_path, output_dir, mode):
    """
    Process a single file and generate analysis report
    """
    # Get file name and extension
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    output_file = f"{file_base}.md"
    output_path = os.path.join(output_dir, output_file)
    
    # Check if output file exists and decide processing strategy based on mode
    if mode == "update" and os.path.exists(output_path):
        logger.info(f"Skipping {file_name} as it already exists in output directory (update mode)")
        return True
    
    logger.info(f"Processing {file_name}")
    
    # Extract text from file
    content = extract_text(file_path)
    if not content:
        logger.error(f"Failed to extract text from {file_name}")
        return False
    
    # Analyze content
    analysis_result = analyze_content(content)
    if not analysis_result:
        logger.error(f"Failed to analyze content from {file_name}")
        return False
    
    # Generate markdown report
    markdown_content = generate_markdown_report(file_name, analysis_result)
    
    # Write to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Successfully generated analysis report for {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error writing to {output_path}: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse student research interests from documents")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="private_file", 
        help="Path to input directory containing student documents (default: private_file)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="interests", 
        help="Path to output directory for analysis reports (default: interests)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="update", 
        choices=["update", "replace"], 
        help="Processing mode: update (only new files) or replace (all files) (default: update)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if LibreOffice is available for Word to PDF conversion
    libreoffice_available = is_libreoffice_available()
    if libreoffice_available:
        logger.info("LibreOffice is available, will use it for Word to PDF conversion when needed")
    else:
        logger.info("LibreOffice is not available, will use python-docx to extract text from Word files directly")
    
    # Get all files in input directory
    all_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.pdf', '.doc', '.docx']:
                all_files.append(file_path)
    
    if not all_files:
        logger.info("No supported files found in input directory")
        return
    
    logger.info(f"Found {len(all_files)} supported files in input directory")
    
    # Step 1: Organize files and handle Word to PDF conversion if needed
    files_to_process = []
    processed_files = set()  # Track processed files by path
    file_hashes = set()       # Track processed files by hash for deduplication
    
    # First pass: process files and handle conversions
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Skip if file has already been processed
        if file_path in processed_files:
            logger.info(f"Skipping duplicate file path: {file_path}")
            continue
        
        if file_ext in ['.doc', '.docx']:
            # Word file found
            if libreoffice_available:
                # Check if corresponding PDF exists
                pdf_base = os.path.splitext(file_path)[0]
                pdf_path = f"{pdf_base}.pdf"
                
                if os.path.exists(pdf_path):
                    # PDF already exists, use it instead
                    logger.info(f"Found existing PDF {pdf_path} for Word file {file_path}, will use PDF for processing")
                    files_to_process.append((pdf_path, 'pdf'))
                    processed_files.add(file_path)  # Mark Word file as processed
                    processed_files.add(pdf_path)   # Mark PDF file as processed
                else:
                    # No PDF exists, convert Word to PDF
                    logger.info(f"No existing PDF found for {file_path}, converting to PDF")
                    if convert_word_to_pdf(file_path, pdf_path):
                        # Conversion successful, process the new PDF
                        files_to_process.append((pdf_path, 'pdf'))
                        processed_files.add(file_path)  # Mark Word file as processed
                        processed_files.add(pdf_path)   # Mark PDF file as processed
                    else:
                        # Conversion failed, fall back to direct Word processing
                        logger.error(f"Failed to convert {file_path} to PDF, will process Word file directly")
                        files_to_process.append((file_path, 'word'))
                        processed_files.add(file_path)  # Mark Word file as processed
            else:
                # LibreOffice not available, process Word file directly
                logger.info(f"Processing Word file {file_path} directly since LibreOffice is not available")
                files_to_process.append((file_path, 'word'))
                processed_files.add(file_path)  # Mark Word file as processed
        elif file_ext == '.pdf':
            # PDF file found
            files_to_process.append((file_path, 'pdf'))
            processed_files.add(file_path)  # Mark PDF file as processed
    
    # Step 2: Deduplicate files based on content hash
    unique_files_to_process = []
    
    for file_path, file_type in files_to_process:
        # Calculate file hash for deduplication
        file_hash = get_file_hash(file_path)
        if file_hash and file_hash in file_hashes:
            # Duplicate content found, skip
            logger.info(f"Skipping file with duplicate content: {file_path}")
            continue
        
        # Add to unique processing list
        unique_files_to_process.append((file_path, file_type))
        if file_hash:
            file_hashes.add(file_hash)
    
    logger.info(f"After deduplication, {len(unique_files_to_process)} unique files to process")
    
    # Step 3: Process unique files
    success_count = 0
    failure_count = 0
    
    for file_path, file_type in tqdm(unique_files_to_process, desc="Processing files"):
        if process_file(file_path, args.output_dir, args.mode):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    logger.info(f"Processing completed: {success_count} successes, {failure_count} failures")
    print(f"Processing completed: {success_count} successes, {failure_count} failures")


if __name__ == "__main__":
    main()
