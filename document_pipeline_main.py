# document_pipeline_main.py

import os
import csv
import argparse
import yaml
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable, Any
from tqdm import tqdm

# Import pipeline modules
from pipeline.ingestion import ingest_documents
from pipeline.extract_pbac_metadata_gemini import extract_metadata as extract_metadata_gemini
from pipeline.local_extractor import extract_metadata_local
from pipeline.validator import validate_and_clean_json
from pipeline.embedding_generator import generate_embeddings_for_document

# --- MODIFIED: Import the new central logger ---
from utils.logger import setup_logger
from pipeline.utils import get_pdf_page_count, split_pdf

PDF_PAGE_LIMIT = 200

def create_directories(paths_config: Dict, logger):
    for key, path in paths_config.items():
        if key in ['raw_json', 'validated_json', 'embeddings', 'logs', 'cache', 'split_pdfs', 'reports']:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                logger.error(f"Could not create directory {path}: {e}")
                raise

def resolve_paths(config: Dict):
    paths = config['paths']
    output_base = paths.get('output_base', '')
    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = val.replace('{paths.output_base}', output_base)
    return config

def process_document(doc_path: str, config: Dict, extractor_fn: Callable) -> Dict:
    logger = config['logger']
    paths = config['paths']
    force_refresh = config.get('force_refresh', False)
    doc_filename_base = os.path.splitext(os.path.basename(doc_path))[0]
    
    final_embedding_path = os.path.join(paths['embeddings'], f"{doc_filename_base}.json")
    if os.path.exists(final_embedding_path) and not force_refresh:
        return {'file': os.path.basename(doc_path), 'status': 'SKIPPED', 'details': 'Final embedding file already exists.'}
    
    logger.info(f"Stage 1: Starting extraction for '{doc_path}' using '{config['extraction']['provider']}' provider.")
    raw_json_path = os.path.join(paths['raw_json'], f"{doc_filename_base}.json")
    raw_json_data = extractor_fn(doc_path, config, logger)
    
    if raw_json_data.get("error"):
        return {'file': os.path.basename(doc_path), 'status': 'ERROR', 'details': f"Extraction failed: {raw_json_data['error']}"}

    with open(raw_json_path, 'w', encoding='utf-8') as f:
        json.dump(raw_json_data, f, indent=2)

    validated_json_path = os.path.join(paths['validated_json'], f"{doc_filename_base}.json")
    validated_data, report = validate_and_clean_json(raw_json_data, source_filename=os.path.basename(doc_path))
    if report['status'] == 'error':
        return {'file': os.path.basename(doc_path), 'status': 'ERROR', 'details': f"Validation failed: {report['errors']}"}
    
    with open(validated_json_path, 'w', encoding='utf-8') as f:
        json.dump(validated_data, f, indent=2)

    final_output = generate_embeddings_for_document(validated_data, config, logger)
    if final_output and final_output.get("chunks"):
        with open(final_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
    else:
        return {'file': os.path.basename(doc_path), 'status': 'WARNING', 'details': 'No chunks were produced during embedding.'}

    return {'file': os.path.basename(doc_path), 'status': 'SUCCESS', 'details': f"Successfully processed and saved to {final_embedding_path}"}

def run_pipeline(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config = resolve_paths(config)
    paths = config['paths']
    
    # --- MODIFIED: Setup the central logger ---
    log_path = os.path.join(paths['logs'], 'pipeline.log')
    logger = setup_logger(name='pipeline', log_file=log_path)
    config['logger'] = logger
    
    provider = config.get('extraction', {}).get('provider', 'local')
    if provider == 'gemini':
        if not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("GOOGLE_API_KEY must be set to use the 'gemini' extractor.")
        extractor_function = extract_metadata_gemini
    elif provider == 'local':
        extractor_function = extract_metadata_local
    else:
        raise ValueError(f"Invalid extraction provider '{provider}' in config.yaml. Choose 'gemini' or 'local'.")
    
    logger.info("="*20 + " PIPELINE STARTED " + "="*20)
    logger.info(f"Using extraction provider: {provider}")

    try:
        paths['split_pdfs'] = os.path.join(paths['output_base'], '0_split_pdfs_cache')
        create_directories(paths, logger)
    except Exception:
        logger.error("Failed to create necessary directories. Exiting.")
        return

    page_limit = config.get('processing', {}).get('pdf_page_limit', PDF_PAGE_LIMIT)
    initial_documents = ingest_documents(paths['input'], config.get('supported_formats', ['pdf']), logger)
    documents_to_process: List[str] = []

    logger.info("--- Starting Pre-processing: Checking for large or corrupt PDFs ---")
    for doc_path in initial_documents:
        if doc_path.lower().endswith('.pdf'):
            page_count = get_pdf_page_count(doc_path, logger)
            if page_count == 0:
                continue
            elif page_count > page_limit:
                logger.warning(f"Found large PDF: {os.path.basename(doc_path)} ({page_count} pages). Splitting into chunks of {page_limit}...")
                split_parts = split_pdf(doc_path, page_limit, paths['split_pdfs'], logger)
                documents_to_process.extend(split_parts)
            else:
                documents_to_process.append(doc_path)
        else:
            documents_to_process.append(doc_path)
    logger.info("--- Pre-processing Complete ---")
    
    if not documents_to_process:
        logger.warning("No new documents found to process.")
        logger.info("="*21 + " PIPELINE ENDED " + "="*21)
        return

    logger.info(f"Total processing queue size: {len(documents_to_process)} files/parts.")
    
    run_summary = []
    with ThreadPoolExecutor(max_workers=config.get('max_threads', 1)) as executor:
        futures = {executor.submit(process_document, doc_path, config, extractor_function): doc_path for doc_path in documents_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Documents", unit="file"):
            try:
                result = future.result()
                run_summary.append(result)
            except Exception as e:
                doc_path = futures[future]
                logger.error(f"A critical error occurred while processing {doc_path}: {e}", exc_info=True)
                run_summary.append({'file': os.path.basename(doc_path), 'status': 'CRITICAL_ERROR', 'details': str(e)})

    if run_summary:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(paths['reports'], f'pipeline_run_report_{timestamp}.csv')
        
        try:
            with open(report_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['file', 'status', 'details']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(run_summary)
            logger.info(f"📊 Pipeline run summary saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write summary report: {e}")

    logger.info("="*21 + " PIPELINE ENDED " + "="*21)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-stage document processing pipeline.")
    parser.add_argument("--config", required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()
    
    run_pipeline(args.config)