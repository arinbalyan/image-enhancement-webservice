#!/usr/bin/env python
"""
Google Drive Batch Processor for Image Enhancement.

This script processes images from a Google Drive folder, enhances them using
the image enhancement pipeline, and saves them to an output folder.

Features:
- OAuth2 authentication with Google Drive
- Recursive folder processing
- Folder structure preservation
- Resume capability (skip already processed images)
- Progress tracking with progress bar
- Error handling and logging
- Parallel processing (optional)

Usage:
    python scripts/batch_processor.py --input "raw_images" --output "image-enhancer"
    python scripts/batch_processor.py --credentials path/to/credentials.json
    python scripts/batch_processor.py --resume  # Resume from previous session
"""

import os
import sys
import json
import pickle
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from algorithms.manager import AlgorithmManager
from utils.logger import get_logger
from config.settings import get_settings

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Image file extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class GoogleDriveBatchProcessor:
    """Batch processor for Google Drive images."""
    
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        progress_file: str = "processing_progress.json"
    ):
        """
        Initialize the batch processor.
        
        Args:
            credentials_path: Path to Google OAuth credentials file
            token_path: Path to save/load OAuth token
            progress_file: Path to save processing progress for resume
        """
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.progress_file = Path(progress_file)
        
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.service = None
        self.algorithm_manager = None
        
        # Processing state
        self.processed_files: set = set()
        self.failed_files: Dict[str, str] = {}
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None
        }
    
    def authenticate(self) -> None:
        """Authenticate with Google Drive API."""
        creds = None
        
        # Load existing token if available
        if self.token_path.exists():
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or create new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}\n"
                        "Please download OAuth credentials from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        self.logger.info("Successfully authenticated with Google Drive")
    
    def get_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Get folder ID from folder path.
        
        Args:
            folder_path: Folder path like "My Drive/raw_images"
            
        Returns:
            Folder ID or None if not found
        """
        parts = folder_path.strip('/').split('/')
        parent_id = 'root'
        
        for part in parts:
            query = f"name='{part}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                return None
            
            parent_id = files[0]['id']
        
        return parent_id
    
    def create_folder(self, name: str, parent_id: str = 'root') -> str:
        """
        Create a folder in Google Drive.
        
        Args:
            name: Folder name
            parent_id: Parent folder ID
            
        Returns:
            Created folder ID
        """
        # Check if folder already exists
        query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, fields="files(id)").execute()
        
        if results.get('files'):
            return results['files'][0]['id']
        
        # Create new folder
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        folder = self.service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        return folder['id']
    
    def list_images(
        self,
        folder_id: str,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all image files in a folder.
        
        Args:
            folder_id: Folder ID to search
            recursive: Whether to search subfolders
            
        Returns:
            List of file metadata dictionaries
        """
        images = []
        folders_to_process = [(folder_id, "")]
        
        while folders_to_process:
            current_folder_id, current_path = folders_to_process.pop(0)
            
            # Query for files in current folder
            query = f"'{current_folder_id}' in parents and trashed=false"
            page_token = None
            
            while True:
                results = self.service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageToken=page_token
                ).execute()
                
                for file in results.get('files', []):
                    file_name = file['name']
                    file_path = f"{current_path}/{file_name}" if current_path else file_name
                    
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        if recursive:
                            folders_to_process.append((file['id'], file_path))
                    else:
                        # Check if it's an image
                        ext = Path(file_name).suffix.lower()
                        if ext in IMAGE_EXTENSIONS:
                            images.append({
                                'id': file['id'],
                                'name': file_name,
                                'path': file_path,
                                'size': int(file.get('size', 0)),
                                'mimeType': file['mimeType']
                            })
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
        
        return images
    
    def download_file(self, file_id: str, local_path: str) -> None:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: File ID to download
            local_path: Local path to save file
        """
        request = self.service.files().get_media(fileId=file_id)
        
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
    
    def upload_file(
        self,
        local_path: str,
        folder_id: str,
        filename: str
    ) -> str:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Local file path
            folder_id: Destination folder ID
            filename: Name for the uploaded file
            
        Returns:
            Uploaded file ID
        """
        # Determine mime type
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'application/octet-stream')
        
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(local_path, mimetype=mime_type)
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file['id']
    
    def load_progress(self) -> None:
        """Load processing progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get('processed_files', []))
                self.failed_files = data.get('failed_files', {})
                self.stats = data.get('stats', self.stats)
            self.logger.info(f"Loaded progress: {len(self.processed_files)} files already processed")
    
    def save_progress(self) -> None:
        """Save processing progress to file."""
        data = {
            'processed_files': list(self.processed_files),
            'failed_files': self.failed_files,
            'stats': self.stats
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def process_image(
        self,
        image_info: Dict[str, Any],
        output_folder_id: str,
        output_structure: Dict[str, str]
    ) -> bool:
        """
        Process a single image.
        
        Args:
            image_info: Image metadata
            output_folder_id: Output folder ID
            output_structure: Mapping of paths to folder IDs
            
        Returns:
            True if successful, False otherwise
        """
        file_id = image_info['id']
        file_name = image_info['name']
        file_path = image_info['path']
        
        # Skip if already processed
        if file_id in self.processed_files:
            self.stats['skipped'] += 1
            return True
        
        try:
            # Create temporary files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_input = Path(tmp_dir) / file_name
                tmp_output = Path(tmp_dir) / "output"
                tmp_output.mkdir(exist_ok=True)
                
                # Download image
                self.download_file(file_id, str(tmp_input))
                
                # Initialize algorithm manager if needed
                if self.algorithm_manager is None:
                    self.algorithm_manager = AlgorithmManager()
                
                # Process image
                enhanced_path = self.algorithm_manager.enhance_image(
                    image_path=str(tmp_input),
                    output_dir=str(tmp_output),
                    algorithms=None,  # Auto-select
                    preserve_original=True
                )
                
                # Determine output folder
                path_parts = Path(file_path).parent.parts
                current_folder_id = output_folder_id
                
                for part in path_parts:
                    if part:
                        folder_key = '/'.join(path_parts[:path_parts.index(part) + 1])
                        if folder_key not in output_structure:
                            output_structure[folder_key] = self.create_folder(part, current_folder_id)
                        current_folder_id = output_structure[folder_key]
                
                # Upload enhanced image
                enhanced_name = Path(enhanced_path).name
                self.upload_file(enhanced_path, current_folder_id, enhanced_name)
            
            self.processed_files.add(file_id)
            self.stats['completed'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_name}: {e}")
            self.failed_files[file_id] = str(e)
            self.stats['failed'] += 1
            return False
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        resume: bool = True,
        parallel: bool = False,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Process all images in a Google Drive folder.
        
        Args:
            input_folder: Input folder path in Google Drive
            output_folder: Output folder path in Google Drive
            resume: Whether to resume from previous session
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Starting batch processing: {input_folder} -> {output_folder}")
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Load progress if resuming
        if resume:
            self.load_progress()
        
        # Get input folder ID
        input_folder_id = self.get_folder_id(input_folder)
        if not input_folder_id:
            raise ValueError(f"Input folder not found: {input_folder}")
        
        # Create output folder
        output_parts = output_folder.strip('/').split('/')
        output_folder_id = 'root'
        for part in output_parts:
            output_folder_id = self.create_folder(part, output_folder_id)
        
        # List all images
        images = self.list_images(input_folder_id)
        self.stats['total'] = len(images)
        self.logger.info(f"Found {len(images)} images to process")
        
        # Track folder structure for output
        output_structure: Dict[str, str] = {}
        
        # Process images
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(images), desc="Processing")
        except ImportError:
            progress_bar = None
        
        if parallel and max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_image,
                        img,
                        output_folder_id,
                        output_structure
                    ): img for img in images
                }
                
                for future in as_completed(futures):
                    if progress_bar:
                        progress_bar.update(1)
                    self.save_progress()
        else:
            # Sequential processing
            for i, image in enumerate(images):
                self.process_image(image, output_folder_id, output_structure)
                if progress_bar:
                    progress_bar.update(1)
                
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    self.save_progress()
        
        if progress_bar:
            progress_bar.close()
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.save_progress()
        
        # Print summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Processing Complete!")
        self.logger.info(f"Total: {self.stats['total']}")
        self.logger.info(f"Completed: {self.stats['completed']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"{'='*50}")
        
        return self.stats


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Process images from Google Drive folder"
    )
    parser.add_argument(
        '--input', '-i',
        default='raw_images',
        help='Input folder path in Google Drive'
    )
    parser.add_argument(
        '--output', '-o',
        default='image-enhancer',
        help='Output folder path in Google Drive'
    )
    parser.add_argument(
        '--credentials', '-c',
        default='credentials.json',
        help='Path to Google OAuth credentials file'
    )
    parser.add_argument(
        '--token', '-t',
        default='token.json',
        help='Path to save/load OAuth token'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from previous session'
    )
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel processing'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore previous progress'
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = GoogleDriveBatchProcessor(
        credentials_path=args.credentials,
        token_path=args.token
    )
    
    # Authenticate
    print("Authenticating with Google Drive...")
    processor.authenticate()
    
    # Process folder
    print(f"Processing images from '{args.input}' to '{args.output}'...")
    stats = processor.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        resume=not args.no_resume,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    print("\nDone!")
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
