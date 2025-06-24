#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Union, TypedDict
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Annotations
from mcp import McpError
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file'
]

class Settings(BaseSettings):
    """Server configuration settings"""
    GOOGLE_SERVICE_ACCOUNT_PATH: str | None = None
    GOOGLE_CREDENTIALS_PATH: str = "./client_secret.json"  # Changed to current directory
    GOOGLE_TOKEN_PATH: str = "./token.json"  # Changed to current directory
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_prefix = "GOOGLE_SHEETS_MCP_"

class GoogleSheetsError(Exception):
    """Base exception for Google Sheets operations"""
    pass

class CredentialsError(GoogleSheetsError):
    """Raised when there are issues with credentials"""
    pass

class SheetNotFoundError(GoogleSheetsError):
    """Raised when a sheet is not found"""
    pass

class SheetRange(BaseModel):
    sheet_name: str
    cell_range: str
    
    def to_a1_notation(self) -> str:
        return f"'{self.sheet_name}'!{self.cell_range}"

# Create the MCP server instance
mcp = FastMCP("Google Sheets MCP")

class GoogleSheetsMCP:
    """
    Google Sheets MCP Server implementation
    """
    def __init__(self, service_account_path: Optional[str] = None, credentials_path: str = "./client_secret.json"):
        """
        Initialize the Google Sheets MCP Server
        
        Args:
            service_account_path: Path to service account JSON file
            credentials_path: Path to OAuth credentials JSON file
        """
        self.service_account_path = service_account_path
        self.credentials_path = credentials_path
        self.sheets_service = None
        self.drive_service = None
        self._initialize_services()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_api_request(self, request_func, *args, **kwargs):
        """Make an API request with retries"""
        try:
            return await request_func(*args, **kwargs)
        except HttpError as e:
            if e.resp.status in [429, 500, 502, 503, 504]:
                raise  # Retry on these status codes
            raise  # Don't retry on other errors

    def _initialize_services(self):
        """Initialize Google API services"""
        try:
            credentials = self._get_credentials()
            self.sheets_service = build('sheets', 'v4', credentials=credentials)
            self.drive_service = build('drive', 'v3', credentials=credentials)
            logger.info("Successfully initialized Google services")
        except Exception as e:
            logger.error(f"Failed to initialize Google services: {str(e)}")
            # Don't fail initialization - we'll check for services before use
    
    def _get_credentials(self):
        """Get Google API credentials"""
        # Try service account first
        if self.service_account_path and os.path.exists(self.service_account_path):
            try:
                return service_account.Credentials.from_service_account_file(
                    self.service_account_path, scopes=SCOPES
                )
            except Exception as e:
                logger.error(f"Error loading service account credentials: {str(e)}")
        
        # Fall back to user OAuth if available
        token_path = "./token.json"  # Changed to current directory
        credentials_path = self.credentials_path
        
        credentials = None
        if os.path.exists(token_path):
            try:
                with open(token_path, 'r') as token_file:
                    credentials = Credentials.from_authorized_user_info(
                        json.load(token_file), SCOPES
                    )
            except Exception:
                logger.warning("Failed to load token, will attempt to create new one")
        
        # If no valid credentials available, prompt the user to log in
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    logger.info("Refreshed expired credentials")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    credentials = None
            
            if not credentials:
                if not os.path.exists(credentials_path):
                    logger.error(f"No credentials available at {credentials_path}. Please provide OAuth credentials.")
                    raise FileNotFoundError(f"No Google API credentials available at {credentials_path}")
                    
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                credentials = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(credentials.to_json())
                logger.info("Saved new credentials")
        
        return credentials

    # CLI Command Methods
    async def cli_list_sheets(self) -> None:
        """CLI command to list all Google Sheets"""
        try:
            files = await self.list_files()
            if not files:
                print("No Google Sheets found.")
                return
            
            print("\nAvailable Google Sheets:")
            print("-" * 60)
            for i, file in enumerate(files, 1):
                sheet_id = str(file.uri).replace("sheets://", "")  # Convert to string first
                print(f"{i}. {file.name}")
                print(f"   ID: {sheet_id}")
                if file.annotations:
                    print(f"   Modified: {file.annotations.modified_at}")
                print()
        except Exception as e:
            print(f"Error listing sheets: {e}")


    async def cli_read_sheet(self, sheet_id: str, range_name: str = "A1:Z1000") -> None:
        """CLI command to read data from a sheet"""
        try:
            data = await self.read_file(sheet_id, {"range": range_name, "headers": "true"})
            parsed_data = json.loads(data)
            
            if not parsed_data:
                print("No data found in the specified range.")
                return
            
            print(f"\nData from sheet {sheet_id} (range: {range_name}):")
            print("-" * 60)
            
            if isinstance(parsed_data, list) and parsed_data:
                if isinstance(parsed_data[0], dict):
                    # Data with headers
                    headers = list(parsed_data[0].keys())
                    print("\t".join(headers))
                    print("-" * 60)
                    for row in parsed_data[:10]:  # Show first 10 rows
                        print("\t".join(str(row.get(h, '')) for h in headers))
                    if len(parsed_data) > 10:
                        print(f"... and {len(parsed_data) - 10} more rows")
                else:
                    # Raw 2D array
                    for row in parsed_data[:10]:
                        print("\t".join(str(cell) for cell in row))
                    if len(parsed_data) > 10:
                        print(f"... and {len(parsed_data) - 10} more rows")
            else:
                print(json.dumps(parsed_data, indent=2))
                
        except Exception as e:
            print(f"Error reading sheet: {e}")

    async def cli_create_sheet(self, title: str) -> None:
        """CLI command to create a new sheet"""
        try:
            result = await self.create_sheet(title)
            result_data = json.loads(result)
            sheet_id = result_data.get('spreadsheetId')
            print(f"Successfully created sheet: {title}")
            print(f"Sheet ID: {sheet_id}")
            print(f"URL: https://docs.google.com/spreadsheets/d/{sheet_id}/edit")
        except Exception as e:
            print(f"Error creating sheet: {e}")

    async def cli_write_sheet(self, sheet_id: str, data_file: str, range_name: str = "A1") -> None:
        """CLI command to write data to a sheet from a JSON file"""
        try:
            if not os.path.exists(data_file):
                print(f"Data file {data_file} not found.")
                return
            
            with open(data_file, 'r') as f:
                data = f.read()
            
            # Construct file_id with range
            file_path = f"{sheet_id}/Sheet1/{range_name}"
            
            result = await self.write_file(file_path, data)
            result_data = json.loads(result)
            print(f"Successfully wrote data to sheet {sheet_id}")
            print(f"Updated {result_data.get('updated_cells', 0)} cells")
            print(f"Updated range: {result_data.get('updated_range', range_name)}")
        except Exception as e:
            print(f"Error writing to sheet: {e}")

    async def cli_search_sheets(self, query: str) -> None:
        """CLI command to search for sheets"""
        try:
            files = await self.search_files(query)
            if not files:
                print(f"No sheets found matching '{query}'.")
                return
            
            print(f"\nSheets matching '{query}':")
            print("-" * 60)
            for i, file in enumerate(files, 1):
                sheet_id = str(file.uri).replace("sheets://", "")  # Convert to string first
                print(f"{i}. {file.name}")
                print(f"   ID: {sheet_id}")
                if file.annotations:
                    print(f"   Modified: {file.annotations.modified_at}")
                print()
        except Exception as e:
            print(f"Error searching sheets: {e}")

    # Core MCP Methods
    async def list_files(
        self, 
        path: str = "", 
        page_size: int = 100,
        page_token: Optional[str] = None
    ) -> List[Resource]:
        """
        List Google Sheets as files
        
        Args:
            path: Path to list (ignored for Google Sheets)
            page_size: Number of files to return per page
            page_token: Token to use for pagination
            
        Returns:
            List of Resource objects
        """
        if not self.drive_service:
            self._initialize_services()
            if not self.drive_service:
                raise HTTPException(status_code=500, detail="Google Drive service unavailable")
                
        try:
            # Search for Google Sheets files
            results = self.drive_service.files().list(
                q="mimeType='application/vnd.google-apps.spreadsheet'",
                pageSize=page_size,
                pageToken=page_token,
                fields="files(id, name, createdTime, modifiedTime, owners)"
            ).execute()
            
            sheets = results.get('files', [])
            
            # Convert to MCP Resource objects
            files = []
            for sheet in sheets:
                files.append(Resource(
                    uri=f"sheets://{sheet.get('id')}",
                    name=sheet.get("name"),
                    mimeType="application/vnd.google-apps.spreadsheet",
                    size=None,  # Size not applicable for Google Sheets
                    annotations=Annotations(
                        modified_at=sheet.get("modifiedTime"),
                        created_at=sheet.get("createdTime")
                    )
                ))
            
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to list Google Sheets: {str(e)}")
    
    async def read_file(self, file_id: str, query: Dict[str, Any] = None) -> str:
        """
        Read Google Sheet data
        
        Args:
            file_id: Google Sheet ID or path
            query: Query parameters
            
        Returns:
            Sheet data as JSON string
        """
        if not self.sheets_service:
            self._initialize_services()
            if not self.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            # Parse file_id to extract sheet_id, sheet_name, and range
            if '/' in file_id:
                parts = file_id.strip('/').split('/')
                sheet_id = parts[0]
                sheet_name = parts[1] if len(parts) > 1 else ""
                cell_range = parts[2] if len(parts) > 2 else "A1"
            else:
                sheet_id = file_id
                sheet_name = ""
                cell_range = "A1"
                
            # Build range_name
            if sheet_name and cell_range:
                range_name = f"'{sheet_name}'!{cell_range}"
            else:
                range_name = cell_range
                
            # Process query parameters
            if query:
                # Override range if specified in query
                if 'range' in query:
                    range_name = query['range']
                    
                # Add sheet name to range if provided
                if sheet_name and 'range' in query and not query['range'].startswith(f"'{sheet_name}'!"):
                    range_name = f"'{sheet_name}'!{query['range']}"
                    
            # Set options
            value_render_option = query.get('valueRenderOption', 'FORMATTED_VALUE') if query else 'FORMATTED_VALUE'
            date_time_render_option = query.get('dateTimeRenderOption', 'FORMATTED_STRING') if query else 'FORMATTED_STRING'
            
            # Read from sheet
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=sheet_id,
                range=range_name,
                valueRenderOption=value_render_option,
                dateTimeRenderOption=date_time_render_option
            ).execute()
            
            values = result.get('values', [])
            
            # Process data
            if not values:
                return json.dumps([])
                
            # If the first row might be headers
            if query and query.get('headers', 'true').lower() in ('true', '1', 't'):
                if len(values) > 0:
                    headers = values[0]
                    data = []
                    
                    for row in values[1:]:
                        # Pad the row if it's shorter than headers
                        padded_row = row + [''] * (len(headers) - len(row))
                        data.append(dict(zip(headers, padded_row)))
                        
                    return json.dumps(data)
            
            # Return as 2D array
            return json.dumps(values)
            
        except HttpError as error:
            if error.resp.status == 404:
                raise McpError(f"Sheet with ID {sheet_id} not found")
            logger.error(f"Google API error: {str(error)}")
            raise HTTPException(status_code=error.resp.status, detail=str(error))
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    
    async def write_file(self, file_id: str, content: str) -> str:
        """
        Write data to a Google Sheet
        
        Args:
            file_id: Google Sheet ID or path
            content: Data to write (JSON string)
            
        Returns:
            Result of write operation
        """
        if not self.sheets_service:
            self._initialize_services()
            if not self.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            # Parse file_id to extract sheet_id, sheet_name, and range
            if '/' in file_id:
                parts = file_id.strip('/').split('/')
                sheet_id = parts[0]
                sheet_name = parts[1] if len(parts) > 1 else "Sheet1"
                cell_range = parts[2] if len(parts) > 2 else "A1"
            else:
                sheet_id = file_id
                sheet_name = "Sheet1"
                cell_range = "A1"
                
            # Build range_name
            if sheet_name and cell_range:
                range_name = f"'{sheet_name}'!{cell_range}"
            else:
                range_name = cell_range
            
            # Parse the content
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                data = [[content]]
            
            # Convert to 2D array
            if isinstance(data, dict):
                # Convert dict to array of key-value pairs
                values = [["Key", "Value"]] + [[k, str(v)] for k, v in data.items()]
            elif isinstance(data, list):
                if data and all(isinstance(item, dict) for item in data):
                    # Array of objects - extract keys from first object
                    keys = list(data[0].keys())
                    values = [keys] + [[str(item.get(k, '')) for k in keys] for item in data]
                elif data and all(isinstance(item, list) for item in data):
                    # Already a 2D array
                    values = [[str(cell) if cell is not None else '' for cell in row] for row in data]
                else:
                    # 1D array, make it a column
                    values = [[str(item)] for item in data]
            else:
                # Single value
                values = [[str(data)]]
            
            # Write to sheet
            result = self.sheets_service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body={"values": values}
            ).execute()
            
            return json.dumps({
                "updated_cells": result.get('updatedCells'),
                "updated_rows": result.get('updatedRows'),
                "updated_columns": result.get('updatedColumns'),
                "updated_range": result.get('updatedRange')
            })
            
        except HttpError as error:
            if error.resp.status == 404:
                raise McpError(f"Sheet with ID {sheet_id} not found")
            logger.error(f"Google API error: {str(error)}")
            raise HTTPException(status_code=error.resp.status, detail=str(error))
        except Exception as e:
            logger.error(f"Error writing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
    
    async def search_files(self, query: str) -> List[Resource]:
        """
        Search for Google Sheets
        
        Args:
            query: Search query string
            
        Returns:
            List of matching Resource objects
        """
        if not self.drive_service:
            self._initialize_services()
            if not self.drive_service:
                raise HTTPException(status_code=500, detail="Google Drive service unavailable")
                
        try:
            # Search for Google Sheets files by name
            q = f"mimeType='application/vnd.google-apps.spreadsheet' and name contains '{query}'"
            results = self.drive_service.files().list(
                q=q,
                pageSize=100,
                fields="files(id, name, createdTime, modifiedTime, owners)"
            ).execute()
            
            sheets = results.get('files', [])
            
            # Convert to MCP Resource objects
            files = []
            for sheet in sheets:
                files.append(Resource(
                    uri=f"sheets://{sheet.get('id')}",
                    name=sheet.get("name"),
                    mimeType="application/vnd.google-apps.spreadsheet",
                    size=None,  # Size not applicable for Google Sheets
                    annotations=Annotations(
                        modified_at=sheet.get("modifiedTime"),
                        created_at=sheet.get("createdTime")
                    )
                ))
            
            return files
        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to search files: {str(e)}")

    # MCP Tool Methods
    @staticmethod
    @mcp.tool(
        name="create_sheet",
        description="Create a new Google Sheet"
    )
    async def create_sheet(title: str) -> str:
        """Create a new Google Sheet"""
        instance = mcp._instance
        if not instance:
            raise HTTPException(status_code=500, detail="MCP instance not initialized")
            
        try:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Failed to initialize Google Sheets service")
                
            spreadsheet = {
                'properties': {
                    'title': title
                }
            }
            spreadsheet = instance.sheets_service.spreadsheets().create(
                body=spreadsheet,
                fields='spreadsheetId'
            ).execute()
            return json.dumps({"spreadsheetId": spreadsheet.get('spreadsheetId')})
        except Exception as e:
            logger.error(f"Error creating sheet: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @mcp.tool(
        name="format_range",
        description="Format a range of cells in a sheet"
    )
    async def format_range(file_id: str, range: str, format: dict) -> str:
        """Format a range of cells in a sheet"""
        instance = mcp._instance
        if not instance.sheets_service:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            requests = [{
                'repeatCell': {
                    'range': {'sheetId': 0, 'range': range},
                    'cell': {'userEnteredFormat': format},
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            }]
            instance.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=file_id,
                body={'requests': requests}
            ).execute()
            return json.dumps({"status": "success"})
        except Exception as e:
            logger.error(f"Error formatting range: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @mcp.tool(
        name="write_formula",
        description="Write a formula to a range of cells"
    )
    async def write_formula(file_id: str, range: str, formula: str) -> str:
        """Write a formula to a range of cells"""
        instance = mcp._instance
        if not instance.sheets_service:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            values = [[{'userEnteredValue': {'formulaValue': formula}}]]
            instance.sheets_service.spreadsheets().values().update(
                spreadsheetId=file_id,
                range=range,
                valueInputOption='USER_ENTERED',
                body={'values': values}
            ).execute()
            return json.dumps({"status": "success"})
        except Exception as e:
            logger.error(f"Error writing formula: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @mcp.tool(
        name="add_sheet",
        description="Add a new sheet to an existing spreadsheet"
    )
    async def add_sheet(file_id: str, title: str) -> str:
        """Add a new sheet to an existing spreadsheet"""
        instance = mcp._instance
        if not instance.sheets_service:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            requests = [{
                'addSheet': {
                    'properties': {
                        'title': title
                    }
                }
            }]
            instance.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=file_id,
                body={'requests': requests}
            ).execute()
            return json.dumps({"status": "success"})
        except Exception as e:
            logger.error(f"Error adding sheet: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @mcp.tool(
        name="delete_sheet",
        description="Delete a sheet from a spreadsheet"
    )
    async def delete_sheet(file_id: str, sheet_id: int) -> str:
        """Delete a sheet from a spreadsheet"""
        instance = mcp._instance
        if not instance.sheets_service:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            requests = [{
                'deleteSheet': {
                    'sheetId': sheet_id
                }
            }]
            instance.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=file_id,
                body={'requests': requests}
            ).execute()
            return json.dumps({"status": "success"})
        except Exception as e:
            logger.error(f"Error deleting sheet: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @mcp.tool(
        name="get_sheet_properties",
        description="Get properties of all sheets in a spreadsheet"
    )
    @lru_cache(maxsize=100)
    async def get_sheet_properties(file_id: str) -> str:
        """Get properties of all sheets in a spreadsheet"""
        instance = mcp._instance
        if not instance.sheets_service:
            instance._initialize_services()
            if not instance.sheets_service:
                raise HTTPException(status_code=500, detail="Google Sheets service unavailable")
                
        try:
            spreadsheet = instance.sheets_service.spreadsheets().get(
                spreadsheetId=file_id,
                fields='sheets.properties'
            ).execute()
            return json.dumps(spreadsheet.get('sheets', []))
        except Exception as e:
            logger.error(f"Error getting sheet properties: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Google Sheets MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--service-account", help="Path to service account JSON file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--credentials-path", default="./client_secret.json", help="Path to OAuth credentials file")
    parser.add_argument("--token-path", default="./token.json", help="Path to OAuth token file")
    
    # CLI commands
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--list-sheets", action="store_true", help="List all Google Sheets")
    parser.add_argument("--read-sheet", help="Read data from a sheet (provide sheet ID)")
    parser.add_argument("--create-sheet", help="Create a new sheet (provide title)")
    parser.add_argument("--write-sheet", help="Write data to a sheet (provide sheet ID)")
    parser.add_argument("--search-sheets", help="Search for sheets (provide search query)")
    parser.add_argument("--range", default="A1:Z1000", help="Range to read from sheet (default: A1:Z1000)")
    parser.add_argument("--data-file", help="JSON file containing data to write to sheet")
    
    return parser.parse_args()

async def run_cli_commands(args, mcp_server):
    """Run CLI commands"""
    if args.list_sheets:
        await mcp_server.cli_list_sheets()
    
    if args.read_sheet:
        await mcp_server.cli_read_sheet(args.read_sheet, args.range)
    
    if args.create_sheet:
        await mcp_server.cli_create_sheet(args.create_sheet)
    
    if args.write_sheet:
        if not args.data_file:
            print("Error: --data-file is required when using --write-sheet")
            return
        await mcp_server.cli_write_sheet(args.write_sheet, args.data_file, args.range)
    
    if args.search_sheets:
        await mcp_server.cli_search_sheets(args.search_sheets)

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create MCP server
    mcp_server = GoogleSheetsMCP(
        service_account_path=args.service_account,
        credentials_path=args.credentials_path
    )
    
    # Store the instance in the mcp object
    mcp._instance = mcp_server
    
    # Initialize services before starting
    try:
        mcp_server._initialize_services()
        if not mcp_server.sheets_service:
            logger.error("Failed to initialize Google Sheets service")
            sys.exit(1)
        logger.info("Successfully initialized Google Sheets service")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        sys.exit(1)
    
    # Check if running in CLI mode
    if args.cli or args.list_sheets or args.read_sheet or args.create_sheet:
        import asyncio
        asyncio.run(run_cli_commands(args, mcp_server))
        return
    
    # Start server using mcp.run()
    logger.info(f"Starting Google Sheets MCP Server on {args.host}:{args.port}")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()