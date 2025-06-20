import pytest
from unittest.mock import Mock, patch
import json
from google_sheets import GoogleSheetsMCP

@pytest.fixture
def mcp_server():
    return GoogleSheetsMCP(service_account_path="test_credentials.json")

@pytest.mark.asyncio
async def test_create_sheet(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets(), 'create') as mock_create:
        mock_create.return_value.execute.return_value = {'spreadsheetId': 'test123'}
        result = await mcp_server.create_sheet("Test Sheet")
        assert json.loads(result)['spreadsheetId'] == 'test123'

@pytest.mark.asyncio
async def test_format_range(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets(), 'batchUpdate') as mock_update:
        mock_update.return_value.execute.return_value = {}
        result = await mcp_server.format_range(
            "test123",
            "Sheet1!A1:B10",
            {"backgroundColor": {"red": 1, "green": 0, "blue": 0}}
        )
        assert json.loads(result)['status'] == 'success'

@pytest.mark.asyncio
async def test_write_formula(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets().values(), 'update') as mock_update:
        mock_update.return_value.execute.return_value = {}
        result = await mcp_server.write_formula(
            "test123",
            "Sheet1!A1",
            "=SUM(A1:A10)"
        )
        assert json.loads(result)['status'] == 'success'

@pytest.mark.asyncio
async def test_add_sheet(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets(), 'batchUpdate') as mock_update:
        mock_update.return_value.execute.return_value = {}
        result = await mcp_server.add_sheet("test123", "New Sheet")
        assert json.loads(result)['status'] == 'success'

@pytest.mark.asyncio
async def test_delete_sheet(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets(), 'batchUpdate') as mock_update:
        mock_update.return_value.execute.return_value = {}
        result = await mcp_server.delete_sheet("test123", 0)
        assert json.loads(result)['status'] == 'success'

@pytest.mark.asyncio
async def test_get_sheet_properties(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets(), 'get') as mock_get:
        mock_get.return_value.execute.return_value = {
            'sheets': [
                {'properties': {'title': 'Sheet1', 'sheetId': 0}},
                {'properties': {'title': 'Sheet2', 'sheetId': 1}}
            ]
        }
        result = await mcp_server.get_sheet_properties("test123")
        sheets = json.loads(result)
        assert len(sheets) == 2
        assert sheets[0]['properties']['title'] == 'Sheet1'
        assert sheets[1]['properties']['title'] == 'Sheet2'

@pytest.mark.asyncio
async def test_list_files(mcp_server):
    with patch.object(mcp_server.drive_service.files(), 'list') as mock_list:
        mock_list.return_value.execute.return_value = {
            'files': [
                {
                    'id': '123',
                    'name': 'Test Sheet',
                    'createdTime': '2024-03-19T00:00:00Z',
                    'modifiedTime': '2024-03-19T00:00:00Z',
                    'owners': [{'emailAddress': 'test@example.com'}]
                }
            ]
        }
        files = await mcp_server.list_files()
        assert len(files) == 1
        assert files[0].name == 'Test Sheet'
        assert files[0].id == '123'

@pytest.mark.asyncio
async def test_read_file(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets().values(), 'get') as mock_get:
        mock_get.return_value.execute.return_value = {
            'values': [
                ['Header1', 'Header2'],
                ['Value1', 'Value2']
            ]
        }
        result = await mcp_server.read_file("test123", {'headers': 'true'})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]['Header1'] == 'Value1'
        assert data[0]['Header2'] == 'Value2'

@pytest.mark.asyncio
async def test_write_file(mcp_server):
    with patch.object(mcp_server.sheets_service.spreadsheets().values(), 'update') as mock_update:
        mock_update.return_value.execute.return_value = {
            'updatedCells': 2,
            'updatedRows': 1,
            'updatedColumns': 2,
            'updatedRange': 'Sheet1!A1:B1'
        }
        result = await mcp_server.write_file(
            "test123",
            json.dumps([{'Header1': 'Value1', 'Header2': 'Value2'}])
        )
        data = json.loads(result)
        assert data['updatedCells'] == 2
        assert data['updatedRows'] == 1
        assert data['updatedColumns'] == 2 

print("All tests passed!")