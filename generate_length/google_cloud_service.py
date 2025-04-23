from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# TODO: insert credentials here
# Path to your service account credentials
SERVICE_ACCOUNT_FILE = "path/toyour/credentials/json"

# Scopes needed for Drive API access
SCOPES = ['https://www.googleapis.com/auth/drive']

def list_spreadsheets():
    try:
        # Load service account credentialspytho
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Build the Drive API client
        drive_service = build('drive', 'v3', credentials=creds)

        # Query to list all Google Sheets files (MIME type for Sheets is 'application/vnd.google-apps.spreadsheet')
        query = "mimeType='application/vnd.google-apps.spreadsheet'"

        # List the files
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        if not files:
            print("No spreadsheets found.")
        else:
            print("Spreadsheets owned by the service account:")
            for file in files:
                print(f"ID: {file['id']} - Name: {file['name']}")

    except HttpError as error:
        print(f"An error occurred: {error}")


def delete_spreadsheets():
    try:
        # Load service account credentials
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Build the Drive API client
        drive_service = build('drive', 'v3', credentials=creds)

        # Query to list all Google Sheets files (MIME type for Sheets is 'application/vnd.google-apps.spreadsheet')
        query = "mimeType='application/vnd.google-apps.spreadsheet'"

        # List the files
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        if not files:
            print("No spreadsheets found.")
        else:
            print("Deleting the following spreadsheets:")
            for file in files:
                print(f"Deleting: {file['name']} (ID: {file['id']})")
                # Delete the spreadsheet
                drive_service.files().delete(fileId=file['id']).execute()
            print("Deletion completed.")

    except HttpError as error:
        print(f"An error occurred: {error}")
