from src.utility import get_project_dir
import os

def connect_gdrive():
    try:
        from google.colab import drive
        drive.mount("/gdrive")
    except:
        print("You dont seem to be on google colab.")