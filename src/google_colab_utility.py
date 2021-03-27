import sys

def connect_gdrive():
    try:
        from google.colab import drive
        drive.mount("/gdrive")
    except:
        print("You dont seem to be on google colab.")

if __name__ == "__main__":
    globals()[sys.argv[1]]()