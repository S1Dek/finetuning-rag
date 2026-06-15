import time
from ingest import ingest_folder

def watch():
    db = ingest_folder()

    print("Watching folder...")

    while True:
        time.sleep(10)
        db = ingest_folder()