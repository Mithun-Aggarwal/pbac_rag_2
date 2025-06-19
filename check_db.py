# check_db.py
import chromadb

# This path must match the 'vector_store' path in your config.yaml
DB_PATH = "/home/mit/Learning_and_growing/AI_DATA_EXTRACTION_AND_SEARCH_V1/Curated_information/documents/output_folder/4_vector_store"

try:
    # Connect to the persistent database
    client = chromadb.PersistentClient(path=DB_PATH)

    # Get the list of all collections in the database
    collections = client.list_collections()

    if not collections:
        print("Database is connected, but it contains NO collections.")
    else:
        print("Database is connected. Found the following collections:")
        for collection in collections:
            # Get the number of items in each collection
            count = collection.count()
            print(f"- Collection Name: '{collection.name}', Items: {count}")

except Exception as e:
    print(f"An error occurred: {e}")