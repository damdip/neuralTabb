from copyreg import pickle
import weaviate
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pandas as pd
from datetime import datetime
from weaviate.classes.config import Configure
from weaviate.classes.data import DataObject
import pickle 
import numpy as np

def create_weaviate_client():
    weaviate_client = weaviate.connect_to_local()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return weaviate_client


def checkExistingCollection(weaviate_client, collection_name):
    """
    Controlla se una collezione esiste in Weaviate
    """
    return weaviate_client.collections.exists(collection_name)



def resetSchema(weaviate_client , collection_name):
    """
    Resetta lo schema di una collezione in Weaviate
    """
    if weaviate_client.collections.exists(collection_name):
        weaviate_client.collections.delete(collection_name)



def closeConnection(weaviate_client):
    weaviate_client.close()


def createSchema(weaviate_client, collections_name, collection_properties):
    
    try:    
        weaviate_client.collections.create(
        name=collections_name,
        properties=collection_properties,
        vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
    except weaviate.exceptions.SchemaValidationException as e:
        print(f"An error occurred while creating the schema: {e}")

def retrieveElements(weaviate_client, collection_name, attribute, limit = 5):
    # Basic search (fetch objects)
    collection = weaviate_client.collections.get(collection_name)
    response = collection.query.fetch_objects(
        limit=limit,
        return_properties=[attribute]
    )

    for item in response.objects:
        print(item.properties)  # Print the raw text of each item

def createElementData(columns, row):
    """
    Crea un dizionario di dati per un elemento da inserire in Weaviate,
    convertendo i tipi numpy in tipi Python nativi e gestendo valori non JSON compliant.
    """
    element_data = {}
    for column in columns:
        value = row[column]
        
        # Gestisci valori NaN, None, inf
        if pd.isna(value) or value is None:
            element_data[column] = None
            continue
            
        # Gestisci valori infiniti
        if isinstance(value, (float, np.floating)) and (np.isinf(value)):
            element_data[column] = None  # o un valore di default appropriato
            continue
            
        # Converti tipi numpy in tipi Python nativi
        if isinstance(value, np.generic):
            value = value.item()
            
        # Controllo finale per float fuori range
        if isinstance(value, float):
            if not (-1.7976931348623157e+308 <= value <= 1.7976931348623157e+308):
                element_data[column] = None
                continue
                
        element_data[column] = value
        
    return element_data




def Key4Gemini():
        ### --- CONFIGURAZIONE --- ###
    try:
        with open("config/configLLM.txt", "r") as f:
            GEMINI_KEY = f.read().strip()
    except Exception as e:
        print(f"Errore durante l'apertura del file di config: {e}")
    genai.configure(api_key=GEMINI_KEY)


"""
Metodi per inserimento di elementi in weaviate

"""

def insertElement(weaviate_client, collection_name, element_data):
    """
    Inserisce un elemento in una collezione Weaviate
    """
    collection = weaviate_client.collections.get(collection_name)
    try:
        collection.data.insert(
            properties=element_data
        )
    except weaviate.exceptions.SchemaValidationException as e:
        print(f"An error occurred while inserting the element: {e}")


def insertManyElements(weaviate_client, collection_name, df):
    """
    Inserisce più elementi in una collezione Weaviate
    """
    properties_list = []
    for index, row in df.iterrows():
        element_data = createElementData(df.columns, row)
        DataObject(
            properties=element_data
        )
        properties_list.append(element_data)

    collection = weaviate_client.collections.get(collection_name)
    try:
        collection.data.insert_many(properties_list)
    except weaviate.exceptions.SchemaValidationException as e:
        print(f"An error occurred while inserting the elements: {e}")

def insertElementsWithDynamicBatchFix(client, collection_name, df):
    """
    Inserimento batch corretto per client v4.
    Usa il context manager di client.batch.configure(...).
    """
    try:
        with client.batch.dynamic() as batch:
            for _, row in df.iterrows():
                data = createElementData(df.columns, row)
                batch.add_object(
                    properties=data,
                    collection=collection_name,
                )
        print("Inserimento batch completato.")
    except Exception as e:
        print(f"An error occurred during batch insertion: {e}")

def insertElementsWithDynamicBatchFixProgressBar(client, collection_name, df):
    """
    Inserimento batch corretto per client v4 con progress tracking.
    Usa il context manager di client.batch.configure(...).
    """
    total_rows = len(df)
    processed_rows = 0
    batch_size = 100  # Dimensione batch per il progress
    
    print(f"Inizio inserimento di {total_rows} righe...")
    print("=" * 50)
    
    try:
        with client.batch.dynamic() as batch:
            for index, row in df.iterrows():
                data = createElementData(df.columns, row)
                batch.add_object(
                    properties=data,
                    collection=collection_name,
                )
                
                processed_rows += 1
                
                # Stampa progress ogni batch_size righe
                if processed_rows % batch_size == 0 or processed_rows == total_rows:
                    percentage = (processed_rows / total_rows) * 100
                    progress_bar = "█" * int(percentage // 2) + "░" * (50 - int(percentage // 2))
                    print(f"\r[{progress_bar}] {processed_rows}/{total_rows} righe ({percentage:.1f}%)", end="", flush=True)
                
                # Stampa dettagliata ogni 1000 righe
                if processed_rows % 1000 == 0:
                    print(f"\n✓ Processate {processed_rows}/{total_rows} righe...")
        
        print(f"\n{'='*50}")
        print(f"✅ Inserimento batch completato!")
        print(f"📊 Totale righe inserite: {processed_rows}/{total_rows}")
        
    except Exception as e:
        print(f"\n❌ Errore dopo {processed_rows} righe processate")
        print(f"An error occurred during batch insertion: {e}")
        print(f"Righe completate prima dell'errore: {processed_rows}/{total_rows}")

def get_df_size_mb_pickle(df):
    """Calcola dimensione reale usando pickle"""
    return len(pickle.dumps(df)) / (1024 * 1024)

def extractChunksAndInsertIntoWeaviate(weaviate_client, collection_name, df):
    num_partitions = int(get_df_size_mb_pickle(df)//100 + 1)
    total_length = len(df)

    chunk_size = total_length // num_partitions
    for i in range(0, num_partitions):
        start = i * chunk_size

        if( i == num_partitions - 1):
            end = total_length
        else:
            end = (i + 1) * chunk_size
        
        df_chunked = df.iloc[start:end]
    
        insertManyElements(weaviate_client, collection_name, df_chunked)
                  
def extractChunksAndInsertIntoWeaviateProgressBar(weaviate_client, collection_name, df):
    num_partitions = int((get_df_size_mb_pickle(df) // 0.07) + 1)
    total_length = len(df)
    chunk_size = total_length // num_partitions

    print(f"Inizio inserimento in {num_partitions} chunk, {total_length} righe totali.")

    for i in range(num_partitions):
        start = i * chunk_size
        if i == num_partitions - 1:
            end = total_length
        else:
            end = (i + 1) * chunk_size

        df_chunked = df.iloc[start:end]

        print(f"Inserimento di {df_chunked.shape[0]} righe (righe {start} a {end})...")
        insertManyElements(weaviate_client, collection_name, df_chunked)

        progress = (i + 1) / num_partitions * 100
        print(f"Chunk {i + 1}/{num_partitions} inserito ({progress:.1f}%).")