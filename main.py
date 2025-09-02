import pandas as pd
from python_web_app.pandasToWeaviate import interactive_weaviate_types, get_properties_from_map,deterministic_weaviate_types
from python_web_app.weaviateMain import create_weaviate_client, checkExistingCollection, extractChunksAndInsertIntoWeaviate, extractChunksAndInsertIntoWeaviateProgressBar, insertElementsWithDynamicBatch, insertElementsWithDynamicBatchFix, insertElementsWithDynamicBatchFixProgressBar, insertManyElements, resetSchema,closeConnection, createSchema, insertElement, createElementData, retrieveElements
path_book_data = "dataset/readyToUse/book_data_processed_and_cleaned_small.xlsx"

##path_book_reviews = "dataset/readyToUse/book_reviews_sampled.xlsx"

book_data = pd.read_excel(path_book_data)
# Carica il file CSV
df = pd.read_excel(path_book_data)
# Mostra le prime righe del DataFrame
print(df.head())

#Inizializzazione weaviate:
try:
    client = create_weaviate_client()
    print("Connessione a Weaviate riuscita.")
    
except Exception as e:
    print(f"Errore di connessione a Weaviate: {e}")
    exit(1)

#resetSchema(client, "book_data")

if(not checkExistingCollection(client, "book_data")):
    #Crea mappa a partire dai dati del dataframe
    mappings, type_map = deterministic_weaviate_types(df)
    #Crea properties da passare a weaviate per la creazione della classe
    properties = get_properties_from_map(type_map)
    print("Creazione della collezione 'book_data'")
    createSchema(client, "book_data", properties)
    print("La collezione 'book_data' è stata creata correttamente.")
else:
    print("La collezione 'book_data' esiste già.")




try:
    print("Inserimento degli elementi nella collezione 'book_data'")
    #extractChunksAndInsertIntoWeaviateProgressBar(client, "book_data", df)
    extractChunksAndInsertIntoWeaviateProgressBar(client, "book_data", df)
    print("Elementi inseriti correttamente:")
    retrieveElements(client, "book_data", "title", 3)
except Exception as e:
    print(f"Errore durante l'inserimento degli elementi: {e}")
finally:
    closeConnection(client)
<<<<<<< Updated upstream
=======



    
>>>>>>> Stashed changes


