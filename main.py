import pandas as pd







# Specifica il percorso del file Excel
file_path = "dataset\Online Retail.xlsx"

# Carica il file Excel
df = pd.read_excel(file_path)

# Stampa tutte le righe
print(df.head())


for row in df.iterrows():
    # Stampa il numero della riga e il contenuto della riga
    print(f"Row {row[0]}: {row[1]}")

