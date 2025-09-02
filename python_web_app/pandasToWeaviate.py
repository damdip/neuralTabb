import pandas as pd
import numpy as np
import re
import weaviate.classes.config as wc



def interactive_weaviate_types(df, df_name="DataFrame"):
    """
    Suggerisce i tipi Weaviate con conferma interattiva dell'utente
    Restituisce: (mappings_completi, mappa_semplice)
    """
    print(f"\n🎯 MAPPATURA INTERATTIVA TIPI WEAVIATE - {df_name}")
    print("=" * 70)

    
    # Lista completa dei tipi Weaviate disponibili (in MAIUSCOLO)
    available_types = [
        'TEXT', 'STRING', 'INT', 'NUMBER', 'BOOLEAN', 
        'DATE', 'UUID', 'GEOCOORDINATES', 'PHONENUMBER', 'BLOB',
        'TEXT[]', 'STRING[]', 'INT[]', 'NUMBER[]', 'BOOLEAN[]', 
        'DATE[]', 'UUID[]', 'OBJECT', 'OBJECT[]'
    ]
    
    confirmed_mappings = []
    
    print("📋 Per ogni colonna, conferma il tipo predetto o scegline uno diverso.\n")
    
    for i, col in enumerate(df.columns, 1):
        print(f"\n{'='*60}")
        print(f"📋 COLONNA {i}/{len(df.columns)}: '{col}'")
        print(f"{'='*60}")
        
        pandas_type = str(df[col].dtype)
        
        # Predici il tipo automaticamente
        predicted_type, reason = predict_weaviate_type(df[col], pandas_type)
        
        # Mostra info sulla colonna
        print(f"🔍 Tipo Pandas: {pandas_type}")
        print(f"📊 Valori non-null: {df[col].count():,}/{len(df):,}")
        
        # Mostra esempi
        sample_values = df[col].dropna().head(3).tolist()
        if sample_values:
            print(f"📝 Esempi: {sample_values}")
        
        print(f"\n💡 Tipo predetto: '{predicted_type}'")
        print(f"   Motivo: {reason}")
        
        # Chiedi conferma
        while True:
            print(f"\n❓ Confermi il tipo '{predicted_type}' per la colonna '{col}'?")
            choice = input("   Digita 'y' per sì, 'n' per no, 'info' per vedere tutti i tipi: ").lower().strip()
            
            if choice in ['y', 'yes', 'si', 's']:
                final_type = predicted_type
                print(f"   ✅ Confermato: {col} → {final_type}")
                break
                
            elif choice in ['n', 'no']:
                print(f"\n🔧 Scegli un tipo diverso per '{col}':")
                print("   Tipi disponibili:")
                
                # Mostra tipi in colonne per leggibilità
                for j, wtype in enumerate(available_types):
                    print(f"   {j+1:2d}. {wtype:15}", end="")
                    if (j + 1) % 3 == 0:  # 3 colonne
                        print()
                if len(available_types) % 3 != 0:
                    print()
                
                while True:
                    try:
                        type_choice = input("\n   Inserisci il numero del tipo (1-{}): ".format(len(available_types)))
                        type_index = int(type_choice) - 1
                        
                        if 0 <= type_index < len(available_types):
                            final_type = available_types[type_index]
                            print(f"   ✅ Scelto: {col} → {final_type}")
                            break
                        else:
                            print(f"   ❌ Numero non valido. Inserisci un numero tra 1 e {len(available_types)}")
                    except ValueError:
                        print("   ❌ Inserisci un numero valido")
                break
                
            elif choice == 'info':
                print(f"\n📚 TIPI WEAVIATE DISPONIBILI:")
                print("   🔤 Tipi Base:")
                print("      • text: Testo lungo, indicizzato per ricerca")
                print("      • string: Stringhe brevi, non tokenizzate")
                print("      • int: Numeri interi")
                print("      • number: Numeri decimali")
                print("      • boolean: true/false")
                print("      • date: Date ISO 8601")
                print("      • uuid: Identificatori unici")
                print("      • geoCoordinates: Coordinate geografiche")
                print("      • phoneNumber: Numeri telefono")
                print("      • blob: Dati binari")
                print("   📦 Tipi Array: Aggiungi [] a qualsiasi tipo base")
                print("   🔗 Tipi Oggetto: object, object[]")
                continue
            else:
                print("   ❌ Risposta non valida. Digita 'y', 'n' o 'info'")
        
        confirmed_mappings.append({
            'column': col,
            'original_name': col,
            'pandas_type': pandas_type,
            'weaviate_type': final_type,
            'predicted_type': predicted_type,
            'user_confirmed': final_type == predicted_type
        })
    
    print(f"\n🎉 MAPPATURA COMPLETATA!")
    print("=" * 50)
    
    # Riepilogo finale
    for mapping in confirmed_mappings:
        status = "✅ Confermato" if mapping['user_confirmed'] else "🔧 Modificato"
        print(f"{mapping['column']:25} → {mapping['weaviate_type']:15} {status}")
    
    # Crea e stampa la mappa semplice
    type_map = create_type_mapping(confirmed_mappings)
    print(f"\n📋 MAPPA FINALE - {df_name}")
    print("=" * 50)
    
    for field_name, weaviate_type in type_map.items():
        print(f"'{field_name}' → '{weaviate_type}'")
    
    print(f"\n💾 Mappa Python:")
    print("type_map = {")
    for field_name, weaviate_type in type_map.items():
        print(f"    '{field_name}': '{weaviate_type}',")
    print("}")
    
    return confirmed_mappings, type_map

def deterministic_weaviate_types(df, df_name="DataFrame"):
    """
    Assegna automaticamente i tipi Weaviate senza interazione utente
    Restituisce: (mappings_completi, mappa_semplice)
    """
    print(f"\n🤖 MAPPATURA AUTOMATICA TIPI WEAVIATE - {df_name}")
    print("=" * 70)
    
    confirmed_mappings = []
    
    print("📋 Analizzando automaticamente ogni colonna...\n")
    
    for i, col in enumerate(df.columns, 1):
        print(f"📋 COLONNA {i}/{len(df.columns)}: '{col}'")
        
        pandas_type = str(df[col].dtype)
        
        # Predici il tipo automaticamente
        predicted_type, reason = predict_weaviate_type(df[col], pandas_type)
        
        # Mostra info della decisione
        print(f"   🔍 Tipo Pandas: {pandas_type}")
        print(f"   💡 Tipo assegnato: '{predicted_type}' - {reason}")
        
        # Mostra esempi per verifica visiva
        sample_values = df[col].dropna().head(2).tolist()
        if sample_values:
            print(f"   📝 Esempi: {sample_values}")
        
        confirmed_mappings.append({
            'column': col,
            'original_name': col,
            'pandas_type': pandas_type,
            'weaviate_type': predicted_type,
            'predicted_type': predicted_type,
            'user_confirmed': True,  # Automaticamente confermato
            'reason': reason
        })
        print()
    
    print(f"🎉 MAPPATURA AUTOMATICA COMPLETATA!")
    print("=" * 50)
    
    # Riepilogo finale
    for mapping in confirmed_mappings:
        print(f"{mapping['column']:25} → {mapping['weaviate_type']:15} ✅ Auto")
    
    # Crea e stampa la mappa semplice
    type_map = create_type_mapping(confirmed_mappings)
    print(f"\n📋 MAPPA FINALE - {df_name}")
    print("=" * 50)
    
    for field_name, weaviate_type in type_map.items():
        print(f"'{field_name}' → '{weaviate_type}'")
    
    print(f"\n💾 Mappa Python:")
    print("type_map = {")
    for field_name, weaviate_type in type_map.items():
        print(f"    '{field_name}': '{weaviate_type}',")
    print("}")
    
    return confirmed_mappings, type_map

def predict_weaviate_type(series, pandas_type):
    """
    Predice il tipo Weaviate e restituisce anche il motivo
    Include rilevamento di tipi array
    """
    if pandas_type == 'object':
        non_null_values = series.dropna()
        
        if len(non_null_values) == 0:
            return 'TEXT', "Colonna vuota - default text"
        
        sample_values = non_null_values.tolist()
        
        # ========== CONTROLLO ARRAY ==========
        # Verifica se i valori sono liste/array
        list_samples = [v for v in sample_values[:20] if isinstance(v, (list, tuple))]
        
        if len(list_samples) > len(sample_values) * 0.5:  # Almeno 50% sono array
            print(f"   🔍 Rilevato tipo array - analizzando contenuto...")
            
            # Analizza il contenuto degli array
            all_elements = []
            for lst in list_samples[:10]:  # Prendi primi 10 array
                if isinstance(lst, (list, tuple)) and len(lst) > 0:
                    all_elements.extend(lst)
            
            if all_elements:
                # Converti elementi in stringa per analisi
                str_elements = [str(elem) for elem in all_elements[:50]]
                
                # Controllo elementi numerici
                try:
                    numeric_elements = [float(elem) for elem in str_elements if str(elem).replace('.', '').replace('-', '').isdigit()]
                    if len(numeric_elements) > len(str_elements) * 0.8:
                        # Controlla se sono interi
                        if all(float(elem).is_integer() for elem in numeric_elements):
                            return 'INT[]', f"Array di interi (esempi: {list_samples[:3]})"
                        else:
                            return 'NUMBER[]', f"Array di numeri (esempi: {list_samples[:3]})"
                except:
                    pass
                
                # Controllo elementi booleani
                bool_elements = [elem for elem in str_elements if str(elem).lower() in ['true', 'false', '1', '0', 'yes', 'no']]
                if len(bool_elements) > len(str_elements) * 0.8:
                    return 'BOOLEAN[]', f"Array di booleani (esempi: {list_samples[:3]})"
                
                # Controllo date
                date_elements = [elem for elem in str_elements if re.match(r'^\d{4}-\d{2}-\d{2}', str(elem))]
                if len(date_elements) > len(str_elements) * 0.8:
                    return 'DATE[]', f"Array di date (esempi: {list_samples[:3]})"
                
                # Controllo UUID
                uuid_elements = [elem for elem in str_elements if len(str(elem)) == 36 and str(elem).count('-') == 4]
                if len(uuid_elements) > len(str_elements) * 0.8:
                    return 'UUID[]', f"Array di UUID (esempi: {list_samples[:3]})"
                
                # Controllo lunghezza stringhe per TEXT[] vs STRING[]
                avg_length = sum(len(str(elem)) for elem in str_elements) / len(str_elements)
                if avg_length > 50:
                    return 'TEXT[]', f"Array di testo lungo (media {avg_length:.0f} char, esempi: {list_samples[:3]})"
                else:
                    return 'STRING[]', f"Array di stringhe (media {avg_length:.0f} char, esempi: {list_samples[:3]})"
            else:
                return 'STRING[]', f"Array vuoti - default string[] (esempi: {list_samples[:3]})"
        
        # ========== CONTROLLO TIPI SINGOLI ==========
        # Se non è array, procedi con controlli normali
        sample_strings = [str(v) for v in sample_values]
        
        # UUID pattern
        if any(len(str(v)) == 36 and str(v).count('-') == 4 for v in sample_strings[:10]):
            return 'UUID', "Pattern UUID rilevato"
        
        # Phone pattern  
        elif any(re.match(r'^[\+]?[1-9]?[0-9]{7,15}$', str(v)) for v in sample_strings[:10]):
            return 'PHONENUMBER', "Pattern telefono rilevato"
        
        # Date pattern
        elif any(re.match(r'^\d{4}-\d{2}-\d{2}', str(v)) for v in sample_strings[:10]):
            return 'DATE', "Pattern data ISO rilevato"
        
        # Boolean pattern
        elif all(str(v).lower() in ['true', 'false', 'yes', 'no', '1', '0'] for v in sample_strings[:20]):
            return 'BOOLEAN', "Pattern booleano rilevato"
        
        # Categorico con pochi valori
        elif series.nunique() / len(series) < 0.05:
            return 'STRING', f"Categorico ({series.nunique()} valori unici)"
        
        # Testo lungo
        elif np.mean([len(str(v)) for v in sample_strings]) > 50:
            return 'TEXT', f"Testo lungo (media {np.mean([len(str(v)) for v in sample_strings]):.0f} caratteri)"
        
        # Stringhe brevi
        else:
            return 'STRING', "Stringhe brevi"
    
    # Controllo per interi mascherati da float
    elif pandas_type in ['float64', 'float32']:
        if series.dropna().apply(lambda x: x.is_integer()).all():
            return 'INT', "Float che sono tutti interi"
        else:
            return 'NUMBER', f"Numeri decimali ({pandas_type})"
    
    elif pandas_type in ['int64', 'int32']:
        return 'INT', f"Numeri interi ({pandas_type})"
    
    elif pandas_type == 'bool':
        return 'BOOLEAN', "Tipo booleano"
    
    elif 'datetime' in pandas_type:
        return 'DATE', f"Tipo data ({pandas_type})"
    
    else:
        return 'TEXT', f"Mapping default da {pandas_type}"
    
def create_type_mapping(mappings):
    """
    Crea una mappa semplice: nome_campo → tipo_weaviate
    """
    type_map = {}
    for mapping in mappings:
        type_map[mapping['column']] = mapping['weaviate_type']
    return type_map

def get_properties_from_map(type_map):
    """
    Crea una lista di proprietà Weaviate da una mappa nome->tipo
    
    Args:
        type_map (dict): Mappa con chiave=nome_campo, valore=tipo_weaviate
                        Es: {'title': 'TEXT', 'price': 'NUMBER', 'authors': 'STRING[]'}
    
    Returns:
        list: Lista di wc.Property per Weaviate
    """
    properties = []
    
    # Mappatura tipi stringa -> DataType Weaviate
    data_type_mapping = {
        'TEXT': wc.DataType.TEXT,
        'STRING': wc.DataType.TEXT,  # In Weaviate v4 STRING è deprecato, usa TEXT
        'INT': wc.DataType.INT,
        'NUMBER': wc.DataType.NUMBER,
        'BOOLEAN': wc.DataType.BOOL,
        'DATE': wc.DataType.DATE,
        'UUID': wc.DataType.UUID,
        'GEOCOORDINATES': wc.DataType.GEO_COORDINATES,
        'PHONENUMBER': wc.DataType.PHONE_NUMBER,
        'BLOB': wc.DataType.BLOB,
        'TEXT[]': wc.DataType.TEXT_ARRAY,
        'STRING[]': wc.DataType.TEXT_ARRAY,  # STRING[] -> TEXT_ARRAY
        'INT[]': wc.DataType.INT_ARRAY,
        'NUMBER[]': wc.DataType.NUMBER_ARRAY,
        'BOOLEAN[]': wc.DataType.BOOL_ARRAY,
        'DATE[]': wc.DataType.DATE_ARRAY,
        'UUID[]': wc.DataType.UUID_ARRAY,
        'OBJECT': wc.DataType.OBJECT,
        'OBJECT[]': wc.DataType.OBJECT_ARRAY
    }
    
    for field_name, field_type in type_map.items():
        # Converti il tipo in maiuscolo per sicurezza
        field_type_upper = field_type.upper()
        
        if field_type_upper in data_type_mapping:
            weaviate_data_type = data_type_mapping[field_type_upper]
            properties.append(
                wc.Property(name=field_name, data_type=weaviate_data_type)
            )
            print(f"✅ Aggiunta proprietà: {field_name} → {field_type_upper}")
        else:
            print(f"⚠️  Tipo non riconosciuto per '{field_name}': {field_type}. Usando TEXT come default.")
            properties.append(
                wc.Property(name=field_name, data_type=wc.DataType.TEXT)
            )
    
    return properties




# Proteggi l'esecuzione quando il file è importato
if __name__ == "__main__":
    # Codice di test o esempio che gira solo quando esegui direttamente questo file
    print("Modulo pandasToWeaviate caricato correttamente!")
    # Esempio di utilizzo (opzionale)
    df = pd.read_csv("test.csv")
    mappings, type_map = interactive_weaviate_types(df)