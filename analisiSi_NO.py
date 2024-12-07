import os
import pandas as pd

def analizza_risultati(cartella_risultati, nome_interessato):
    # Lista per salvare i risultati di ogni CSV
    risultati = []

    # Cicla attraverso tutti i file nella cartella risultati
    for filename in os.listdir(cartella_risultati):
        if filename.endswith(".csv"):
            filepath = os.path.join(cartella_risultati, filename)

            # Leggi il CSV
            df = pd.read_csv(filepath)

            # Utilizza groupby per ottenere i conteggi desiderati
            conteggi = df.groupby(['user', 'check']).size().unstack(fill_value=0)

            # Conta quanti "Si" ha ricevuto il nome interessato
            si_nome_interessato = conteggi.loc[nome_interessato, 'si']

            # Conta quanti "Si" hanno ricevuto tutti gli altri
            si_altri = conteggi.drop(nome_interessato, errors='ignore')['si'].sum()

            # Salva i risultati per il CSV corrente
            risultati.append({'filename': filename, f'si_{nome_interessato}': si_nome_interessato, 'si_altri': si_altri})

    # Ordina i risultati in base al criterio definito
    risultati_ordinati = sorted(risultati, key=lambda x: x[f'si_{nome_interessato}'] - x['si_altri'], reverse=True)

    # Seleziona i primi 3 risultati
    primi_3 = risultati_ordinati[:3]

    # Stampa i risultati
    print("I 3 migliori CSV:")
    for i, result in enumerate(primi_3, 1):
        print(f"{i}. {result['filename']}: Si {nome_interessato} = {result[f'si_{nome_interessato}']}, Si Altri = {result['si_altri']}")

# Sostituisci 'percorso/risultati' con il percorso effettivo della tua cartella risultati
# Sostituisci 'NomeInteressato' con il nome specifico che vuoi cercare
analizza_risultati('RNN2E_8Blabel3/0','ConteTeresa')


#ConteTeresa AlessandroMercurio MassimilianoRak AntonioFeretti FrancescoEsposito
#RNN2E_8B