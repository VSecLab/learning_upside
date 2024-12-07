# learningVR

# Riconoscitore utenti

Questo progetto consiste in un'applicazione web per l'analisi e la predizione dei dati biomeccanici di utenti che utilizzano il dispositivo Meta Quest, focalizzata sull'uso di modelli di machine learning, in particolare reti neurali ricorrenti (RNN) e reti neurali LSTM Autoencoder (Long Short-Term Memory).
L'obiettivo è un riconoscitore che distingue gli utenti in base ai loro movimenti biomeccanici

## Descrizione dei file

- **global_ml.py**: Questo file contiene le principali funzioni per l'applicazione come model_and_evaluate e check per la valutazione e verifica dei risultati.
- **models.py**: Questo file contiene la definizione dei modelli di rete neurale utilizzati per l'analisi e la predizione dei dati.
- **VisorData.py**: Questo file contiene funzioni utili per l'accesso e la manipolazione dei dati utilizzati nell'applicazione.
- **Flask_app.py**:Questo è il file principale dell'applicazione Flask. Gestisce le richieste HTTP e gestendo le diverse funzioni ML.


## Guida all'installazione

1. Assicurarsi di avere Python installato nel proprio sistema. È consigliabile utilizzare Python 3.11.4 o versioni successive.
2. Clonare il repository dal repository remoto:

    ```
    git clone <url_del_repository>
    ```

3. Passare alla directory del progetto:

    ```
    cd nome_del_progetto
    ```

4. Creare un ambiente virtuale (opzionale ma consigliato):

    ```
    python3 -m venv nome_ambiente_virtuale
    ```

5. Attivare l'ambiente virtuale:

    - Su Windows:

    ```
    nome_ambiente_virtuale\Scripts\activate
    ```

    - Su macOS e Linux:

    ```
    source nome_ambiente_virtuale/bin/activate
    ```

6. Installare le dipendenze del progetto:

    ```
    pip install -r requirements.txt
    ```

7. Avviare l'applicazione:

    ```
    python global_ml.py
    ```

8. Aprire un browser web e visitare `http://127.0.0.1:5000` per accedere all'applicazione.

## Guida all'utilizzo

L'applicazione offre tre principali funzionalità:

1. **Model and Evaluate**: Questa funzione permette di addestrare un modello di rete neurale e valutarlo utilizzando i dati forniti.
2. **Check**: Questa funzione permette di eseguire una verifica utilizzando un modello di rete neurale pre-addestrato e di valutare la sua precisione e recall e matrici di confusione.
3. **Run Experiments**: Questa funzione permette di eseguire una serie di esperimenti utilizzando diversi parametri e modelli per valutare le prestazioni complessive del sistema.














