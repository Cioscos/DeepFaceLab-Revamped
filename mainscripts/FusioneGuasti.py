"""I codici di guasto del servizio di fusione. Dati puri, nessun import:
stessa deroga e stessa ragione di DettaglioGuasti (gui/testi.py li legge
senza pagare torch). Il codice VIAGGIA nel dizionario d'errore accanto al
motivo; la GUI sceglie la frase dal codice, mai dal testo del motivo.

Come li', qui stanno SOLO i guasti che i dati o la macchina di chi usa il
programma possono davvero produrre: un errore di libreria imprevisto
resta senza codice di proposito, e la GUI lo mostra col suo motivo.
"""

# `--model-dir` non esiste, o nessun modello col nome chiesto sta li'.
MODELLO_ASSENTE = "modello_assente"
# `--aligned-dir` non esiste: senza allineati non c'e' niente da fondere.
ALLINEATI_ASSENTI = "allineati_assenti"
# `--input-dir` esiste ma non contiene immagini.
NESSUN_FRAME = "nessun_frame"
# Un comando con `op` che il servizio non conosce.
COMANDO_SCONOSCIUTO = "comando_sconosciuto"
# Un processo di compositing e' morto con un frame in mano. L'host lo SA --
# Subprocessor gli restituisce il dato -- ma non sa perche': il motivo
# grezzo resta nel campo `motivo`, e il frame torna in coda.
CLIENT_CADUTO = "client_caduto"
# Lo emette il CLIENT: il processo e' morto con una richiesta in sospeso.
SERVIZIO_INTERROTTO = "servizio_interrotto"

# I due insiemi restano distinti perche' le reti che li tengono onesti
# stanno da due lati diversi del tubo: un codice del servizio si prova
# facendo fallire il servizio per davvero, uno del client facendo tacere
# il trasporto.
CODICI_SERVIZIO = (MODELLO_ASSENTE, ALLINEATI_ASSENTI, NESSUN_FRAME,
                   COMANDO_SCONOSCIUTO, CLIENT_CADUTO)
CODICI_CLIENT = (SERVIZIO_INTERROTTO,)

# Niente codice per la sessione non corrispondente ne' per l'esaurimento
# di memoria: la prima viaggia nel campo `ripresa` dell'evento `pronto`,
# non in un `error`; la seconda non e' distinguibile da un'altra caduta del
# client (`on_data_return` vede il pool morire, non il perche'), quindi
# viaggia come CLIENT_CADUTO col motivo grezzo accanto.

CODICI = CODICI_SERVIZIO + CODICI_CLIENT
