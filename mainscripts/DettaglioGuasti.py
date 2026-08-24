"""I codici di guasto del servizio di dettaglio.

Il dizionario d'errore del protocollo porta un `codice` accanto al
`motivo`. Il motivo e' una stringa d'implementazione -- in italiano, come
tutto il resto del pacchetto -- e non e' un testo da mostrare a chi
guarda: la GUI mappa il codice a una frase inglese sua (gui/testi.py) e
tiene il motivo come dettaglio tecnico.

Il codice VIAGGIA, non si indovina. Riconoscere un guasto confrontando il
testo italiano del motivo si romperebbe in silenzio alla prima
riformulazione del messaggio, cioe' peggio di non riconoscerlo affatto.

Dati puri, e deve restare tale: stesso ruolo e stessa ragione di
MotoriCatalog -- l'interfaccia grafica e la sua suite leggera importano
questo modulo, e non devono pagare torch per leggere una tabella. Import
consentiti: nessuno.

Qui stanno SOLO i guasti che i dati o la macchina di chi usa il programma
possono davvero produrre. Una violazione del protocollo o un errore di
libreria imprevisto restano senza codice di proposito: la GUI ha un
ripiego generico che li mostra comunque col loro motivo, e un codice
inventato per un caso ipotetico prometterebbe una diagnosi che nessuno ha
mai visto accadere.

Gli ultimi due non li emette il servizio: li emette il CLIENT, e sono qui
lo stesso perche' chi mostra il guasto non distingue -- e non deve
distinguere -- da quale lato del tubo e' arrivato.
"""

# `DFLJPG.load` torna None: il file manca, e' troncato, o non e' affatto
# un JPEG. NON basta che gli manchino i metadati DFL -- un JPEG qualunque
# si carica benissimo, con un dizionario vuoto -- quindi la frase da dire
# a chi guarda parla del file illeggibile, non dei metadati. Prima si
# accusava il file di non portare dati DFL per QUALUNQUE guasto, su volti
# sani: la reazione naturale a quella frase e' cancellare il volto.
FILE_ILLEGGIBILE = "file_illeggibile"

# Il fotogramma di partenza non e' nella cartella dei frame: il volto non
# si puo' ritagliare di nuovo, e senza di lui non c'e' niente su cui
# rifare il rilevamento.
FRAME_ASSENTE = "frame_assente"

# Il volto non porta la matrice fotogramma -> allineato, quindi le
# modifiche non hanno un verso per tornare sul fotogramma.
SENZA_MATRICE = "senza_matrice"

# Il volto non porta il rettangolo del rilevamento, che e' cio' su cui
# l'allineatore andrebbe fatto girare.
SENZA_RETTANGOLO = "senza_rettangolo"

# I landmark proposti non producono un allineamento: coincidenti, o
# comunque tali da far tornare una matrice non finita.
ALLINEAMENTO_NON_VALIDO = "allineamento_non_valido"

# Il servizio non ha risposto entro il tempo concesso. Non e' un guasto
# del volto: e' una macchina carica, o un servizio morto per inattivita'
# che la richiesta successiva riavviera' da se'.
SERVIZIO_MUTO = "servizio_muto"

# E' arrivata la risposta di una richiesta PRECEDENTE, rimasta indietro
# dopo un tempo scaduto. Il client la scarta invece di consegnarla al
# posto di quella corrente -- consegnarla sposterebbe lo sfasamento sulla
# richiesta dopo, e non si riassorbirebbe mai. Il motivo grezzo porta i
# due numeri di sequenza, che a chi guarda non dicono niente.
RISPOSTA_FUORI_SEQUENZA = "risposta_fuori_sequenza"

# I due insiemi restano distinti perche' le reti che li tengono onesti
# sono due e stanno in suite diverse: un codice del servizio si prova
# facendo fallire `FacesetDetail.rispondi` per davvero, uno del client
# facendo tacere il trasporto. Chi MOSTRA il guasto non li distingue --
# per lui c'e' solo CODICI -- ma chi aggiunge un codice deve dire da quale
# lato del tubo nasce, o non avra' nessuna rete addosso.
CODICI_SERVIZIO = (FILE_ILLEGGIBILE, FRAME_ASSENTE, SENZA_MATRICE,
                   SENZA_RETTANGOLO, ALLINEAMENTO_NON_VALIDO)
CODICI_CLIENT = (SERVIZIO_MUTO, RISPOSTA_FUORI_SEQUENZA)

CODICI = CODICI_SERVIZIO + CODICI_CLIENT
