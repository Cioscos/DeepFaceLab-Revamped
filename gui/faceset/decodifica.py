"""Decodifica delle celle, al volo e fuori dal thread dell'interfaccia.

Nessuna miniatura su disco: un volto allineato e' gia' una miniatura, e la
lettura scalata di Qt (come i IMREAD_REDUCED_* di OpenCV) sfrutta la
scalatura DCT di libjpeg -- 3,64 ms per volto misurati sul percorso lento.
Con otto thread la griglia decodifica piu' in fretta di quanto si possa
scorrere, e la pagina si risparmia 750 MB di duplicati per cartella con la
loro invalidazione.

Le richieste uscite dal viewport si DIMENTICANO invece di accodarsi: dopo
dieci secondi di scorrimento una coda ingenua lavorerebbe su celle che
nessuno guarda piu'.

La consegna dal thread di lavoro al thread dell'interfaccia passa da un
segnale Qt interno, mai da una chiamata diretta a un metodo che tocca la
cache: '_cache' e '_in_corso' non sono protette da un lucchetto, e con
MAX_THREAD lavori attivi insieme una chiamata diretta le muterebbe da
thread diversi in parallelo -- una corsa critica che si vede una volta ogni
cento scorrimenti, non ogni volta. Un segnale emesso da un thread diverso
da quello del ricevente viaggia in coda sull'event loop del thread
proprietario di Decodificatore (qui il thread dell'interfaccia, che ha un
event loop in corsa): la consegna vera avviene sempre li', mai in
parallelo con un'altra.

Lo stesso segnale porta anche i fallimenti (file corrotto, sparito fra la
scansione e la decodifica, una directory al posto di un file): con
l'immagine a None. Anche un fallimento libera '_in_corso', o quella cella
resterebbe segnata "in volo" per sempre e non si ricaricherebbe piu' --
nemmeno dopo che l'utente ripristina il file da questa stessa pagina.
"""
from collections import OrderedDict

from PyQt5.QtCore import QObject, QRunnable, QSize, QThreadPool, pyqtSignal

LATI = (64, 96, 128, 192, 256)
MAX_THREAD = 8


def peso_immagine(immagine):
    """I byte veri di un'immagine decodificata, zero se non c'e'.

    `bytesPerLine() * height()` e non `width() * height()`: una riga di
    QImage e' allineata a quattro byte, e la differenza e' reale appena la
    larghezza non e' un multiplo.

    Sta qui e non in `griglia.py` -- che la usa per la cache delle maschere
    e la importa da qui -- perche' due copie della stessa misura sono il
    modo in cui una delle due smette di misurare la stessa cosa; la
    direzione dell'import e' obbligata, `griglia` importa `decodifica` e
    mai il contrario.
    """
    if immagine is None:
        return 0
    return immagine.bytesPerLine() * immagine.height()


class _Lavoro(QRunnable):
    def __init__(self, percorso, lato, generazione, emettitore):
        super().__init__()
        self._percorso = percorso
        self._lato = lato
        self._generazione = generazione
        self._emettitore = emettitore

    def run(self):
        from PyQt5.QtGui import QImageReader
        if self._generazione != self._emettitore.generazione():
            # La generazione e' gia' cambiata: 'dimentica_le_richieste()' ha
            # gia' ripulito '_in_corso' per intero, quindi qui non c'e'
            # niente da segnalare indietro.
            return
        immagine = None
        try:
            lettore = QImageReader(str(self._percorso))
            lettore.setAutoTransform(True)
            dimensione = lettore.size()
            if dimensione.isValid() and dimensione.width() > 0:
                scala = min(1.0, float(self._lato) / max(dimensione.width(),
                                                         dimensione.height()))
                lettore.setScaledSize(QSize(max(1, int(dimensione.width() * scala)),
                                            max(1, int(dimensione.height() * scala))))
            letta = lettore.read()
            if letta is not None and not letta.isNull():
                immagine = letta
        except Exception:
            immagine = None
        # Non 'self._emettitore.consegna(...)': quella e' una chiamata
        # diretta, eseguita qui sul thread di lavoro. Il segnale, invece,
        # attraversa verso il thread proprietario -- vedi il docstring.
        # Emesso SEMPRE, anche con 'immagine' a None: e' l'unico modo per
        # liberare '_in_corso' anche su un fallimento, sul thread giusto --
        # altrimenti quella chiave resta segnata "in volo" per sempre e la
        # cella non si ricarica piu', nemmeno dopo che il file torna leggibile.
        self._emettitore._consegna.emit(self._percorso, self._lato, immagine,
                                        self._generazione)


class Decodificatore(QObject):
    pronta = pyqtSignal(object, int, object)   # percorso, lato, QImage

    # Segnale interno: unico punto di passaggio fra il thread di lavoro e
    # quello proprietario di Decodificatore. Mai emesso ne' connesso fuori
    # da questo modulo.
    _consegna = pyqtSignal(object, int, object, int)

    # Il tetto e' in BYTE, non in voci, ed e' la stessa lezione della cache
    # delle maschere accanto (`Griglia.TETTO_MASCHERE_BYTE`): una voce non
    # pesa sempre uguale. Alle stesse 3 000 voci che questo tetto contava
    # prima, una miniatura da 64 pesa 16 384 byte e una da 256 ne pesa
    # 262 144 -- cioe' 46,9 MiB contro 750,0 MiB con la stessa costante e
    # senza che nulla lo dica. La pagina deve reggere oltre 50 000 volti su
    # una macchina scelta per la poca VRAM, dove 750 MiB di host non sono un
    # dettaglio di prestazioni.
    #
    # 64 MiB sono 256 miniature a lato 256 o 4 096 a lato 64: molto piu' di
    # una schermata (~28 celle a 256 su uno schermo 1080p, ~480 a 64), che
    # e' cio' che serve perche' scorrere avanti e indietro non ridecodifichi.
    TETTO_CACHE_BYTE = 64 * 1024 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache = OrderedDict()
        self._peso_cache = 0
        self._in_corso = set()
        self._generazione = 0
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(MAX_THREAD)
        self._consegna.connect(self._esegui_consegna)

    def generazione(self):
        return self._generazione

    def dimentica_le_richieste(self):
        """Le richieste ancora in coda smettono di interessare. La cache resta."""
        self._generazione += 1
        self._in_corso.clear()

    def svuota(self):
        """La cache non serve piu' a nessuno: cambio di cartella o di
        progetto.

        Senza, le voci della cartella di prima restano dentro come peso
        morto fino alla chiusura della pagina -- e dopo un sort non
        tornerebbero utili nemmeno rientrando nella stessa cartella, perche'
        il sort rinomina i file e la chiave della cache e' il percorso.
        """
        self._cache.clear()
        self._peso_cache = 0

    def in_cache(self, percorso, lato):
        chiave = (str(percorso), lato)
        immagine = self._cache.get(chiave)
        if immagine is not None:
            self._cache.move_to_end(chiave)
        return immagine

    def richiedi(self, percorso, lato):
        chiave = (str(percorso), lato)
        if chiave in self._cache:
            # Colpo di cache: stessa recenza di 'in_cache()', o la LRU
            # sfoltirebbe una cella che l'utente sta guardando proprio ora.
            self._cache.move_to_end(chiave)
            self.pronta.emit(percorso, lato, self._cache[chiave])
            return
        if chiave in self._in_corso:
            return
        self._in_corso.add(chiave)
        self._pool.start(_Lavoro(percorso, lato, self._generazione, self))

    def _esegui_consegna(self, percorso, lato, immagine, generazione):
        """Slot del segnale '_consegna': gira sempre sul thread proprietario
        di Decodificatore, mai su quello di un lavoro in corso -- e' qui,
        non in '_Lavoro.run()', che la cache va toccata.

        'immagine' a None significa che il lavoro e' fallito (file
        corrotto, sparito, o una directory): si libera comunque la chiave
        da '_in_corso' -- altrimenti resterebbe segnata "in volo" per
        sempre e una richiesta successiva la troverebbe li' e non farebbe
        niente -- ma non si scrive in cache e non si emette 'pronta'."""
        if generazione != self._generazione:
            return
        chiave = (str(percorso), lato)
        self._in_corso.discard(chiave)
        if immagine is None:
            return
        self._cache[chiave] = immagine
        self._cache.move_to_end(chiave)
        self._peso_cache += peso_immagine(immagine)
        while self._peso_cache > self.TETTO_CACHE_BYTE and len(self._cache) > 1:
            # `> 1` e non `while self._cache`: l'ultima voce rimasta e'
            # quella appena consegnata, cioe' la cella che l'utente sta
            # guardando adesso. Se un giorno una singola immagine sfondasse
            # il tetto da sola, meglio tenerla che ridecodificarla a ogni
            # repaint.
            _chiave, caduta = self._cache.popitem(last=False)
            self._peso_cache -= peso_immagine(caduta)
        self.pronta.emit(percorso, lato, immagine)
