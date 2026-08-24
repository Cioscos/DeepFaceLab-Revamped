"""Il client del servizio di dettaglio: una riga JSON per richiesta.

Il servizio si avvia PIGRAMENTE al primo doppio click e muore da se' dopo
TIMEOUT_INATTIVITA_S senza comandi (cinque minuti, mainscripts/FacesetDetail.py).
Se muore per conto suo, la richiesta successiva lo riavvia: una richiesta
senza risposta non blocca niente, chi guarda vede il volto (il JPEG c'e'
comunque) e legge il guasto vero -- non una diagnosi sul file, che era
falsa su un volto sano.

La finestra che lo consuma sta in gui/dettaglio/finestra.py: qui resta il
solo client, che entrambe le pagine costruiscono e passano a lei.

`trasporto` e' iniettabile perche' un test non deve avviare un processo
per verificare che il comando parta con il percorso giusto.
"""
import json
import time
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from mainscripts import DettaglioGuasti

TIMEOUT_MS = 8000


class ServizioMuto(OSError):
    """Il servizio non ha risposto entro TIMEOUT_MS.

    E' una classe sua, e non un `OSError` nudo, per una ragione sola: chi
    la intercetta la riconosce per TIPO. Riconoscerla dal testo del
    messaggio si romperebbe in silenzio alla prima riformulazione, che e'
    esattamente il difetto che il catalogo dei codici esiste per togliere.
    """


class ClienteDettaglio(QObject):
    pronto = pyqtSignal(dict)
    volti_pronti = pyqtSignal(dict)
    riallineato = pyqtSignal(dict)
    salvato = pyqtSignal(dict)
    rilevato = pyqtSignal(dict)
    # Due argomenti: il motivo grezzo del servizio e il suo codice
    # (mainscripts/DettaglioGuasti.py), None quando il guasto non e'
    # catalogato. Il motivo resta il PRIMO, cosi' uno slot che ne prende
    # uno solo -- Qt li accetta -- riceve esattamente quello che riceveva
    # prima.
    fallito = pyqtSignal(object, object)

    def __init__(self, workdir, trasporto=None, parent=None):
        super().__init__(parent)
        self._workdir = Path(workdir)
        self._trasporto = trasporto
        self._id = 0
        self._processo = None

    def apri(self, percorso):
        self._chiedi({"op": "open", "path": str(percorso)}, "opened", self.pronto)

    def volti_del_frame(self, aligned_dir, nome_frame):
        """I volti gia' su disco per un fotogramma. Consegna la risposta
        INTERA: chi ascolta ha bisogno anche dei campi che nessun
        namedtuple porta -- il nome del file della maschera -- e
        distillarla qui vorrebbe dire decidere al posto suo."""
        self._chiedi({"op": "frame", "aligned_dir": str(aligned_dir),
                      "frame": str(nome_frame)}, "framed", self.volti_pronti)

    def riallinea(self, percorso, frame_dir, source_landmarks):
        """L'anteprima del riallineamento: nessuna scrittura nel progetto.

        `source_landmarks` passa cosi' com'e', a differenza dei due percorsi
        che passano da `str()`: un `np.ndarray` (la forma in cui i landmark
        vivono altrove nel pacchetto) non e' serializzabile in JSON e fa
        fallire `json.dumps` dentro `_chiedi` -- l'errore arriva comunque a
        `fallito`, ma come un `TypeError` che non dice da dove viene."""
        self._chiedi({"op": "riallinea", "path": str(percorso),
                      "frame_dir": str(frame_dir),
                      "source_landmarks": source_landmarks},
                     "riallineato", self.riallineato)

    def salva(self, percorso, frame_dir, source_landmarks):
        """L'unica richiesta che riscrive il file allineato.

        Stessa avvertenza di `riallinea` su `source_landmarks`: passa raw."""
        self._chiedi({"op": "salva", "path": str(percorso),
                      "frame_dir": str(frame_dir),
                      "source_landmarks": source_landmarks},
                     "salvato", self.salvato)

    def rileva(self, percorso, frame_dir, modo, allineatore, rilevatore=None):
        """Proposte dai motori. Il rilevatore si manda SOLO nel modo
        `volto`: nel modo `landmarks` il servizio non lo costruisce
        nemmeno, e mandarne uno lascerebbe credere il contrario a chi
        legge il protocollo."""
        comando = {"op": "rileva", "path": str(percorso),
                   "frame_dir": str(frame_dir), "modo": modo,
                   "allineatore": allineatore}
        if rilevatore is not None:
            comando["rilevatore"] = rilevatore
        self._chiedi(comando, "rilevato", self.rilevato)

    def _chiedi(self, comando, op_attesa, segnale):
        """Il giro comune a ogni operazione verso il servizio: numera,
        invia, decodifica, e emette `segnale` oppure `fallito`. Non solleva
        mai -- un servizio morto per inattivita' entra proprio da qui, e la
        richiesta successiva lo riavvia."""
        self._id += 1
        comando = dict(comando, id=self._id)
        try:
            risposta = self._invia(json.dumps(comando) + "\n")
        except ServizioMuto as e:
            self.fallito.emit(e, DettaglioGuasti.SERVIZIO_MUTO)
            return
        except Exception as e:
            self.fallito.emit(e, None)
            return
        try:
            dati = json.loads(risposta)
        except (TypeError, ValueError) as e:
            self.fallito.emit(e, None)
            return
        if dati.get("op") != op_attesa:
            # Il codice viaggia insieme al motivo: e' cio' che permette a
            # chi mostra il guasto di dire una frase sua invece di
            # ripetere il testo d'implementazione del servizio. Un guasto
            # senza codice arriva con None, non sparisce.
            self.fallito.emit(dati.get("motivo"), dati.get("codice"))
            return
        if dati.get("id") != self._id:
            # Un timeout ha sfasato le consegne: questa risposta e' di una
            # richiesta PRECEDENTE, arrivata solo ora. Consegnarla come se
            # fosse quella corrente sposterebbe lo sfasamento sulla
            # richiesta successiva, e non si riassorbirebbe mai. Il motivo
            # passa da una variabile, non da un letterale nudo dentro
            # `.emit(...)`: una guardia della suite cammina l'AST di gui/ per
            # ricavare il vocabolario dei comandi verso il trainer da ogni
            # chiamata a `.emit(...)` con un argomento costante, e un
            # letterale qui ci finirebbe dentro senza avere niente a che
            # fare con quel canale.
            motivo = "id fuori sequenza (atteso %r, ricevuto %r)" % (self._id, dati.get("id"))
            self.fallito.emit(motivo, DettaglioGuasti.RISPOSTA_FUORI_SEQUENZA)
            return
        segnale.emit(dati)

    def _invia(self, riga):
        if self._trasporto is not None:
            return self._trasporto(riga)
        return self._invia_al_processo(riga)

    def _invia_al_processo(self, riga):
        from PyQt5.QtCore import QProcess
        if self._processo is None or self._processo.state() == QProcess.NotRunning:
            self._avvia()
        self._processo.write(riga.encode("utf-8"))
        return self._leggi_una_riga_completa()

    def _leggi_una_riga_completa(self):
        """Aspetta che il buffer del processo contenga una riga intera,
        non solo dei byte.

        `waitForReadyRead` torna appena arrivano DEI byte, non
        necessariamente una riga completa -- e `readLine()` consegna
        quello che c'e' nel buffer anche senza il `\\n` finale. Una sola
        lettura dopo l'attesa (come faceva questo metodo prima) puo'
        quindi restituire meta' risposta e lasciare il resto nel buffer:
        alla richiesta successiva `waitForReadyRead` torna subito (i dati
        ci sono gia') e `readLine()` consegna la coda della risposta
        precedente spacciandola per quella nuova -- da li' in poi ogni
        richiesta e' sfasata di un messaggio rispetto a quella vera.
        `canReadLine()` dice se il buffer contiene davvero un `\\n`;
        finche' non e' cosi' si continua ad aspettare, rispettando il
        timeout complessivo invece di uno per ogni lettura parziale."""
        scadenza = time.monotonic() + TIMEOUT_MS / 1000.0
        while not self._processo.canReadLine():
            rimanente_ms = int((scadenza - time.monotonic()) * 1000)
            if rimanente_ms <= 0 or not self._processo.waitForReadyRead(rimanente_ms):
                raise ServizioMuto("il servizio di dettaglio non ha risposto")
        return bytes(self._processo.readLine()).decode("utf-8", "replace")

    def _avvia(self):
        from PyQt5.QtCore import QProcess
        from gui.faceset.avvio import comando_servizio
        programma, argomenti = comando_servizio(self._workdir)
        self._processo = QProcess(self)
        self._processo.setProcessChannelMode(QProcess.SeparateChannels)
        self._processo.start(programma, argomenti)
        self._processo.waitForStarted(TIMEOUT_MS)

    def ferma(self):
        if self._processo is not None:
            self._processo.kill()
            self._processo = None
