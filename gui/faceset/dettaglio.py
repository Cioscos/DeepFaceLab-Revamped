"""Il client del servizio di dettaglio: una riga JSON per richiesta.

Il servizio si avvia PIGRAMENTE al primo doppio click e muore da se' dopo
TIMEOUT_INATTIVITA_S senza comandi (cinque minuti, mainscripts/FacesetDetail.py).
Se muore per conto suo, la richiesta successiva lo riavvia: una richiesta
senza risposta non blocca niente, chi guarda vede il volto (il JPEG c'e'
comunque) e legge il guasto vero -- non una diagnosi sul file, che era
falsa su un volto sano.

La finestra che lo consuma sta in gui/dettaglio/finestra.py: qui resta il
solo client, che entrambe le pagine costruiscono e passano a lei.

**Il giro verso il processo NON aspetta.** `show()` accoda un disegno, non
lo esegue: se `_chiedi` blocca dentro un `waitFor*` -- che non fa girare
l'event loop -- il primo disegno arriva solo a risposta ricevuta, e per
tutto quel tempo la finestra e' una superficie mai dipinta. La consegna
passa dal segnale `readyReadStandardOutput` del processo: si scrive e si
torna subito, e la riga arriva a `_su_dati_pronti` al giro dopo
dell'event loop. Le richieste in volo si tengono in `_richieste`, chiavate
sull'`id` del protocollo, e ogni riga in arrivo trova la sua per
correlazione -- non per un ordine presunto, che con piu' richieste in
volo e un tempo scaduto in mezzo non si riassorbe mai.

`trasporto` e' iniettabile perche' un test non deve avviare un processo
per verificare che il comando parta con il percorso giusto: se c'e', la
consegna resta SINCRONA come prima, nello stesso giro della chiamata --
e' quello che tiene in piedi la suite del client senza riscriverla.
"""
import json
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

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
        self._processo_collegato = None   # il processo a cui il segnale e' gia' agganciato
        self._buffer = b""                # i byte del processo non ancora tagliati in righe
        # Le richieste in volo del solo ramo ASINCRONO, chiavate sull'id
        # del protocollo: id -> (op_attesa, segnale). Il ramo sincrono
        # (trasporto iniettato) non ne ha bisogno, consegna nello stesso
        # giro della chiamata.
        self._richieste = {}
        self._timer_di = {}

    def apri(self, percorso):
        """Torna l'id assegnato alla richiesta: e' cio' con cui chi chiama
        puo' riconoscere LA SUA risposta fra quelle che arrivano dopo,
        vedi `FinestraDettaglio._su_pronto`."""
        return self._chiedi({"op": "open", "path": str(percorso)}, "opened", self.pronto)

    def volti_del_frame(self, percorsi):
        """I volti gia' su disco per un fotogramma, come PERCORSI ESPLICITI.

        Chi li risolve e' gui/faceset/indice.py::mappa_per_fotogramma: il
        servizio non cerca piu' da se', perche' cercare per nome di file
        smetteva di funzionare al primo sort.

        Consegna la risposta INTERA: chi ascolta ha bisogno anche dei campi
        che nessun namedtuple porta -- il nome del file della maschera -- e
        distillarla qui vorrebbe dire decidere al posto suo.

        Torna l'id assegnato, per chi -- `PaginaEstrazione._assicura_volti`
        -- deve riconoscere la propria risposta fra quelle in ritardo.
        """
        return self._chiedi({"op": "frame", "percorsi": [str(p) for p in percorsi]},
                            "framed", self.volti_pronti)

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
        """Il giro comune a ogni operazione verso il servizio: numera e
        invia. Non solleva mai -- un servizio morto per inattivita' entra
        proprio dal ramo asincrono, e la richiesta successiva lo riavvia,
        e una codifica JSON che fallisce (`riallinea`/`salva` passano un
        `np.ndarray` grezzo) emette `fallito` invece di risalire fin dentro
        uno slot Qt.

        Torna l'id assegnato alla richiesta, in ENTRAMBI i rami: e' quello
        che permette a chi chiama di riconoscere la propria risposta,
        vedi `apri`. None se la codifica e' fallita -- non e' mai partita
        nessuna richiesta con quell'id, quindi nessuna risposta potra' mai
        combaciarci.
        """
        self._id += 1
        id_ = self._id
        try:
            riga = json.dumps(dict(comando, id=id_)) + "\n"
        except (TypeError, ValueError) as e:
            self.fallito.emit(e, None)
            return None
        if self._trasporto is not None:
            self._chiedi_sincrono(riga, id_, op_attesa, segnale)
            return id_
        self._richieste[id_] = (op_attesa, segnale)
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._su_timeout(id_))
        timer.start(TIMEOUT_MS)
        self._timer_di[id_] = timer
        try:
            self._invia_al_processo(riga)
        except Exception as e:
            # L'avvio del processo sta a VALLE della promessa in cima a
            # questo docstring, ed e' l'unico pezzo del giro che puo'
            # ancora sollevare: `comando_servizio` su un modulo `avvio`
            # non configurato, un eseguibile che non c'e'. Risalendo,
            # l'eccezione arrivava dentro lo slot Qt del chiamante --
            # per PyQt5 un `qFatal`, cioe' la finestra che sparisce.
            # Nessun codice del catalogo descrive un servizio che non
            # parte: si emette col ripiego generico, che mostra comunque
            # il motivo.
            self._dimentica_richiesta(id_)
            self.fallito.emit(e, None)
            return None
        return id_

    def _dimentica_richiesta(self, id_):
        """Toglie di mezzo una richiesta che non e' mai partita: senza,
        il suo timer scadrebbe fra TIMEOUT_MS annunciando un secondo
        guasto -- «servizio muto» -- per una domanda che nessuno ha mai
        fatto."""
        timer = self._timer_di.pop(id_, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        self._richieste.pop(id_, None)

    def _chiedi_sincrono(self, riga, id_, op_attesa, segnale):
        """Il ramo del `trasporto` iniettato: risponde SUBITO, nello stesso
        giro della chiamata -- e' cio' che tiene in piedi la suite del
        client senza processi ne' segnali."""
        try:
            risposta = self._trasporto(riga)
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
        if dati.get("id") != id_:
            # Una consegna sincrona resta comunque sfasabile se il
            # trasporto risponde con l'id di una richiesta PRECEDENTE (i
            # test del catalogo dei guasti lo fanno apposta). Il motivo
            # passa da una variabile, non da un letterale nudo dentro
            # `.emit(...)`: una guardia della suite cammina l'AST di gui/ per
            # ricavare il vocabolario dei comandi verso il trainer da ogni
            # chiamata a `.emit(...)` con un argomento costante, e un
            # letterale qui ci finirebbe dentro senza avere niente a che
            # fare con quel canale.
            motivo = "id fuori sequenza (atteso %r, ricevuto %r)" % (id_, dati.get("id"))
            self.fallito.emit(motivo, DettaglioGuasti.RISPOSTA_FUORI_SEQUENZA)
            return
        segnale.emit(dati)

    def _su_timeout(self, id_):
        """Il tempo e' scaduto senza risposta: il servizio non risponde
        piu', o e' morto per inattivita' -- la richiesta successiva lo
        riavvia da sola."""
        timer = self._timer_di.pop(id_, None)
        # `deleteLater()`, non solo `stop()`: senza, ogni richiesta
        # scaduta lascia un QTimer figlio di questo client per sempre --
        # un trascinamento e' una richiesta, e la finestra ne fa tante.
        if timer is not None:
            timer.deleteLater()
        voce = self._richieste.pop(id_, None)
        if voce is None:
            return  # la risposta e' arrivata nell'istante fra lo scadere e qui
        self.fallito.emit(ServizioMuto("il servizio di dettaglio non ha risposto"),
                          DettaglioGuasti.SERVIZIO_MUTO)

    def _invia_al_processo(self, riga):
        from PyQt5.QtCore import QProcess
        if self._processo is None or self._processo.state() == QProcess.NotRunning:
            self._avvia()
        if self._processo is not self._processo_collegato:
            self._processo.readyReadStandardOutput.connect(self._su_dati_pronti)
            # Un processo che finisce da solo -- crash, kill esterno, EOF
            # sulla pipe -- non manda mai piu' niente: aspettare il
            # timeout per accorgersene sarebbe inutile quando Qt lo sa
            # gia' subito.
            self._processo.finished.connect(self._su_processo_morto)
            self._processo.errorOccurred.connect(self._su_processo_morto)
            self._processo_collegato = self._processo
        self._processo.write(riga.encode("utf-8"))

    def _su_processo_morto(self, *_args):
        """Fallisce sul colpo ogni richiesta in volo, come `ferma()` --
        ma NON tocca `self._processo`: la richiesta successiva lo trova
        `NotRunning` e lo riavvia da sola, com'e' sempre stato."""
        for timer in self._timer_di.values():
            timer.stop()
            timer.deleteLater()
        self._timer_di.clear()
        richieste, self._richieste = self._richieste, {}
        for _op_attesa, _segnale in richieste.values():
            self.fallito.emit(ServizioMuto("il servizio di dettaglio si e' fermato"),
                              DettaglioGuasti.SERVIZIO_MUTO)

    def _su_dati_pronti(self):
        """Lo slot del segnale del processo: accumula nel buffer e
        consegna ogni riga completa che vi si trova, mai una lettura
        bloccante.

        Il segnale puo' arrivare anche dopo `ferma()`: il sistema operativo
        consegna byte gia' scritti dal figlio prima del `kill()` con un
        giro di ritardo. `_scarta_processo` disconnette proprio per
        impedirlo, ma questa guardia resta comunque -- uno slot che
        solleva e' un `qFatal` per PyQt5, non un'eccezione qualunque."""
        if self._processo is None:
            return
        self._buffer += bytes(self._processo.readAllStandardOutput())
        riga = self._leggi_una_riga_completa()
        while riga is not None:
            self._consegna(riga)
            riga = self._leggi_una_riga_completa()

    def _leggi_una_riga_completa(self):
        """Una riga intera dal buffer accumulato, o None se non c'e'
        ancora.

        Una risposta puo' arrivare a pezzi su piu' segnali
        `readyReadStandardOutput`, non tutta insieme: `self._buffer`
        sopravvive fra una chiamata e l'altra, quindi un frammento senza
        `\\n` finale resta li' ad aspettare il pezzo successivo invece di
        essere consegnato a meta'."""
        if b"\n" not in self._buffer:
            return None
        i = self._buffer.index(b"\n") + 1
        riga, self._buffer = self._buffer[:i], self._buffer[i:]
        return riga.decode("utf-8", "replace")

    def _consegna(self, riga):
        """Instrada una riga arrivata dal processo verso la richiesta che
        l'ha causata, correlando per `id`: ogni risposta trova la sua
        richiesta, anche se un'altra e' partita nel frattempo -- a
        differenza del ramo sincrono, qui non c'e' un ordine da presumere."""
        try:
            dati = json.loads(riga)
        except (TypeError, ValueError) as e:
            self.fallito.emit(e, None)
            return
        id_ricevuto = dati.get("id")
        if id_ricevuto is None:
            # Il servizio risponde cosi' -- {"op":"error","id":None,...} --
            # quando la riga che ha ricevuto non si capiva come JSON: non
            # c'e' un comando da cui leggere l'id, quindi NESSUNA
            # richiesta in `_richieste` puo' mai combaciarci per
            # correlazione. Senza questo ramo la riga verrebbe scartata in
            # silenzio (id sconosciuto) e la richiesta vera resterebbe
            # appesa fino al timeout, che la accuserebbe di essere un
            # servizio muto -- falso, il servizio ha risposto benissimo.
            self.fallito.emit(dati.get("motivo"), dati.get("codice"))
            return
        voce = self._richieste.pop(id_ricevuto, None)
        timer = self._timer_di.pop(id_ricevuto, None)
        if timer is not None:
            # Ferma E cancella: uno `stop()` da solo lascia il QTimer
            # figlio di questo client per sempre, uno per richiesta
            # servita -- 200 trascinamenti, 200 QTimer morti ma vivi.
            timer.stop()
            timer.deleteLater()
        if voce is None:
            # Nessuna richiesta in attesa con questo id: e' scaduta prima
            # di arrivare, o non e' mai partita da questo client. Si
            # scarta, senza avvisare nessuno -- non e' un guasto, e'
            # semplicemente in ritardo su qualcosa che non aspetta piu'.
            return
        op_attesa, segnale = voce
        if dati.get("op") != op_attesa:
            self.fallito.emit(dati.get("motivo"), dati.get("codice"))
            return
        segnale.emit(dati)

    def _scarta_processo(self):
        """Disconnette i segnali dal processo corrente e ne programma la
        distruzione, se ce n'e' uno.

        Senza questo, un `QProcess` -- figlio del client -- sopravvive per
        sempre a ogni riavvio del servizio o a `ferma()`: resta agganciato
        a `_su_dati_pronti`/`_su_processo_morto` e un byte consegnato in
        ritardo dal sistema operativo raggiunge uno slot che non se lo
        aspetta piu'."""
        processo = self._processo
        if processo is not None and processo is self._processo_collegato:
            processo.readyReadStandardOutput.disconnect(self._su_dati_pronti)
            processo.finished.disconnect(self._su_processo_morto)
            processo.errorOccurred.disconnect(self._su_processo_morto)
        if processo is not None:
            processo.deleteLater()
        self._processo = None
        self._processo_collegato = None

    def _avvia(self):
        from PyQt5.QtCore import QProcess
        from gui.faceset.avvio import comando_servizio
        self._scarta_processo()
        programma, argomenti = comando_servizio(self._workdir)
        self._processo = QProcess(self)
        self._processo.setProcessChannelMode(QProcess.SeparateChannels)
        # Un frammento del processo MORTO non deve incollarsi davanti alla
        # prima risposta di quello nuovo: senza questa riga un `open` a
        # meta' buffer avvelena la decodifica della richiesta che riavvia
        # il servizio, e chi guarda vede un guasto senza codice seguito
        # da un secondo «servizio muto» falso 8 s dopo.
        self._buffer = b""
        self._processo.start(programma, argomenti)
        self._processo.waitForStarted(TIMEOUT_MS)

    def ferma(self):
        """Uccide il processo e svuota lo stato in volo.

        Fallisce ogni richiesta ancora in attesa (`_su_processo_morto`,
        lo stesso giro di un crash): senza, un cursore o un indicatore
        acceso in attesa di quella risposta non tornerebbe mai indietro --
        `setOverrideCursor` e' uno stack GLOBALE di QApplication, e
        resterebbe girato su tutta l'interfaccia, non solo su questa
        finestra.

        Il processo va anche scartato (`_scarta_processo`), non solo
        ucciso: senza, chiudere la scheda con una richiesta in volo puo'
        far arrivare un `readyReadStandardOutput` tardivo su un client che
        crede di non avere piu' nessun processo -- abortisce l'intera
        GUI, non solo la finestra che si stava chiudendo."""
        if self._processo is not None:
            self._processo.kill()
        self._scarta_processo()
        self._buffer = b""
        self._su_processo_morto()
