"""Il trasporto verso un servizio a righe JSON in sottoprocesso.

Lo usano la sessione manuale di estrazione e la pagina di fusione: cio'
che finisce in un log o sotto gli occhi dell'utente non nomina nessuna
delle due, o un guasto della fusione si leggerebbe come un guasto
dell'estrazione.

Due modi, e la differenza e' cio' che rende usabile il rettangolo mosso da
tastiera:

- `invia` e' sincrono, e va bene per `frame` e `salva`: sono gesti
  singoli, l'utente ha appena cliccato e aspetta il risultato.
- `invia_ultimo` e' per `rileva`, che si genera decine di volte al
  secondo tenendo premuta una freccia. Vale L'ULTIMA: una richiesta
  superata da una piu' recente si butta invece di accodarsi -- e' la
  stessa scelta che gui/faceset/decodifica.py fa per le miniature uscite
  dal viewport, e per la stessa ragione. Accodarle vorrebbe dire vedere
  il rettangolo inseguire il tasto per secondi dopo averlo lasciato.

Il figlio serve una richiesta alla volta: mentre una e' scritta e non
ancora risposta la successiva non parte, resta in `_in_attesa` -- al
massimo UNA -- e quella nuova sostituisce quella che c'era. Se la
richiesta gia' scritta viene superata prima che risponda, la sua risposta
arriva comunque (il figlio non si puo' interrompere a meta') ma si
scarta: e' la bandiera `valida` dentro `_in_volo`.

Le risposte si appaiano per ID, non per ordine di arrivo: e' la sola
difesa contro il mescolamento fra una richiesta sincrona e una asincrona
sullo stesso canale. Una risposta con un id che non corrisponde a niente
in attesa si scarta.

Se il canale muore mentre una richiesta e' in volo o in attesa -- il
figlio crasha, esaurisce la memoria, o viene ucciso da fuori --
`_in_volo` non deve restare occupato per sempre: senza una liberazione
esplicita ogni `invia_ultimo` successiva prenderebbe il ramo "occupato" e
non scriverebbe mai piu' niente sul canale. Vedi `_su_guasto`: chi aveva
una richiesta in sospeso riceve un fallimento sintetico, poi lo slot si
libera. La richiesta SUCCESSIVA pero' non riavvia piu' il processo da
sola (voce 3.83, estesa da `invia_subito` a `invia_ultimo`): un canale
gia' morto risponde subito col guasto sintetico invece di far ripartire
un processo intero -- un modello caricato in VRAM -- per un comando
qualunque, come faceva il vecchio trasporto sincrono che ricontrollava
lo stato a ogni chiamata.

Il canale e' iniettabile (parametro `canale`): i test non devono avviare
un processo con torch dentro. Il canale reale (`_CanaleProcesso`) e' un
QProcess col protocollo a righe JSON gia' visto in
gui/faceset/dettaglio.py::ClienteDettaglio, spostato qui da
gui/estrazione/pagina.py -- con una differenza voluta: legge in modo NON
bloccante (readyReadStandardOutput), non con un ciclo di
waitForReadyRead, perche' e' proprio l'attesa bloccante che questo
modulo esiste per togliere di mezzo. Il collegamento a `finished` e
`errorOccurred` segue lo stesso precedente di gui/execution/jobs.py.

Lo STDERR del figlio si legge anche lui (`readyReadStandardError`), ma
resta un canale separato per davvero: `SeparateChannels`, mai
`MergedChannels`. `rispondi` (mainscripts/ExtractManual.py) gira sotto
`contextlib.redirect_stdout(sys.stderr)` proprio per tenere il protocollo
a righe JSON su stdout pulito da qualunque traceback -- unificare i
canali qui vorrebbe dire rompere quella cautela dall'altro capo. Le
ultime righe si accumulano in un anello di `_MAX_RIGHE_STDERR` righe
(`TrasportoAsincrono.stderr_recente()`), non in una lista che cresce: il
servizio gira per ore, e prima di questo modulo lo stderr del figlio non
lo leggeva nessuno -- il traceback vero di un fallimento (`rispondi`
cattura ogni eccezione e risponde solo `{"op": "error", "motivo":
str(e)}`) finiva buttato via dal sistema operativo appena il processo
terminava.
"""
import collections
import json
import time
from pathlib import Path

from core.interact import interact as io
from gui.estrazione import avvio as avvio_mod
from gui.estrazione import servizio as servizio_mod

# La risposta sintetica che _su_guasto consegna a chi aveva una richiesta
# in sospeso quando il canale muore: stessa forma di un "error" vero dal
# figlio (Servizio._invia la riconosce dal campo "op"), cosi' un
# chiamante come Servizio.rileva_quando_puoi non ha bisogno di sapere che
# questa risposta non e' mai passata dal processo. Il modulo e' condiviso
# da estrazione e fusione: il motivo resta senza nome di servizio, e chi
# lo mostra all'utente (i due servizio.py) ci mette il nome che conosce.
_RISPOSTA_GUASTO = {"op": "error", "motivo": "il servizio si e' interrotto"}

# Quante righe di stderr tenere in memoria. Un traceback Python tipico
# (compresa una catena di eccezioni concatenate con "During handling of
# the above exception") sta comodamente sotto un centinaio di righe; 200
# lascia margine senza far crescere la memoria di un servizio che gira
# per ore -- un anello, non una lista che si accumula.
_MAX_RIGHE_STDERR = 200

# Quanto si aspetta un processo appena ucciso, per non lasciare al
# distruttore di QProcess un processo vivo da reclamare (millisecondi in
# pratica: e' gia' morto).
_ATTESA_MORTE_MS = 200


class _CanaleProcesso:
    """Il canale reale: un QProcess, protocollo a righe JSON su
    stdin/stdout.

    La consegna passa SEMPRE da `readyReadStandardOutput`, mai da una
    lettura bloccante: e' il segnale che rende il canale non bloccante
    per davvero. `aspetta()` esiste solo per l'uso sincrono di
    `TrasportoAsincrono.invia` -- fa avanzare l'event loop del processo
    finche' arrivano dei byte, ma e' ancora `_su_dati_pronti` (agganciato
    al segnale) a spezzare il buffer in righe e a consegnarle, mai
    `aspetta()` stessa.

    Una riga si consegna solo quando e' completa (`\\n` trovato nel
    buffer): la stessa cautela di
    ClienteDettaglio._leggi_una_riga_completa, qui ottenuta accumulando
    invece di aspettare `canReadLine()`, perche' la lettura di partenza
    (`readyReadStandardOutput`) e' gia' non bloccante da sola.

    `finished` e `errorOccurred` sono collegati entrambi, come in
    gui/execution/jobs.py: un crash emette il primo, un processo che non
    parte mai (eseguibile assente) puo' non emettere il secondo -- vedi
    JobRun._on_error_occurred li' per lo stesso ragionamento.

    Lo stderr si legge allo stesso modo (`readyReadStandardError`,
    accumulo per riga completa) ma finisce in un anello separato
    (`_stderr`), mai nel buffer del protocollo: non e' `_ricevitore` a
    consegnarlo, e non passa da `_su_riga`. L'anello sopravvive a
    `_su_morte`/`_su_errore` -- quei due azzerano solo `self._processo`,
    apposta: e' proprio nel momento del crash che le righe servono.
    """

    def __init__(self, workdir, comando_servizio=None, ambiente=None):
        self.workdir = workdir
        self._comando_servizio = comando_servizio or avvio_mod.comando_servizio
        self._ambiente = dict(ambiente or {})
        self._processo = None
        self._buffer = b""
        self._buffer_stderr = b""
        self._stderr = collections.deque(maxlen=_MAX_RIGHE_STDERR)
        self._ricevitore = None
        self._gestore_guasto = None
        # "mai avviato" e "morto" sono stati diversi anche se in entrambi
        # self._processo e' None: un processo mai partito deve valere
        # ATTIVO (scrivi() lo avvia al primo comando -- e' cosi' che parte
        # la sessione manuale), uno partito e poi crashato deve valere
        # morto. La bandiera si spegne alla prima chiamata di scrivi().
        self._mai_avviato = True

    def collega(self, ricevitore):
        self._ricevitore = ricevitore

    def collega_guasto(self, gestore):
        self._gestore_guasto = gestore

    def righe_stderr(self):
        """Le ultime righe di stderr del figlio, piu' vecchia per prima."""
        return list(self._stderr)

    def scrivi(self, comando):
        from PyQt5.QtCore import QProcess
        if self._processo is None or self._processo.state() == QProcess.NotRunning:
            self._avvia()
        self._mai_avviato = False
        self._processo.write((json.dumps(comando) + "\n").encode("utf-8"))

    def aspetta(self, timeout_ms):
        if self._processo is None:
            return False
        return self._processo.waitForReadyRead(max(0, timeout_ms))

    def attivo(self):
        """C'e' un processo VIVO all'altro capo -- oppure non ne e' MAI
        partito uno.

        Distingue «il figlio sta lavorando» e «nessun comando e' ancora
        stato scritto» (entrambi ATTIVO: `scrivi()` avvia il processo al
        primo comando, ed e' cosi' che parte una sessione manuale) da «il
        figlio non c'e' piu' e `scrivi` lo farebbe RIPARTIRE» (morto): per
        `chiudi`/`invia_ultimo` la differenza e' fra rispondere subito col
        guasto sintetico e riavviare un processo con un modello dentro
        solo per consegnargli un comando qualunque."""
        if self._mai_avviato:
            return True
        from PyQt5.QtCore import QProcess
        return self._processo is not None and self._processo.state() != QProcess.NotRunning

    def _avvia(self):
        from PyQt5.QtCore import QProcess, QProcessEnvironment
        programma, argomenti = self._comando_servizio(self.workdir)
        # Azzera ENTRAMBI i buffer di ricomposizione (stdout e stderr):
        # un figlio precedente puo' essere morto a meta' riga, senza '\n'
        # finale, lasciando un frammento. Senza questo azzeramento il
        # primo output del processo nuovo si concatenerebbe al residuo
        # del vecchio -- su stdout la riga fusa fallisce json.loads in
        # _su_riga, che inghiotte l'eccezione e ritorna None: la risposta
        # si perde in silenzio. Difetto preesistente per `_buffer`,
        # stessa causa e stessa correzione per `_buffer_stderr` (aggiunto
        # insieme all'anello dello stderr, non introdotto qui).
        self._buffer = b""
        self._buffer_stderr = b""
        self._processo = QProcess()
        self._processo.setProcessChannelMode(QProcess.SeparateChannels)
        if self._ambiente:
            env = QProcessEnvironment.systemEnvironment()
            for chiave, valore in self._ambiente.items():
                env.insert(chiave, str(valore))
            self._processo.setProcessEnvironment(env)
        self._processo.readyReadStandardOutput.connect(self._su_dati_pronti)
        self._processo.readyReadStandardError.connect(self._su_stderr_pronto)
        self._processo.finished.connect(self._su_morte)
        self._processo.errorOccurred.connect(self._su_errore)
        self._processo.start(programma, argomenti)
        self._processo.waitForStarted(servizio_mod.TIMEOUT_MS)

    def _su_dati_pronti(self):
        # La guardia non e' difensiva: un segnale emesso poco prima del
        # kill viene consegnato DOPO che il campo e' stato azzerato, e
        # un'eccezione dentro uno slot Qt e' un qFatal -- cioe' la GUI
        # intera che aborta, non una traccia da leggere (voce 3.78).
        if self._processo is None:
            return
        self._buffer += bytes(self._processo.readAllStandardOutput())
        while b"\n" in self._buffer:
            grezza, self._buffer = self._buffer.split(b"\n", 1)
            if self._ricevitore is not None:
                self._ricevitore(grezza.decode("utf-8", "replace"))

    def _su_stderr_pronto(self):
        if self._processo is None:
            return                            # come sopra, voce 3.78
        self._buffer_stderr += bytes(self._processo.readAllStandardError())
        while b"\n" in self._buffer_stderr:
            grezza, self._buffer_stderr = self._buffer_stderr.split(b"\n", 1)
            self._stderr.append(grezza.decode("utf-8", "replace"))

    def _su_morte(self, _codice, _stato):
        self._processo = None
        if self._gestore_guasto is not None:
            self._gestore_guasto()

    def _su_errore(self, _errore):
        # Un processo che non parte mai (eseguibile assente) puo' non
        # emettere mai `finished` -- vedi il commento gemello in
        # gui/execution/jobs.py::_on_error_occurred. Idempotente verso
        # `_su_morte`: il secondo dei due segnali trova lo stato gia'
        # azzerato e il gestore non ha piu' niente da liberare.
        self._processo = None
        if self._gestore_guasto is not None:
            self._gestore_guasto()

    def chiudi(self):
        """Scollega PRIMA di uccidere, e solo dopo azzera il campo.

        Scollegare e' la difesa strutturale: cio' che il figlio ha scritto
        un istante prima del kill e' gia' in coda come segnale, e senza
        questo verrebbe consegnato a uno slot che trova `self._processo`
        gia' a None. La guardia nei due slot resta come seconda rete, per
        i segnali che `_su_morte`/`_su_errore` azzerano per conto loro.
        """
        processo, self._processo = self._processo, None
        if processo is None:
            return
        for segnale, slot in ((processo.readyReadStandardOutput, self._su_dati_pronti),
                              (processo.readyReadStandardError, self._su_stderr_pronto),
                              (processo.finished, self._su_morte),
                              (processo.errorOccurred, self._su_errore)):
            segnale.disconnect(slot)
        processo.kill()
        # Il distruttore di QProcess aspetta comunque un processo ancora
        # vivo, stampando "Destroyed while process is still running": qui
        # e' gia' ucciso, quindi l'attesa e' di millisecondi e il messaggio
        # non compare.
        processo.waitForFinished(_ATTESA_MORTE_MS)


class TrasportoAsincrono:
    def __init__(self, workdir, canale=None, su_evento=None, comando_servizio=None, ambiente=None):
        self.workdir = Path(workdir) if workdir is not None else None
        self._canale = canale if canale is not None else \
            _CanaleProcesso(self.workdir, comando_servizio, ambiente)
        self._su_evento = su_evento
        self._canale.collega(self._su_riga)
        self._canale.collega_guasto(self._su_guasto)
        self._prossimo_id = 0
        # (id, valida, callback) della richiesta scritta sul canale e non
        # ancora risposta, o None. `valida` distingue "questa risposta ci
        # interessa ancora" da "e' stata superata mentre era in volo": il
        # figlio la calcola comunque (non si interrompe una richiesta gia'
        # partita), ma senza questa bandiera la sua risposta arriverebbe
        # comunque alla callback -- proprio cio' che "l'ultimo vince" deve
        # evitare.
        self._in_volo = None
        self._in_attesa = None    # (comando, callback), al massimo uno
        self._chiuso = False

    # -- sincrono ---------------------------------------------------------

    def invia(self, comando):
        """Per `frame` e `salva`: gesti singoli, l'utente aspetta.

        Butta la richiesta asincrona che fosse in attesa -- un gesto
        esplicito non deve aspettare il turno dietro un rilevamento che
        l'utente potrebbe aver gia' smesso di guardare -- e tira
        attivamente dal canale finche' la propria risposta non arriva,
        senza dipendere da un ciclo eventi esterno."""
        if self._chiuso:
            return None
        risultato = {}

        def _cattura(risposta):
            risultato["r"] = risposta

        self.invia_ultimo(comando, _cattura)
        if self._chiuso:
            return None
        scadenza = time.monotonic() + servizio_mod.TIMEOUT_MS / 1000.0
        while "r" not in risultato and not self._chiuso:
            rimanente_ms = int((scadenza - time.monotonic()) * 1000)
            if rimanente_ms <= 0 or not self._canale.aspetta(rimanente_ms):
                break
        return risultato.get("r")

    # -- asincrono ----------------------------------------------------------

    def invia_ultimo(self, comando, quando_pronto):
        """Per `rileva`: se il canale e' libero parte subito, altrimenti
        sostituisce la richiesta che era in attesa e invalida quella gia'
        in volo -- la sua risposta, quando arriva, si scarta.

        Un figlio GIA' MORTO non si riavvia per un comando qualunque
        (una freccia, un `cfg`): stessa guardia di `invia_subito`, vedi
        il suo docstring per la ragione (voce 3.83)."""
        if self._chiuso:
            return
        if not self._canale_attivo():
            # Come invia_subito: un figlio gia' morto non si riavvia per un
            # comando qualunque -- riavviare vuol dire ricaricare un modello
            # intero. Chi ha chiesto riceve il guasto sintetico; gli slot
            # restano liberi.
            self._in_volo = None
            self._in_attesa = None
            self._consegna(quando_pronto, dict(_RISPOSTA_GUASTO))
            return
        self._prossimo_id += 1
        comando = dict(comando)
        comando["id"] = self._prossimo_id
        if self._in_volo is None:
            self._scrivi(comando, quando_pronto)
        else:
            id_in_volo, _valida, callback_in_volo = self._in_volo
            self._in_volo = (id_in_volo, False, callback_in_volo)
            self._in_attesa = (comando, quando_pronto)

    def invia_subito(self, comando, quando_pronto):
        """Per `chiudi`: la riga va scritta ORA, non dietro la coalescenza.

        `invia_ultimo` accoda dietro la richiesta in volo, e con un
        caricamento di modello in corso quella resta in volo per minuti: il
        comando di chiusura non partirebbe mai, e chi lo ha mandato
        ucciderebbe un figlio che non ha mai saputo di dover salvare. Qui
        la richiesta in volo si invalida (la sua risposta non interessa
        piu' a nessuno, si sta chiudendo) e quella in attesa si butta.

        Un figlio GIA' MORTO non si riavvia: `_scrivi` passa da
        `_CanaleProcesso.scrivi`, che un processo fermo lo fa ripartire --
        cioe' ricaricherebbe un modello intero solo per consegnargli un
        `chiudi`, mentre chi aspetta la risposta resta fermo fino al
        proprio timeout (voce 3.83). Si risponde subito col guasto
        sintetico, la stessa forma che `_su_guasto` consegna, e si liberano
        ENTRAMBI gli slot: la risposta della richiesta in volo, se mai
        arrivasse, non corrisponderebbe piu' a nessun id e si scarterebbe."""
        if self._chiuso:
            return
        if not self._canale_attivo():
            # Nessuna delle due avra' mai una risposta vera, e lo slot deve
            # restare LIBERO -- non perche' una richiesta futura riavvii il
            # processo (non lo fa piu', voce 3.87: anche `invia_ultimo`
            # porta questa guardia), ma perche' con `_in_volo` occupato per
            # sempre ogni chiamata successiva prenderebbe il ramo
            # "occupato" invece di rispondere subito col guasto.
            self._in_volo = None
            self._in_attesa = None
            self._consegna(quando_pronto, dict(_RISPOSTA_GUASTO))
            return
        self._prossimo_id += 1
        comando = dict(comando)
        comando["id"] = self._prossimo_id
        self._in_attesa = None
        # `_scrivi` rimpiazza `_in_volo`: la risposta della richiesta
        # superata non corrispondera' piu' a nessun id in attesa e si
        # scarta da se' (`_su_riga`).
        self._scrivi(comando, quando_pronto)

    def stderr_recente(self):
        """Le ultime righe di stderr del figlio, per la diagnosi -- mai
        per il protocollo, che resta esclusivamente su stdout. Delega al
        canale: e' li' che l'anello vive, perche' sopravviva a un canale
        che si azzera e si riavvia (`_CanaleProcesso.scrivi`) mentre
        `TrasportoAsincrono` stesso resta lo stesso oggetto per tutta la
        sessione."""
        return self._canale.righe_stderr()

    def consegna_tutto(self):
        """Aiutante dei test: pompa il canale finche' ha risposte pronte.
        Il canale vero si consegna da solo via readyReadStandardOutput --
        questo non serve mai in produzione."""
        while self._canale.aspetta(0):
            pass

    def chiudi(self):
        self._in_volo = None
        self._in_attesa = None
        self._chiuso = True
        self._canale.chiudi()

    # -- interno ------------------------------------------------------------

    def _canale_attivo(self):
        """Il canale ha un processo vivo. Un canale finto che non sappia
        rispondere (i test, e chiunque inietti un trasporto) vale sempre
        vivo: e' il canale a processo che ha qualcosa da dire qui."""
        attivo = getattr(self._canale, "attivo", None)
        return True if attivo is None else bool(attivo())

    def _scrivi(self, comando, callback):
        self._in_volo = (comando["id"], True, callback)
        self._canale.scrivi(comando)

    def _consegna(self, callback, risposta):
        if callback is None:
            return
        try:
            callback(risposta)
        except Exception as errore:
            # La callback arriva da uno slot Qt: un'eccezione li' chiama
            # qFatal e si porta via il processo con dentro ogni training
            # aperto (misurato, EXIT=134) -- si inghiotte, ma non in
            # silenzio: un difetto nella callback del chiamante altrimenti
            # sparisce dietro un rettangolo che smette di aggiornarsi
            # senza una riga da nessuna parte.
            io.log_err("trasporto: una callback ha sollevato: %s" % errore)

    def _su_riga(self, riga):
        try:
            risposta = json.loads(riga)
        except (TypeError, ValueError):
            return None
        if isinstance(risposta, dict) and risposta.get("id") is None:
            # Un EVENTO del figlio (frame_pronto, avanzamento...): non e' la
            # risposta a nessuna richiesta. Il servizio di estrazione non ne
            # manda; quello di fusione si'.
            if self._su_evento is not None:
                self._consegna(self._su_evento, risposta)
            return None
        if self._in_volo is None or not isinstance(risposta, dict) \
                or risposta.get("id") != self._in_volo[0]:
            return None    # non corrisponde a nessuna richiesta in attesa
        _id, valida, callback = self._in_volo
        self._in_volo = None
        if valida:
            self._consegna(callback, risposta)
        if self._in_attesa is not None:
            prossimo_comando, prossima_callback = self._in_attesa
            self._in_attesa = None
            if self._canale_attivo():
                self._scrivi(prossimo_comando, prossima_callback)
            else:
                # Il figlio ha appena risposto (questa riga) e poi e'
                # morto prima che la richiesta coalescente partisse --
                # `readyReadStandardOutput` puo' arrivare prima di
                # `finished`. Senza questa guardia `_scrivi` chiamerebbe
                # `_CanaleProcesso.scrivi`, che troverebbe il processo
                # `NotRunning` e lo farebbe RIPARTIRE: lo stesso riavvio
                # che la voce 3.87 chiude sugli altri due ingressi.
                self._consegna(prossima_callback, dict(_RISPOSTA_GUASTO))
        return risposta

    def _su_guasto(self):
        """Il canale (il processo figlio) non c'e' piu'. Non chiude il
        trasporto -- libera lo slot e avvisa chi stava aspettando, invece
        di lasciarlo occupato per sempre. La richiesta SUCCESSIVA non
        riavvia piu' il processo da sola (voce 3.87): trova
        `_canale_attivo()` falso e riceve subito lo stesso guasto
        sintetico, senza toccare `_CanaleProcesso.scrivi`. Idempotente: se
        non c'era niente in volo o in attesa non fa nulla, e un secondo
        avviso (finished ED errorOccurred possono arrivare entrambi per lo
        stesso evento) trova lo stato gia' vuoto."""
        if self._in_volo is not None:
            _id, valida, callback = self._in_volo
            self._in_volo = None
            if valida:
                self._consegna(callback, dict(_RISPOSTA_GUASTO))
        if self._in_attesa is not None:
            _comando, callback = self._in_attesa
            self._in_attesa = None
            self._consegna(callback, dict(_RISPOSTA_GUASTO))
