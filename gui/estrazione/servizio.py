"""Il client del protocollo di estrazione manuale.

Non importa mainscripts: parla il protocollo. Il trasporto e' iniettabile
perche' i test non devono avviare un processo con torch dentro.

Ogni numero che entra da qui passa da gui/numeri.py con ENTRAMBI i
predicati: la finitezza non basta, 1e300 e' finito e uccide comunque
l'int() di un paintEvent.
"""
from gui import numeri

TIMEOUT_MS = 10000


def _punti_utilizzabili(punti):
    if not isinstance(punti, list) or not punti:
        return None
    fuori = []
    for punto in punti:
        if not isinstance(punto, (list, tuple)) or len(punto) != 2:
            return None
        x, y = punto
        if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
            return None
        if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
            return None
        fuori.append((float(x), float(y)))
    return fuori


def _volti_utilizzabili(risposta):
    """[(rect, landmarks), ...] dal campo "volti" di una risposta
    "rileva", scartando ogni volto malformato invece di sollevare. La
    versione sincrona (`rileva()`) e' stata rimossa: `_rileva` in
    gui/estrazione/pagina.py passa sempre da
    `rileva_quando_puoi`, e una funzione viva solo nei suoi test e' codice
    dormiente -- questo filtro resta perche' `rileva_quando_puoi` lo usa
    ancora."""
    fuori = []
    for volto in (risposta.get("volti") or []):
        if not isinstance(volto, dict):
            continue
        r = volto.get("rect")
        if not isinstance(r, list) or len(r) != 4:
            continue
        if not all(numeri.numero_finito(v) and numeri.intero_qt_utilizzabile(v)
                   for v in r):
            continue
        lmrks = _punti_utilizzabili(volto.get("landmarks"))
        if lmrks is None:
            continue
        fuori.append((r, lmrks))
    return fuori


class Servizio(object):
    def __init__(self, trasporto):
        self._trasporto = trasporto
        # Il motivo dell'ultimo guasto, o None se l'ultima richiesta e'
        # andata a buon fine. Serve a distinguere "nessun volto" (una
        # risposta legittima con volti=[]) da un guasto vero (pesi
        # mancanti, memoria esaurita): senza, i due casi tornano entrambi
        # [] da `rileva` e un guasto sistemico sparirebbe dentro una
        # sessione intera che sembra "nessun volto in ogni fotogramma" --
        # proprio il caso peggiore, perche' "nessun volto" e' la normalita'
        # di questa pagina (206 frame su 983 nel materiale dell'utente).
        self.ultimo_errore = None
        # Le ultime righe di stderr del figlio catturate AL MOMENTO del
        # guasto che ha valorizzato `ultimo_errore` -- non un puntatore
        # all'anello vivo del trasporto (quello non si svuota mai da solo,
        # `TrasportoAsincrono._MAX_RIGHE_STDERR`), altrimenti una richiesta
        # riuscita dopo un guasto lascerebbe il tooltip appiccicato a un
        # traceback vecchio. Si azzera insieme a `ultimo_errore`.
        self.ultimo_stderr = []

    def _stderr_del_trasporto(self):
        """Le righe di stderr del trasporto, o [] se il trasporto non le
        espone. Chiamata dai due percorsi di guasto qui sotto, entrambi
        raggiungibili da uno slot Qt (`rileva_quando_puoi` sempre,
        `landmark`/`frame`/`salva` quando l'utente sta correggendo a mano):
        un'eccezione in uno slot chiama qFatal e si porta via il processo
        con dentro ogni training aperto, quindi questo non deve MAI
        sollevare -- un trasporto finto nei test (o un canale futuro) che
        non implementi `stderr_recente` deve degradare a lista vuota."""
        ottieni = getattr(self._trasporto, "stderr_recente", None)
        if ottieni is None:
            return []
        try:
            return list(ottieni())
        except AttributeError:
            return []

    def _invia(self, comando):
        # M3 della revisione finale: un id proprio qui era vestigiale --
        # TrasportoAsincrono.invia_ultimo fa una COPIA di `comando` e lo
        # riscrive comunque col PROPRIO contatore prima di scriverlo sul
        # canale (era gia' cosi' quando il trasporto passava da sincrono
        # ad asincrono), quindi l'id assegnato qui non arrivava mai al
        # figlio. Due contatori dove ne basta uno: quello vero e' del
        # trasporto.
        risposta = self._trasporto.invia(comando)
        if not isinstance(risposta, dict):
            self.ultimo_errore = "risposta non valida dal servizio"
            self.ultimo_stderr = self._stderr_del_trasporto()
            return None
        if risposta.get("op") == "error":
            self.ultimo_errore = risposta.get("motivo")
            self.ultimo_stderr = self._stderr_del_trasporto()
            return None
        self.ultimo_errore = None
        self.ultimo_stderr = []
        return risposta

    def frame(self, path):
        risposta = self._invia({"op": "frame", "path": str(path)})
        if risposta is None:
            return None, None
        return risposta.get("raster"), risposta.get("shape")

    def landmark(self, centro, punta):
        risposta = self._invia({"op": "landmark",
                                "centro": [float(centro[0]), float(centro[1])],
                                "punta": [float(punta[0]), float(punta[1])]})
        if risposta is None:
            return None, None
        return risposta.get("rect"), _punti_utilizzabili(risposta.get("landmarks"))

    def salva(self, **campi):
        """Chiede al servizio di salvare il volto corrente.

        `face_idx` e' opzionale nel protocollo ma vale 0 se non lo si
        passa: due chiamate dallo stesso frame senza `face_idx` esplicito
        si sovrascrivono in silenzio, perche' il servizio nomina il file
        col nome del frame E l'indice del volto (`face_idx`), non con un
        contatore che tiene lui. Chi salva piu' volti dallo stesso frame
        deve passare `face_idx` a mano.
        """
        campi["op"] = "salva"
        risposta = self._invia(campi)
        return None if risposta is None else risposta.get("file")

    def rileva_quando_puoi(self, path, rect, face_type, accurato, quando_pronto,
                           rilevatore=None, allineatore=None, tieni_in_memoria=True):
        """I volti che il motore trova nel frame, o dentro `rect` se dato --
        `rect=None` e' il rilevamento automatico all'apertura del frame (il
        rilevatore cerca da solo), un `rect` esplicito e' il rettangolo che
        l'utente sta muovendo: si salta il rilevatore e si passa
        direttamente all'allineatore, che e' cio' che rende il gesto
        fluido.

        Non aspetta: passa da `invia_ultimo` del trasporto, non da `invia`
        (bloccherebbe l'interfaccia a ogni pressione di freccia), e
        consegna il risultato a `quando_pronto` quando (e se) arriva -- una
        richiesta superata da una piu' recente non consegna mai niente
        (TrasportoAsincrono, "l'ultimo vince").

        La callback riceve sempre una lista, vuota sia per "nessun volto"
        sia per un guasto (pesi mancanti, memoria esaurita): e' chiamata da
        uno slot, e uno slot che solleva chiama qFatal come un paintEvent.
        I due casi restano indistinguibili DA QUESTO VALORE DI RITORNO --
        chi vuole saperlo legge `self.ultimo_errore` subito dopo essere
        stato richiamato.

        `rilevatore`/`allineatore` sono le CHIAVI del registro
        (mainscripts/MotoriCatalog.py), non le etichette che la tendina
        mostra, e viaggiano sullo stesso comando di `face_type`: nessun
        canale nuovo. `None` = il default del registro, risolto dal figlio.
        `tieni_in_memoria=False` e' la spunta tolta nella barra: il figlio
        lascia vivi solo i due correnti e libera gli altri subito."""
        comando = {"op": "rileva", "path": str(path),
                  "rect": None if rect is None else [int(v) for v in rect],
                  "face_type": str(face_type), "accurato": bool(accurato),
                  "rilevatore": rilevatore, "allineatore": allineatore,
                  "tieni_in_memoria": bool(tieni_in_memoria)}

        def _su_risposta(risposta):
            if not isinstance(risposta, dict) or risposta.get("op") == "error":
                self.ultimo_errore = (risposta.get("motivo")
                                      if isinstance(risposta, dict)
                                      else "risposta non valida dal servizio")
                self.ultimo_stderr = self._stderr_del_trasporto()
                quando_pronto([])
                return
            self.ultimo_errore = None
            self.ultimo_stderr = []
            quando_pronto(_volti_utilizzabili(risposta))

        self._trasporto.invia_ultimo(comando, _su_risposta)

    def libera_altri(self, rilevatore, allineatore, face_type):
        """Chiede al figlio di lasciar andare ogni motore che non sia
        questa coppia. Non aspetta e non torna niente.

        E' l'operazione che rende vera la promessa della spunta ("the
        others are freed right away"): la politica viaggia anche sul
        comando `rileva`, ma `rileva` vuole un fotogramma corrente, e senza
        -- pellicola filtrata a vuoto, appena entrati in sessione -- non
        partiva niente e la VRAM restava occupata in silenzio. Qui non c'e'
        nessun fotogramma da nominare.

        La risposta si ignora, ma la callback deve esserci lo stesso e non
        deve poter sollevare: e' chiamata da uno slot Qt, e uno slot che
        solleva chiama qFatal e si porta via il processo con dentro ogni
        training aperto. `_su_risposta` qui non fa niente per costruzione:
        e' il modo piu' corto di garantirlo.
        """
        comando = {"op": "libera", "rilevatore": rilevatore,
                   "allineatore": allineatore, "face_type": str(face_type)}

        def _su_risposta(_risposta):
            return None

        self._trasporto.invia_ultimo(comando, _su_risposta)

    def ferma(self):
        self._trasporto.chiudi()
