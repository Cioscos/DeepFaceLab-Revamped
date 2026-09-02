"""Il client del protocollo di fusione. Non importa mainscripts (salvo il
catalogo dei codici, dati puri); il trasporto e' iniettabile.

Ogni metodo e' asincrono e passa da `invia_ultimo`: la pagina non deve
mai bloccarsi aspettando il pool. La callback riceve il dizionario di
risposta, o None su guasto -- e non solleva mai fuori di qui, perche' e'
chiamata da uno slot Qt."""
from core.interact import interact as io
from mainscripts import FusioneGuasti


def _e_il_guasto_sintetico(risposta):
    """Il canale morto risponde con un `error` senza `codice`, che porta
    il motivo grezzo del trasporto (condiviso con l'estrazione, senza nome
    di servizio). Import qui dentro, non in testa: `trasporto.py` importa
    `gui.estrazione.servizio` per `TIMEOUT_MS`, ed evitare un ciclo fra
    moduli vale piu' di qualche microsecondo per chiamata."""
    from gui.estrazione import trasporto as trasporto_mod
    return risposta.get("motivo") == trasporto_mod._RISPOSTA_GUASTO["motivo"]


class Servizio(object):
    def __init__(self, trasporto):
        self._trasporto = trasporto
        self.ultimo_errore = None
        self.ultimo_codice = None
        self.ultimo_stderr = []

    def _stderr(self):
        ottieni = getattr(self._trasporto, "stderr_recente", None)
        if ottieni is None:
            return []
        try:
            return list(ottieni())
        except Exception:
            return []

    def _invia(self, comando, quando_pronto, subito=False):
        def _su_risposta(risposta):
            if not isinstance(risposta, dict):
                # Nessuna risposta valida: il trasporto e' caduto senza
                # nemmeno consegnare un `error` -- e' il solo caso in cui
                # il client inventa un codice.
                self.ultimo_errore = "risposta non valida dal servizio"
                self.ultimo_codice = FusioneGuasti.SERVIZIO_INTERROTTO
                self.ultimo_stderr = self._stderr()
                esito = None
            elif risposta.get("op") == "error":
                # Un `error` che risponde a un COMANDO: `comando_sconosciuto`,
                # o un guasto imprevisto col solo motivo (`codice: null`). La
                # caduta di un processo del pool non passa di qui -- e' un
                # evento senza `id`, porta `client_caduto` e va a `su_evento`
                # della pagina, che non ferma la sessione per un frame.
                #
                # Il guasto sintetico del canale morto (`_RISPOSTA_GUASTO`)
                # e' anche lui un `error` senza `codice`: senza
                # riconoscerlo qui il motivo grezzo (in italiano, senza
                # nome di servizio) arriverebbe intatto fino allo schermo
                # -- `testi.fusione_guasto` sceglie la frase inglese giusta
                # solo se il codice viaggia.
                # Il codice del figlio viene PRIMA: un `error` vero che
                # portasse per caso quel motivo perderebbe il proprio.
                codice = risposta.get("codice")
                if codice is None and _e_il_guasto_sintetico(risposta):
                    codice = FusioneGuasti.SERVIZIO_INTERROTTO
                self.ultimo_codice = codice
                self.ultimo_errore = risposta.get("motivo")
                self.ultimo_stderr = self._stderr()
                esito = None
            else:
                self.ultimo_errore = None
                self.ultimo_codice = None
                self.ultimo_stderr = []
                esito = risposta
            try:
                quando_pronto(esito)
            except Exception as errore:
                io.log_err("servizio fusione: una callback ha sollevato: %s" % errore)
        if subito:
            self._trasporto.invia_subito(comando, _su_risposta)
        else:
            self._trasporto.invia_ultimo(comando, _su_risposta)

    def stato(self, quando_pronto):
        self._invia({"op": "stato"}, quando_pronto)

    def vai(self, idx, quando_pronto):
        self._invia({"op": "vai", "idx": int(idx)}, quando_pronto)

    def cfg(self, campi, quando_pronto):
        self._invia({"op": "cfg", "campi": dict(campi)}, quando_pronto)

    def propaga(self, verso, fino_a, quando_pronto):
        self._invia({"op": "propaga", "verso": verso, "fino_a": fino_a}, quando_pronto)

    def batch(self, acceso, quando_pronto):
        self._invia({"op": "batch", "acceso": bool(acceso)}, quando_pronto)

    def salva_sessione(self, quando_pronto):
        self._invia({"op": "salva_sessione"}, quando_pronto)

    def keyframe(self, idx, acceso, quando_pronto):
        self._invia({"op": "keyframe", "idx": int(idx), "acceso": bool(acceso)}, quando_pronto)

    def piano(self, interpola, anteprima, quando_pronto):
        self._invia({"op": "piano", "interpola": bool(interpola), "anteprima": bool(anteprima)},
                    quando_pronto)

    def sonda(self, n, quando_pronto):
        self._invia({"op": "sonda", "n": int(n)}, quando_pronto)

    def chiudi(self, quando_chiuso=None):
        """La chiusura garbata: spegne il batch, salva la sessione e
        risponde per ULTIMA. Va scritta SUBITO -- non dietro la richiesta in
        volo, che durante il caricamento del modello resta tale per minuti
        -- e chi la manda deve aspettare la risposta prima di uccidere il
        figlio, o il salvataggio non avviene affatto."""
        self._invia({"op": "chiudi"}, quando_chiuso or (lambda _r: None), subito=True)

    def ferma(self):
        self._trasporto.chiudi()
