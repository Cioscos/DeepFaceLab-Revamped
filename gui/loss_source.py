"""La storia della loss, da due sorgenti che non hanno lo stesso peso.

`<modello>_loss_history.csv` -- scritto a ogni salvataggio, una riga
per iterazione -- e' la verita', ma avanza a scatti: viene appeso ogni
venticinque minuti col valore di default. Gli eventi `iter` del canale
eventi sono vivi ma radi (al piu' due al secondo).

La regola e' una sola: i punti vivi riempiono soltanto la coda che il CSV non
ha ancora, e vengono buttati appena il CSV li raggiunge. Cosi' non si conta
mai due volte la stessa iterazione, ne' si disegnano due curve sovrapposte
leggermente diverse.

Qui e' anche dove i valori con cui non si puo' disegnare si fermano, ed e'
il posto giusto perche' e' quello dove entrano e vengono ricordati. Un
training che diverge scrive `nan` da entrambe le porte -- nel CSV, dove
`"%.6f" % nan` lascia la stringa `nan`, e sul canale eventi, dove
`json.dumps` lo scrive senza chiedere niente -- e da li' in poi quel valore
resta nella storia della loss per sempre. Un buco al suo posto (`None`)
invece di un punto: la serie accanto, che sta benissimo, continua a
disegnarsi, e `scartati` dice a chi guarda quanti valori sono spariti,
perche' una curva che si interrompe senza spiegazione e' peggio di nessuna
curva.
"""
from pathlib import Path

from gui.numeri import iterazione_utilizzabile, numero_finito


class LossSource(object):
    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self._iter = []          # iterazioni lette dal CSV
        self._serie = []         # una lista di loss per iterazione, allineata a _iter
        self._vivi = []          # (iterazione, loss) dagli eventi, oltre la fine del CSV
        self._pos = 0            # cursore in byte dentro il CSV
        self._resto = ""         # riga incompleta in attesa del suo newline
        self._ino = None
        #Il guasto dell'ultima lettura, se c'e' stato. Un file che non c'e'
        #ancora non e' un guasto -- e' ogni modello prima del primo
        #salvataggio; un file che c'e' e non si apre lo e'.
        self.errore = None
        #Quanti valori di loss sono diventati un buco perche' non erano
        #numeri finiti. Non e' una diagnostica: la scheda lo mette nella
        #riga di stato, o l'utente vedrebbe una curva bucata e nessuna
        #ragione.
        self.scartati = 0

    def _azzera(self):
        self._iter, self._serie = [], []
        self._vivi = []
        self._pos, self._resto, self._ino = 0, "", None
        self.scartati = 0

    def ricarica(self):
        """Legge la coda del CSV. True se qualcosa e' cambiato.

        Non solleva: chi la chiama puo' essere uno slot Qt, e un'eccezione
        dentro uno slot non risale al chiamante -- chiude il processo. Il
        guasto finisce in `self.errore`, che resta leggibile dopo.
        """
        self.errore = None
        try:
            stat = self.path.stat()
        except OSError:
            return False
        if self._ino is not None and (stat.st_ino != self._ino or stat.st_size < self._pos):
            # Il file e' stato riscritto (rollback a un checkpoint piu' vecchio) o ricreato:
            # un cursore rimasto indietro punterebbe in mezzo a una riga. Questo funziona
            # perche' il CSV viene sempre riscritto da capo solo quando si accorcia (cioe'
            # quando ha meno righe della storia in memoria); non viene mai ricreato
            # riscrivendo il prefisso e appendendo in seguito. Se questo cambiasse,
            # l'assunzione si romperebbe in silenzio.
            self._azzera()
        self._ino = stat.st_ino
        if stat.st_size == self._pos:
            return False
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                testo = f.read()
                self._pos = f.tell()
        except OSError as guasto:
            #Lo stat era riuscito e l'apertura no: permessi, oppure una
            #directory al posto del file. Il cursore in byte resta dov'era,
            #quindi la lettura successiva riprende da li'.
            self.errore = guasto
            return False
        testo = self._resto + testo
        righe = testo.split("\n")
        self._resto = righe.pop()
        cambiato = False
        for riga in righe:
            valori = self._parse(riga)
            if valori is None:
                continue
            self._iter.append(valori[0])
            self._serie.append(valori[1])
            cambiato = True
        if cambiato and self._iter:
            ultima = self._iter[-1]
            self._vivi = [v for v in self._vivi if v[0] > ultima]
        return cambiato

    def _parse(self, riga):
        """(iterazione, valori) o None se la riga non e' una riga.

        La distinzione fra le due uscite conta: `spazzatura`,
        l'intestazione, un campo che non e' affatto un numero e
        un'iterazione inutilizzabile **saltano la riga intera**, perche' non
        sono un training che diverge; una *loss* che c'e' ma non si puo'
        disegnare (`nan`, `inf`) diventa invece un buco al suo posto e
        lascia intatte le altre serie della stessa iterazione.
        """
        campi = riga.strip().split(",")
        if len(campi) < 3:
            return None
        try:
            iterazione = int(campi[0])
            valori = [float(v) for v in campi[2:]]
        except ValueError:
            return None      # l'intestazione e qualunque riga storta
        #L'iterazione e' l'identita' della riga, non uno dei suoi valori:
        #se non e' utilizzabile la riga sparisce invece di diventare un
        #buco. `int()` accetta qualunque grandezza -- e un `int` di 400
        #cifre non entra in un `float`, cosa che nel disegno e nella
        #consegna del CSV si pagava con il processo -- oltre a lasciar
        #passare i negativi, che sono innocui solo finche' nessuno se ne
        #serve per decidere quale punto viene dopo.
        if not iterazione_utilizzabile(iterazione):
            return None
        return iterazione, [self._disegnabile(v) for v in valori]

    def _disegnabile(self, valore):
        """Il valore, o None (un buco) se non e' un numero finito."""
        if numero_finito(valore):
            return float(valore)
        self.scartati += 1
        return None

    def aggiungi_vivo(self, iterazione, losses):
        """Aggiunge un punto vivo. **True se e' stato accettato**, False se no.

        L'esito torna al chiamante perche' il pannello tiene una seconda
        copia degli stessi punti -- gli serve per travasarli nella sorgente
        nuova quando la lettura del CSV arriva -- e due proprietari della
        stessa lista con due regole diverse non sono due copie: sono due
        storie. Un punto fuori ordine rifiutato qui e accettato la' tornava
        indietro al primo `hello`, cioe' molto dopo, quando nessuno lo lega
        piu' all'evento che l'aveva portato.
        """
        #L'iterazione e' l'asse delle x, e un NaN la' dentro e' peggio che
        #sulle y: passa i due controlli d'ordine qui sotto -- ogni confronto
        #con lui e' falso -- e arriva fino al calcolo dell'indice di colonna,
        #dove diventa l'indice stesso. Stessa regola della colonna `iter`
        #del CSV: una sola per le iterazioni, da qualunque porta entrino.
        #Chi chiama di solito l'ha gia' validata; questa classe non se lo
        #fa dire.
        if not iterazione_utilizzabile(iterazione):
            return False
        if self._iter and iterazione <= self._iter[-1]:
            return False
        if self._vivi and iterazione <= self._vivi[-1][0]:
            return False
        self._vivi.append((iterazione, [self._disegnabile(v) for v in losses]))
        return True

    def ultima_iterazione(self):
        if self._vivi:
            return self._vivi[-1][0]
        return self._iter[-1] if self._iter else None

    def punti(self, fino_a=None):
        """Iterazioni e serie, opzionalmente troncate a `fino_a` compreso."""
        iters = list(self._iter) + [v[0] for v in self._vivi]
        serie = list(self._serie) + [v[1] for v in self._vivi]
        if fino_a is None:
            return iters, serie
        taglio = len(iters)
        for indice, valore in enumerate(iters):
            if valore > fino_a:
                taglio = indice
                break
        return iters[:taglio], serie[:taglio]
