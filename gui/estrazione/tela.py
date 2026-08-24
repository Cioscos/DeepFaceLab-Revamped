"""La tela interattiva dell'estrazione manuale.

Il modello vero: un rettangolo che si POSIZIONA, si BLOCCA e si AFFINA.
Da sbloccato insegue il mouse (`_ricentra`), un click sinistro lo blocca; da
bloccato le frecce lo spostano di un pixel (`muovi`) e `+`/`-` lo
ridimensionano tenendo il centro (`ridimensiona`) -- il blocco ferma il
MOUSE, non il rettangolo, ed e' il passo prima dell'affinamento con le
frecce. Il calcolo dei landmark e la scrittura stanno dall'altra parte, nel
servizio (gui/estrazione/servizio.py).

**Il vettore e' un RIPIEGO**, per il frame su cui il rilevatore non aggancia
per niente: resta raggiungibile dal comando `vettore` (`V`), che era il
click destro della finestra `cv2` che questa tela sostituisce. Non
attraversa nessun processo, quindi non c'e' latenza da nascondere.

**Due spazi di coordinate, e confonderli e' un difetto silenzioso.** Il
frame arriva a risoluzione nativa (1920x1080 e' il caso normale) e la tela
e' larga quanto la scheda: il pixmap si scala per starci tutto dentro
(proporzioni conservate, centrato), quindi le coordinate del widget e
quelle del frame **non coincidono piu'**. Rect e landmark arrivano in
coordinate del FRAME e si portano a quelle del widget solo per disegnarli;
il vettore tracciato col mouse nasce in coordinate del WIDGET e si riporta
a quelle del frame prima di emetterlo -- il servizio salva il volto sul
frame a risoluzione piena, e un fattore dimenticato mette il rettangolo da
un'altra parte senza che niente lo segnali. E' la stessa scala che faceva
la finestra cv2 sostituita da questa tela (`view_scale` in
mainscripts/Extractor.py), che scalava anche in su.

Ogni numero che arriva da fuori passa da gui/numeri.py con ENTRAMBI i
predicati: numero_finito da solo non basta, 1e300 e' finito e uccide
comunque l'int() di un paintEvent. Un paintEvent che solleva chiama qFatal
e porta via il processo con dentro ogni training aperto -- e un'eccezione
in uno slot costa esattamente lo stesso. **Il controllo si ripete DOPO la
scala**, non solo sul dato in ingresso: un frame alto un pixel dentro una
tela alta 900 da' un fattore di 900, e 1e7 (che entra) diventa 9e9 (che
non entra piu' in un int a 32 bit).
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from gui import numeri
from gui.dettaglio.gruppi import GRUPPI_68
from gui.estrazione import volti as volti_mod

RAGGIO_PUNTO = 2
# Il pallino della mascella e' piu' grosso degli altri, come nel disegno
# originale (`cv2.circle(..., 2)` per la mascella, `1` per tutto il resto).
RAGGIO_MASCELLA = 2
RAGGIO_LANDMARK = 1


def gruppi_landmark(punti):
    """[(nome, punti_del_gruppo, chiusa), ...] in coordinate del FRAME.

    Vuota per qualunque numero di punti diverso da 68: FAN puo' produrne 98,
    e il rapporto per frame non porta landmark affatto. Collegare indici che
    non significano quello disegnerebbe una faccia inventata -- meglio i soli
    pallini, che restano veri qualunque sia il modello.
    """
    if len(punti) != 68:
        return []
    return [(nome, tuple(punti[i] for i in indici), chiusa)
            for nome, indici, chiusa in GRUPPI_68]
LATO_PREDEFINITO = 200     # il 100 di rect_size in Extractor.py e' un
                           # SEMI-lato: qui il lato intero e' il doppio.
LATO_MINIMO = 10
FRAZIONE_RIDIMENSIONA = 0.05


def _coppie_utilizzabili(punti):
    """Le coppie (x, y) su cui si puo' fare aritmetica, in coordinate del
    frame. Non si costruisce ancora nessun QPoint: la scala viene dopo, e
    con lei il solo posto in cui il limite dell'int a 32 bit conta."""
    if not punti:
        return []
    fuori = []
    for punto in punti:
        try:
            x, y = punto
        except (TypeError, ValueError):
            continue
        if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
            continue
        if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
            continue
        fuori.append((float(x), float(y)))
    return fuori


def _rects_utilizzabili(rect):
    """Zero, uno o piu' rettangoli (l, t, r, b) in coordinate del frame.

    Accetta una tupla sola -- come la passa la sessione manuale -- o una
    lista di tuple, che e' cio' che il rapporto porta per un frame con piu'
    volti. Un rettangolo illeggibile si salta senza portarsi via gli altri:
    ne basta uno storto in una voce scritta a meta' da uno Stop.
    """
    if not rect:
        return []
    candidati = rect if isinstance(rect, (list, tuple)) and rect and \
        isinstance(rect[0], (list, tuple)) else [rect]
    fuori = []
    for uno in candidati:
        try:
            l, t, r, b = uno
        except (TypeError, ValueError):
            continue
        if all(numeri.numero_finito(v) and numeri.intero_qt_utilizzabile(v)
               for v in (l, t, r, b)):
            fuori.append((float(l), float(t), float(r), float(b)))
    return fuori


def _dimensione_utilizzabile(dimensione, pixmap):
    """(larghezza, altezza) dello spazio in cui vivono rettangoli e
    landmark, o None se non c'e' un pixmap.

    Regge qualunque cosa arrivi da fuori e non solleva mai: da qui si passa
    dentro un paintEvent, e un'eccezione che ne risale chiama qFatal.
    """
    if pixmap is None or pixmap.isNull():
        return None
    if dimensione is not None:
        try:
            larghezza, altezza = dimensione
        except (TypeError, ValueError):
            larghezza, altezza = None, None
        if numeri.numero_finito(larghezza) and numeri.numero_finito(altezza) \
                and larghezza > 0 and altezza > 0:
            return float(larghezza), float(altezza)
    return float(pixmap.width()), float(pixmap.height())


def _punto_qt(x, y):
    """QPoint, non QPointF: drawEllipse su un QPointF resta in virgola
    mobile e un 1e300 ci passa dentro senza sollevare -- la guardia
    servirebbe a niente. QPoint(int(x), int(y)) e' il punto in cui PyQt5
    solleva OverflowError sopra i 2**31, cioe' il posto dove la seconda
    meta' del controllo (intero_qt_utilizzabile) diventa davvero
    necessaria e non solo dichiarata (vedi gui/dettaglio/tela.py).

    Torna None invece di sollevare: qui si arriva da un paintEvent."""
    if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
        return None
    if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
        return None
    return QtCore.QPoint(int(x), int(y))


class Tela(QtWidgets.QWidget):
    vettore_tracciato = QtCore.pyqtSignal(object, object)
    rettangolo_cambiato = QtCore.pyqtSignal(object)
    blocco_cambiato = QtCore.pyqtSignal(bool)
    # (percorso, (x, y)) in coordinate del FOTOGRAMMA. Il percorso e non il
    # Volto: chi ascolta e' la pagina, che deve aprire un file, e un
    # namedtuple che attraversa un segnale invita a tenerne una copia che
    # nessuno aggiorna. Il punto perche' il percorso puo' essere None --
    # "non ho volti caricati, non so rispondere" -- e in quel caso la
    # pagina deve poter ripetere la ricerca dopo averli chiesti.
    volto_scelto = QtCore.pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self._pixmap = None
        self._dim_frame = None
        self._rects = []
        self._punti = []
        self._centro = None
        self._punta = None
        self._bloccato = False
        self._modo_vettore = False
        self._sessione_manuale = False
        self._messaggio = ""
        # Livello DIVERSO da `_rects`: quelli vengono dal rapporto e sono
        # cio' che si disegna, questi vengono dal disco e sono cio' che si
        # puo' aprire. Nel caso normale coincidono; quando non coincidono
        # il rapporto e' piu' vecchio della cartella.
        self._volti = []
        self._mostra_rect = True
        self._mostra_landmarks = False
        self._mostra_maschera = False

    def mostra(self, pixmap, rect, landmarks, dimensione_frame=None):
        """`rect` e `landmarks` sono in coordinate del FRAME, come li
        produce il servizio: la scala la applica il disegno.

        `rect` e' una tupla (l, t, r, b) sola -- come la manda la sessione
        manuale -- o una lista di tuple, cio' che porta il rapporto per un
        frame con piu' volti.

        `dimensione_frame` e' (larghezza, altezza) di QUELLO spazio. None
        vuol dire «il pixmap E' il frame», che e' il caso della sessione
        manuale: il figlio scrive il raster a risoluzione nativa. La
        revisione invece mostra un'anteprima ridotta a LATO_ANTEPRIMA
        mentre il rapporto porta coordinate del frame intero, e finche' la
        scala si ricavava dal solo pixmap il rettangolo usciva spostato e
        ingrandito di 1,875x su un 1920x1080 (misurato il 2026-08-21).
        """
        self._pixmap = pixmap
        self._dim_frame = _dimensione_utilizzabile(dimensione_frame, pixmap)
        self._rects = _rects_utilizzabili(rect)
        self._punti = _coppie_utilizzabili(landmarks)
        self.update()

    def imposta_messaggio(self, testo):
        """Il testo dipinto al centro quando non c'e' niente da mostrare.
        Stringa vuota per spegnerlo. Non ha un gemello `messaggio()`: chi
        lo imposta e' la pagina, che sa gia' cosa gli ha dato."""
        testo = str(testo or "")
        if testo == self._messaggio:
            return
        self._messaggio = testo
        self.update()

    # -- il rettangolo della sessione manuale -------------------------------

    def imposta_rettangolo(self, rect):
        """Il setter di chi sta fuori: NON emette rettangolo_cambiato (vedi
        il docstring del modulo, la nota sull'anello)."""
        self._rects = _rects_utilizzabili(rect)
        self.update()

    def rettangolo(self):
        return self._rects[0] if len(self._rects) == 1 else None

    def imposta_landmarks(self, punti):
        self._punti = _coppie_utilizzabili(punti)
        self.update()

    def imposta_volti(self, volti):
        """I volti che gli allineati di questo fotogramma dichiarano, gia'
        normalizzati da gui/estrazione/volti.py e con la maschera gia'
        colorata."""
        self._volti = list(volti or ())
        self.update()

    def volti(self):
        return list(self._volti)

    def volto_sotto(self, pos):
        """Il volto che sta sotto un punto in coordinate del WIDGET, o
        None. Fa la conversione da se' perche' e' l'unico posto in cui il
        mouse incontra i volti, e farla fuori invita a dimenticarla."""
        if not self._volti:
            return None
        x, y = self._al_frame(pos.x(), pos.y())
        return volti_mod.volto_al_punto(self._volti, x, y)

    def punto_in_rapporto(self, punto):
        """`punto` (x, y) in coordinate del FRAME cade dentro almeno uno
        dei rettangoli che il rapporto ha registrato per questo fotogramma
        -- gli stessi `_rects` che il paintEvent disegna. Un click fuori da
        ognuno di loro e' un gesto neutro sul fondo del fotogramma: chi
        chiama non deve scrivere niente ne' chiedere niente al servizio.
        L'unico posto che sa fare questa aritmetica, per non duplicarla nel
        chiamante."""
        try:
            x, y = punto
        except (TypeError, ValueError):
            return False
        return any(l <= x <= r and t <= y <= b for l, t, r, b in self._rects)

    def imposta_sovrapposizioni(self, rect, landmarks, maschera):
        """Le tre spunte della revisione. Valgono per cio' che si DISEGNA,
        mai per cio' che ESISTE: il click apre il volto qualunque sia la
        combinazione accesa."""
        self._mostra_rect = bool(rect)
        self._mostra_landmarks = bool(landmarks)
        self._mostra_maschera = bool(maschera)
        self.update()

    def blocca(self, acceso):
        acceso = bool(acceso)
        if acceso == self._bloccato:
            return
        self._bloccato = acceso
        self.blocco_cambiato.emit(acceso)

    def bloccato(self):
        return self._bloccato

    def imposta_modo_vettore(self, acceso):
        self._modo_vettore = bool(acceso)
        # Uscire dal modo vettore a meta' trascinamento non deve lasciare la
        # linea appesa nel paintEvent -- ed entrarci con un centro stantio
        # da una sessione precedente sarebbe lo stesso difetto al contrario.
        self._centro = None
        self._punta = None
        self.update()

    def modo_vettore(self):
        return self._modo_vettore

    def imposta_sessione_manuale(self, acceso):
        """Fuori dalla sessione manuale (in revisione, o mentre il servizio
        non e' avviato) la tela e' una superficie di sola lettura: senza
        questo flag, il solo passaggio del mouse -- setMouseTracking(True)
        consegna mouseMoveEvent senza nessun click -- ricentrerebbe o
        creerebbe un rettangolo sopra gli N rettangoli di `mostra`, che la
        revisione deve poter solo guardare."""
        self._sessione_manuale = bool(acceso)

    def sessione_manuale(self):
        return self._sessione_manuale

    def passo_ridimensiona(self):
        """Proporzionale al lato, come il `diff` di Extractor.py: un passo
        fisso e' inutilmente fine su un volto grande e troppo grosso su uno
        piccolo."""
        r = self.rettangolo()
        if r is None:
            return float(LATO_PREDEFINITO) * FRAZIONE_RIDIMENSIONA
        return max(1.0, (r[2] - r[0]) * FRAZIONE_RIDIMENSIONA)

    def muovi(self, dx, dy):
        if not self._bloccato:
            return
        r = self.rettangolo()
        if r is None:
            return
        self._applica(self._dentro_al_frame((r[0] + dx, r[1] + dy,
                                             r[2] + dx, r[3] + dy)))

    def ridimensiona(self, delta):
        r = self.rettangolo()
        if r is None:
            return
        l, t, rr, b = r
        cx, cy = (l + rr) / 2.0, (t + b) / 2.0
        semi_x = max(LATO_MINIMO / 2.0, (rr - l) / 2.0 + delta)
        semi_y = max(LATO_MINIMO / 2.0, (b - t) / 2.0 + delta)
        self._applica(self._dentro_al_frame(
            (cx - semi_x, cy - semi_y, cx + semi_x, cy + semi_y)))

    def _applica(self, rect):
        """L'unico altro scrittore di _rects oltre a mostra/imposta_rettangolo
        -- deve passare dallo stesso validatore, altrimenti l'invariante
        «tutto cio' che sta in _rects ha passato gui/numeri.py» non vale piu'
        e rettangolo() consegnerebbe al chiamante (il servizio, poi
        salva_volto) un valore che imposta_rettangolo avrebbe rifiutato."""
        rects = _rects_utilizzabili(rect)
        if not rects:
            return
        self._rects = rects
        self.update()
        self.rettangolo_cambiato.emit(rects[0])

    def _dentro_al_frame(self, rect):
        """Trasla il rettangolo dentro i limiti del pixmap conservandone la
        misura -- deformarlo contro il bordo darebbe un ritaglio con
        proporzioni diverse da tutti gli altri della sessione. Senza pixmap
        non si ritaglia: non si inventa un limite. Se il rettangolo e' piu'
        grande del frame lo si lascia com'e' invece di schiacciarlo -- un
        ritaglio deformato e' peggio di uno che sborda, che salva_volto
        gestisce gia' da sempre."""
        if self._pixmap is None or self._pixmap.isNull():
            return rect
        l, t, r, b = rect
        larghezza, altezza = r - l, b - t
        limite_l, limite_a = float(self._pixmap.width()), float(self._pixmap.height())
        if larghezza <= limite_l:
            if l < 0.0:
                l, r = 0.0, larghezza
            elif r > limite_l:
                l, r = limite_l - larghezza, limite_l
        if altezza <= limite_a:
            if t < 0.0:
                t, b = 0.0, altezza
            elif b > limite_a:
                t, b = limite_a - altezza, limite_a
        return (l, t, r, b)

    def _ricentra(self, x, y):
        """Da sbloccato: il rettangolo insegue il mouse mantenendo la
        LARGHEZZA corrente come lato, o LATO_PREDEFINITO se non c'e' ancora
        nessun rettangolo -- il caso del frame su cui il rilevatore non
        aggancia, senza il quale non ci sarebbe modo di cominciare.

        Usa un solo lato per entrambi gli assi DI PROPOSITO: il rettangolo
        manuale e' sempre un quadrato, come il ritaglio di Extractor.py.
        Un rettangolo non quadrato (arrivato da imposta_rettangolo, o da
        una risposta del servizio) diventa quadrato al primo movimento del
        mouse -- non e' un difetto, e' la stessa forma che avrebbe avuto se
        tracciato qui da capo."""
        r = self.rettangolo()
        lato = (r[2] - r[0]) if r is not None else float(LATO_PREDEFINITO)
        semi = lato / 2.0
        self._applica(self._dentro_al_frame((x - semi, y - semi, x + semi, y + semi)))

    # -- la scala ----------------------------------------------------------

    # Un velo, non una toppa: sotto deve restare visibile il volto, che e'
    # cio' che si sta guardando.
    OPACITA_MASCHERA = 0.45

    def trasformazione_maschera(self, volto, lato_raster):
        """La QTransform che porta un pixel del raster della maschera nello
        spazio del widget, o None.

        QUATTRO spazi e TRE trasformazioni, e confonderli e' un difetto
        silenzioso (vedi il docstring del modulo, che ne dichiarava due):

            raster della maschera --(lato_allineato / lato_raster)-->
            spazio dell'allineato --(inversa dell'affine)-->
            spazio del fotogramma --(fattore, dx, dy)--> widget

        La prima scala NON e' pleonastica: DFLJPG.set_xseg_mask ricomprime
        la maschera a piacere (ripiega su JPEG sopra i 50 000 byte), quindi
        il raster non ha la dimensione dell'allineato.

        L'ordine di moltiplicazione di Qt e' "prima il sinistro", ed e'
        stato verificato eseguendo contro l'aritmetica manuale su tre
        punti -- non dedotto dalla documentazione.

        `volto.affine` porta gia' i sei argomenti nell'ordine di Qt e ha
        gia' escluso le matrici singolari: `inverted()` qui riesce sempre,
        e il suo flag si legge lo stesso perche' da qui si arriva a un
        paintEvent.
        """
        if volto.affine is None:
            return None
        trasformazione = self.trasformazione()
        if trasformazione is None:
            return None
        if not (numeri.numero_finito(lato_raster) and lato_raster > 0):
            return None
        lato_allineato = volto.lato_allineato
        if lato_allineato is None:
            # Una voce senza `shape`: disegnare la maschera alla scala del
            # proprio raster la metterebbe nel posto sbagliato con un
            # risultato plausibile, che e' peggio del non disegnarla.
            return None
        allineato_al_frame, invertibile = QtGui.QTransform(*volto.affine).inverted()
        if not invertibile:
            return None
        fattore, dx, dy = trasformazione
        s = float(lato_allineato) / float(lato_raster)
        return (QtGui.QTransform().scale(s, s)
                * allineato_al_frame
                * QtGui.QTransform().translate(dx, dy).scale(fattore, fattore))

    def _disegna_maschere(self, pittore):
        for volto in self._volti:
            if volto.maschera is None or volto.maschera.isNull():
                continue
            totale = self.trasformazione_maschera(
                volto, float(volto.maschera.width()))
            if totale is None:
                continue
            pittore.save()
            pittore.setOpacity(self.OPACITA_MASCHERA)
            pittore.setTransform(totale, False)
            pittore.drawImage(QtCore.QPointF(0.0, 0.0), volto.maschera)
            pittore.restore()

    def trasformazione(self):
        """(fattore, dx, dy) per andare dal frame al widget, o None.

        Il fattore non si tiene in un attributo e non si imposta da fuori:
        e' interamente determinato da self._dim_frame -- lo spazio in cui
        vivono rettangoli e landmark -- e dalla dimensione del widget, e un
        attributo sarebbe solo una copia da tenere in sincronia con un
        resizeEvent. Il pixmap ci entra solo come RIPIEGO: quando
        `dimensione_frame` passato a mostra() e' None, _dim_frame diventa la
        dimensione del pixmap perche' in quel caso e' lui il frame (la
        sessione manuale, che scrive il raster a risoluzione nativa).

        None quando la trasformazione non esiste -- nessun pixmap, pixmap
        nullo, nessuno spazio dichiarato, un lato a zero, il widget non
        ancora dimensionato. Chi chiama e' un paintEvent o uno slot: qui non
        si divide mai per zero.
        """
        if self._pixmap is None or self._pixmap.isNull() or self._dim_frame is None:
            return None
        larghezza_frame, altezza_frame = self._dim_frame
        if larghezza_frame <= 0 or altezza_frame <= 0:
            return None
        if self.width() <= 0 or self.height() <= 0:
            return None
        fattore = min(self.width() / float(larghezza_frame),
                      self.height() / float(altezza_frame))
        if not numeri.numero_finito(fattore) or fattore <= 0.0:
            return None
        dx = (self.width() - larghezza_frame * fattore) / 2.0
        dy = (self.height() - altezza_frame * fattore) / 2.0
        return fattore, dx, dy

    def _al_widget(self, x, y):
        t = self.trasformazione()
        if t is None:
            return float(x), float(y)
        fattore, dx, dy = t
        return x * fattore + dx, y * fattore + dy

    def _al_frame(self, x, y):
        """L'inversa. Senza di lei il rettangolo finirebbe in un punto
        diverso da quello indicato -- il difetto peggiorerebbe invece di
        risolversi, perche' oggi (frame piu' piccolo della tela) i due
        spazi coincidono per caso."""
        t = self.trasformazione()
        if t is None:
            return float(x), float(y)
        fattore, dx, dy = t
        return (x - dx) / fattore, (y - dy) / fattore

    # -- il mouse ----------------------------------------------------------

    def mousePressEvent(self, evento):
        if evento.button() != QtCore.Qt.LeftButton:
            return
        if self._modo_vettore:
            self._centro = evento.pos()
            self._punta = evento.pos()
            self.update()
        else:
            if not self._sessione_manuale:
                # In revisione la tela e' di sola lettura per il
                # RETTANGOLO, non per il click: qui il gesto apre il volto
                # che sta sotto, e non tocca ne' `_rects` ne' `_ricentra`.
                punto = self._al_frame(evento.pos().x(), evento.pos().y())
                volto = self.volto_sotto(evento.pos())
                if volto is not None:
                    self.volto_scelto.emit(volto.percorso, punto)
                elif not self._volti:
                    # Nessun volto CARICATO -- non "nessun volto sotto il
                    # mouse". La tela non puo' rispondere e chiede a chi
                    # puo': la pagina li recuperera' e ripetera' la ricerca
                    # su questo punto. Coi volti in mano, invece, un click
                    # sul fondo resta un gesto neutro.
                    self.volto_scelto.emit(None, punto)
                return
            self.blocca(not self._bloccato)

    def mouseMoveEvent(self, evento):
        if self._modo_vettore:
            if self._centro is not None:
                self._punta = evento.pos()
                self.update()
            return
        # setMouseTracking(True) consegna questo evento senza nessun click:
        # in revisione (sessione manuale spenta) deve restare un no-op, o il
        # solo passaggio del mouse cancellerebbe gli N rettangoli del
        # rapporto (_ricentra sostituisce _rects con uno solo).
        if not self._sessione_manuale:
            # Il cursore dice dove si puo' cliccare. Cambia solo se i volti
            # sono gia' stati caricati: nessun recupero al semplice
            # passaggio del mouse, che farebbe partire il servizio mentre
            # si scorre la pellicola.
            self.setCursor(QtCore.Qt.PointingHandCursor
                           if self.volto_sotto(evento.pos()) is not None
                           else QtCore.Qt.ArrowCursor)
            return
        if self._bloccato:
            return
        self._ricentra(*self._al_frame(evento.pos().x(), evento.pos().y()))

    def mouseReleaseEvent(self, evento):
        if not self._modo_vettore:
            return
        if evento.button() == QtCore.Qt.LeftButton and self._centro is not None:
            centro = self._al_frame(self._centro.x(), self._centro.y())
            punta = self._al_frame(evento.pos().x(), evento.pos().y())
            self._centro = None
            self._punta = None
            self.update()
            self.vettore_tracciato.emit(centro, punta)

    def wheelEvent(self, evento):
        # M1 della revisione finale: fuori dalla sessione manuale (in
        # revisione) la tela e' di sola lettura, come mousePressEvent e
        # mouseMoveEvent gia' impongono -- ma wheelEvent non consultava
        # `_sessione_manuale` e ridimensionava comunque il rettangolo del
        # rapporto sotto il mouse. Nessun dato si perde (rettangolo() non
        # e' definito con N rettangoli mostrati), ma la sovrapposizione
        # disegnata smette di corrispondere al rapporto. M2 gemello: da'
        # finalmente un chiamante a `sessione_manuale()`, che non ne aveva
        # nessuno.
        if not self.sessione_manuale():
            return
        # y() == 0 e' uno scroll ORIZZONTALE (trackpad, o una rotella
        # inclinabile): senza questa uscita cadrebbe nel ramo negativo e
        # rimpicciolirebbe invece di restare inerte.
        if evento.angleDelta().y() == 0:
            return
        segno = 1.0 if evento.angleDelta().y() > 0 else -1.0
        self.ridimensiona(segno * self.passo_ridimensiona())

    # -- il disegno --------------------------------------------------------

    def _disegna_landmark(self, pittore, punti):
        """I landmark collegati, come li disegna DFL da sempre.

        Un volto e' fatto di spezzate, non di una nuvola di puntini: senza i
        collegamenti non si vede se la bocca e' chiusa o se un occhio e'
        finito su una guancia, che e' proprio cio' che si guarda mentre si
        corregge un fotogramma a mano.

        Ogni segmento si salta se uno dei due estremi non e' utilizzabile
        DOPO la scala, invece di saltare il gruppo intero: un fattore grande
        puo' portare fuori scala un punto solo, e perdere l'intera mascella
        per quello sarebbe peggio del buco. Qui si arriva da un paintEvent,
        quindi non si solleva mai.

        `punti` arriva da fuori: e' `self._punti` per i landmark vivi della
        sessione manuale, oppure `volto.punti` per un allineato del
        rapporto -- lo stesso disegno serve a entrambi.
        """
        pittore.setPen(QtGui.QPen(QtGui.QColor(250, 220, 90), 1))
        for _nome, gruppo, chiusa in gruppi_landmark(punti):
            qpunti = [_punto_qt(*self._al_widget(x, y)) for x, y in gruppo]
            coppie = list(zip(qpunti, qpunti[1:]))
            if chiusa and len(qpunti) > 2:
                coppie.append((qpunti[-1], qpunti[0]))
            for uno, altro in coppie:
                if uno is not None and altro is not None:
                    pittore.drawLine(uno, altro)
        # I pallini restano, e sono l'unica cosa disegnata quando i punti non
        # sono 68 -- vedi gruppi_landmark, che li' torna vuota.
        mascella = set(GRUPPI_68[0][1]) if len(punti) == 68 else set()
        for indice, (x, y) in enumerate(punti):
            punto = _punto_qt(*self._al_widget(x, y))
            if punto is None:
                continue
            raggio = (RAGGIO_MASCELLA if indice in mascella
                      else RAGGIO_LANDMARK if mascella else RAGGIO_PUNTO)
            pittore.drawEllipse(punto, raggio, raggio)

    def paintEvent(self, _evento):
        pittore = QtGui.QPainter(self)
        pittore.fillRect(self.rect(), self.palette().window())
        trasformazione = self.trasformazione()
        if trasformazione is None and self._messaggio:
            # Solo senza fotogramma: sopra un frame sarebbe peggio del
            # vuoto che rimpiazza.
            pittore.setPen(QtGui.QPen(self.palette().windowText().color()))
            pittore.drawText(self.rect(),
                             QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                             self._messaggio)
        if trasformazione is not None:
            fattore, dx, dy = trasformazione
            # Senza questo il ridimensionamento di un 1080p e' a vicino piu'
            # prossimo: sui capelli e sui bordi del volto e' proprio cio'
            # che serve guardare per decidere dove tracciare il vettore.
            pittore.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            larghezza_frame, altezza_frame = self._dim_frame
            destinazione = QtCore.QRect(
                int(dx), int(dy),
                max(1, int(larghezza_frame * fattore)),
                max(1, int(altezza_frame * fattore)))
            pittore.drawPixmap(destinazione, self._pixmap)
        if self._volti and self._mostra_maschera:
            # Sotto tutto il resto: e' un velo sul volto, e rettangoli e
            # landmark devono restare leggibili sopra di lei.
            self._disegna_maschere(pittore)
        if self._rects and self._mostra_rect:
            pittore.setPen(QtGui.QPen(QtGui.QColor(90, 200, 250), 2))
            for l, t, r, b in self._rects:
                alto_sinistra = _punto_qt(*self._al_widget(l, t))
                basso_destra = _punto_qt(*self._al_widget(r, b))
                if alto_sinistra is not None and basso_destra is not None:
                    pittore.drawRect(QtCore.QRect(alto_sinistra, basso_destra))
        if self._punti:
            # I landmark VIVI della sessione manuale: vengono dal motore,
            # non dal disco, e le tre spunte non li governano -- nasconderli
            # combatterebbe il lavoro che si sta facendo.
            self._disegna_landmark(pittore, self._punti)
        if self._volti and self._mostra_landmarks:
            for volto in self._volti:
                if volto.punti:
                    self._disegna_landmark(pittore, volto.punti)
        if self._centro is not None and self._punta is not None:
            # Gia' in coordinate del widget: nascono dal mouse e muoiono al
            # rilascio, non passano mai dal frame.
            pittore.setPen(QtGui.QPen(QtGui.QColor(250, 120, 90), 1))
            pittore.drawLine(self._centro, self._punta)
        pittore.end()
