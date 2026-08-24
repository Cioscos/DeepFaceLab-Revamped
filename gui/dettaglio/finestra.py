"""La finestra del volto allineato: guardarlo, e ripararlo.

**Il client si INIETTA.** La finestra non ne costruisce uno suo: un
secondo processo figlio che importa torch mentre il primo e' vivo e'
l'errore che la revisione finale del ciclo precedente ha trovato e
corretto. `cliente=None` e' ammesso e lascia la finestra in sola lettura,
per lo strumento che ne fa gli scatti.

**Le frecce e la striscia guardano cose diverse, ed e' voluto.**
`imposta_ordine` riceve cio' che la PAGINA ha deciso -- l'intera griglia
dalla cura, i soli fratelli dall'estrazione -- e la finestra non sceglie
quale dei due. La striscia guarda una relazione sola, `source_filename`, e
la guarda sempre.

**La modifica si spegne in un posto solo**, `_aggiorna_modificabilita`.
Una regola applicata in due gestori finisce applicata in uno.

**Due spazi, e la verita' e' uno solo.** I `source_landmarks` sono in
pixel del FOTOGRAMMA e sono cio' che si salva; la tela mostra il ritaglio
ALLINEATO. Si proietta con `image_to_face_mat` per disegnare e si torna
indietro con la sua inversa una volta per trascinamento, convertendo i
SOLI indici che si sono mossi: il giro andata-ritorno non e' esatto in
virgola mobile, e riconvertire anche i punti fermi li farebbe derivare a
ogni gesto. La pila di undo e il confronto che decide se chiedere
l'anteprima vivono entrambi in spazio fotogramma, cioe' nello stesso
spazio di cio' che si invia.
"""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gui import testi, theme
from gui.dettaglio import gruppi as gruppi_mod
from gui.dettaglio import proiezione
from gui.dettaglio.colori import ColoriGruppi
from gui.dettaglio.fratelli import StrisciaFratelli
from gui.dettaglio.pannello import PannelloAree
from gui.dettaglio.storia import Storia
from gui.dettaglio.tela import Tela
from gui.faceset.griglia import MASCHERA_OFF, MODI_MASCHERA
from gui.numeri import intero_qt_utilizzabile
# Dati puri (import consentiti: collections, niente altro), ed e' l'unica
# cosa di mainscripts che gui/ importa -- come fa gia' la pagina di
# estrazione.
from mainscripts import MotoriCatalog

# I due modi di `rileva`, come li nomina il protocollo del servizio
# (mainscripts/FacesetDetail.py::_rispondi_rileva).
MODO_LANDMARKS = "landmarks"
MODO_VOLTO = "volto"


def _poligoni_utilizzabili(polys):
    """[(tipo, [(x, y), ...]), ...] dal campo `polys` del protocollo.

    La forma vera e' quella di `SegIEPolys.dump()`: un dizionario con una
    lista di poligoni, ognuno col suo `type` (1 include, 0 esclude) e i
    suoi `pts`. Tutto il resto -- `None`, una lista al posto del
    dizionario, un poligono che non e' un dizionario, dei punti che non
    sono punti -- si scarta invece di sollevare: questi dati arrivano da
    un altro processo, e il posto dove morirebbero e' un paintEvent.

    Il predicato e' `intero_qt_utilizzabile` e non il solo `numero_finito`:
    `1e300` e' finito, e muore dentro l'int() che consegna il punto a Qt.
    """
    if not isinstance(polys, dict):
        return []
    fuori = []
    for poligono in polys.get("polys") or ():
        if not isinstance(poligono, dict):
            continue
        punti = []
        for punto in poligono.get("pts") or ():
            try:
                x, y = punto
            except (TypeError, ValueError):
                continue
            if intero_qt_utilizzabile(x) and intero_qt_utilizzabile(y):
                punti.append((float(x), float(y)))
        if len(punti) < 2:
            continue
        tipo = poligono.get("type")
        fuori.append((tipo if isinstance(tipo, int) else 1, punti))
    return fuori


def _riempi_di_motori(tendina, motori):
    """Le voci di un selettore di motori: la `label` si mostra, la `key` si
    porta con `itemData`, e il `help` del catalogo diventa il suggerimento
    della singola voce -- senza, le due tendine mostrerebbero entrambe il
    suggerimento del pulsante e nessuno direbbe cosa distingue S3FD da
    RetinaFace. L'ordine e' quello del catalogo, che e' append-only."""
    for motore in motori:
        tendina.addItem(motore.label, motore.key)
        tendina.setItemData(tendina.count() - 1, motore.help,
                            QtCore.Qt.ToolTipRole)


def _indici_differenti(punti, altri):
    """Gli indici in cui due liste di punti non coincidono.

    Serve a decidere se il riallineamento vale il viaggio, quindi guarda
    solo l'uguaglianza: un punto illeggibile e' diverso da qualunque
    altro, che e' il verso che chiede un giro di troppo invece di
    saltarne uno necessario.
    """
    if altri is None or len(punti) != len(altri):
        return frozenset(range(len(punti)))
    return frozenset(i for i, punto in enumerate(punti)
                     if list(punto) != list(altri[i]))


class FinestraDettaglio(QtWidgets.QWidget):
    def __init__(self, workdir, cliente=None, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self._workdir = Path(workdir)
        self._cliente = cliente
        self._percorso = None
        self._ordine = []
        # Chi risolve i fratelli di un fotogramma: `fn(nome_frame) ->
        # [percorsi]`, vedi `imposta_risolutore_fratelli`. None per
        # default -- la striscia resta vuota, lo stesso ripiego di un
        # volto senza `source_filename`.
        self._risolutore_fratelli = None
        self._frame_dir = None
        self._dati = None
        self._storia = None
        self._raster_mostrato = None
        self._maschera_mostrata = None
        self._mat = None            # la 2x3 fotogramma -> allineato, gia' convalidata
        self._punti_allineati = []  # cio' che la tela mostra ORA, vedi _ricorda_i_punti_mostrati
        self._punti_riallineati = None
        self._colori = ColoriGruppi()
        # La scelta dell'utente sulla maschera, che sopravvive al cambio di volto.
        self._modo_maschera = MASCHERA_OFF
        self._motori_avvisati = False
        self._modo_rilevamento = None
        # Il filtro di appartenenza di `_su_pronto`: l'id della richiesta
        # `apri` che ha aperto il volto CORRENTE, e la bandierina che
        # copre il solo caso in cui il client consegna nello stesso giro
        # della chiamata (il `trasporto` sincrono dei test) -- li' la
        # risposta arriva PRIMA che `_id_apertura` sia assegnato.
        self._id_apertura = None
        self._attesa_apertura = False

        self.tela = Tela()
        self.pannello = PannelloAree(self._colori)
        self.striscia = StrisciaFratelli()
        # Due etichette e non una: il MOTIVO della sola lettura descrive il
        # volto aperto e vale finche' quel volto resta, i messaggi di
        # `etichetta_stato` durano quanto la richiesta che li ha prodotti.
        # Su una sola, un guasto qualunque -- e `fallito` porta anche quelli
        # delle richieste della pagina -- cancellava la spiegazione, che non
        # tornava piu' fino al cambio di volto: proprio l'utente che ha piu'
        # bisogno di leggerla la perdeva.
        self.etichetta_motivo = QtWidgets.QLabel("")
        self.etichetta_motivo.setWordWrap(True)
        self.etichetta_motivo.setVisible(False)
        self.etichetta_stato = QtWidgets.QLabel("")
        self.etichetta_stato.setWordWrap(True)
        self.spunta_landmark = QtWidgets.QCheckBox(testi.FACESET_LANDMARKS)
        self.selettore_maschera = theme.tendina()
        for chiave, etichetta in MODI_MASCHERA:
            self.selettore_maschera.addItem(etichetta, chiave)
        # I due pulsanti dei motori, con la tendina che sceglie il motore.
        # Elenco, ordine ed etichette vengono da MotoriCatalog e qui non si
        # riscrive nessuna voce: la tendina mostra la `label` e porta la
        # `key`, come fa il selettore della pagina di estrazione.
        self.tendina_allineatore = theme.tendina()
        _riempi_di_motori(self.tendina_allineatore, MotoriCatalog.ALLINEATORI)
        self.tendina_allineatore.setToolTip(testi.DETTAGLIO_RILEVA_LANDMARKS_TIP)
        self.bottone_landmarks = QtWidgets.QPushButton(testi.DETTAGLIO_RILEVA_LANDMARKS)
        self.bottone_landmarks.setToolTip(testi.DETTAGLIO_RILEVA_LANDMARKS_TIP)
        self.tendina_rilevatore = theme.tendina()
        _riempi_di_motori(self.tendina_rilevatore, MotoriCatalog.RILEVATORI)
        self.tendina_rilevatore.setToolTip(testi.DETTAGLIO_RILEVA_VOLTO_TIP)
        self.bottone_volto = QtWidgets.QPushButton(testi.DETTAGLIO_RILEVA_VOLTO)
        self.bottone_volto.setToolTip(testi.DETTAGLIO_RILEVA_VOLTO_TIP)
        self.bottone_disfa = QtWidgets.QPushButton(testi.DETTAGLIO_DISFA)
        self.bottone_rifa = QtWidgets.QPushButton(testi.DETTAGLIO_RIFA)
        self.bottone_revert = QtWidgets.QPushButton(testi.DETTAGLIO_REVERT)
        self.bottone_revert.setToolTip(testi.DETTAGLIO_REVERT_TIP)
        self.bottone_salva = QtWidgets.QPushButton(testi.DETTAGLIO_SALVA)
        self.bottone_salva.setToolTip(testi.DETTAGLIO_SALVA_TIP)

        scorrevole = QtWidgets.QScrollArea()
        scorrevole.setWidget(self.tela)
        scorrevole.setWidgetResizable(False)
        self._scorrevole = scorrevole   # per posizionare l'indicatore, sotto
        # L'attesa dell'apertura, ora che il client non blocca piu' il
        # thread della GUI: indeterminata (min e max a zero e' l'idioma
        # gia' in uso nel pacchetto per «sta lavorando e non so quanto ci
        # mette»). NON entra in nessun layout -- figlio diretto di
        # `scorrevole`, posizionato a mano (`_riposiziona_indicatore`) e
        # tenuto sopra con `raise_()`: un widget gestito da un layout
        # sposta i suoi vicini quando appare o sparisce, e la tela
        # scattava di 35 px in giu' e ritorno a ogni apertura -- lo stesso
        # difetto «sgraziato» che questo indicatore doveva togliere,
        # spostato dal bianco a uno scatto.
        self.indicatore_attesa = QtWidgets.QProgressBar(scorrevole.viewport())
        self.indicatore_attesa.setRange(0, 0)
        self.indicatore_attesa.setTextVisible(False)
        self.indicatore_attesa.setToolTip(testi.DETTAGLIO_ATTESA)
        self.indicatore_attesa.setVisible(False)

        # Anche la striscia scorre: il suo minimo cresce col numero di
        # fratelli (misurato: 150 px a due miniature, 1554 a venti), e
        # senza area scorrevole la larghezza minima della finestra
        # seguirebbe il fotogramma piu' affollato del progetto.
        self._scorrevole_fratelli = QtWidgets.QScrollArea()
        self._scorrevole_fratelli.setWidget(self.striscia)
        self._scorrevole_fratelli.setWidgetResizable(True)
        self._scorrevole_fratelli.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self._scorrevole_fratelli.setFixedHeight(
            self.striscia.LATO_MINIATURA + 24)
        self._scorrevole_fratelli.setVisible(False)

        barra = QtWidgets.QHBoxLayout()
        barra.addWidget(self.spunta_landmark)
        barra.addWidget(self.selettore_maschera)
        barra.addStretch(1)
        for b in (self.bottone_disfa, self.bottone_rifa,
                  self.bottone_revert, self.bottone_salva):
            barra.addWidget(b)
        # I quattro controlli dei motori su una riga PROPRIA, come nella
        # pagina di estrazione e per la stessa ragione misurata: dentro la
        # barra portavano la larghezza minima della finestra da 575 a 1191
        # px alla scala normale e da 690 a 1458 alla xlarge -- piu' di un
        # portatile 1366 -- e un QHBoxLayout non va a capo, schiaccia: la
        # prima cosa a sparire sarebbe proprio l'etichetta del pulsante.
        # Su una riga sua la stessa finestra sta in 632 / 718 / 784.
        riga_motori = QtWidgets.QHBoxLayout()
        for w in (self.tendina_allineatore, self.bottone_landmarks,
                  self.tendina_rilevatore, self.bottone_volto):
            riga_motori.addWidget(w)
        riga_motori.addStretch(1)
        centro = QtWidgets.QHBoxLayout()
        centro.addWidget(scorrevole, 1)
        centro.addWidget(self.pannello)
        fuori = QtWidgets.QVBoxLayout(self)
        fuori.addLayout(barra)
        fuori.addLayout(riga_motori)
        fuori.addLayout(centro, 1)
        fuori.addWidget(self._scorrevole_fratelli)
        fuori.addWidget(self.etichetta_motivo)
        fuori.addWidget(self.etichetta_stato)

        self.spunta_landmark.toggled.connect(self.mostra_landmark)
        self.selettore_maschera.currentIndexChanged.connect(self._su_modo_maschera)
        self.pannello.aree_cambiate.connect(self.tela.imposta_aree_attive)
        self.pannello.colori_cambiati.connect(self.tela.imposta_colori)
        self.tela.punti_mossi.connect(self._su_punti_mossi)
        self.tela.trascinamento_finito.connect(self._su_trascinamento_finito)
        self.striscia.scelto.connect(self.apri_volto)
        self.bottone_landmarks.clicked.connect(self._su_rileva_landmarks)
        self.bottone_volto.clicked.connect(self._su_rileva_volto)
        self.bottone_disfa.clicked.connect(self.disfa)
        self.bottone_rifa.clicked.connect(self.rifa)
        self.bottone_revert.clicked.connect(self.revert)
        self.bottone_salva.clicked.connect(self.salva)
        if self._cliente is not None:
            self._cliente.pronto.connect(self._su_pronto)
            self._cliente.volti_pronti.connect(self._su_volti_pronti)
            self._cliente.riallineato.connect(self._su_riallineato)
            self._cliente.salvato.connect(self._su_salvato)
            self._cliente.rilevato.connect(self._su_rilevato)
            self._cliente.fallito.connect(self._su_fallito)
        # Le frecce sono scorciatoie della FINESTRA e non un ramo di
        # keyPressEvent: la QScrollArea attorno alla tela prende il fuoco al
        # primo click sul volto -- il gesto per cui questa finestra esiste --
        # e da li' in poi si TIENE Left/Right per scorrere, quindi l'evento
        # non arriverebbe mai fin qui; con il fuoco su un pulsante la freccia
        # sposterebbe il fuoco. Una scorciatoia si consuma prima di
        # entrambi, misurata in tutti e tre gli stati. Si perde lo
        # scorrimento orizzontale da tastiera, che ha la barra e la rotella,
        # mentre sfogliare i volti e' cio' per cui la finestra si apre.
        for tasto, verso in ((QtCore.Qt.Key_Right, 1), (QtCore.Qt.Key_Left, -1)):
            scorciatoia = QtWidgets.QShortcut(QtGui.QKeySequence(tasto), self)
            scorciatoia.activated.connect(
                lambda passo=verso: self._sposta(passo))
        # Il pannello non annuncia niente alla costruzione -- nessun
        # ascoltatore c'e' ancora -- quindi le sue sette aree accese vanno
        # spinte a mano una volta: la tela nasce con nessuna area attiva e
        # senza questa riga disegnerebbe tutto smorzato mentre il pannello
        # mostra sette spunte accese.
        self.tela.imposta_colori(self.pannello.colori())
        self.tela.imposta_aree_attive(self.pannello.aree_attive())
        self.setWindowTitle(testi.FACESET_DETAIL_TITLE)
        self._aggiorna_selettore_maschera(False)
        # `_aggiorna_modificabilita` e non `_aggiorna_comandi`: senza volto
        # aperto non si modifica niente, e i due pulsanti dei motori
        # nascono accesi -- premerli chiederebbe di rilevare su un
        # percorso None. La tela nasce modificabile, ed e' la stessa riga
        # che la spegne.
        self._aggiorna_modificabilita()
        self._riposiziona_indicatore()

    #override
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._riposiziona_indicatore()

    def _riposiziona_indicatore(self):
        """La geometria dell'indicatore, ricalcolata a mano perche' non e'
        in nessun layout (vedi il commento alla costruzione): centrato
        sull'area visibile di `scorrevole`, vicino al bordo superiore."""
        area = self._scorrevole.viewport().rect()
        larghezza = max(40, min(200, area.width() - 16))
        x = max(0, (area.width() - larghezza) // 2)
        self.indicatore_attesa.setGeometry(x, 8, larghezza, 18)
        self.indicatore_attesa.raise_()

    # -- cio' che la pagina imposta -------------------------------------

    def percorso(self):
        return self._percorso

    def cliente(self):
        return self._cliente

    def frame_dir(self):
        return self._frame_dir

    def imposta_frame_dir(self, percorso):
        self._frame_dir = None if percorso is None else Path(percorso)
        self._aggiorna_modificabilita()

    def imposta_ordine(self, percorsi):
        """L'insieme che sfogliano le FRECCE, e la finestra non decide
        quale sia: l'intera griglia dalla pagina di cura, i soli fratelli
        da quella di estrazione. La striscia guarda un'altra cosa."""
        self._ordine = [Path(p) for p in percorsi]

    def imposta_risolutore_fratelli(self, fn):
        """Chi trova i fratelli di un fotogramma: `fn(nome_frame)` deve
        tornare i percorsi dei suoi allineati, se stesso compreso.

        Un CALLABLE e non una mappa gia' calcolata: le frecce cambiano
        volto, e il volto nuovo puo' venire da un altro fotogramma -- una
        mappa passata una volta sola sarebbe stantia alla prima freccia.
        `None` (il default) lascia la striscia vuota, per chi -- lo
        strumento degli scatti -- non ha nessun indice da interrogare.
        """
        self._risolutore_fratelli = fn

    # -- il volto mostrato ----------------------------------------------

    def mostra(self, percorso, dati):
        """DISEGNA `dati` sul volto `percorso`, e non decide niente: la
        pila delle modifiche riparte da qui, quindi chiamarla per cambiare
        volto e' il modo di perdere il lavoro in silenzio. Dall'esterno si
        passa da `apri_volto`; qui dentro ci arriva la risposta del
        servizio sul volto gia' aperto."""
        # Un messaggio di stato appartiene alla richiesta che lo ha
        # prodotto, e quella parlava del volto di prima: a cancellarlo era
        # `_aggiorna_modificabilita`, che ci scriveva sopra il motivo della
        # sola lettura -- ora il motivo ha un'etichetta sua.
        self.etichetta_stato.setText("")
        self._percorso = Path(percorso)
        self._dati = dati
        self._raster_mostrato = None
        self._maschera_mostrata = None
        self._mat = proiezione.matrice((dati or {}).get("mat"))
        maschera = self._maschera_da(dati)
        self.tela.imposta_volto(
            QtGui.QPixmap(str(percorso)),
            self._da_mostrare(dati),
            maschera,
            (dati or {}).get("face_type"),
            _poligoni_utilizzabili((dati or {}).get("polys")))
        self._ricorda_i_punti_mostrati()
        # La verita' e' in spazio FOTOGRAMMA: la pila la tiene li', perche'
        # e' li' che si salva e li' che si confronta.
        punti = [list(p) for p in ((dati or {}).get("source_landmarks") or [])]
        # La pila e' PER VOLTO: passando a un fratello si ricomincia. Un
        # Ctrl+Z che disfa una modifica fatta su un altro file e' la stessa
        # classe di errore dell'undo della cancellazione che apparteneva
        # alla pagina invece che alla cartella.
        self._storia = Storia(punti)
        self._punti_riallineati = punti
        self._aggiorna_selettore_maschera(maschera is not None)
        self._aggiorna_spunta_landmark(self._punti_allineati)
        self._chiedi_i_fratelli()
        self._aggiorna_modificabilita()
        self._aggiorna_comandi()
        self._aggiorna_titolo()

    def raster_mostrato(self):
        return self._raster_mostrato

    def maschera_mostrata(self):
        return self._maschera_mostrata

    # -- i due spazi ------------------------------------------------------

    def _da_mostrare(self, dati):
        """I punti per la TELA, cioe' in spazio allineato.

        Senza una matrice utilizzabile non si proietta niente -- e' il caso
        di FaceType.MARK_ONLY, e di una matrice singolare o non finita --
        ma il file porta gia' i suoi `landmarks` in spazio allineato: si
        mostrano quelli, e la modifica resta spenta da
        `_aggiorna_modificabilita`. Mostrare i punti del fotogramma sul
        ritaglio li metterebbe fuori dal riquadro.
        """
        sorgente = (dati or {}).get("source_landmarks") or []
        if self._mat is not None and sorgente:
            return proiezione.proietta(sorgente, self._mat)
        return (dati or {}).get("landmarks") or []

    def _ricorda_i_punti_mostrati(self):
        """La copia di cio' che la tela ha ORA, riletta da lei.

        E' il riferimento con cui il trascinamento successivo riconosce
        QUALI punti si sono mossi, e si rilegge invece di ricostruirla:
        ricalcolare la proiezione darebbe numeri diversi da quelli
        mostrati dopo un'anteprima, che porta i landmark calcolati dal
        servizio e non dalla nostra aritmetica.

        Si chiama dove i punti della tela vengono SOSTITUITI -- apertura,
        anteprima, undo -- e non dopo un trascinamento, che li muove sul
        posto: li' il riferimento resta indietro dei soli indici gia'
        convertiti, e riconvertirne uno con la stessa matrice da' lo
        stesso numero. Chiamarla anche li' sarebbe una riga che nessuna
        prova puo' distinguere.
        """
        self._punti_allineati = self.tela.punti()

    def _verita(self):
        """I punti in spazio FOTOGRAMMA: cio' che si salva, e cio' su cui
        si decide se si puo' modificare.

        NON quelli della tela. Senza `source_landmarks` -- che il servizio
        torna None apposta per un file scritto da una versione piu'
        vecchia -- la tela ripiega sui `landmarks` allineati, che sono 68
        lo stesso: decidere su quelli lascerebbe la finestra modificabile,
        il punto si muoverebbe sullo schermo, e il trascinamento sparirebbe
        senza che nessuno lo dica, perche' non c'e' niente da cui tornare
        indietro.
        """
        return [] if self._storia is None else self._storia.corrente()

    def _in_spazio_fotogramma(self, punti_allineati):
        """I punti del fotogramma dopo un trascinamento, o None se non si
        puo' tornare indietro.

        Si convertono i SOLI indici che si sono mossi: il giro
        andata-ritorno non e' esatto in virgola mobile (misurato: 700.0
        torna 699.9999999999999), e riconvertire anche i punti fermi li
        farebbe derivare a ogni gesto -- e farebbe sembrare "diverso" ogni
        punto a chi decide se l'anteprima serve.
        """
        indietro = proiezione.inversa(self._mat)
        if indietro is None or self._storia is None:
            return None
        fuori = self._storia.corrente()
        if len(punti_allineati) != len(fuori):
            return None
        for i in _indici_differenti(punti_allineati, self._punti_allineati):
            convertito = proiezione.proietta_punto(punti_allineati[i], indietro)
            # Un punto illeggibile non muove niente: quello di prima resta,
            # ed e' l'unico verso che non inventa una coordinata.
            if convertito is not None:
                fuori[i] = convertito
        return fuori

    def _maschera_da(self, dati):
        return self._immagine_della_workdir((dati or {}).get("mask"))

    def _maschera_anteprima(self, dati):
        """La maschera annunciata da `riallinea`: stesso posto, altra
        chiave. Il servizio la nomina solo dopo averla rinominata, quindi
        quando il nome arriva il file e' completo."""
        return self._immagine_della_workdir((dati or {}).get("maschera"))

    def _immagine_della_workdir(self, nome):
        if not nome:
            return None
        candidata = QtGui.QImage(str(self._workdir / str(nome)))
        return None if candidata.isNull() else candidata

    def _aggiorna_titolo(self):
        nome = "" if self._percorso is None else self._percorso.name
        self.setWindowTitle(testi.dettaglio_titolo(nome, self.modificata()))

    # -- gli interruttori della barra -------------------------------------

    def mostra_landmark(self, valore):
        """La spunta e' l'unico stato: chiamarla da fuori muove anche lei,
        o il comando e la tela direbbero due cose diverse. La seconda
        `setChecked` col valore che c'e' gia' non riemette niente, quindi
        il giro si chiude qui."""
        valore = bool(valore)
        self.spunta_landmark.setChecked(valore)
        self.tela.imposta_landmark_visibili(valore)

    def landmark_visibili(self):
        return self.spunta_landmark.isChecked()

    def imposta_modo_maschera(self, modo):
        """Come `mostra_landmark`: la tendina e' l'unico stato, quindi
        chiamarla da fuori muove anche lei. E la scelta si RICORDA: e'
        quella che il volto dopo eredita."""
        self._modo_maschera = modo
        self._mostra_modo_maschera(modo)

    def _mostra_modo_maschera(self, modo):
        """Porta tendina e tela su `modo` senza toccare la scelta
        ricordata: e' il transito per un volto che la maschera non ce l'ha
        -- o non ce l'ha ancora, perche' i dati arrivano dopo."""
        indice = self.selettore_maschera.findData(modo)
        if indice >= 0:
            bloccato = self.selettore_maschera.blockSignals(True)
            self.selettore_maschera.setCurrentIndex(indice)
            self.selettore_maschera.blockSignals(bloccato)
        self.tela.imposta_modo_maschera(modo)

    def _su_modo_maschera(self, indice_voce):
        modo = self.selettore_maschera.itemData(indice_voce)
        if modo is not None:
            self.imposta_modo_maschera(modo)

    def _aggiorna_selettore_maschera(self, c_e):
        """Il selettore vale per il volto mostrato ORA.

        Navigando con le frecce si passa da un volto con maschera a uno
        senza: lasciare il modo acceso mostrerebbe la maschera del volto
        precedente -- la tela la tiene finche' non gliene si da' un'altra
        -- cioe' la cosa peggiore che questa finestra possa fare.

        Ma la SCELTA dell'utente non si perde, e torna appena una maschera
        c'e': ogni cambio di volto passa da `mostra(percorso, None)`, che
        maschera non ne ha ancora, quindi spegnerla per sempre vorrebbe
        dire riaccendere il modo a ogni freccia.
        """
        self.selettore_maschera.setEnabled(c_e)
        self.selettore_maschera.setToolTip(
            testi.FACESET_MASK_TIP if c_e else testi.FACESET_NO_MASKS)
        self._mostra_modo_maschera(self._modo_maschera if c_e else MASCHERA_OFF)

    def _aggiorna_spunta_landmark(self, punti):
        """Stessa regola della maschera: un interruttore acceso su un
        volto che non ha landmark promette dei punti che non esistono."""
        self.spunta_landmark.setEnabled(bool(punti))
        self.spunta_landmark.setToolTip(
            testi.FACESET_LANDMARKS_TIP if punti else testi.FACESET_NO_LANDMARKS)

    # -- l'unico posto che decide se si puo' modificare -------------------

    def _aggiorna_modificabilita(self):
        """Un posto solo. La lezione del ciclo precedente e' che una
        regola applicata in due gestori finisce applicata in uno: li' le
        spunte delle sovrapposizioni entravano nella sessione manuale in
        entrambi i versi, perche' lo stato gia' applicato non veniva
        toccato da chi disabilitava il controllo."""
        motivo = ""
        if self._cliente is None or self._dati is None:
            si_puo = False
        elif self._frame_dir is None or self._dati.get("source_filename") is None \
                or not (self._frame_dir / str(self._dati["source_filename"])).exists():
            si_puo, motivo = False, testi.DETTAGLIO_SOLA_LETTURA_SENZA_FRAME
        elif self._mat is None:
            si_puo, motivo = False, testi.DETTAGLIO_SOLA_LETTURA_MATRICE
        elif len(self._verita()) != 68:
            si_puo, motivo = False, testi.DETTAGLIO_SOLA_LETTURA_PUNTI
        else:
            si_puo = True
        self.tela.imposta_modificabile(si_puo)
        # I motori seguono la stessa regola della tela, e nello stesso
        # posto: una proposta e' una modifica, quindi dove non si modifica
        # non si rileva nemmeno. Le tendine si spengono coi pulsanti e
        # cambiano suggerimento, come fa gia' il selettore della maschera:
        # una scelta che si puo' fare e non porta da nessuna parte e'
        # peggio di un controllo grigio che dice perche'.
        self.bottone_landmarks.setEnabled(si_puo)
        self.bottone_volto.setEnabled(si_puo)
        for tendina, suggerimento in (
                (self.tendina_allineatore, testi.DETTAGLIO_RILEVA_LANDMARKS_TIP),
                (self.tendina_rilevatore, testi.DETTAGLIO_RILEVA_VOLTO_TIP)):
            tendina.setEnabled(si_puo)
            tendina.setToolTip(suggerimento if si_puo
                               else testi.DETTAGLIO_MOTORI_SPENTI)
        # Nascosta quando non c'e' motivo: un'etichetta vuota si porterebbe
        # via una riga di altezza sotto ogni volto modificabile.
        self.etichetta_motivo.setText(motivo)
        self.etichetta_motivo.setVisible(bool(motivo))
        self._aggiorna_comandi()

    def _aggiorna_comandi(self):
        modificabile = self.tela.modificabile()
        self.bottone_salva.setEnabled(modificabile and self.modificata())
        self.bottone_revert.setEnabled(modificabile and self.modificata())
        self.bottone_disfa.setEnabled(
            modificabile and self._storia is not None and self._storia.puo_disfare())
        self.bottone_rifa.setEnabled(
            modificabile and self._storia is not None and self._storia.puo_rifare())

    # -- il trascinamento ------------------------------------------------

    def _su_punti_mossi(self, _punti):
        """Durante il gesto: SOLO il titolo. Nessuna richiesta al
        servizio -- una per `mouseMoveEvent` sommergerebbe il servizio di
        un comando per pixel spostato, anche con un client che non
        blocca piu' il thread della GUI."""
        self._aggiorna_titolo()

    def _su_trascinamento_finito(self, punti_allineati):
        punti = self._in_spazio_fotogramma(punti_allineati)
        if punti is None:
            return
        self._storia.applica(punti)
        self._aggiorna_comandi()
        self._aggiorna_titolo()
        self._chiedi_riallineamento(punti)

    def _chiedi_riallineamento(self, punti):
        """Il viaggio verso il servizio, se serve davvero.

        Sui punti che non muovono il ritaglio si SALTA: la mascella di un
        `whole_face` non entra in `get_transform_mat`, quindi il raster
        tornerebbe identico a quello che gia' si vede, e il viaggio non
        serve a niente anche se non blocca piu' l'interfaccia. Su un
        volto `head` gli stessi punti muovono il ritaglio, e allora si
        chiede: l'insieme non e' una costante, e' funzione del face type.

        Il confronto e' con i punti dell'ULTIMO invio, non con quelli
        dell'apertura: cosi' anche undo, redo e revert passano di qui e
        chiedono l'anteprima solo se il ritaglio puo' essere cambiato. E
        se non e' cambiato NIENTE non si chiede: e' raggiungibile -- un
        Undo dopo un trascinamento inerte riporta esattamente ai punti
        dell'ultimo invio -- e sarebbe un giro bloccante che per
        definizione non puo' cambiare il ritaglio.

        `punti` e' in spazio FOTOGRAMMA, come vuole il protocollo.
        """
        if self._cliente is None or not self.tela.modificabile():
            return
        influenti = gruppi_mod.indici_influenti((self._dati or {}).get("face_type"))
        if not (_indici_differenti(punti, self._punti_riallineati) & influenti):
            return
        self._punti_riallineati = [list(p) for p in punti]
        self._cliente.riallinea(self._percorso, self._frame_dir, punti)

    def _su_riallineato(self, dati):
        if self._storia is None:
            return
        raster = dati.get("raster")
        if raster:
            # La matrice torna NUOVA: il ritaglio e' cambiato, e da qui in
            # poi e' lei a legare i due spazi. Tenere la vecchia farebbe
            # disegnare ogni anteprima successiva con la trasformazione
            # sbagliata. Si adotta INSIEME ai pixel che descrive: una
            # risposta senza raster (oggi il servizio non ne manda --
            # un fallimento di codifica torna un errore) lascerebbe
            # altrimenti convertire i trascinamenti contro una matrice che
            # non corrisponde a cio' che si vede.
            mat = proiezione.matrice(dati.get("mat"))
            if mat is not None:
                self._mat = mat
            maschera = self._maschera_anteprima(dati)
            # `imposta_volto` azzera la selezione, che qui non e' cambiata:
            # senza rimetterla, trascinare due volte di fila lo stesso
            # gruppo sarebbe impossibile -- il secondo gesto ripartirebbe
            # da zero punti.
            selezione = self.tela.selezione()
            self.tela.imposta_volto(
                QtGui.QPixmap(str(self._workdir / str(raster))),
                self._punti_dell_anteprima(dati), maschera,
                (self._dati or {}).get("face_type"),
                _poligoni_utilizzabili((self._dati or {}).get("polys")))
            self.tela.imposta_selezione(selezione)
            self._ricorda_i_punti_mostrati()
            self._raster_mostrato = raster
            self._aggiorna_selettore_maschera(maschera is not None)
        self._maschera_mostrata = dati.get("maschera")
        self._aggiorna_comandi()

    def _punti_dell_anteprima(self, dati):
        """I landmark che il servizio ha calcolato sul ritaglio nuovo, gia'
        in spazio allineato: sono la sua aritmetica sul suo raster, e
        rifarli qui vorrebbe dire disegnare i punti di una trasformazione
        accanto ai pixel di un'altra. Se mancassero si proietta la verita'
        con la matrice nuova."""
        punti = dati.get("landmarks")
        corrente = self._storia.corrente()
        if isinstance(punti, list) and len(punti) == len(corrente):
            return punti
        return proiezione.proietta(corrente, self._mat)

    # -- i due pulsanti dei motori ---------------------------------------

    def _su_rileva_landmarks(self):
        """I landmark rifatti sul rettangolo che il volto ha GIA'. Il
        rilevatore non si manda: il servizio in questo modo non lo
        costruisce nemmeno, e mandarne uno lascerebbe credere il
        contrario a chi legge il protocollo."""
        self._rileva(MODO_LANDMARKS, None)

    def _su_rileva_volto(self):
        """Il rilevatore sull'intero fotogramma: puo' tornare un
        rettangolo diverso, piu' d'uno, o nessuno."""
        self._rileva(MODO_VOLTO, self.tendina_rilevatore.currentData())

    def _rileva(self, modo, rilevatore):
        """Il giro comune ai due pulsanti: l'avviso, e la richiesta.

        Nessuna guardia sul cliente o sul volto aperto, a differenza di
        `salva`, che e' pubblico: qui si arriva solo da due pulsanti che
        `_aggiorna_modificabilita` spegne dove non si modifica -- lo
        stesso posto che decide per la tela. Ripetere li' la regola
        sarebbe scriverla in due posti, ed e' cosi' che finisce applicata
        in uno.
        """
        # Il modo si ricorda perche' la risposta non lo porta, e le due
        # risposte vuote non vogliono dire la stessa cosa: in `volto` il
        # rilevatore ha cercato sul fotogramma e non ha trovato niente, in
        # `landmarks` il rettangolo c'era gia' ed e' l'allineatore a non
        # aver reso punti utilizzabili.
        self._modo_rilevamento = modo
        self._avvisa_motori_una_volta()
        # L'etichetta si RIDIPINGE subito: la richiesta e' asincrona e
        # torna prima che l'event loop giri da solo, quindi senza questa
        # riga l'avviso resterebbe accodato dietro la risposta invece di
        # comparire per primo. L'indicatore e' lo stesso che copre
        # l'apertura -- questo e' lo scambio piu' lento della finestra,
        # perche' ci sta dentro la costruzione dei motori.
        self.etichetta_stato.repaint()
        self.indicatore_attesa.setVisible(True)
        self._cliente.rileva(self._percorso, self._frame_dir, modo,
                             self.tendina_allineatore.currentData(), rilevatore)

    def _avvisa_motori_una_volta(self):
        """Il costo in memoria video, detto PRIMA della richiesta.

        I pesi si costruiscono al primo uso e restano caricati per i
        cinque minuti del sorvegliante d'inattivita': su una macchina che
        sta anche addestrando, quella e' memoria che il training non ha
        piu'. Dirlo dopo non servirebbe a niente -- e dirlo ogni volta
        coprirebbe le risposte, che nella stessa etichetta ci scrivono
        quante proposte sono arrivate.
        """
        if self._motori_avvisati:
            return
        self._motori_avvisati = True
        self.etichetta_stato.setText(testi.DETTAGLIO_MOTORI_IN_MEMORIA)

    def _su_rilevato(self, dati):
        """Le proposte dei motori: nessuna scrittura, un passo della pila.

        La prima si applica con `storia.applica`, cioe' come un
        trascinamento qualunque: e' un gesto che cambia i punti, quindi si
        disfa. Con piu' di una il numero si SCRIVE -- sceglierne una al
        posto di chi guarda e' gia' abbastanza, farlo senza dirlo e'
        peggio.

        La guardia sul volto aperto e' quella di `_su_salvato`, e per la
        stessa ragione: `rilevato` e' un segnale del CLIENT, che le pagine
        condividono, e un'eccezione in uno slot costa quanto una in un
        paintEvent.
        """
        self.indicatore_attesa.setVisible(False)
        if self._storia is None:
            return
        proposte = self._proposte_utilizzabili(dati)
        if not proposte:
            self.etichetta_stato.setText(
                testi.DETTAGLIO_NESSUN_LANDMARK
                if self._modo_rilevamento == MODO_LANDMARKS
                else testi.DETTAGLIO_NESSUNA_PROPOSTA)
            return
        if len(proposte) > 1:
            self.etichetta_stato.setText(testi.dettaglio_proposte(len(proposte)))
        self._storia.applica(proposte[0])
        self._applica_punti(proposte[0])

    def _su_fallito(self, motivo, codice=None):
        """Il guasto del servizio, scritto DOVE l'utente sta guardando.

        Senza questa riga un rilevamento caduto era completamente muto:
        `_rispondi_rileva` torna `op=error` per un `.npy` che manca, per
        il fotogramma assente e per un volto senza `source_rect`, e la
        finestra restava ferma sul testo di prima -- mentre dietro la
        pagina di cura scriveva che il file non ha dati DFL, cioe' una
        diagnosi sbagliata.

        Il testo lo sceglie il CODICE, non il motivo: quello che arriva
        dal servizio e' italiano d'implementazione e non e' una frase da
        leggere. `fallito` e' condiviso e porta anche i guasti di `open` e
        `frame`, che appartengono alle pagine, quindi la frase non presume
        quale richiesta sia caduta.

        `codice` ha un default perche' non ogni emettitore di questo
        segnale lo porta; il motivo puo' essere un'eccezione, una stringa
        o None -- il client emette tutte e tre le forme -- e nessuna deve
        sollevare dentro uno slot.

        Spegne anche l'indicatore di attesa, se acceso: `fallito` e'
        condiviso con le richieste della pagina, quindi non si sa SE il
        guasto riguardi l'apertura in corso -- ma un indicatore che resta
        acceso per sempre e' peggio di uno spento un giro troppo presto.
        """
        self.indicatore_attesa.setVisible(False)
        self.etichetta_stato.setText(testi.dettaglio_guasto(codice, motivo))

    def _proposte_utilizzabili(self, dati):
        """I soli `source_landmarks` che possono sostituire i punti aperti:
        una lista lunga quanto loro.

        Le proposte arrivano in JSON da un altro processo. Una senza punti
        -- o con un numero diverso -- non e' una proposta: infilarla nella
        pila lascerebbe uno stato che nessun altro metodo sa piu' leggere,
        a cominciare da `_aggiorna_modificabilita`, che di quei punti conta
        la lunghezza. Le coordinate DENTRO invece passano intatte: hanno
        gia' la rete della tela, un punto per volta, e scartarne una
        cambierebbe il significato di ogni indice dopo di lei.
        """
        quanti = len(self._storia.corrente())
        fuori = []
        for proposta in (dati.get("proposte") or ()):
            if not isinstance(proposta, dict):
                continue
            punti = proposta.get("source_landmarks")
            if isinstance(punti, list) and len(punti) == quanti:
                fuori.append(punti)
        return fuori

    # -- salvataggio, undo, revert ---------------------------------------

    def modificata(self):
        return self._storia is not None and self._storia.modificata()

    def salva(self):
        if self._cliente is None or not self.tela.modificabile():
            return
        # La pila tiene la verita' in spazio FOTOGRAMMA, che e' cio' che il
        # protocollo vuole: la tela mostra il ritaglio, non il fotogramma.
        self._cliente.salva(self._percorso, self._frame_dir, self._storia.corrente())

    def _su_salvato(self, dati):
        # `salvato` e' un segnale del client, cioe' condiviso come
        # `volti_pronti`: senza questa guardia una risposta che arriva
        # quando nessun volto e' aperto solleverebbe dentro uno slot, e
        # un'eccezione in uno slot costa quanto una in un paintEvent.
        if self._storia is None:
            return
        self._storia.segna_salvata()
        if dati.get("maschera") == "trasportata":
            self.etichetta_stato.setText(testi.DETTAGLIO_MASCHERA_DEGRADA)
        self._aggiorna_comandi()
        self._aggiorna_titolo()

    def disfa(self):
        if self._storia is None:
            return
        self._storia.disfa()
        self._applica_punti(self._storia.corrente())

    def rifa(self):
        if self._storia is None:
            return
        self._storia.rifa()
        self._applica_punti(self._storia.corrente())

    def revert(self):
        """Torna all'ultimo salvataggio, o all'apertura se non si e' mai
        salvato. E' un passo della pila come gli altri: si puo' disfare,
        o l'unico modo di annullare un Revert sbagliato sarebbe rifare a
        mano ogni trascinamento."""
        if self._storia is None:
            return
        punti = self._storia.stato_salvato()
        if punti is None:
            return
        self._storia.applica(punti)
        self._storia.segna_salvata()
        self._applica_punti(punti)

    def _applica_punti(self, punti):
        """Porta i punti sulla tela e chiede l'anteprima. `punti` e' in
        spazio FOTOGRAMMA, come tutto cio' che esce dalla pila: sulla tela
        ci arrivano proiettati. NON aggiunge un passo alla pila: chi chiama
        ha gia' deciso se e' un passo nuovo (una proposta dei motori) o uno
        gia' dentro (undo, redo, revert)."""
        self.tela.imposta_punti(proiezione.proietta(punti, self._mat)
                                if self._mat is not None else punti)
        self._ricorda_i_punti_mostrati()
        self._aggiorna_comandi()
        self._aggiorna_titolo()
        self._chiedi_riallineamento(punti)

    # -- navigazione ------------------------------------------------------

    def apri_volto(self, percorso):
        """L'UNICA via per cambiare volto: ci passano le frecce, la
        striscia e le due pagine.

        E' una sola perche' la domanda e il cambio non si separano.
        Quando le pagine chiamavano `mostra` e `cliente.apri` di fila,
        un click su un altro volto buttava via le modifiche vive senza
        chiedere niente; e proteggere il solo `mostra` peggiorava le cose,
        perche' il `cliente.apri` di dopo restava e portava i dati del
        volto nuovo sopra quello rimasto aperto -- plausibile, e
        sbagliato. Chi chiama tiene cio' che e' suo (l'ordine delle
        frecce, la cartella dei fotogrammi, il cursore d'attesa); cio'
        che non e' suo e' decidere SE cambiare volto.

        Torna True se la finestra mostra ORA `percorso` -- anche se ci
        era gia' -- e False se l'abbandono e' stato rifiutato.
        """
        percorso = Path(percorso)
        if percorso == self._percorso:
            return True
        if self.modificata() and not self._conferma_abbandono():
            # La striscia marca il bottone cliccato prima di sapere se il
            # volto si apre davvero: rifiutando l'abbandono va rimessa sul
            # volto che resta aperto, o mostrerebbe marcato un fratello
            # che non si sta guardando.
            self.striscia.imposta_fratelli(self.striscia.percorsi(),
                                           str(self._percorso))
            return False
        self.mostra(percorso, None)
        if self._cliente is not None:
            self.indicatore_attesa.setVisible(True)
            # Con un client ASINCRONO la risposta arriva a un giro
            # successivo dell'event loop: `_id_apertura` deve gia' essere
            # pronto per quando arriva. La bandierina copre il ramo
            # SINCRONO (il `trasporto` iniettato dei test): li' `apri`
            # consegna dentro questa stessa chiamata, cioe' PRIMA che
            # l'assegnazione qui sotto sia avvenuta.
            self._attesa_apertura = True
            try:
                self._id_apertura = self._cliente.apri(percorso)
            finally:
                self._attesa_apertura = False
        return True

    def _su_pronto(self, dati):
        """Filtro di appartenenza: si accetta solo la risposta della
        richiesta di apertura CORRENTE, riconosciuta per `id`.

        Con un client sincrono era impossibile distinguere una risposta
        in ritardo, perche' non ne arrivavano mai: reso asincrono, una
        `opened` rimasta indietro puo' arrivare DOPO che le frecce hanno
        gia' aperto un altro volto -- disegnarla sopra sarebbe la cosa
        peggiore che questa finestra possa fare. Il ramo della bandierina
        serve al `trasporto` sincrono dei test, dove la consegna avviene
        prima che `_id_apertura` sia stato scritto (vedi `apri_volto`).
        """
        if self._percorso is None:
            return
        if not (self._attesa_apertura or dati.get("id") == self._id_apertura):
            return
        self.indicatore_attesa.setVisible(False)
        self.mostra(self._percorso, dati)

    def _sposta(self, passo):
        if self._percorso is None or self._percorso not in self._ordine:
            return
        i = self._ordine.index(self._percorso) + passo
        if 0 <= i < len(self._ordine):
            self.apri_volto(self._ordine[i])

    def vai_avanti(self):
        self._sposta(1)

    def vai_indietro(self):
        self._sposta(-1)

    def _conferma_abbandono(self):
        """Metodo suo perche' i test lo sostituiscono: una QMessageBox in
        un test resterebbe aperta per sempre."""
        risposta = QtWidgets.QMessageBox.question(
            self, testi.DETTAGLIO_SALVA, testi.DETTAGLIO_ABBANDONA,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return risposta == QtWidgets.QMessageBox.Yes

    #override
    def closeEvent(self, event):
        """Chiudere e' l'altro modo di perdere il lavoro.

        Ci passano la croce della finestra, l'Esc e la chiusura
        dell'applicazione (`su_chiusura_scheda` chiude questa finestra,
        che e' una Qt.Window senza genitore). Senza modifiche vive non si
        chiede niente: una domanda in mezzo alla chiusura
        dell'applicazione, quando non c'e' niente da salvare, sarebbe una
        finestra che non si lascia chiudere.
        """
        if self.modificata() and not self._conferma_abbandono():
            event.ignore()
            return
        super().closeEvent(event)

    #override
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            # La prima Esc svuota la selezione, la seconda chiude: chi ha
            # appena preso venti punti col laccio vuole liberarli, non
            # perdere la finestra.
            if self.tela.selezione():
                self.tela.imposta_selezione(())
            else:
                self.close()
        elif event.key() == QtCore.Qt.Key_Z \
                and event.modifiers() & QtCore.Qt.ControlModifier:
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self.rifa()
            else:
                self.disfa()
        else:
            super().keyPressEvent(event)

    # -- i fratelli --------------------------------------------------------

    def _chiedi_i_fratelli(self):
        """I percorsi li trova il risolutore iniettato, non piu' il
        servizio: `ClienteDettaglio.volti_del_frame` vuole percorsi
        espliciti, e questa finestra non ha un indice da interrogare da
        se'. Senza risolutore, o senza fratelli per QUESTO fotogramma, la
        striscia resta vuota -- zero percorsi non vale un viaggio verso il
        servizio."""
        nome = (self._dati or {}).get("source_filename")
        if self._cliente is None or nome is None or self._percorso is None \
                or self._risolutore_fratelli is None:
            self._mostra_fratelli([], None)
            return
        percorsi = self._risolutore_fratelli(nome) or []
        if not percorsi:
            self._mostra_fratelli([], None)
            return
        self._cliente.volti_del_frame(percorsi)

    def _su_volti_pronti(self, dati):
        """ATTENZIONE: `volti_pronti` e' un segnale CONDIVISO. Il client e'
        lo stesso oggetto che la pagina usa, e `gui/estrazione/pagina.py`
        chiama `volti_del_frame` per conto suo -- quindi ogni risposta
        arriva a entrambi, e senza filtro la striscia si riempirebbe coi
        volti di un fotogramma che questa finestra non sta guardando.

        Il filtro e' l'appartenenza: si accetta solo una risposta che
        contiene il volto aperto. E' la stessa forma del controllo sull'id
        dentro `_chiedi`, e per la stessa ragione -- due richieste sullo
        stesso canale che non si distinguono si consegnano a vicenda.
        """
        percorsi = [v.get("path") for v in (dati.get("volti") or []) if v.get("path")]
        if self._percorso is None or str(self._percorso) not in percorsi:
            return
        self._mostra_fratelli(percorsi, str(self._percorso))

    def _mostra_fratelli(self, percorsi, corrente):
        """L'area scorrevole segue la striscia, che decide da se' se
        mostrarsi: la regola dei "meno di due volti" resta in un posto
        solo, e qui si legge soltanto la sua conclusione."""
        self.striscia.imposta_fratelli(percorsi, corrente)
        self._scorrevole_fratelli.setVisible(not self.striscia.isHidden())
