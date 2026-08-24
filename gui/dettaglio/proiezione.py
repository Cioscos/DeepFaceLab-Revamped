"""La 2x3 fra spazio fotogramma e spazio allineato, e il suo inverso.

**Niente Qt e niente numpy.** Una trasformazione affine e' aritmetica
elementare -- `x' = a*x + b*y + c` -- e `gui/` non importa numpy da
nessuna parte: non e' il posto per cominciare. Sta qui e non nella
finestra per la stessa ragione di `gruppi` e `selezione`: e' la parte che
puo' sbagliare in modi che a schermo si notano solo dopo, e si prova senza
un QApplication.

**Perche' esiste.** La verita' di un volto sono i suoi `source_landmarks`,
in pixel del FOTOGRAMMA; la tela mostra il ritaglio ALLINEATO, 1:1.
`image_to_face_mat` e' il ponte, e si attraversa una volta per verso: si
proietta per disegnare, si torna indietro con l'inversa una volta per
trascinamento. La strada opposta -- tenere la verita' in spazio allineato
e invertire al salvataggio -- farebbe passare i punti dall'inversa a OGNI
ciclo modifica-salva-modifica, e siccome la matrice viene ricalcolata
proprio da quei punti l'errore entrerebbe in un anello di retroazione.

**Niente solleva.** La matrice arriva in JSON da un altro processo, e chi
chiama sta gia' proteggendo un paintEvent o uno slot.
"""
from gui.numeri import intero_qt_utilizzabile

# Sotto questo determinante la matrice non si inverte: e' la stessa soglia
# del processo figlio (mainscripts/ExtractorLib.py). Le due condizioni --
# determinante troppo piccolo e coefficiente non finito -- coprono classi
# disgiunte, quindi servono entrambe.
DETERMINANTE_MINIMO = 1e-12


def matrice(grezza):
    """`(a, b, c, d, e, f)` da `[[a, b, c], [d, e, f]]`, o None.

    None anche per una matrice **che non si inverte**, benche' scritta
    bene: questo ponte serve in entrambi i versi -- si proietta per
    disegnare e si torna indietro per salvare -- e accettarne una di sola
    andata lascerebbe trascinare i punti senza che il trascinamento arrivi
    da nessuna parte, cioe' il guasto silenzioso peggiore. Chi la riceve
    None mostra i `landmarks` che il file porta gia' in spazio allineato e
    spegne la modifica, dicendolo.

    Il predicato e' `intero_qt_utilizzabile`, che comincia gia' da
    `numero_finito`: un coefficiente da 1e300 e' finito, moltiplica una
    coordinata e la porta fuori dalla firma a 32 bit di Qt dentro un
    paintEvent. Un `image_to_face_mat` vero sta in poche unita' di scala e
    in qualche migliaio di pixel di traslazione, quindi il tetto non
    esclude niente di reale.
    """
    try:
        righe = list(grezza)
    except TypeError:
        return None
    if len(righe) != 2:
        return None
    valori = []
    for riga in righe:
        try:
            coefficienti = list(riga)
        except TypeError:
            return None
        if len(coefficienti) != 3:
            return None
        for v in coefficienti:
            if not intero_qt_utilizzabile(v):
                return None
            valori.append(float(v))
    mat = tuple(valori)
    return mat if inversa(mat) is not None else None


def inversa(mat):
    """La matrice che riporta indietro, o None se non esiste.

    None anche quando esiste in teoria ma i suoi coefficienti non sono
    consegnabili: un determinante appena sopra la soglia produce numeri
    enormi, e la soglia da sola non basterebbe.

    Convalida anche cio' che RICEVE, e non solo cio' che produce: in
    produzione ci arriva sempre l'uscita di `matrice()`, che il tetto lo
    ha gia' applicato, ma questa funzione e' pubblica e si prova da sola --
    una regola che vale solo per chi passa dalla porta principale non e'
    una regola.
    """
    try:
        a, b, c, d, e, f = mat
    except (TypeError, ValueError):
        return None
    for v in (a, b, c, d, e, f):
        if not intero_qt_utilizzabile(v):
            return None
    det = a * e - b * d
    if abs(det) < DETERMINANTE_MINIMO:
        return None
    fuori = (e / det, -b / det, (b * f - c * e) / det,
             -d / det, a / det, (c * d - a * f) / det)
    for v in fuori:
        if not intero_qt_utilizzabile(v):
            return None
    return fuori


def proietta_punto(punto, mat):
    """`[x, y]` trasformato, o None se il punto non e' due numeri su cui
    si possa fare aritmetica."""
    if mat is None:
        return None
    try:
        x, y = punto
    except (TypeError, ValueError):
        return None
    if not (intero_qt_utilizzabile(x) and intero_qt_utilizzabile(y)):
        return None
    a, b, c, d, e, f = mat
    return [a * x + b * y + c, d * x + e * y + f]


def proietta(punti, mat):
    """Una NUOVA lista, un punto per punto.

    Un punto che non si puo' proiettare passa INTATTO invece di sparire:
    toglierlo accorcerebbe la lista e cambierebbe il significato di ogni
    indice dopo di lui, cioe' quale punto appartiene a quale gruppo. Chi
    disegna ha gia' la sua rete, un punto per volta.
    """
    fuori = []
    for punto in punti or ():
        convertito = proietta_punto(punto, mat)
        fuori.append(punto if convertito is None else convertito)
    return fuori
