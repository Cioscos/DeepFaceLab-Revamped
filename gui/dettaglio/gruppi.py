"""I sette gruppi facciali dei 68 landmark: dati puri, niente Qt.

Sta qui e non nella tela perche' lo leggono in tre -- il disegno, la
selezione e i pulsanti delle aree -- e una copia per consumatore e' il
modo in cui questa famiglia di difetti nasce.

La tabella ricopia `landmarks_68_pt` e `draw_landmarks` di
facelib/LandmarksProcessor.py: `gui/` non puo' importare `facelib`, quindi
i sette gruppi vivono qui come dato ricopiato, non importato. Quattro
spezzate APERTE e tre CHIUSE, esattamente come le due chiamate a
`cv2.polylines` dell'originale.

**Indici e spezzata sono due cose diverse.** La spezzata del naso ripete
l'indice 30 in coda per chiudere l'arco delle narici sulla punta del
dorso -- nell'originale `np.concatenate((nose, [nose[-6]]))`, altrimenti
la punta resta appesa. E' un espediente di DISEGNO, non un punto
recuperato -- il commento `# missed one point` di
facelib/LandmarksProcessor.py induce in errore, perche' `slice(27, 36)`
il 30 ce l'ha gia'. Chi seleziona deve vedere il 30 UNA volta, o un
trascinamento di gruppo lo muove del doppio e un Ctrl+click lo aggiunge e
lo toglie nello stesso gesto. `test_i_gruppi_replicano_la_topologia_a_68_punti`
e' cio' che tiene le due tabelle allineate se l'originale cambia.
"""

# Spostata da gui/estrazione/tela.py, che ora la importa da qui: era
# l'unica definizione e resta l'unica. (nome, indici, chiusa)
GRUPPI_68 = (
    ("mascella",              tuple(range(0, 17)),          False),
    ("sopracciglio-destro",   tuple(range(17, 22)),         False),
    ("sopracciglio-sinistro", tuple(range(22, 27)),         False),
    ("naso",                  tuple(range(27, 36)) + (30,), False),
    ("occhio-destro",         tuple(range(36, 42)),         True),
    ("occhio-sinistro",       tuple(range(42, 48)),         True),
    ("bocca",                 tuple(range(48, 68)),         True),
)

NOMI = tuple(nome for nome, _indici, _chiusa in GRUPPI_68)

_INDICI = {nome: frozenset(indici) for nome, indici, _chiusa in GRUPPI_68}
_SPEZZATE = {nome: tuple(indici) for nome, indici, _chiusa in GRUPPI_68}
_DI_CHI = {i: nome for nome, indici in _INDICI.items() for i in indici}

# Scelti per contrasto sulla pelle e fra loro. Sono i PREDEFINITI: chi
# vuole altro li cambia dai pulsanti delle aree, e la scelta vive in
# QSettings (gui/dettaglio/colori.py).
COLORI_PREDEFINITI = {
    "mascella":              (120, 200, 255),
    "sopracciglio-destro":   (255, 200,  80),
    "sopracciglio-sinistro": (255, 150,  60),
    "naso":                  (120, 240, 140),
    "occhio-destro":         (255,  90,  90),
    "occhio-sinistro":       (255, 120, 180),
    "bocca":                 (200, 130, 255),
}

# facelib/LandmarksProcessor.py::get_transform_mat stima la similarita' con
#     umeyama(np.concatenate([image_landmarks[17:49], image_landmarks[54:55]]), ...)
# quindi trascinare un punto FUORI da questi non ruota ne' ritaglia niente.
# Riscritto a mano perche' gui/ non puo' importare facelib: se un giorno
# quella funzione cambiasse gli indici, NIENTE lo segnalerebbe qui.
INDICI_ALLINEAMENTO = frozenset(range(17, 49)) | {54}

# Con face_type `head` la stessa funzione chiama estimate_averaged_yaw, che
# legge anche questi per correggere l'offset orizzontale: su un volto head
# la mascella sposta il ritaglio, sugli altri no.
INDICI_YAW = frozenset({0, 1, 2, 14, 15, 16, 27, 28, 29})

# Solo "head", non "head_no_align": in get_transform_mat il ramo che
# chiama estimate_averaged_yaw e' un'uguaglianza stretta con FaceType.HEAD,
# quindi con head_no_align quei nove indici restano inerti e il volto si
# comporta come whole_face.
_HEAD = ("head",)


def indici_gruppo(nome):
    """Gli indici del gruppo, senza duplicati: e' cio' che si seleziona."""
    return _INDICI.get(nome, frozenset())


def spezzata_gruppo(nome):
    """Gli indici nell'ordine in cui si disegnano, duplicati compresi."""
    return _SPEZZATE.get(nome, ())


def gruppo_di(indice):
    """Il nome del gruppo a cui l'indice appartiene, o None."""
    return _DI_CHI.get(indice)


def indici_influenti(face_type):
    """Gli indici che, mossi, cambiano il ritaglio.

    `face_type` e' la STRINGA che DFLJPG salva ('whole_face', 'head', ...),
    non un FaceType: gui/ non puo' importare facelib. Un valore
    sconosciuto o None vale come non-head, che e' il caso in cui si
    promette MENO -- promettere di piu' farebbe apparire inerte un punto
    che invece muove il ritaglio, e quello e' il verso che confonde.
    """
    if face_type in _HEAD:
        return INDICI_ALLINEAMENTO | INDICI_YAW
    return INDICI_ALLINEAMENTO
