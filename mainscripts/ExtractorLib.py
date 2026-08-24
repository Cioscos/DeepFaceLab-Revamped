"""La parte dell'estrazione che i due frontali condividono.

Nessuna UI e nessun `io`: la finestra cv2 di Extractor.py e il servizio
ExtractManual.py chiamano queste funzioni, e sono le stesse. E' lo stesso
taglio gia' fatto da TrainerLib.py sul loop del trainer e da SorterLib.py
sui verbi del sorting, e per la stessa ragione: cio' che si estrae diventa
cio' che si puo' provare senza aprire niente.
"""
import math
import os

import cv2
import numpy as np
import numpy.linalg as npla

from core import mathlib
from core.cv2ex import cv2_imwrite
from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor
from mainscripts import ExtractReport

# La coda del file temporaneo del salvataggio: non e' fra le estensioni di
# `core.pathex`, quindi un temporaneo rimasto orfano non viene raccolto da
# nessuna strada a valle.
SUFFISSO_TEMPORANEO = ".dfltmp"


def landmarks_da_vettore(centro, punta):
    """I 68 punti sintetizzati da un vettore tracciato a mano.

    `centro` e' il punto in cui l'utente ha premuto, `punta` quello in cui
    si trova il puntatore: lunghezza e angolo del vettore danno scala e
    rotazione del template. landmarks_2D ha 51 punti, i 17 zeri davanti
    sono la mandibola che il template non porta.

    Con un vettore di lunghezza nulla non c'e' ne' scala ne' angolo:
    si torna il rettangolo degenere e nessun landmark, mai un'eccezione.
    """
    x, y = float(centro[0]), float(centro[1])
    pt1 = np.float32([x, y])
    pt2 = np.float32([float(punta[0]), float(punta[1])])

    pt_vec = pt2 - pt1
    pt_vec_len = npla.norm(pt_vec)

    rect = (int(x - pt_vec_len), int(y - pt_vec_len),
            int(x + pt_vec_len), int(y + pt_vec_len))

    if pt_vec_len == 0:
        return rect, None

    pt_vec = pt_vec / pt_vec_len

    lmrks = np.concatenate((np.zeros((17, 2), np.float32),
                            LandmarksProcessor.landmarks_2D), axis=0)
    lmrks -= lmrks[30:31, :]
    mat = cv2.getRotationMatrix2D((0, 0),
                                  -np.arctan2(pt_vec[1], pt_vec[0]) * 180 / math.pi,
                                  pt_vec_len)
    mat[:, 2] += (x, y)
    return rect, LandmarksProcessor.transform_points(lmrks, mat).astype(np.float32)


def salva_volto(immagine, rect, image_landmarks, face_type, image_size,
                jpeg_quality, output_filepath, source_filename, manuale=False):
    """Scrive il volto allineato e i suoi metadati. Torna il percorso, o
    None se il volto e' stato scartato.

    Lo scarto e' quello di sempre: se l'area dei landmark supera quattro
    volte l'area del rettangolo del rilevatore, l'allineamento e' andato
    fuori strada. Non si applica ai volti tracciati a mano (`manuale`),
    perche' li' il rettangolo lo ha deciso l'utente. La matrice si calcola
    una volta sola, qui dentro.
    """
    rect = np.array(rect)

    if face_type == FaceType.MARK_ONLY:
        image_to_face_mat = None
        face_image = immagine
        face_image_landmarks = image_landmarks
    else:
        image_to_face_mat = LandmarksProcessor.get_transform_mat(
            image_landmarks, image_size, face_type)
        face_image = cv2.warpAffine(immagine, image_to_face_mat,
                                    (image_size, image_size), cv2.INTER_LANCZOS4)
        face_image_landmarks = LandmarksProcessor.transform_points(
            image_landmarks, image_to_face_mat)

        if not manuale and face_type <= FaceType.FULL_NO_ALIGN:
            landmarks_bbox = LandmarksProcessor.transform_points(
                [(0, 0), (0, image_size - 1), (image_size - 1, image_size - 1),
                 (image_size - 1, 0)], image_to_face_mat, True)
            rect_area = mathlib.polygon_area(
                np.array(rect[[0, 2, 2, 0]]).astype(np.float32),
                np.array(rect[[1, 1, 3, 3]]).astype(np.float32))
            landmarks_area = mathlib.polygon_area(
                landmarks_bbox[:, 0].astype(np.float32),
                landmarks_bbox[:, 1].astype(np.float32))
            if landmarks_area > 4 * rect_area:
                return None

    cv2_imwrite(output_filepath, face_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    dflimg = DFLJPG.load(output_filepath)
    dflimg.set_face_type(FaceType.toString(face_type))
    dflimg.set_landmarks(np.asarray(face_image_landmarks).tolist())
    dflimg.set_source_filename(source_filename)
    dflimg.set_source_rect(rect)
    dflimg.set_source_landmarks(np.asarray(image_landmarks).tolist())
    dflimg.set_image_to_face_mat(image_to_face_mat)
    dflimg.save()
    return output_filepath


def _face_type_da(valore):
    """Il face type come FaceType, accettando anche la stringa che DFLJPG
    salva ('whole_face', 'head', ...).

    Le due forme arrivano da due strade -- il codice di estrazione ha gia'
    l'enum, il protocollo del servizio di dettaglio ha la stringa che ha
    letto dal file -- e convertire in un posto solo evita che ognuna delle
    due se lo converta a modo suo.
    """
    if isinstance(valore, str):
        return FaceType.fromString(valore)
    return valore


def riallinea_volto(immagine, source_landmarks, face_type, image_size):
    """(raster, matrice 2x3, landmark allineati) da un fotogramma e i suoi
    landmark IN COORDINATE DEL FOTOGRAMMA.

    E' la stessa catena di `salva_volto`, senza il disco e senza i
    metadati: serve sia all'anteprima durante il trascinamento sia al
    salvataggio, che di suo aggiunge solo la scrittura.

    Il controllo di scarto di `salva_volto` NON si replica: quello ferma un
    allineatore automatico andato fuori strada, e qui i punti li ha messi
    una persona -- lo stesso motivo per cui `salva_volto` lo salta gia'
    quando `manuale=True`.
    """
    face_type = _face_type_da(face_type)
    punti = np.array(source_landmarks, dtype=np.float32)
    mat = LandmarksProcessor.get_transform_mat(punti, image_size, face_type)
    raster = cv2.warpAffine(immagine, mat, (image_size, image_size),
                            cv2.INTER_LANCZOS4)
    return raster, mat, LandmarksProcessor.transform_points(punti, mat)


def composta_fra_allineamenti(mat_vecchia, mat_nuova):
    """La 2x3 che porta dal VECCHIO spazio allineato al NUOVO.

    Entrambe le matrici vanno da fotogramma ad allineato, quindi la
    composizione e' `mat_nuova @ inversa(mat_vecchia)`. Si promuovono a 3x3
    per moltiplicarle e si riprendono le prime due righe.

    Solleva se la vecchia non e' invertibile: una matrice singolare non ha
    un verso inverso, e restituire dei numeri comunque darebbe una maschera
    trasportata in un posto plausibile e sbagliato -- il difetto peggiore
    disponibile qui. `cv2.invertAffineTransform` non solleva da sola: su una
    matrice degenere torna zeri finiti, non NaN/inf, quindi il controllo va
    fatto prima e su due fronti -- il determinante della parte lineare (la
    singolare vera) e la finitezza (i landmark tutti coincidenti, che
    `get_transform_mat` puo' produrre, tornano una matrice tutta NaN senza
    sollevare). La soglia sul determinante e' assoluta perche' la scala
    tipica qui e' (image_size / estensione del volto)^2: sulla fixture di
    prova vale ~0.02, e resta ordini di grandezza sopra 1e-12 anche per un
    volto minuscolo su un frame 8K con image_size=64.
    """
    mat_vecchia = np.array(mat_vecchia, dtype=np.float32)
    if (not np.all(np.isfinite(mat_vecchia))
            or abs(np.linalg.det(mat_vecchia[:, :2])) < 1e-12):
        raise ValueError("la matrice di partenza non e' invertibile")
    inversa = cv2.invertAffineTransform(mat_vecchia)
    def _a_3x3(m):
        fuori = np.eye(3, dtype=np.float32)
        fuori[:2, :] = np.array(m, dtype=np.float32)
        return fuori
    return (_a_3x3(mat_nuova) @ _a_3x3(inversa))[:2, :]


def trasporta_maschera(dflimg, composta, lato_allineato):
    """Riwarpa la maschera XSeg dal vecchio spazio allineato al nuovo.

    Torna True se c'era una maschera e l'ha trasportata, False se non
    c'era. I poligoni NON passano di qui: sono punti, e si trasformano in
    modo esatto (vedi `riallinea_e_salva`).

    Il lato del raster si LEGGE, non si assume. `DFLJPG.set_xseg_mask`
    ricomprime ma non ridimensiona, quindi il lato e' quello di chi l'ha
    scritta -- la risoluzione del modello XSeg, non quella dell'allineato.
    mainscripts/FacesetResizer.py assume 256 e su una maschera di lato
    diverso sposterebbe la segmentazione.

    Assume la maschera quadrata (usa `maschera.shape[0]` per entrambi i
    lati): `DFLJPG.get_xseg_mask` non lo garantisce, ma tutte le maschere
    che questo ciclo scrive sono quadrate (XSegNet lavora su input
    quadrati).
    """
    if not dflimg.has_xseg_mask():
        return False
    maschera = np.asarray(dflimg.get_xseg_mask())
    lato_maschera = maschera.shape[0]
    # La catena e' maschera -> allineato -> (composta) -> allineato ->
    # maschera, con S = lato_allineato / lato_maschera all'andata e 1/S al
    # ritorno. Facendo i conti su un punto p in coordinate della maschera:
    #     (1/S) * ( A*(S*p) + b )  =  A*p + b/S
    # cioe' la PARTE LINEARE resta identica (le due scale si elidono su di
    # lei) e solo la TRASLAZIONE va divisa per S. Non e' una
    # semplificazione estetica: scalare anche A darebbe una maschera
    # ruotata in modo plausibile, che e' il difetto peggiore disponibile
    # qui perche' non solleva niente.
    giu = lato_maschera / float(lato_allineato)
    mat = np.array(composta, dtype=np.float32).copy()
    mat[:, 2] *= giu
    fuori = cv2.warpAffine(maschera, mat, (lato_maschera, lato_maschera),
                           flags=cv2.INTER_LANCZOS4)
    # LANCZOS4 sovraoscilla: senza la soglia la maschera esce con valori
    # sopra 1 e sotto 0, e non e' piu' binaria. Stessa riga di
    # FacesetResizer.
    fuori[fuori < 0.5] = 0
    fuori[fuori >= 0.5] = 1
    dflimg.set_xseg_mask(fuori)
    return True


def riallinea_e_salva(percorso_allineato, immagine_frame, source_landmarks):
    """Riscrive un volto allineato coi landmark modificati, PORTANDOSI
    DIETRO la maschera e i poligoni.

    Torna {"mat": [[...]], "landmarks": [[x, y], ...],
           "maschera": "trasportata"|"assente", "poligoni": <quanti>}.

    NON passa da `salva_volto`, e non e' una svista: quella riscrive il
    JPEG da zero e ricostruisce i metadati campo per campo, senza mai
    chiamare set_xseg_mask ne' set_seg_ie_polys -- su un volto mascherato
    cancella la maschera in silenzio. La ricetta giusta e' quella di
    mainscripts/FacesetResizer.py: get_dict prima, set_dict dopo, e i
    campi che cambiano sovrascritti sopra.
    """
    percorso_allineato = str(percorso_allineato)
    dflimg = DFLJPG.load(percorso_allineato)
    if dflimg is None:
        raise ValueError("non e' un JPEG DFL: %s" % (percorso_allineato,))
    mat_vecchia = dflimg.get_image_to_face_mat()
    if mat_vecchia is None:
        raise ValueError("questo volto non ha una matrice di allineamento")
    face_type = dflimg.get_face_type()
    lato = int(dflimg.get_shape()[0])
    dfl_dict = dflimg.get_dict()

    raster, mat_nuova, lmrks = riallinea_volto(
        immagine_frame, source_landmarks, face_type, lato)
    if not np.all(np.isfinite(mat_nuova)):
        # composta_fra_allineamenti controlla solo mat_vecchia: landmark
        # tutti coincidenti (l'utente trascina ogni punto sullo stesso
        # pixel) fanno tornare a get_transform_mat una mat_nuova tutta NaN
        # senza sollevare. Il controllo va fatto qui, PRIMA di scrivere
        # qualunque cosa su disco: una composta NaN riscriverebbe il file
        # del progetto con un raster e dei metadati insensati.
        raise ValueError("i landmark forniti non producono un allineamento valido")
    composta = composta_fra_allineamenti(mat_vecchia, mat_nuova)

    # Scrittura atomica: tutto il lavoro (raster, poligoni, maschera,
    # landmark) va su un file temporaneo accanto, e l'originale si
    # sostituisce solo alla fine con os.replace. Fra il warp del raster e
    # l'ultimo set_* il file e' incompleto -- niente landmark, niente
    # image_to_face_mat, niente maschera -- e set_xseg_mask puo' sollevare
    # di suo: scrivere in-place lascerebbe il volto allineato dell'utente
    # (i cui source_landmarks non vivono altrove) irrecuperabile se il
    # processo si interrompe in quella finestra.
    # E il temporaneo NON finisce per un'estensione di `core.pathex`: un
    # `00000.tmp.jpg` dentro `aligned/` e' un volto DFL che
    # `get_image_paths` raccoglie, e un'interruzione fra `save()` e
    # `os.replace` (SIGKILL, OOM, corrente) lo lascerebbe li' come
    # duplicato silenzioso del volto vero, dentro l'addestramento.
    percorso_tmp = percorso_allineato + SUFFISSO_TEMPORANEO
    try:
        # I byte si codificano a mano perche' `cv2_imwrite` deduce il
        # formato dal suffisso del NOME: su `.dfltmp` `imencode` fallirebbe
        # e la scrittura verrebbe saltata in silenzio. Il buffer e' lo
        # stesso che scriveva prima, JPEG di qualita' 100.
        riuscita, buffer = cv2.imencode(".jpg", raster,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if not riuscita:
            raise ValueError("il raster riallineato non si codifica in JPEG")
        with open(percorso_tmp, "wb") as f:
            f.write(buffer)
        dflimg = DFLJPG.load(percorso_tmp)
        dflimg.set_dict(dfl_dict)

        polys = dflimg.get_seg_ie_polys()
        quanti = 0
        for poly in polys.get_polys():
            # Punti, non pixel: la trasformazione e' ESATTA, e lo resta
            # quante volte si voglia. E' l'unica parte del lavoro manuale
            # sulle maschere che non si degrada mai.
            poly.set_points(LandmarksProcessor.transform_points(poly.get_pts(), composta))
            quanti += 1
        dflimg.set_seg_ie_polys(polys)

        trasportata = trasporta_maschera(dflimg, composta, lato)

        dflimg.set_landmarks(np.asarray(lmrks).tolist())
        dflimg.set_source_landmarks(np.asarray(source_landmarks, dtype=np.float32).tolist())
        dflimg.set_image_to_face_mat(mat_nuova)
        dflimg.save()
        os.replace(percorso_tmp, percorso_allineato)
    except Exception:
        if os.path.exists(percorso_tmp):
            os.remove(percorso_tmp)
        raise
    return {"mat": np.asarray(mat_nuova).astype(float).tolist(),
            "landmarks": np.asarray(lmrks).astype(float).tolist(),
            "maschera": "trasportata" if trasportata else "assente",
            "poligoni": quanti}


def voce_da_data(data, luminanza, motore=None):
    """Da un ExtractSubprocessor.Data alla voce di rapporto.

    `n_volti` conta i volti **davvero scritti su disco**, non i rilevamenti
    che hanno dei landmark: `salva_volto` ne scarta una parte (la regola
    dell'area, sopra), e `data.faces_detected` -- il numero che la riga di
    comando stampa in fondo, e quello che l'altro produttore dello stesso
    rapporto ottiene contando i file (`ExtractIndex._volti_per_frame`) --
    conta gia' i soli scritti. Con la conta dei rilevamenti, un frame il cui
    unico volto e' stato scartato leggeva `n_volti=1` dopo l'estrazione e
    `n_volti=0` dopo una ricostruzione dell'indice, e **sfuggiva al filtro
    «senza volto»**, che e' la ragione per cui la pagina esiste.

    Chi lo sa e' `final_stage`, che scrive gli indici salvati su
    `data.indici_salvati` mentre salva. `None` significa «quello stadio non
    e' passato di qui» -- non «nessuno salvato» -- e allora si ricade sui
    rilevamenti con landmark: in produzione non succede mai, perche' il
    rapporto si scrive solo dopo uno stadio 'final' o 'all'.

    Il lato e' il maggiore fra larghezza e altezza del rettangolo: e' cio'
    che serve al filtro "volto piccolo", ed e' confrontabile fra frame di
    risoluzione diversa solo insieme alla dimensione del frame, che il
    lettore ha gia'.

    La posa va stimata nello spazio ALLINEATO, non in quello del frame:
    LandmarksProcessor.estimate_pitch_yaw_roll si aspetta i landmark del
    volto gia' allineato (e' cosi' che la usa Sample.get_pitch_yaw_roll),
    non quelli grezzi nello spazio sorgente -- passarle questi ultimi da'
    una posa sbagliata in silenzio, perche' dipenderebbe da dove il volto
    si trova nel fotogramma invece che dalla sua geometria.

    `motore` e' la coppia rilevatore+allineatore che ha prodotto la voce
    (o "manual" per i due rami tracciati a mano). None -- il ripiego di
    default -- vuol dire sconosciuto, mai un motore dedotto.
    """
    salvati = getattr(data, "indici_salvati", None)
    volti = []
    for indice, (rect, lmrks) in enumerate(zip(getattr(data, "rects", None) or [],
                                               getattr(data, "landmarks", None) or [])):
        if lmrks is None:
            continue
        if salvati is not None and indice not in salvati:
            continue
        l, t, r, b = [int(v) for v in rect]
        mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
        allineati = LandmarksProcessor.transform_points(lmrks, mat)
        posa = LandmarksProcessor.estimate_pitch_yaw_roll(allineati, size=256)
        volti.append({"rect": [l, t, r, b],
                      "posa": [float(p) for p in posa],
                      "lato": max(r - l, b - t)})
    return ExtractReport.voce(data.filepath, volti=volti, luminanza=luminanza,
                              stato=ExtractReport.STATO_AUTOMATICO, motore=motore)
