"""La parte dell'estrazione che i due frontali condividono.

Nessuna UI e nessun `io`: la finestra cv2 di Extractor.py e il servizio
ExtractManual.py chiamano queste funzioni, e sono le stesse. E' lo stesso
taglio gia' fatto da TrainerLib.py sul loop del trainer e da SorterLib.py
sui verbi del sorting, e per la stessa ragione: cio' che si estrae diventa
cio' che si puo' provare senza aprire niente.
"""
import math

import cv2
import numpy as np
import numpy.linalg as npla

from core import mathlib
from core.cv2ex import cv2_imwrite
from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor
from mainscripts import ExtractReport


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
