import math
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy import linalg as npla

from core import imagelib, mathlib, pathex
from core.cv2ex import *
from core.interact import interact as io
from DFLIMG import *
from mainscripts import SorterCatalog, SorterLib
from mainscripts.SorterLib import absdiff_batch


def _ordina_per_campo(input_path, chiave, campi, estrai, etichetta, reverse=True):
    """Il modello comune ai nove metodi che ordinano per un campo solo.

    Il costo di questi metodi e' la decodifica del JPEG, non l'aritmetica:
    il caricatore li rende paralleli tutti insieme, dove prima sette di loro
    aprivano un'immagine alla volta in un processo solo.
    """
    io.log_info (f"Sorting by {etichetta}...")

    validi, scarti = SorterLib.carica_descrittori(
        pathex.get_image_paths(input_path), campi)
    if not validi:
        return [], _scarti_per(chiave, scarti)

    io.log_info ("Sorting...")
    valori = [ estrai(d) for d in validi ]
    ordine = sorted(range(len(validi)), key=lambda i: valori[i], reverse=reverse)

    img_list = [ (validi[i].path, valori[i]) for i in ordine ]
    return img_list, _scarti_per(chiave, scarti)


def sort_by_blur(input_path, par):
    return _ordina_per_campo(input_path, 'blur', {SorterLib.C_SHARPNESS},
                             lambda d: d.sharpness, "blur")


def sort_by_motion_blur(input_path, par):
    return _ordina_per_campo(input_path, 'motion-blur',
                             {SorterLib.C_MOTION_BLUR},
                             lambda d: d.motion_blur, "motion blur")


def sort_by_face_yaw(input_path, par):
    return _ordina_per_campo(input_path, 'face-yaw', {SorterLib.C_POSA},
                             lambda d: d.yaw, "face yaw")


def sort_by_face_pitch(input_path, par):
    return _ordina_per_campo(input_path, 'face-pitch', {SorterLib.C_POSA},
                             lambda d: d.pitch, "face pitch")


def sort_by_face_source_rect_size(input_path, par):
    def area(d):
        r = np.array(d.source_rect)
        return mathlib.polygon_area(r[[0, 2, 2, 0]].astype(np.float32),
                                    r[[1, 1, 3, 3]].astype(np.float32))
    return _ordina_per_campo(input_path, 'face-source-rect-size',
                             {SorterLib.C_METADATI}, area, "face rect size")


def sort_by_origname(input_path, par):
    return _ordina_per_campo(input_path, 'origname', {SorterLib.C_METADATI},
                             lambda d: d.source_filename or "",
                             "original filename", reverse=False)


def sort_by_brightness(input_path, par):
    return _ordina_per_campo(input_path, 'brightness', {SorterLib.C_HSV},
                             lambda d: d.brightness, "brightness")


def sort_by_hue(input_path, par):
    return _ordina_per_campo(input_path, 'hue', {SorterLib.C_HSV},
                             lambda d: d.hue, "hue")


def sort_by_black(input_path, par):
    return _ordina_per_campo(input_path, 'black', {SorterLib.C_BLACK},
                             lambda d: d.black, "amount of black pixels",
                             reverse=False)


def _scarti_per(chiave, scarti):
    """Gli scarti diventano cestino solo se il descrittore lo dichiara.

    L'aggancio strutturale fra cio' che un metodo promette e cio' che fa: un
    metodo che dichiara di riordinare senza cestinare lascia dov'e' il file
    che non ha potuto leggere, e lo dice. Il rischio che questo previene e'
    il peggiore di questo lavoro, perche' l'interfaccia grafica costruira'
    sopra quelle dichiarazioni.
    """
    if not scarti:
        return []
    if SorterCatalog.CESTINA in SorterCatalog.per_chiave(chiave).produces:
        return [(p,) for p in scarti]
    io.log_info(f"{len(scarti)} file(s) could not be read; left in place.")
    return []


def _descrittore_hist(input_path):
    """Il descrittore condiviso dai due metodi gemelli -- non il device.

    Il device dipende da quanta memoria in piu' oltre a P il chiamante
    tiene, e i due gemelli non tengono la stessa cosa: catena_greedy vive
    di P soltanto, _somme_bhattacharyya ci aggiunge un blocco x N. Sceglierlo
    qui avrebbe forzato una stima sola per i due, giusta al piu' per uno.
    """
    image_paths = pathex.get_image_paths(input_path)
    validi, scarti = SorterLib.carica_descrittori(
        image_paths, {SorterLib.C_HIST})
    if not validi:
        return None, None, scarti
    P = SorterLib.descrittore_istogrammi([d.hist for d in validi])
    return validi, P, scarti


def sort_by_hist(input_path, par):
    io.log_info ("Sorting by histogram similarity...")

    validi, P, scarti = _descrittore_hist(input_path)
    if validi is None:
        return [], _scarti_per('hist', scarti)

    io.log_info ("Sorting...")
    # catena_greedy tiene sul device solo P (piu' due vettori O(N)
    # trascurabili rispetto a P): la stima e' P.nbytes, non un multiplo
    # arbitrario che non corrisponde a cosa la funzione alloca davvero.
    device = SorterLib.scegli_device(len(validi), P.nbytes)
    ordine = SorterLib.catena_greedy(P, device=device)

    img_list = [ (validi[i].path,) for i in ordine ]
    return img_list, _scarti_per('hist', scarti)


# Righe per blocco in _somme_bhattacharyya: sia il default della funzione
# sia la stima di memoria che sceglie il device leggono la stessa costante,
# cosi' non possono divergere silenziosamente l'una dall'altra.
_BLOCCO_HIST_DISSIM = 2048


def _somme_bhattacharyya(P, device, blocco=_BLOCCO_HIST_DISSIM):
    """Le somme per riga della matrice di Bhattacharyya, senza materializzarla.

    SorterLib.bhattacharyya_matrice costruisce l'N x N intera -- a 20 000
    immagini 1.6 GB in float32 -- ma hist-dissim usa solo la somma per riga.
    Un blocco di righe alla volta contro l'intero P tiene la memoria a
    blocco x N invece che a N x N.

    Un solo buffer (blocco x N, o N x N quando N < blocco) resta vivo per
    tutto il ciclo e viene riscritto sul posto: la GEMM scrive con out=,
    poi mul_/add_/clamp_min_/sqrt_ in sequenza al posto di
    "1.0 - (...)" seguito da torch.sqrt(...), che altrimenti tengono due
    copie della stessa forma vive nello stesso istante. Una prima stesura
    senza queste due attenzioni (ne' out=, ne' il buffer riusato) arrivava
    a tenere fino a tre copie della stessa forma: misurato su CUDA a
    20 000 immagini, 535.5 MiB di picco contro i 223.1 MiB di questa
    versione, ~8 MiB oltre P.nbytes + blocco*N*4 a ogni N provato
    (9000-30000) -- il residuo di P.nbytes + blocco*N*4 e' comodamente
    dentro il margine con cui scegli_device viene chiamata, non serve
    correggerlo nella formula.
    """
    t = torch.as_tensor(P, device=device)
    n = t.shape[0]
    somme = torch.empty(n, dtype=torch.float32, device=device)
    m0 = min(blocco, n)
    buf = torch.empty((m0, n), dtype=torch.float32, device=device)
    for a in range(0, n, blocco):
        fine = min(a + blocco, n)
        vista = buf[:fine - a]
        torch.mm(t[a:fine], t.T, out=vista)
        vista.mul_(-1.0).add_(1.0).clamp_min_(0).sqrt_()
        somme[a:fine] = vista.sum(dim=1)
    return somme.cpu().numpy().astype(np.float32)


def sort_by_hist_dissim(input_path, par):
    io.log_info ("Sorting by histogram dissimilarity...")

    validi, P, scarti = _descrittore_hist(input_path)
    if validi is None:
        return [], _scarti_per('hist-dissim', scarti)

    io.log_info ("Sorting...")
    # _somme_bhattacharyya tiene P piu' un blocco x N temporaneo (il primo
    # blocco, che e' il piu' grande quando N < blocco): la stima riflette
    # entrambi, non solo P come per hist qui sopra.
    n = len(validi)
    blocco_eff = min(_BLOCCO_HIST_DISSIM, n)
    device = SorterLib.scegli_device(n, P.nbytes + blocco_eff * n * 4)

    # La somma per riga della matrice: la distanza di un'immagine da tutte
    # le altre. La diagonale e' nulla, quindi non va esclusa. A blocchi
    # invece che sulla matrice intera -- vedi _somme_bhattacharyya.
    somme = _somme_bhattacharyya(P, device, blocco=_BLOCCO_HIST_DISSIM)
    ordine = np.argsort(-somme, kind="stable")

    img_list = [ (validi[i].path,) for i in ordine ]
    return img_list, _scarti_per('hist-dissim', scarti)

def sort_by_oneface_in_image(input_path, par):
    io.log_info ("Sort by one face in images...")
    image_paths = pathex.get_image_paths(input_path)

    # <frame>_<faccia> e' la forma che l'estrazione produce. Un nome che non
    # la rispetta non dice niente su quante facce c'erano in quel frame:
    # resta dov'e'. Prima, i nomi fuori forma non entravano nell'array degli
    # indici ma gli indici venivano poi usati sulla lista intera, quindi il
    # cestino riceveva un file al posto di un altro.
    facce_di = {}
    fuori_forma = []
    for path in image_paths:
        pezzi = Path(path).stem.split('_')
        if len(pezzi) == 2 and pezzi[0].isdigit() and pezzi[1].isdigit():
            facce_di.setdefault(pezzi[0], []).append(int(pezzi[1]))
        else:
            fuori_forma.append(path)

    if not facce_di:
        io.log_info ("Nothing found. Possible recover original filenames first.")
        return [], []

    frame_multipli = { frame for frame, indici in facce_di.items()
                       if any(i != 0 for i in indici) }

    img_list = []
    trash_img_list = []
    for path in image_paths:
        pezzi = Path(path).stem.split('_')
        if len(pezzi) != 2 or not pezzi[0].isdigit() or not pezzi[1].isdigit():
            continue
        if pezzi[0] in frame_multipli:
            trash_img_list.append ( (path,) )
        else:
            img_list.append ( (path,) )

    if fuori_forma:
        io.log_info ("%d file(s) are not named <frame>_<face>; left in place."
                     % (len(fuori_forma)) )

    io.log_info ("Found %d images." % (len(trash_img_list)) )
    return img_list, trash_img_list

# Il numero di gradi di imbardata in cui il faceset viene diviso. Non e'
# math.pi/2 ma -1.2..+1.2 perche' l'imbardata massima che i landmark 2DFAN
# producono davvero sta in quell'intervallo.
GRADI_YAW = 128
YAW_MIN, YAW_MAX = -1.2, 1.2

# Quanti volti per grado tenere prima di ordinarli per nitidezza: si tiene
# dieci volte il bisogno, si ordina, si taglia.
FATTORE_NITIDEZZA = 10


def sort_best_faster(input_path, par):
    return sort_best(input_path, par, faster=True)


def sort_best(input_path, par, faster=False):
    target_count = par.quanti_volti()
    if target_count <= 0:
        raise ValueError(
            f"target number of faces must be positive, got {target_count}")

    io.log_info ("Performing sort by best faces.")
    if faster:
        io.log_info("Using faster algorithm. Faces will be sorted by source-rect-area instead of blur.")

    campi = {SorterLib.C_POSA, SorterLib.C_HIST}
    campi.add(SorterLib.C_METADATI if faster else SorterLib.C_SHARPNESS)

    validi, scarti = SorterLib.carica_descrittori(
        pathex.get_image_paths(input_path), campi)
    trash_img_list = _scarti_per('final-fast' if faster else 'final', scarti)
    if not validi:
        return [], trash_img_list

    # Con un obiettivo piu' piccolo del numero di gradi, un grado per volto:
    # prima questo caso dava zero volti per grado e nessun ordinamento.
    gradi = max(1, min(GRADI_YAW, target_count))
    imgs_per_grad = max(1, round(target_count / gradi))

    if faster:
        punteggio = np.array([
            mathlib.polygon_area(
                np.array(d.source_rect)[[0, 2, 2, 0]].astype(np.float32),
                np.array(d.source_rect)[[1, 1, 3, 3]].astype(np.float32))
            for d in validi], dtype=np.float32)
    else:
        punteggio = np.array([d.sharpness for d in validi], dtype=np.float32)

    # Il segno: il codice originale binna su -yaw.
    yaw = np.array([-d.yaw for d in validi], dtype=np.float32)
    bordi_yaw = np.linspace(YAW_MIN, YAW_MAX, gradi)
    grado_di = SorterLib.bin_lineare(yaw, bordi_yaw)

    # Un vero faceset e' quasi tutto frontale: la maggior parte dei gradi
    # resta sotto quota, o vuota. Il codice originale compensava alzando la
    # quota per grado di quanto i gradi scarsi mancavano in totale -- senza
    # questo, un faceset sbilanciato tiene una piccola frazione del target
    # invece di avvicinarlo. bincount sostituisce il ciclo su range(gradi)
    # dell'originale con un conteggio vettorizzato sullo stesso grado_di gia'
    # calcolato per il binning.
    conteggi = np.bincount(grado_di, minlength=gradi)
    total_lack = np.maximum(imgs_per_grad - conteggi, 0).sum()
    imgs_per_grad += total_lack // gradi

    tetto = imgs_per_grad * FATTORE_NITIDEZZA

    P = SorterLib.descrittore_istogrammi([d.hist for d in validi])
    # bhattacharyya_matrice qui non vede mai l'intero P: nel_grado viene
    # troncato a `tetto` prima di essere diviso per beccheggio, quindi il
    # secchio piu' grande passato a un confronto e' min(tetto, N), non N.
    # A differenza di hist/hist-dissim (dove la stima giusta e' l'intero
    # dataset), qui la stima onesta e' quel secchio: la lezione del task
    # precedente era una stima lineare passata a un'allocazione quadratica,
    # e la trappola gemella sarebbe qui una stima sull'intero N quando cio'
    # che viene davvero allocato e' al piu' tetto x tetto.
    secchio_max = min(tetto, len(validi))
    stima_bytes = (secchio_max * P.shape[1] + secchio_max * secchio_max) * P.itemsize
    device = SorterLib.scegli_device(len(validi), stima_bytes)

    final_img_list = []
    pitch = np.array([d.pitch for d in validi], dtype=np.float32)

    for g in io.progress_bar_generator (range(gradi), "Fetching the best"):
        nel_grado = np.flatnonzero(grado_di == g)
        if nel_grado.size == 0:
            continue

        # Per nitidezza (o per area), poi tagliati a dieci volte il bisogno.
        nel_grado = nel_grado[np.argsort(-punteggio[nel_grado], kind="stable")]
        if nel_grado.size > tetto:
            trash_img_list += [ (validi[i].path,) for i in nel_grado[tetto:] ]
            nel_grado = nel_grado[:tetto]

        # Dentro il grado, per beccheggio.
        pitch_gradi = max(1, imgs_per_grad)
        bordi_pitch = np.linspace(-math.pi / 2, math.pi / 2, pitch_gradi)
        pg_di = SorterLib.bin_lineare(pitch[nel_grado], bordi_pitch)

        secchi = []
        for pg in range(pitch_gradi):
            dentro = nel_grado[pg_di == pg]
            if dentro.size == 0:
                continue
            # Dentro il secchio, il piu' dissimile per primo.
            sotto = SorterLib.bhattacharyya_matrice(P[dentro], device=device)
            dentro = dentro[np.argsort(-sotto.sum(axis=1), kind="stable")]
            secchi.append(list(dentro))

        # A giro fra i secchi, uno per volta, come faceva il codice originale.
        n = imgs_per_grad
        while n > 0 and secchi:
            avanzato = False
            for secchio in secchi:
                if not secchio:
                    continue
                final_img_list.append( (validi[secchio.pop(0)].path,) )
                avanzato = True
                n -= 1
                if n == 0:
                    break
            if not avanzato:
                break

        for secchio in secchi:
            trash_img_list += [ (validi[i].path,) for i in secchio ]

    return final_img_list, trash_img_list

"""
def sort_by_vggface(input_path):
    io.log_info ("Sorting by face similarity using VGGFace model...")

    model = VGGFace()

    final_img_list = []
    trash_img_list = []

    image_paths = pathex.get_image_paths(input_path)
    img_list = [ (x,) for x in image_paths ]
    img_list_len = len(img_list)
    img_list_range = [*range(img_list_len)]

    feats = [None]*img_list_len
    for i in io.progress_bar_generator(img_list_range, "Loading"):
        img = cv2_imread( img_list[i][0] ).astype(np.float32)
        img = imagelib.normalize_channels (img, 3)
        img = cv2.resize (img, (224,224) )
        img = img[..., ::-1]
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
        feats[i] = model.predict( img[None,...] )[0]

    tmp = np.zeros( (img_list_len,) )
    float_inf = float("inf")
    for i in io.progress_bar_generator ( range(img_list_len-1), "Sorting" ):
        i_feat = feats[i]

        for j in img_list_range:
            tmp[j] = npla.norm(i_feat-feats[j]) if j >= i+1 else float_inf

        idx = np.argmin(tmp)

        img_list[i+1], img_list[idx] = img_list[idx], img_list[i+1]
        feats[i+1], feats[idx] = feats[idx], feats[i+1]

    return img_list, trash_img_list
"""

# Il tetto vale sul fabbisogno TOTALE in memoria, non sulla sola matrice
# delle distanze. Sorvegliare la matrice sarebbe sorvegliare il termine
# sbagliato: su 400 volti a 768 pixel i soli pixel grezzi sono 707 MB contro
# 0.6 MiB di matrice, mille volte tanto, e il processo esaurirebbe la memoria
# molto prima che un tetto sul quadratico avesse modo di rifiutare.
#
# Otto gigabyte: la meta' dei sedici che una macchina su cui gira DeepFaceLab
# ha come minimo realistico. Lascia passare l'uso normale -- 400 volti a 768
# pixel ne stimano 3.30, e sono il caso peggiore per risoluzione fra quelli
# misurati -- e ferma cio' che non entrerebbe comunque.
TETTO_MEMORIA_BYTE = 8 * 1024**3


def _stima_byte_absdiff(n, byte_immagine):
    """I byte che il confronto alloca, prima che ne allochi uno.

    Tre termini, e sono i tre che il codice alloca davvero: i pixel grezzi
    impilati (uno per byte), la copia in virgola mobile che l1_a_blocchi ne
    fa (quattro per byte, e l'originale resta vivo accanto), e la matrice
    delle distanze. Il termine lineare non e' trascurabile: a piena
    risoluzione e' quello che domina, ed e' l'errore che questa funzione
    esiste per non rifare.
    """
    return 5 * n * byte_immagine + 4 * n * n


def _controlla_tetto(n, byte_immagine, rimedio):
    """Il rifiuto prima del caricamento, che e' il solo momento utile.

    `rimedio` e' la coda del messaggio e la passa il chiamante: il controllo
    e' comune alle due varianti, quindi un testo cablato qui direbbe alla
    variante veloce di usare se stessa.
    """
    stima = _stima_byte_absdiff(n, byte_immagine)
    if stima > TETTO_MEMORIA_BYTE:
        raise ValueError (
            f"{n} images of {byte_immagine // 1024} KiB need about "
            f"{stima // 1024**2} MiB of memory (the images, their float copy "
            f"and the distance matrix), over the "
            f"{TETTO_MEMORIA_BYTE // 1024**2} MiB cap; {rimedio}")
    return stima


def _catena_absdiff(X, paths, par, byte_immagine):
    """La parte comune ai due metodi: la matrice, la catena, l'ordine."""
    n = len(paths)
    is_sim = par.per_simili()
    # La stessa stima del tetto, e non una piu' piccola: il device deve
    # rispondere per cio' che ci finisce sopra davvero, copia in virgola
    # mobile compresa.
    device = SorterLib.scegli_device(n, _stima_byte_absdiff(n, byte_immagine))

    io.log_info ("Computing...")
    D = SorterLib.l1_a_blocchi(X, device=device)

    io.log_info ("Sorting...")
    ordine = SorterLib.catena_da_matrice(D, piu_lontano=not is_sim)
    return [ (paths[i],) for i in ordine ], []


def sort_by_absdiff(input_path, par):
    io.log_info ("Sorting by absolute difference...")

    image_paths = pathex.get_image_paths(input_path)
    if len(image_paths) == 0:
        io.log_info ("No images found.")
        return [], []

    io.log_info (
        "Comparing every pixel of every pair: this is the exact order, and "
        "the slow one. The 'absdiff-fast' entry compares a 32x32 thumbnail "
        "instead -- about five times faster on the whole command, and a "
        "DIFFERENT order, not an approximation of this one.")

    # I pixel grezzi non passano dal caricatore: quello restituisce
    # descrittori e mai immagini, per tenere la memoria costante. Questo e'
    # il solo metodo che ha davvero bisogno dei pixel in blocco, e li legge
    # da se'.
    #
    # La prima immagine si legge prima delle altre perche' il fabbisogno si
    # stima solo conoscendo i byte per immagine, e stimarlo dopo aver
    # caricato tutto sarebbe rifiutare a memoria gia' esaurita.
    prima = cv2_imread(image_paths[0])
    if prima is None:
        raise ValueError (f"unable to read {Path(image_paths[0]).name}")
    _controlla_tetto(len(image_paths), prima.nbytes,
                     "use absdiff-fast instead")

    # Un array solo, riempito riga per riga. La lista seguita da np.stack
    # teneva in vita due copie dei pixel nell'istante in cui la seconda
    # nasce, cioe' il doppio del termine piu' grande della stima -- e una
    # stima che non descrive il picco non e' una stima.
    X = np.empty((len(image_paths), prima.size), dtype=prima.dtype)
    for i, path in enumerate(io.progress_bar_generator(image_paths, "Loading")):
        img = prima if i == 0 else cv2_imread(path)
        if img is None:
            raise ValueError (f"unable to read {Path(path).name}")
        # Anche il dtype, e non la sola forma: l'array e' allocato col dtype
        # della prima immagine, quindi un PNG a 16 bit in mezzo a dei JPEG
        # verrebbe troncato in silenzio invece di far sbagliare rumorosamente.
        if img.shape != prima.shape or img.dtype != prima.dtype:
            raise ValueError (
                f"mixed resolution: {Path(path).name} is {img.shape[:2]} "
                f"{img.dtype} but the first image is {prima.shape[:2]} "
                f"{prima.dtype}; absdiff compares raw pixels and needs one "
                f"resolution -- use absdiff-fast instead")
        X[i] = img.reshape(-1)

    return _catena_absdiff(X, image_paths, par, prima.nbytes)


def sort_by_absdiff_fast(input_path, par):
    io.log_info ("Sorting by absolute difference (faster)...")
    io.log_info (
        "Comparing a 32x32 thumbnail of each face instead of every pixel: "
        "about five times faster than the exact 'absdiff' on the whole "
        "command, and a DIFFERENT order -- the thumbnail averages away the "
        "fine detail 'absdiff' compares, so this is not an approximation of "
        "that order.")

    image_paths = pathex.get_image_paths(input_path)
    # I byte per immagine sono noti senza aprire niente: la miniatura ha lo
    # stesso lato per tutte. Qui il rimedio non e' un altro metodo -- questo
    # E' gia' quello veloce.
    byte_miniatura = SorterLib.LATO_MINIATURA ** 2 * 3
    _controlla_tetto(len(image_paths), byte_miniatura,
                     "sort a smaller folder at a time")

    validi, scarti = SorterLib.carica_descrittori(
        image_paths, {SorterLib.C_MINIATURA})
    if not validi:
        return [], _scarti_per('absdiff-fast', scarti)

    X = np.stack([np.asarray(d.miniatura) for d in validi])
    img_list, _ = _catena_absdiff(X, [d.path for d in validi],
                                  par, byte_miniatura)
    return img_list, _scarti_per('absdiff-fast', scarti)

# Bit di differenza fra due hash percettivi sotto i quali due volti contano
# come duplicati. Zero, e non un numero piu' generoso: i gruppi sono
# componenti connesse, quindi su un faceset estratto da un piano sequenza
# ogni frame somiglia al vicino e la catena si richiude su tutto il faceset.
# Misurato su 1619 frame consecutivi di un video vero: gia' a soglia 2 il
# gruppo piu' grande e' 1011 volti, a soglia 6 sono 1616 su 1619. Il valore
# e' misurato, non scelto; il verbale delle misure porta la corsa completa.
SOGLIA_DUPLICATI = 0


def sort_by_dedup(input_path, par):
    io.log_info ("Removing duplicate faces...")
    soglia = par.soglia_duplicati()

    validi, scarti = SorterLib.carica_descrittori(
        pathex.get_image_paths(input_path),
        {SorterLib.C_HASH, SorterLib.C_SHARPNESS})
    trash_img_list = _scarti_per('dedup', scarti)
    if not validi:
        return [], trash_img_list

    # La stima non e' lineare come sembrerebbe: coppie_simili tiene sul device
    # X (n x 64 float32) PIU', a ogni passo, un blocco di distanze e la sua
    # maschera booleana, che sono quadratici nel lato del blocco. Contare solo
    # X e' l'errore gemello di quello gia' evitato in sort_best -- una stima
    # lineare passata a un'allocazione quadratica -- e a blocco=2048 sono
    # 16 MiB di float32 piu' 4 MiB di maschera che mancherebbero all'appello.
    lato = min(SorterLib.BLOCCO_HASH, len(validi))
    stima_bytes = len(validi) * 64 * 4 + lato * lato * 5
    device = SorterLib.scegli_device(len(validi), stima_bytes)
    coppie = SorterLib.coppie_simili([d.hash for d in validi], soglia,
                                     device=device)
    gruppi = SorterLib.gruppi_da_coppie(len(validi), coppie)

    da_buttare = set()
    for gruppo in gruppi:
        # Di ogni gruppo sopravvive la piu' nitida: fra due frame quasi
        # identici, quello meno mosso.
        gruppo = sorted(gruppo, key=lambda i: validi[i].sharpness, reverse=True)
        da_buttare.update(gruppo[1:])

    io.log_info ("Found %d duplicate(s) in %d group(s)."
                 % (len(da_buttare), len(gruppi)) )

    img_list = [ (validi[i].path, validi[i].sharpness)
                 for i in range(len(validi)) if i not in da_buttare ]
    trash_img_list += [ (validi[i].path,) for i in sorted(da_buttare) ]
    return img_list, trash_img_list


def sort_by_coverage(input_path, par):
    # Il testo dice cio' che il metodo fa davvero, non cio' che il nome
    # promette: massimizzare la distanza minima fra i pochi scelti NON e' la
    # copertura uniforme di un intervallo -- prende bene gli estremi e puo'
    # lasciare buchi in mezzo. Misurato su un faceset allineato vero da 400
    # volti (whole_face), dividendo l'imbardata in venti intervalli: dei 19
    # popolati, un obiettivo di 20 ne raggiunge 11 e uno di 40 ne raggiunge
    # 14. I buchi si chiudono al crescere dell'obiettivo, ma la soglia
    # dipende dal faceset: nessuna percentuale da promettere qui.
    io.log_info ("Selecting the most varied faces...")
    io.log_info ("These are the faces most different from each other, not a "
                 "uniform sample: a small target favors the extremes and can "
                 "leave gaps in between. Measured on a 400-face faceset, a "
                 "target of 20 reached 11 of the 19 populated yaw ranges, and "
                 "40 reached 14 -- ask for more faces if the middle of the "
                 "range matters too.")
    target_count = par.quanti_volti()
    if target_count <= 0:
        raise ValueError (
            f"target number of faces must be positive, got {target_count}")

    validi, scarti = SorterLib.carica_descrittori(
        pathex.get_image_paths(input_path),
        {SorterLib.C_POSA, SorterLib.C_HSV})
    trash_img_list = _scarti_per('coverage', scarti)
    if not validi:
        return [], trash_img_list

    # Posa e illuminazione, normalizzate: sono le due grandezze per cui un
    # faceset "copre" o non copre cio' che il training incontrera'.
    #
    # Il rollio NON e' fra le colonne, ed e' una scelta, non una dimenticanza:
    # i volti allineati sono gia' ruotati in verticale per costruzione, quindi
    # il loro rollio e' gia' stato corretto via prima di arrivare qui, e cio'
    # che ne resta e' un artefatto, non un segnale di posa. Misurato su due
    # faceset allineati veri (whole_face, 400 volti l'uno): il rollio satura
    # al bordo del clip +-pi/2 nel 97.2% e nel 99.2% dei volti, con uno scarto
    # 30 e 93 volte quello dell'imbardata. Rimesso fra le colonne, la
    # normalizzazione -- che equalizza la varianza, non l'informazione -- gli
    # ridarebbe lo stesso peso dell'imbardata.
    X = SorterLib.normalizza_colonne(np.array(
        [[d.yaw, d.pitch, d.brightness] for d in validi],
        dtype=np.float32))

    scelti = SorterLib.campiona_lontani(X, target_count)
    tenuti = set(scelti.tolist())

    io.log_info ("Keeping %d of %d faces." % (len(tenuti), len(validi)) )

    img_list = [ (validi[i].path,) for i in scelti ]
    trash_img_list += [ (validi[i].path,) for i in range(len(validi))
                        if i not in tenuti ]
    return img_list, trash_img_list


def sort_by_upscale_factor(input_path, par):
    # L'ingrandimento vero, letto dalla matrice di allineamento che il
    # ritaglio ha davvero usato: il calcolo sta nel caricatore, qui resta
    # solo l'ordine. Un volto la cui matrice manca o e' degenere non ha un
    # ingrandimento da misurare, e il caricatore lo restituisce come scarto
    # -- resta dov'e', contato e detto, invece di ricevere un numero inventato.
    return _ordina_per_campo(input_path, 'upscale-factor',
                             {SorterLib.C_INGRANDIMENTO}, lambda d: d.upscale,
                             "upscale factor", reverse=False)


# Il lato della griglia di posa su cui si confrontano le due distribuzioni.
# Sedici e non centoventotto: qui serve la forma della distribuzione, non un
# bin per volto, e una griglia troppo fine renderebbe ogni bin vuoto.
GRADI_MATCH = 16


def sort_by_match_dst(input_path, par):
    io.log_info ("Sorting by pose coverage of the reference faceset...")

    ref_dir = par.cartella_riferimento(input_path)

    validi, scarti = SorterLib.carica_descrittori(
        pathex.get_image_paths(input_path), {SorterLib.C_POSA})
    trash_img_list = _scarti_per('match-dst', scarti)
    if not validi:
        return [], trash_img_list

    # Imbardata e beccheggio, senza il rollio: su volti allineati il rollio e'
    # gia' stato corretto dall'allineamento, e cio' che ne resta satura al
    # bordo del clip +-pi/2 nella quasi totalita' dei volti -- un asse in piu'
    # che non porta posa.
    bordi_yaw = np.linspace(YAW_MIN, YAW_MAX, GRADI_MATCH)
    bordi_pitch = np.linspace(-math.pi / 2, math.pi / 2, GRADI_MATCH)

    densita = np.zeros((GRADI_MATCH, GRADI_MATCH), dtype=np.float32)
    ha_riferimento = False
    if ref_dir is None:
        # I due modi di restare senza riferimento vanno detti separatamente:
        # con un solo messaggio, un refuso in --ref-dir e' indistinguibile
        # dal non averne passato nessuno, e l'utente cerca il difetto dove
        # non e'.
        io.log_info ("No reference faceset: order left unchanged.")
    elif not Path(ref_dir).is_dir():
        io.log_info ("Reference faceset directory does not exist: %s\n"
                     "Order left unchanged." % ref_dir )
    else:
        rif, _ = SorterLib.carica_descrittori(
            pathex.get_image_paths(ref_dir), {SorterLib.C_POSA})
        if rif:
            ry = SorterLib.bin_lineare([d.yaw for d in rif], bordi_yaw)
            rp = SorterLib.bin_lineare([d.pitch for d in rif], bordi_pitch)
            np.add.at(densita, (ry, rp), 1.0)
            densita /= len(rif)
            ha_riferimento = True
            io.log_info ("Reference: %d faces from %s" % (len(rif), ref_dir) )
        else:
            io.log_info ("Reference faceset is empty: order left unchanged.")

    iy = SorterLib.bin_lineare([d.yaw for d in validi], bordi_yaw)
    ip = SorterLib.bin_lineare([d.pitch for d in validi], bordi_pitch)
    punteggi = densita[iy, ip]

    if ha_riferimento:
        # Quanto l'ordine puo' distinguere, con le cifre di QUESTA corsa: un
        # riferimento a soggetto singolo -- la forma normale di un faceset
        # estratto da un video, cioe' il caso d'uso -- occupa una manciata
        # di bin su 256, e i volti si spartiscono altrettanti punteggi.
        distinti = int(np.unique(punteggi).size)
        io.log_info ("Reference covers %d of %d pose bins; %d of %d faces "
                     "score zero; %d distinct scores."
                     % (int(np.count_nonzero(densita)),
                        GRADI_MATCH * GRADI_MATCH,
                        int(np.count_nonzero(punteggi == 0)),
                        len(punteggi), distinti) )
        # Meno di un punteggio distinto ogni dieci volti: l'ordine e' fatto
        # di pochi blocchi enormi, e dentro un blocco non ordina niente. La
        # soglia dice quando vale la pena avvertire, non quando il metodo e'
        # sbagliato: le cifre sopra restano quelle vere in ogni caso.
        if distinti * 10 <= len(punteggi):
            io.log_info ("The reference is concentrated in few poses: this "
                         "order tells %d faces apart in %d groups only, and "
                         "inside a group it sorts nothing."
                         % (len(punteggi), distinti) )

    ordine = np.argsort(-punteggi, kind="stable")
    img_list = [ (validi[i].path, float(punteggi[i])) for i in ordine ]
    return img_list, trash_img_list


def _sposta(src, dst):
    """Sposta src in dst senza mai sovrascrivere.

    Torna [] se e' riuscito, [(src, motivo)] altrimenti. Path.rename
    sovrascrive in silenzio su POSIX e solleva su Windows: il controllo
    esplicito rende le due piattaforme uguali, e sul lato che non perde file.
    """
    if dst.exists():
        return [ (src, f"{dst.name} already exists") ]
    try:
        src.rename (dst)
        return []
    except OSError as e:
        return [ (src, str(e)) ]


def _riporta_falliti(falliti, cosa):
    """Un resoconto, non una riga per file: su un faceset grande scorre via."""
    if not falliti:
        return
    io.log_err (f"{len(falliti)} file(s) could not be {cosa}:")
    for src, motivo in falliti[:10]:
        io.log_err (f"  {src.name}: {motivo}")
    if len(falliti) > 10:
        io.log_err (f"  ... and {len(falliti) - 10} more")


def final_process(input_path, img_list, trash_img_list):
    if len(trash_img_list) != 0:
        parent_input_path = input_path.parent
        trash_path = parent_input_path / (input_path.stem + '_trash')
        trash_path.mkdir (exist_ok=True)

        io.log_info ("Trashing %d items to %s" % ( len(trash_img_list), str(trash_path) ) )

        for filename in pathex.get_image_paths(trash_path):
            Path(filename).unlink()

        falliti = []
        for i in io.progress_bar_generator( range(len(trash_img_list)), "Moving trash", leave=False):
            src = Path (trash_img_list[i][0])
            falliti += _sposta (src, trash_path / src.name)
        _riporta_falliti (falliti, "moved to trash")

        io.log_info ("")

    if len(img_list) != 0:
        # La larghezza viene dall'insieme, non da una costante: con '%.5d' il
        # centomillesimo file diventa piu' lungo del novantanovemila-
        # novecentonovantanovesimo e si ordina prima, che e' esattamente cio'
        # che la rinomina esiste per evitare.
        larghezza = max(5, len(str(len(img_list) - 1)))
        intermedio = '%%.%dd_%%s' % larghezza
        finale = '%%.%dd%%s' % larghezza

        # Due passate: la prima porta ogni file su un nome che nessun altro
        # file dell'insieme puo' avere, la seconda gli da' il nome definitivo.
        # Senza la prima, rinominare A in B mentre B esiste ancora perde B.
        falliti = []
        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming", leave=False):
            src = Path (img_list[i][0])
            falliti += _sposta (src, input_path / (intermedio % (i, src.name)))

        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming"):
            nome = Path (img_list[i][0]).name
            src = input_path / (intermedio % (i, nome))
            falliti += _sposta (src, input_path / (finale % (i, src.suffix)))

        _riporta_falliti (falliti, "renamed")

class Parametri:
    """I parametri per-metodo, dal flag al metodo, col prompt come ripiego.

    Un campo a None significa "il flag non c'era": il metodo chiede, con lo
    stesso testo di prompt di sempre, quindi la chiave sotto cui una
    risposta pre-fornita viene cercata non cambia.
    """

    def __init__(self, target_count=None, ref_dir=None, similar=None,
                 threshold=None):
        self.target_count = target_count
        self.ref_dir = ref_dir
        self.similar = similar
        self.threshold = threshold

    def quanti_volti(self):
        if self.target_count is not None:
            return self.target_count
        return io.input_int("Target number of faces?", 2000)

    def cartella_riferimento(self, input_path):
        """La cartella di confronto: il flag, poi il fratello, poi la domanda.

        La derivazione dal fratello copre il caso normale (data_src/aligned
        contro data_dst/aligned) senza chiedere niente. Ma se non risolve a
        una cartella che esiste, si chiede: indovinare in silenzio una
        cartella sbagliata produrrebbe un ordine plausibile e falso.

        Un percorso indicato dall'utente -- dal flag o dalla risposta -- torna
        com'e' stato scritto, senza controlli: e' chi lo consuma a dire che
        non esiste, nominandolo, cosi' un refuso non si confonde con
        l'assenza di un riferimento.
        """
        if self.ref_dir is not None:
            return Path(self.ref_dir)

        input_path = Path(input_path)
        gemelli = {"data_src": "data_dst", "data_dst": "data_src"}
        for nome, opposto in gemelli.items():
            if input_path.parent.name == nome:
                candidato = input_path.parent.parent / opposto / input_path.name
                if candidato.is_dir():
                    return candidato

        risposta = io.input_str(
            "Reference faceset directory?", None,
            help_message="The other faceset, whose head poses this one is "
                         "compared against.")
        return Path(risposta) if risposta else None

    def per_simili(self):
        if self.similar is not None:
            return self.similar
        return io.input_bool("Sort by similar?", True,
                             help_message="Otherwise sort by dissimilar.")

    def soglia_duplicati(self):
        if self.threshold is not None:
            return self.threshold
        return io.input_int(
            "Duplicate distance threshold?", SOGLIA_DUPLICATI,
            help_message="Bits of difference between two 64-bit perceptual "
                         "hashes below which two faces count as duplicates. "
                         "At the default 0 only perceptually identical faces "
                         "are grouped; catching near-copies means raising it, "
                         "and that is where it gets expensive. Groups are "
                         "chains: if A matches B and B matches C, all three "
                         "are grouped and only one survives, even when A and "
                         "C look nothing alike. On a faceset extracted from "
                         "video every frame matches its neighbour, so the "
                         "chain closes over the whole sequence very fast. "
                         "Measured on 1619 consecutive frames of a real "
                         "video: 0 trashes 441 of them in groups of at most "
                         "9, while 2 trashes more than half the faceset, "
                         "1011 of the 1619 in a single group. Raise it one "
                         "bit at a time, and check the _trash folder before "
                         "training.")


FUNZIONI = {
    'blur':                   sort_by_blur,
    'motion-blur':            sort_by_motion_blur,
    'face-yaw':               sort_by_face_yaw,
    'face-pitch':             sort_by_face_pitch,
    'face-source-rect-size':  sort_by_face_source_rect_size,
    'hist':                   sort_by_hist,
    'hist-dissim':            sort_by_hist_dissim,
    'brightness':             sort_by_brightness,
    'hue':                    sort_by_hue,
    'black':                  sort_by_black,
    'origname':               sort_by_origname,
    'oneface':                sort_by_oneface_in_image,
    'absdiff':                sort_by_absdiff,
    'final':                  sort_best,
    'final-fast':             sort_best_faster,
    'dedup':                  sort_by_dedup,
    'coverage':               sort_by_coverage,
    'upscale-factor':         sort_by_upscale_factor,
    'match-dst':              sort_by_match_dst,
    'absdiff-fast':           sort_by_absdiff_fast,
}

# L'ordine e la descrizione vengono dal catalogo, non da qui: un elenco
# ripetuto e' un elenco che va fuori sincrono.
sort_func_methods = {
    m.key: (m.label, FUNZIONI[m.key]) for m in SorterCatalog.METODI
}


def main (input_path, sort_by_method=None, par=None):
    io.log_info ("Running sort tool.\r\n")

    if par is None:
        par = Parametri()

    if sort_by_method is None:
        io.log_info(f"Choose sorting method:")

        key_list = list(sort_func_methods.keys())
        for i, key in enumerate(key_list):
            desc, func = sort_func_methods[key]
            io.log_info(f"[{i}] {desc}")

        io.log_info("")
        id = io.input_int("", 5, valid_list=[*range(len(key_list))] )

        sort_by_method = key_list[id]
    else:
        sort_by_method = sort_by_method.lower()

    if sort_by_method not in sort_func_methods:
        raise ValueError (f"unknown sorting method: {sort_by_method}")

    desc, func = sort_func_methods[sort_by_method]
    img_list, trash_img_list = func(input_path, par)

    final_process (input_path, img_list, trash_img_list)
