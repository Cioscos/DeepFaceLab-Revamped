"""Il caricatore dei descrittori e le primitive numeriche del sorting.

Nessuna di queste funzioni conosce le cartelle, i prompt o il cestino:
prendono array e restituiscono array. E' cio' che le rende verificabili
contro un riferimento scritto a mano, senza GPU e senza faceset.
"""
import multiprocessing
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch

from core.cv2ex import cv2_imread
from core.imagelib import estimate_sharpness
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import LandmarksProcessor


def descrittore_istogrammi(hists):
    """Da istogrammi a un descrittore in cui la distanza e' un prodotto scalare.

    La formula di cv2.HISTCMP_BHATTACHARYYA e'

        d(H1,H2) = sqrt( 1 - sum_i sqrt(H1_i H2_i) / sqrt(sum H1 sum H2) )

    e il termine sotto radice si fattorizza: con p = sqrt(H / sum H) vale
    sqrt(H1_i H2_i) / sqrt(sum H1 sum H2) = p1_i p2_i, quindi la somma su i
    e' p1 . p2 e la matrice completa delle distanze e' una sola GEMM.

    `hists` e' una sequenza di istogrammi -- ognuno un array qualunque, o
    una sequenza di array quando i canali sono piu' di uno, che vengono
    concatenati nell'ordine dato.
    """
    righe = []
    for voce in hists:
        if isinstance(voce, (list, tuple)):
            riga = np.concatenate([np.asarray(c, dtype=np.float32).ravel()
                                   for c in voce])
        else:
            riga = np.asarray(voce, dtype=np.float32).ravel()
        righe.append(riga)

    X = np.stack(righe) if righe else np.zeros((0, 0), dtype=np.float32)
    somme = X.sum(axis=1, keepdims=True)
    # Un istogramma tutto a zero non puo' esistere (un'immagine ha pixel),
    # ma se arrivasse dividerebbe per zero: la somma a 1 lo lascia a zero.
    somme[somme == 0] = 1.0
    return np.sqrt(X / somme).astype(np.float32)


def bhattacharyya_matrice(P, device="cpu"):
    """La matrice completa N x N delle distanze, come una GEMM."""
    t = torch.as_tensor(P, device=device)
    D = torch.sqrt((1.0 - t @ t.T).clamp_min_(0))
    return D.cpu().numpy().astype(np.float32)


def catena_greedy(P, device="cpu", partenza=0):
    """L'ordine del vicino piu' prossimo, una riga per passo.

    Non costruisce mai la matrice piena: a ogni passo serve una riga sola,
    ed e' cio' che rende trattabili le decine di migliaia di immagini. La
    radice quadrata e' monotona, quindi per l'argmin si lavora su
    1 - p . p_cur senza calcolarla.

    Sui pareggi esatti sceglie l'indice originale piu' basso. La versione
    precedente permutava la lista con scambi e a parita' di distanza
    sceglieva la posizione piu' bassa fra le rimanenti: sui pareggi in
    float32 i due ordini possono differire, ed e' voluto.
    """
    n = int(np.asarray(P).shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    t = torch.as_tensor(P, device=device)
    usati = torch.zeros(n, dtype=torch.bool, device=device)
    ordine = torch.empty(n, dtype=torch.long, device=device)

    cur = int(partenza)
    usati[cur] = True
    ordine[0] = cur
    for k in range(1, n):
        d = 1.0 - (t @ t[cur])
        d[usati] = float("inf")
        cur = int(torch.argmin(d))
        usati[cur] = True
        ordine[k] = cur

    return ordine.cpu().numpy().astype(np.int64)


def catena_da_matrice(D, piu_lontano=False, partenza=0):
    """L'ordine del vicino (o del lontano) piu' prossimo, da una matrice data.

    La gemella di catena_greedy per i casi in cui la distanza non e' un
    prodotto scalare e la matrice esiste gia'. Sui pareggi sceglie l'indice
    piu' basso, come lei.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    usati = np.zeros(n, dtype=bool)
    ordine = np.empty(n, dtype=np.int64)
    cur = int(partenza)
    usati[cur] = True
    ordine[0] = cur
    for k in range(1, n):
        riga = D[cur].astype(np.float64).copy()
        riga[usati] = -np.inf if piu_lontano else np.inf
        cur = int(np.argmax(riga) if piu_lontano else np.argmin(riga))
        usati[cur] = True
        ordine[k] = cur
    return ordine


def bin_lineare(valori, bordi):
    """L'indice di grado di ogni valore, con gli estremi che catturano la coda.

    Sostituisce un ciclo su gradi x N con tre condizioni per elemento. I due
    bordi non sono simmetrici e non e' un dettaglio: il primo grado prende
    tutto cio' che sta sotto il secondo bordo, l'ultimo tutto cio' che sta
    sopra il proprio -- cosi' nessun valore resta fuori, nemmeno quelli
    oltre l'intervallo nominale.

    Nota: si chiama bin_lineare e non bin_2d perche' il binning per pitch e'
    annidato *dentro* quello per yaw e riusa la stessa funzione una seconda
    volta, invece di essere un secondo asse della stessa griglia.
    """
    valori = np.asarray(valori, dtype=np.float32)
    gradi = len(bordi)
    if gradi <= 1:
        return np.zeros(len(valori), dtype=np.int64)
    # digitize con i bordi interni: il risultato e' gia' 0 per cio' che sta
    # sotto bordi[1] e gradi-1 per cio' che sta sopra bordi[-1].
    return np.digitize(valori, np.asarray(bordi)[1:]).astype(np.int64)


def l1_a_blocchi(X, device="cpu", blocco=256):
    """La matrice N x N delle distanze L1, a blocchi.

    `X` e' (N, D) e omogeneo: le immagini arrivano gia' appiattite e della
    stessa forma, perche' un faceset a risoluzione mista non si puo'
    impilare e il controllo va fatto dove i file si leggono, non qui.

    A blocchi per due ragioni indipendenti: il temporaneo di una cdist
    piena su immagini a piena risoluzione non entrerebbe in memoria, e il
    ciclo Python che questa funzione sostituisce allocava un tensore
    (batch, H, W, 3) per ogni immagine j.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    t = torch.as_tensor(X, device=device)
    if t.dtype != torch.float32:
        t = t.float()

    out = torch.empty((n, n), dtype=torch.float32, device=device)
    for a in range(0, n, blocco):
        fetta_a = t[a:a + blocco]
        for b in range(0, n, blocco):
            out[a:a + fetta_a.shape[0], b:b + t[b:b + blocco].shape[0]] = \
                torch.cdist(fetta_a, t[b:b + blocco], p=1)

    # copy=False e non una copia: `out` e' gia' float32, e su un faceset
    # grande la copia raddoppierebbe il termine quadratico del fabbisogno --
    # quello che il chiamante stima prima di allocare, e che deve descrivere
    # il picco vero.
    return out.cpu().numpy().astype(np.float32, copy=False)


def absdiff_batch(i_images, j_images, device=None):
    """
    db[jj, ii] = somma di |i_images[ii] - j_images[jj]| su tutti i pixel.

    Le ~25 righe TF che questa funzione sostituisce (due placeholder, 512 nodi
    reduce_sum pre-costruiti e due closure che li eseguono) erano il modo di
    esprimere questo in un grafo statico.

    Le due closure erano una sola cosa. func_bs_remain troncava le uscite a
    batch_size_remain ed era scelta quando image_paths_len - j < batch_size,
    cioe' esattamente quando j_images ha batch_size_remain elementi: entrambe
    calcolavano "una uscita per immagine j". Il ciclo qui sotto gira su
    len(j_images) e le copre tutte e due.

    Il dtype non e' un dettaglio: cv2_imread consegna uint8, e in uint8
    255 - 0 e 0 - 255 danno entrambi 255 mentre la somma delle differenze ne
    vuole il valore assoluto vero. Il float32 e' anche cio' che faceva il TF,
    che alimentava un placeholder tf.float32.
    """
    if device is None:
        device = nn.device

    i_t = torch.as_tensor(np.asarray(i_images), dtype=torch.float32, device=device)
    j_t = torch.as_tensor(np.asarray(j_images), dtype=torch.float32, device=device)

    diffs = [ torch.sum(torch.abs(i_t - j_t[i]), dim=[1,2,3]) for i in range(j_t.shape[0]) ]
    return torch.stack(diffs).cpu().numpy()


# Sotto questo numero di immagini la CPU vince: il costo di inizializzare il
# contesto CUDA non viene ripagato. Il valore e' misurato, non scelto; la
# corsa che l'ha localizzato e' a verbale col suo comando.
SOGLIA_GPU = 8000

# Quanta memoria libera pretendere oltre a quella richiesta. Un sort non
# deve far cadere un training che gira nello stesso momento.
MARGINE_VRAM = 1.5


def scegli_device(n, byte_richiesti=0):
    """"cpu" oppure "cuda", e lo dice sempre.

    Tre condizioni in and: il numero di immagini supera SOGLIA_GPU, esiste
    un device CUDA, e la memoria libera copre il richiesto per MARGINE_VRAM.

    La riga di log non e' un abbellimento: la scelta e' implicita, e una
    scelta implicita che si dichiara non e' una scelta nascosta. Chi vede
    trenta secondi invece di cinque deve poter leggere il perche' senza
    aprire il codice.
    """
    if n <= SOGLIA_GPU:
        io.log_info(f"Using CPU: {n} images is below the {SOGLIA_GPU} threshold.")
        return "cpu"

    if not torch.cuda.is_available():
        io.log_info(f"Using CPU: no CUDA device available for {n} images.")
        return "cpu"

    try:
        libera, _totale = torch.cuda.mem_get_info()
    except Exception as e:
        io.log_info(f"Using CPU: cannot read free GPU memory ({e}).")
        return "cpu"

    servono = byte_richiesti * MARGINE_VRAM
    if libera <= servono:
        io.log_info(
            f"Using CPU: {libera // 1024**2} MiB free on the GPU, "
            f"{int(servono) // 1024**2} MiB needed with margin.")
        return "cpu"

    io.log_info(f"Using CUDA for {n} images, "
                f"{libera // 1024**2} MiB free on the device.")
    return "cuda"


# I campi che un metodo puo' chiedere. Il caricatore fa una sola passata sul
# disco e riempie solo quelli chiesti.
C_METADATI = "metadati"          # source_filename, source_rect, shape
C_POSA = "posa"                  # yaw, pitch, roll
C_INGRANDIMENTO = "ingrandimento"  # upscale, dalla matrice di allineamento
C_SHARPNESS = "sharpness"
C_MOTION_BLUR = "motion_blur"
C_HIST = "hist"                  # tre canali, mascherato quando si puo'
C_HSV = "hsv"                    # brightness, hue (radianti su (-pi, pi], non gradi OpenCV [0, 180))
C_BLACK = "black"
C_HASH = "hash"                  # dHash a 64 bit
C_MINIATURA = "miniatura"        # 32x32x3 appiattito, uint8

# Il lato della miniatura. E' l'unico campo del descrittore che porta dei
# pixel, e a questo lato ne porta 3072 -- meno di un'immagine anche piccola,
# ed e' l'invariante che tiene costante la memoria del padre.
LATO_MINIATURA = 32

# I campi che non pretendono i metadati DFL: un JPEG qualunque li produce.
# Per tutti gli altri, un file senza metadati e' uno scarto -- ed e' la
# distinzione che prima non esisteva, perche' lo zero valeva per entrambi.
CAMPI_SENZA_DFL = {C_HSV, C_BLACK, C_HASH, C_HIST, C_MINIATURA}

_CAMPI_DESCRITTORE = [
    "path", "source_filename", "source_rect", "shape",
    "yaw", "pitch", "roll", "upscale", "sharpness", "motion_blur",
    "hist", "brightness", "hue", "black", "hash", "miniatura",
]

Descrittore = namedtuple("Descrittore", _CAMPI_DESCRITTORE)
Descrittore.__new__.__defaults__ = (None,) * (len(_CAMPI_DESCRITTORE) - 1)

# Oltre questo numero di figli il carico diventa I/O-bound e aggiungere CPU
# non paga piu' -- lo stesso ordine di grandezza dei generatori dei campioni.
MAX_FIGLI = 8

# I campi la cui pipeline pixel consuma la maschera dei landmark (sharpness/
# motion_blur la usano per isolare il volto, hist per escludere il fondo).
# Gli altri (hsv, black, hash) non la guardano mai: costruirla per loro e'
# solo un convexHull+fillConvexPoly sprecato, ed era la regressione della
# voce A -- un file coi soli metadati ma senza landmark veniva scartato
# anche per campi che non ne avevano bisogno.
_CAMPI_CON_MASCHERA = {C_SHARPNESS, C_MOTION_BLUR, C_HIST}


def _dhash(gray):
    """dHash a 64 bit: il confronto fra pixel adiacenti di un 9x8.

    Restituisce un intero, non un array: la distanza di Hamming fra hash si
    calcola sui bit, e un intero e' cio' che attraversa un confine di
    processo senza costare niente.
    """
    piccola = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bit = (piccola[:, 1:] > piccola[:, :-1]).ravel()
    valore = 0
    for b in bit:
        valore = (valore << 1) | int(b)
    return valore


def _ingrandimento(mat):
    """Di quanto e' stata ingrandita la regione ritagliata dal fotogramma.

    `image_to_face_mat` e' l'affine che ha portato il fotogramma nel volto
    allineato: il suo fattore di scala lineare -- la radice del modulo del
    determinante, cioe' il rapporto fra il lato dell'immagine allineata e il
    lato equivalente della regione sorgente -- E' l'ingrandimento, senza
    bisogno di invertirla o di mappare angoli. Sopra 1 la faccia e' stata
    tirata su da pochi pixel, sotto 1 rimpicciolita.

    Non si usa il source_rect: quel rettangolo e' il riquadro grezzo del
    rilevatore, mentre la regione davvero ritagliata la decidono i landmark
    e il tipo di volto, e le due non stanno in un rapporto costante nemmeno
    dentro un faceset omogeneo.

    Torna None quando la matrice manca o non e' invertibile -- landmark
    degeneri (tutti nello stesso punto) producono una matrice di NaN, e un
    NaN in una chiave di ordinamento non e' un valore, e' un ordine
    arbitrario. Chi chiama lo tratti come un file che non si e' potuto
    leggere.
    """
    if mat is None:
        return None
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (2, 3) or not np.isfinite(mat).all():
        return None
    det = abs(float(np.linalg.det(mat[:, :2])))
    return float(np.sqrt(det)) if det > 0 else None


class CaricatoreDescrittori(Subprocessor):
    """Una passata parallela sul disco, un descrittore per immagine.

    Restituisce descrittori, mai immagini: la memoria del padre non cresce
    con la risoluzione del faceset. Le classi che questa sostituisce
    trattenevano fino a ventimila immagini decodificate in un figlio.
    """

    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.campi = client_dict['campi']
            # Ogni figlio deve cappare i propri thread: con spawn non
            # eredita niente dal padre, e senza il cap N figli su N core
            # chiedono N*N thread.
            cv2.setNumThreads(1)

            # Diagnosticabilita': senza questa sonda, un pacchetto mancante
            # arriva come N scarti silenziosi indistinguibili da N file
            # corrotti -- e' esattamente cosi' che e' stato scoperto qui che
            # scikit-image (dietro estimate_sharpness, per C_SHARPNESS) non
            # era installato. Un errore solo, chiaro, all'avvio del figlio.
            if C_SHARPNESS in self.campi:
                try:
                    import skimage  # noqa: F401
                except ImportError as e:
                    raise ImportError(
                        "il campo 'sharpness' richiede il pacchetto "
                        f"'scikit-image' (assente: {e})") from e

        #override
        def process_data(self, data):
            try:
                filepath = Path(data[0])
                campi = self.campi
                valori = {"path": str(filepath)}

                serve_dfl = bool(campi - CAMPI_SENZA_DFL)
                dflimg = DFLIMG.load(filepath)
                ha_dati = dflimg is not None and dflimg.has_data()

                if serve_dfl and not ha_dati:
                    return [1, str(filepath)]

                if ha_dati:
                    if C_METADATI in campi:
                        valori["source_filename"] = dflimg.get_source_filename()
                        valori["source_rect"] = dflimg.get_source_rect()
                        valori["shape"] = dflimg.get_shape()
                    if C_POSA in campi:
                        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(
                            dflimg.get_landmarks(), size=dflimg.get_shape()[1])
                        valori.update(yaw=float(yaw), pitch=float(pitch),
                                      roll=float(roll))
                    if C_INGRANDIMENTO in campi:
                        fattore = _ingrandimento(dflimg.get_image_to_face_mat())
                        if fattore is None:
                            return [1, str(filepath)]
                        valori["upscale"] = fattore

                serve_pixel = bool(campi - {C_METADATI, C_POSA,
                                            C_INGRANDIMENTO})
                if serve_pixel:
                    bgr = cv2_imread(str(filepath))
                    if bgr is None:
                        return [1, str(filepath)]

                    # La maschera serve solo ai campi che la consumano
                    # davvero (sharpness/motion_blur/hist): costruirla per
                    # hsv/black/hash e' spreco puro, ed era la regressione
                    # della voce A -- un file coi soli metadati ma senza
                    # landmark veniva scartato anche per quei campi, che non
                    # guardano mai la maschera. Lettura dei landmark in
                    # forma difensiva: un file coi metadati ma senza quella
                    # chiave degrada (maschera assente) invece di sollevare.
                    maschera = None
                    if ha_dati and (campi & _CAMPI_CON_MASCHERA):
                        landmarks = dflimg.get_dict().get('landmarks')
                        if landmarks is not None:
                            maschera = LandmarksProcessor.get_image_hull_mask(
                                bgr.shape, np.array(landmarks))

                    if C_SHARPNESS in campi or C_MOTION_BLUR in campi:
                        if maschera is not None:
                            mascherata = (bgr * maschera).astype(np.uint8)
                        else:
                            mascherata = bgr
                        if C_SHARPNESS in campi:
                            valori["sharpness"] = float(
                                estimate_sharpness(mascherata))
                        if C_MOTION_BLUR in campi:
                            valori["motion_blur"] = float(cv2.Laplacian(
                                mascherata, cv2.CV_64F, ksize=11).var())

                    if C_HIST in campi:
                        # mask= esclude i pixel fuori maschera invece di
                        # contarli come neri: moltiplicare per la maschera
                        # ammucchiava tutto il fondo nel bin 0, dove
                        # dominava la distanza.
                        m8 = None
                        if maschera is not None:
                            m8 = (maschera[..., 0] > 0).astype(np.uint8)
                        valori["hist"] = tuple(
                            cv2.calcHist([bgr], [c], m8, [256], [0, 256])
                            for c in range(3))

                    if C_HSV in campi:
                        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                        valori["brightness"] = float(np.mean(hsv[..., 2]))
                        # La tinta e' circolare: la media va fatta sui
                        # vettori, non sui numeri, o 0 e 179 danno 90.
                        # OpenCV mette la tinta in [0, 180); il campo
                        # restituito e' pero' in RADIANTI su (-pi, pi], non
                        # in unita' OpenCV -- non va assunto [0, 180).
                        ang = hsv[..., 0].astype(np.float32) * (np.pi / 90.0)
                        valori["hue"] = float(np.arctan2(
                            np.sin(ang).mean(), np.cos(ang).mean()))

                    if C_BLACK in campi:
                        valori["black"] = int(np.count_nonzero(bgr == 0))

                    if C_HASH in campi:
                        valori["hash"] = _dhash(
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

                    if C_MINIATURA in campi:
                        piccola = cv2.resize(
                            bgr, (LATO_MINIATURA, LATO_MINIATURA),
                            interpolation=cv2.INTER_AREA)
                        valori["miniatura"] = piccola.reshape(-1)

                return [0, Descrittore(**valori)]

            except Exception as e:
                # Tutto il corpo e' in questo try, non solo il pezzo pixel:
                # un'eccezione fuori da esso veniva ignorata dalla classe
                # base (figlio ucciso, dato ri-accodato) e, quando l'ultimo
                # figlio moriva, il risultato tornava troncato senza
                # sollevare (voce B). Qui invece diventa sempre uno scarto
                # esplicito, mai una sparizione silenziosa -- per questo
                # il messaggio usa data[0] cosi' com'e' (gia' una stringa),
                # non Path(data[0]): costruire un Path qui potrebbe a sua
                # volta sollevare, e l'except di un except non e' catturato
                # da nessuno.
                self.log_err(f"{data[0]}: {e}")
                return [1, str(data[0])]

        #override
        def get_data_name(self, data):
            return data[0]

    #override
    def __init__(self, image_paths, campi):
        self.input_data = list(image_paths)
        self.campi = set(campi)
        self.validi = []
        self.scarti = []
        super().__init__('CaricatoreDescrittori', CaricatoreDescrittori.Cli, 60)

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), MAX_FIGLI)
        io.log_info(f'Running on {cpu_count} CPUs')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'campi': self.campi}

    #override
    def on_clients_initialized(self):
        io.progress_bar("Loading", len(self.input_data))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len(self.input_data) > 0:
            return [self.input_data.pop(0)]
        return None

    #override
    def on_data_return(self, host_dict, data):
        self.input_data.insert(0, data[0])

    #override
    def on_result(self, host_dict, data, result):
        if result[0] == 0:
            self.validi.append(result[1])
        else:
            self.scarti.append(result[1])
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.validi, self.scarti


def carica_descrittori(image_paths, campi):
    """(validi, scarti) in una passata parallela. Su lista vuota non lancia niente."""
    image_paths = list(image_paths)
    if not image_paths:
        return [], []
    return CaricatoreDescrittori(image_paths, campi).run()


# Righe per blocco in coppie_simili: sia il default della funzione sia la
# stima di memoria che sceglie il device leggono questa costante, cosi' non
# possono divergere silenziosamente l'una dall'altra.
BLOCCO_HASH = 2048


def coppie_simili(hashes, soglia, device="cpu", blocco=BLOCCO_HASH):
    """Le coppie (i, j) con i < j la cui distanza di Hamming sta sotto soglia.

    La distanza fra due hash a 64 bit e' un prodotto matriciale: con i bit
    come +1/-1 in una matrice (N, 64), il prodotto scalare di due righe vale
    64 - 2*hamming, quindi hamming = (64 - X @ X.T) / 2.

    A blocchi, e restituendo le coppie invece della matrice: quella piena a
    ventimila immagini sarebbe quattrocento megabyte, e non serve a niente.
    """
    valori = np.asarray(hashes, dtype=np.uint64)
    n = valori.shape[0]
    if n < 2:
        return []

    X = np.empty((n, 64), dtype=np.float32)
    for b in range(64):
        X[:, b] = ((valori >> np.uint64(b)) & np.uint64(1)).astype(np.float32) * 2.0 - 1.0

    t = torch.as_tensor(X, device=device)
    coppie = []
    for a in range(0, n, blocco):
        fetta = t[a:a + blocco]
        for b in range(a, n, blocco):
            dist = (64.0 - fetta @ t[b:b + blocco].T) / 2.0
            righe, colonne = torch.nonzero(dist <= soglia, as_tuple=True)
            for r, c in zip(righe.tolist(), colonne.tolist()):
                i, j = a + r, b + c
                if i < j:
                    coppie.append((i, j))
    return coppie


def gruppi_da_coppie(n, coppie):
    """Le componenti connesse di dimensione maggiore di uno.

    Se A somiglia a B e B a C, i tre stanno nello stesso gruppo anche quando
    A e C non si somigliano: tenerne una sola e' la cosa giusta, e una catena
    di frame consecutivi e' esattamente questo caso.
    """
    padre = list(range(n))

    def radice(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for i, j in coppie:
        ri, rj = radice(i), radice(j)
        if ri != rj:
            padre[rj] = ri

    per_radice = {}
    for x in range(n):
        per_radice.setdefault(radice(x), []).append(x)
    return [g for g in per_radice.values() if len(g) > 1]


def normalizza_colonne(X):
    """Ogni colonna a media zero e scarto uno.

    Senza, la colonna con la scala piu' grande domina la distanza: la
    luminosita' sta fra 0 e 255, gli angoli fra -1.2 e 1.2, e la copertura
    finirebbe per essere solo copertura di luminosita'.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X
    scarti = X.std(axis=0)
    # Una colonna costante non porta informazione: azzerarla invece di
    # dividere per zero.
    scarti[scarti == 0] = 1.0
    return ((X - X.mean(axis=0)) / scarti).astype(np.float32)


def campiona_lontani(X, k):
    """Farthest-point sampling: k punti che massimizzano la distanza minima.

    Parte dal punto piu' vicino al centroide e aggiunge ogni volta quello
    piu' lontano da tutti i gia' scelti, tenendo la distanza minima corrente
    in un vettore -- O(N*k), senza mai costruire una matrice N x N.

    L'ordine di uscita e' quello di scelta: il primo e' il piu'
    rappresentativo, e i successivi aggiungono ognuno il massimo di
    copertura che resta.

    Non e' un campione uniforme, e chi lo chiama non lo presenti come tale:
    massimizzare la distanza minima privilegia gli estremi, e a k piccoli
    puo' lasciare vuoti gli intervalli in mezzo.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n == 0 or k <= 0:
        return np.zeros((0,), dtype=np.int64)
    k = min(k, n)

    centro = X.mean(axis=0)
    primo = int(np.argmin(((X - centro) ** 2).sum(axis=1)))

    scelti = [primo]
    minime = ((X - X[primo]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        # Il gia' scelto esce dalla gara con un valore che nessuna distanza
        # puo' raggiungere: -1 sopravvive al minimo dei passi successivi,
        # mentre uno zero verrebbe pareggiato da un doppione esatto.
        minime[scelti[-1]] = -1.0
        prossimo = int(np.argmax(minime))
        scelti.append(prossimo)
        minime = np.minimum(minime, ((X - X[prossimo]) ** 2).sum(axis=1))

    return np.array(scelti, dtype=np.int64)
