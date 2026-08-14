"""I descrittori dei metodi di ordinamento del faceset.

Sorgente unica di tre cose che prima stavano in tre posti e potevano
derivare: l'elenco dei metodi, l'ordine del menu' (che e' indicizzato per
posizione) e le choices di argparse.

Dati puri, e deve restare tale: l'interfaccia grafica e la sua suite
leggera importano questo modulo, e non devono pagare torch per leggere una
tabella. Import consentiti: collections e pathlib, niente altro.

L'ordine di METODI e' APPEND-ONLY. La memoria di progetto ricorda l'indice
scelto dall'utente, non la chiave: inserire una voce in mezzo farebbe
cambiare significato a una preferenza ricordata, e cambiare significato
verso un metodo che sposta file nel cestino.
"""
from collections import namedtuple
from pathlib import Path

# Cosa il metodo lascia sul disco.
RIORDINA = "reorder"
CESTINA = "trash"

# Cosa il metodo deve leggere per funzionare.
METADATI = "metadata"      # solo i metadati dentro il JPEG, nessuna decodifica
PIXEL = "pixels"           # decodifica l'immagine

# Ordine di grandezza del costo.
ISTANTANEO = "instant"     # non decodifica i pixel
VELOCE = "fast"            # una passata sui pixel, lineare in N
LENTO = "slow"             # quadratico in N, oppure N per il numero di scelti

# Se il metodo puo' usare la GPU.
DEVICE_MAI = "never"
DEVICE_AUTO = "auto"

# Come si torna indietro.
ORIGNAME = "origname"        # rilanciare origname ripristina la sequenza
DAL_CESTINO = "from-trash"   # origname piu' i file da riportare a mano
NESSUNO = "none"             # nessuno stato precedente a cui tornare

MetodoSort = namedtuple("MetodoSort", [
    "key",            # chiave CLI, stabile per sempre
    "label",          # nome breve in inglese, e' cio' che il menu' stampa
    "produces",       # (RIORDINA,) oppure (RIORDINA, CESTINA)
    "artefatti",      # modelli di percorso relativi all'input
    "needs",          # METADATI | PIXEL
    # Cosa il faceset deve avere perche' il risultato sia SENSATO -- non cosa
    # il codice pretende per non cadere. La differenza non e' accademica: blur
    # e motion-blur dichiarano "landmarks" e senza landmark degradano
    # comunque, misurando la nitidezza sull'immagine intera invece che sul
    # solo volto. Chi legge questo campo (l'interfaccia grafica per prima) lo
    # presenti come un avvertimento, mai come un requisito che blocca.
    "prerequisiti",
    "params",         # chiavi dei parametri per-metodo
    "costo",          # ISTANTANEO | VELOCE | LENTO
    "device",         # DEVICE_MAI | DEVICE_AUTO
    "annulla",        # ORIGNAME | DAL_CESTINO | NESSUNO
])

_CESTINO = ("../{stem}_trash",)

METODI = (
    MetodoSort("blur", "blur",
               (RIORDINA, CESTINA), _CESTINO, PIXEL, ("landmarks",), (),
               VELOCE, DEVICE_MAI, DAL_CESTINO),
    MetodoSort("motion-blur", "motion blur",
               (RIORDINA, CESTINA), _CESTINO, PIXEL, ("landmarks",), (),
               VELOCE, DEVICE_MAI, DAL_CESTINO),
    MetodoSort("face-yaw", "face yaw direction",
               (RIORDINA, CESTINA), _CESTINO, METADATI, ("landmarks",), (),
               ISTANTANEO, DEVICE_MAI, DAL_CESTINO),
    MetodoSort("face-pitch", "face pitch direction",
               (RIORDINA, CESTINA), _CESTINO, METADATI, ("landmarks",), (),
               ISTANTANEO, DEVICE_MAI, DAL_CESTINO),
    MetodoSort("face-source-rect-size", "face rect size in source image",
               (RIORDINA, CESTINA), _CESTINO, METADATI, ("source_rect",), (),
               ISTANTANEO, DEVICE_MAI, DAL_CESTINO),
    # prerequisiti vuoti di proposito: l'istogramma usa la maschera del
    # volto quando i landmark ci sono, e l'immagine intera quando non ci
    # sono. Degrada, non cestina -- e' cio' che rende vero il (RIORDINA,)
    # qui accanto anche in una cartella con un JPEG che non viene da DFL.
    MetodoSort("hist", "histogram similarity",
               (RIORDINA,), (), PIXEL, (), (),
               LENTO, DEVICE_AUTO, ORIGNAME),
    MetodoSort("hist-dissim", "histogram dissimilarity",
               (RIORDINA,), (), PIXEL, (), (),
               LENTO, DEVICE_AUTO, ORIGNAME),
    MetodoSort("brightness", "brightness",
               (RIORDINA,), (), PIXEL, (), (),
               VELOCE, DEVICE_MAI, ORIGNAME),
    MetodoSort("hue", "hue",
               (RIORDINA,), (), PIXEL, (), (),
               VELOCE, DEVICE_MAI, ORIGNAME),
    MetodoSort("black", "amount of black pixels",
               (RIORDINA,), (), PIXEL, (), (),
               VELOCE, DEVICE_MAI, ORIGNAME),
    MetodoSort("origname", "original filename",
               (RIORDINA, CESTINA), _CESTINO, METADATI, ("source_filename",), (),
               ISTANTANEO, DEVICE_MAI, NESSUNO),
    MetodoSort("oneface", "one face in image",
               (RIORDINA, CESTINA), _CESTINO, METADATI, ("nome-indicizzato",), (),
               ISTANTANEO, DEVICE_MAI, DAL_CESTINO),
    MetodoSort("absdiff", "absolute pixel difference",
               (RIORDINA,), (), PIXEL, (), ("similar",),
               LENTO, DEVICE_AUTO, ORIGNAME),
    MetodoSort("final", "best faces",
               (RIORDINA, CESTINA), _CESTINO, PIXEL, ("landmarks",),
               ("target_count",), LENTO, DEVICE_AUTO, DAL_CESTINO),
    MetodoSort("final-fast", "best faces faster",
               (RIORDINA, CESTINA), _CESTINO, PIXEL,
               ("landmarks", "source_rect"), ("target_count",),
               VELOCE, DEVICE_AUTO, DAL_CESTINO),
    # --- da qui in poi le voci nuove: SOLO in coda, mai in mezzo ---
    # LENTO e non VELOCE: il catalogo definisce VELOCE come "una passata sui
    # pixel, lineare in N" e LENTO come "quadratico in N", e il confronto qui
    # e' tutte-le-coppie. La costante e' piccola (misurato: 0.24 s la parte
    # quadratica a 20 000 hash su CPU, contro i minuti della decodifica di
    # 20 000 JPEG), ma la scala e' quella dichiarata dal campo, non il tempo
    # sul faceset di oggi.
    #
    # "duplicate faces" e non "near-duplicates", ed e' una promessa, non un
    # dettaglio di stile: alla soglia predefinita (0) il metodo raggruppa i
    # soli volti col dHash IDENTICO -- provato eseguendo, una quasi-copia a 2
    # bit di distanza (variazione di luminosita' e ricompressione JPEG) non
    # viene raggruppata. Le quasi-copie si prendono alzando la soglia, col
    # costo che il prompt dichiara. Il campo e' cio' che il menu' stampa e
    # cio' che l'interfaccia grafica leggera': non deve promettere piu' di
    # quanto il valore predefinito faccia. La forma e' quella degli altri
    # quindici -- un sintagma nominale, mai un imperativo.
    MetodoSort("dedup", "duplicate faces",
               (RIORDINA, CESTINA), _CESTINO, PIXEL, (), ("threshold",),
               LENTO, DEVICE_AUTO, DAL_CESTINO),
    # LENTO nella seconda accezione del campo -- N per il numero di scelti --
    # e non nella prima: nessuna matrice N x N nasce mai qui. DEVICE_MAI
    # perche' il conto e' un prodotto scalare su tre colonne, dove il
    # contesto CUDA costerebbe piu' di cio' che fa risparmiare.
    #
    # "most varied faces" e non "maximum coverage", e vale la stessa regola
    # di "duplicate faces" qui sopra: il campo non deve promettere piu' di
    # quanto il metodo faccia. Il campionamento a punti lontani massimizza la
    # distanza minima fra i pochi scelti, che non e' la copertura uniforme di
    # un intervallo -- misurato su un faceset vero da 400 volti, un obiettivo
    # di 20 raggiunge 11 dei 19 intervalli di imbardata popolati. La forma
    # resta quella degli altri sedici, un sintagma nominale.
    MetodoSort("coverage", "most varied faces",
               (RIORDINA, CESTINA), _CESTINO, PIXEL, ("landmarks",),
               ("target_count",), LENTO, DEVICE_MAI, DAL_CESTINO),
    # Il piu' veloce dei diciotto: legge la sola matrice di allineamento dai
    # metadati e non decodifica mai i pixel. Il prerequisito e' quella
    # matrice e non il source_rect: il rettangolo del rilevatore non e'
    # proporzionale alla regione che l'allineamento ritaglia davvero.
    # Riordina e non cestina -- dove tagliare la coda delle facce ingrandite
    # e' una decisione dell'utente, non una soglia da fissare qui.
    MetodoSort("upscale-factor", "upscale factor",
               (RIORDINA,), (), METADATI, ("image_to_face_mat",), (),
               ISTANTANEO, DEVICE_MAI, ORIGNAME),
    # ISTANTANEO come face-yaw, pur essendo l'unico metodo che apre DUE
    # faceset: il campo dichiara la CLASSE di costo, non il tempo di parete, e
    # due passate di soli metadati restano due passate di soli metadati. Un
    # fattore due non cambia la classe, e un campo che cambia significato da
    # una voce all'altra e' peggio di un campo impreciso -- chi lo legge, a
    # partire dall'interfaccia grafica, applica la stessa regola a tutte e
    # venti le voci.
    MetodoSort("match-dst", "match the other faceset's poses",
               (RIORDINA,), (), METADATI, ("landmarks",), ("ref_dir",),
               ISTANTANEO, DEVICE_MAI, ORIGNAME),
    # In coda e non accanto ad absdiff, dove si leggerebbe meglio: il menu' e'
    # indicizzato per posizione e la memoria di progetto ricorda l'indice.
    #
    # L'etichetta nomina la miniatura, e non si limita a "faster": e' il
    # menu' il posto in cui la scelta si fa, e "faster" da solo prometterebbe
    # lo stesso ordine in meno tempo. Le due voci NON danno lo stesso ordine
    # -- su un volto a dettaglio fine la miniatura media via cio' che l'altra
    # confronta -- e questa differenza e' semantica, non un'approssimazione.
    # Che sia un ordine diverso, e di quanto sia piu' rapida, lo dicono per
    # esteso le due righe che i metodi stampano appena partono.
    # LENTO come absdiff, per quanto sia cinque volte piu' rapido: il campo
    # dichiara la CLASSE di costo, e questa voce materializza la matrice N x N
    # -- tanto che ha bisogno di un tetto in memoria per rifiutarsi. Un
    # fattore cinque non fa scendere di classe un quadratico. Quanto sia piu'
    # rapido lo dice il testo che l'utente legge, che e' il posto giusto per
    # una misura; il campo dice come cresce, non quanto dura oggi.
    MetodoSort("absdiff-fast",
               "absolute pixel difference on 32x32 thumbnails, 5x faster",
               (RIORDINA,), (), PIXEL, (), ("similar",),
               LENTO, DEVICE_AUTO, ORIGNAME),
)

CHIAVI = tuple(m.key for m in METODI)

_PER_CHIAVE = {m.key: m for m in METODI}


def per_chiave(key):
    return _PER_CHIAVE[key]


def risolvi_artefatti(metodo, input_path):
    """I percorsi veri che il metodo fa nascere accanto alla cartella di input.

    Il modello "../{stem}_trash" e' relativo alla cartella di input, non al
    suo genitore: e' la stessa costruzione che final_process usa, tenuta qui
    perche' la GUI deve poter offrire un collegamento alla cartella degli
    scarti senza importare nulla di pesante.
    """
    input_path = Path(input_path)
    risolti = []
    for modello in metodo.artefatti:
        if modello == "../{stem}_trash":
            risolti.append(input_path.parent / (input_path.stem + "_trash"))
        else:
            raise ValueError(f"modello di artefatto non riconosciuto: {modello}")
    return tuple(risolti)
