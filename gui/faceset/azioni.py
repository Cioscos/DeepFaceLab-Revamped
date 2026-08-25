"""Le nove operazioni della pagina, e a quale passo del catalogo
corrispondono.

La corrispondenza e' SCRITTA, non dedotta a runtime: dedurla dagli
argomenti significherebbe indovinare, e un'operazione che lancia il passo
sbagliato riordina o cestina la cartella sbagliata. La guardia in
tests_gui/ verifica che i gemelli src/dst differiscano solo per la
cartella di input, cosi' la scrittura a mano non puo' derivare dal
catalogo.

Diciassette passi, nove operazioni: cinque hanno il gemello dst, quattro
esistono solo per src, tre sono viewer e diventano «Open in file manager»
sul selettore di cartella.

**Le nove `etichetta` sono testi visibili, e vivono qui invece che in
`gui/testi.py`.** Stanno accanto alla riga che dichiara i due passi
gemelli perche' e' li' che si legge cosa quell'etichetta promette, e la
`Operazione` e' una riga sola. La conseguenza va detta: la rete AST di
`tests_gui/test_testi_soltanto_da_un_posto.py` non le vede, perche'
`menu_strumenti.addAction(op.etichetta)` le porta a schermo attraverso un
**attributo**, e la rete guarda solo i letterali dentro il sottoalbero
della chiamata al widget. E' il buco n. 2 gia' dichiarato nel docstring
di quella guardia (una funzione che compone un testo altrove), qui
allargato di nove etichette: chi le riscrive non ha nessuna rete sotto,
solo questa nota.

`passo_dst=None` significa «questa operazione non esiste per il dataset
dst». Non lo dice `passo_per`, che ricade su `passo_src` di proposito
(vedi li'): lo dice il **dato**, ed e' sul dato che i chiamanti devono
filtrare -- vedi `gui/faceset/pagina.py::_ha_il_passo`.
"""
from collections import namedtuple
from pathlib import Path

from gui.catalog.model import KIND_MAIN, PROCESS_BATCH, Invocation, StepDef
from gui.faceset.cestino import SUFFISSO, e_un_cestino

# Non sta nel catalogo perche' non e' un passo del workflow dell'utente:
# nessuno script generato lo lancia, la formalizzazione non lo conosce, e
# registrarlo li' farebbe rosse -- a ragione -- le guardie che camminano
# all_steps() contro commands.toml.
#
# Dichiara `modifies` vuoto: scrive solo nella cache, che vive fuori dal
# progetto (gui/faceset/cache.py), quindi non contende nessun artefatto e
# puo' girare mentre si guarda la griglia. Un solo `--input-dir`, che e'
# cio' che _sostituisci_input_dir pretende (solleva se manca, se e'
# duplicato o se penzola); il valore e' un segnaposto, la pagina lo
# sostituisce sempre con la cartella mostrata.
PASSO_INDICE = StepDef(
    name="Index faceset",
    family="cura-faceset",
    kind=KIND_MAIN,
    process=PROCESS_BATCH,
    summary="Reads pose and masks of every face into the interface's cache.",
    invocations=(Invocation(verb=("facesettool", "index"),
                            args=("--input-dir", "{WORKSPACE}/data_src/aligned")),),
    optional=True,
)

Operazione = namedtuple("Operazione", [
    "chiave", "etichetta", "passo_src", "passo_dst", "solo_allineate",
])

# solo_allineate=True: l'operazione ha senso solo su una cartella di volti
# allineati. Su una cartella di fotogrammi grezzi non e' una scelta
# discutibile, e' un errore -- e va detto, non lasciato provare.
OPERAZIONI = (
    Operazione("sort", "Sort…",
               "4.2) data_src sort",
               "5.2) data_dst sort", False),
    Operazione("pack", "Faceset pack",
               "4.2) data_src util faceset pack",
               "5.2) data_dst util faceset pack", True),
    Operazione("unpack", "Faceset unpack",
               "4.2) data_src util faceset unpack",
               "5.2) data_dst util faceset unpack", True),
    Operazione("resize", "Faceset resize…",
               "4.2) data_src util faceset resize",
               "5.2) data_dst util faceset resize", True),
    Operazione("recover-names", "Recover original filenames",
               "4.2) data_src util recover original filename",
               "5.2) data_dst util recover original filename", True),
    Operazione("enhance", "Faceset enhance…",
               "4.2) data_src util faceset enhance", None, True),
    Operazione("landmarks-debug", "Add landmarks debug images",
               "4.2) data_src util add landmarks debug images", None, True),
    Operazione("metadata-save", "Save faceset metadata",
               "4.2) data_src util faceset metadata save", None, True),
    Operazione("metadata-restore", "Restore faceset metadata",
               "4.2) data_src util faceset metadata restore", None, True),
)

# Elenco dei PERMESSI, non dei divieti -- l'asimmetria dei costi impone
# questa forma: un falso negativo (una cartella legittima che l'elenco non
# prevede) costa un'azione grigiata con la sua ragione scritta accanto,
# visibile e recuperabile (e lanciabile comunque da riga di comando); un
# falso positivo costa file cancellati in silenzio. Solo le tre cartelle
# che DeepFaceLab scrive davvero come faceset di volti: quella allineata
# originale, e le due varianti che facesettool resize/enhance producono.
# `aligned_debug` resta fuori di proposito: mainscripts/Extractor.py la
# scrive come fotogrammi INTERI con landmark e rettangoli disegnati sopra
# (cv2_imwrite semplice, nessun metadato DFL), non come volti allineati --
# un'operazione solo_allineate=True approvata li' degraderebbe in silenzio
# (i caricatori saltano i file non-DFL), e "faceset pack" seguito dal
# default "Delete original files? yes" cancellerebbe le immagini di debug
# da cui non ha impacchettato niente.
BASI_FACESET = ("aligned", "aligned_resized", "aligned_enhanced")

# Il nome che samplelib/PackedFaceset.py scrive (`packed_faceset_filename`).
# Ripetuto qui e non importato: `gui/` non importa MAI samplelib -- parla coi
# verbi di main.py in sottoprocesso e basta. La guardia in tests_gui/ legge la
# costante dal sorgente di samplelib e la confronta con questa, cosi' la
# ripetizione non puo' divergere in silenzio.
NOME_PACCHETTO = "faceset.pak"

# I motivi del rifiuto sono CODICI, non frasi: la frase inglese la sceglie
# gui/testi.py. Stessa scelta di mainscripts/DettaglioGuasti.py, e per la
# stessa ragione -- riconoscere un rifiuto dal testo si romperebbe in
# silenzio alla prima riformulazione.
MOTIVO_NON_ALLINEATA = "aligned-only"
MOTIVO_IMPACCHETTATA = "packed"

# L'unica operazione che ha ancora senso su una cartella impacchettata: e'
# quella che la disfa. Tutte le altre girerebbero su zero file -- pack
# chiederebbe di sovrascrivere il .pak con niente, sort e resize
# concluderebbero in un attimo senza toccare un volto -- e un lavoro che
# finisce bene senza aver fatto niente e' peggio di un'azione grigia con la
# sua ragione scritta accanto.
CHIAVE_UNPACK = "unpack"


def _senza_input_dir(args):
    """Gli argomenti tolto --input-dir e il suo valore, per il confronto
    fra gemelli."""
    fuori = []
    salta = False
    for i, a in enumerate(args):
        if salta:
            salta = False
            continue
        if a == "--input-dir":
            salta = True
            continue
        fuori.append(a)
    return fuori


def passo_per(operazione, dataset):
    if dataset == "dst" and operazione.passo_dst:
        return operazione.passo_dst
    return operazione.passo_src


def e_allineata(cartella):
    nome = Path(cartella).name
    base = nome[:-len(SUFFISSO)] if e_un_cestino(cartella) else nome
    return base in BASI_FACESET


def ha_pacchetto(cartella):
    """Se la cartella porta un `faceset.pak`.

    Solo il disco, nessun contenuto letto: aprire il pacchetto per contare
    cosa c'e' dentro vorrebbe dire caricare samplelib, che qui non si puo'
    e non si deve.
    """
    if cartella is None:
        return False
    return (Path(cartella) / NOME_PACCHETTO).is_file()


def applicabile(operazione, cartella, impacchettata=False):
    """(ammessa, motivo). Il motivo e' vuoto quando e' ammessa.

    `impacchettata` non e' `ha_pacchetto(cartella)`: e' «il pacchetto c'e' E
    non c'e' nessun volto sciolto». Chi risponde «no» a «Delete original
    files?» tiene i due insiemi affiancati, e li' la griglia mostra i volti e
    ogni operazione ha ancora un senso -- rifiutarle sarebbe un falso
    positivo. Il calcolo lo fa la pagina, che la cartella l'ha gia' letta una
    volta e non deve rileggerla.
    """
    if operazione.solo_allineate and not e_allineata(cartella):
        return False, MOTIVO_NON_ALLINEATA
    if impacchettata and operazione.chiave != CHIAVE_UNPACK:
        return False, MOTIVO_IMPACCHETTATA
    return True, ""
