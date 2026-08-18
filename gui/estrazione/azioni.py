"""Corrispondenza fra le operazioni della pagina e i passi del catalogo.

Scritta a mano, come in gui/faceset/azioni.py: il nome di un passo e' una
stringa scritta da un umano in scripts/commands.toml, non un
identificatore, e dedurne l'operazione sarebbe fragile.

Le `etichetta` sono **testo visibile, quindi in inglese**, e vivono qui
invece che in gui/testi.py per la stessa ragione del gemello
(gui/faceset/azioni.py): stanno accanto alla riga che dichiara i due passi
gemelli, ed e' li' che si legge cosa quell'etichetta promette. La rete AST
di tests_gui/test_testi_soltanto_da_un_posto.py non le vede -- arrivano al
widget attraverso un attributo, non un letterale -- quindi chi le riscrive
non ha nessuna rete sotto, solo questa nota.
"""
import collections

from gui.catalog.model import KIND_MAIN, PROCESS_BATCH, Invocation, StepDef

Operazione = collections.namedtuple(
    "Operazione", "chiave etichetta passo_src passo_dst")

# Non sta nel catalogo, come il gemello faceset (gui/faceset/azioni.py::PASSO_INDICE):
# nessuno script generato lo lancia, la formalizzazione non lo conosce.
# `modifies` vuoto: scrive solo nella cache fuori dal progetto
# (gui/faceset/cache.py::id_cartella, riusata da gui/estrazione/pagina.py),
# quindi non contende nessun artefatto. E' il ripiego per le cartelle
# estratte prima che il rapporto per frame esistesse, che non ne hanno mai
# scritto uno: senza un chiamante di produzione sarebbe irraggiungibile.
PASSO_INDICE = StepDef(
    name="Index extraction report",
    family="estrazione",
    kind=KIND_MAIN,
    process=PROCESS_BATCH,
    summary="Rebuilds the per-frame report for a folder extracted before this feature existed, from aligned/ and aligned_debug/, without re-running detection.",
    invocations=(Invocation(verb=("extracttool", "index"),
                            args=("--input-dir", "{WORKSPACE}/data_src")),),
    optional=True,
)

OPERAZIONI = (
    Operazione("auto", "Extract automatically",
               "4) data_src faceset extract",
               "5) data_dst faceset extract"),
    Operazione("manuale", "Extract manually",
               "4) data_src faceset extract MANUAL",
               "5) data_dst faceset extract MANUAL"),
    # Punta allo STESSO passo di "auto" invece di
    # "+ manual fix" -- quel passo resta nel catalogo e resta lanciabile dal
    # `.bat`, che apre davvero la finestra cv2, ma la GUI non lo usa piu':
    # e' PaginaEstrazione._su_job_finito a entrare da sola nella sessione
    # manuale nativa sui frame senza volto, a job finito, sostituendo la
    # finestra esterna invece di aprirla.
    Operazione("auto-con-correzione", "Extract and fix the misses",
               None,
               "5) data_dst faceset extract"),
    Operazione("riestrai-selezione", "Re-extract the selected frames",
               None,
               "5) data_dst faceset MANUAL RE-EXTRACT DELETED ALIGNED_DEBUG"),
)

_PER_CHIAVE = dict((op.chiave, op) for op in OPERAZIONI)


def passo_per(chiave, lato):
    """Torna il nome del passo del catalogo per (operazione, lato).

    Solleva KeyError su una chiave sconosciuta invece di tornare None: un
    None qui diventerebbe un comando composto a meta' molto piu' tardi.
    """
    op = _PER_CHIAVE[chiave]
    nome = op.passo_src if lato == "src" else op.passo_dst
    if nome is None:
        raise KeyError("l'operazione %r non esiste per il lato %r" % (chiave, lato))
    from gui.catalog import extraction
    for passo in extraction.STEPS:
        if passo.name == nome:
            return passo
    raise KeyError("passo non trovato nel catalogo: %r" % nome)
