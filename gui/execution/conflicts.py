"""Which steps may run at the same time.

Two jobs can coexist only if neither touches (produces or modifies) an
artifact the other uses in any way. The relations come from the step
catalog, which mirrors the workflow graph.
"""
from gui.progetti import stesso_workspace


def _touches(step):
    return set(step.produces) | set(step.modifies)


def _uses(step):
    return set(step.consumes) | _touches(step)


def conflict(a, b):
    contested = (_touches(a) & _uses(b)) | (_touches(b) & _uses(a))
    for name in sorted(contested):
        return name
    return None


# Occupanti che tengono un artefatto senza essere un Job del JobManager --
# oggi solo la sessione manuale di estrazione (gui/estrazione/pagina.py),
# un QProcess che il gestore non traccia affatto, quindi `active_jobs()`
# non la vede: senza questo registro un lavoro avviato dalla lista Steps
# (che chiama JobManager.try_start direttamente) o dalla pagina faceset
# (che chiama gui.faceset.conflitti.chi_occupa) sullo stesso `aligned` non
# verrebbe mai rifiutato mentre la sessione manuale scrive li'.
#
# [(identita_workspace, passo), ...]: un passo qualunque -- un vero StepDef
# o un fittizio come gui.faceset.conflitti.PassoFittizio -- purche' porti
# `.name`/`.consumes`/`.produces`/`.modifies`, la stessa forma che
# `conflict()` gia' sa leggere. Una lista e non un dict per lo stesso
# motivo di `chi_occupa`: due identita' diverse possono nominare lo stesso
# workspace (stesso_workspace e' sovrabbondante apposta), quindi il
# confronto e' sempre un giro esplicito, mai una chiave diretta.
_occupanti_esterni = []


def registra_occupante(identita, passo):
    _occupanti_esterni.append((identita, passo))


def libera_occupante(identita, passo):
    try:
        _occupanti_esterni.remove((identita, passo))
    except ValueError:
        pass


def occupanti_di(identita):
    return [passo for ident, passo in _occupanti_esterni
            if stesso_workspace(identita, ident)]
