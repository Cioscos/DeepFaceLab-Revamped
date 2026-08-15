"""Le mutazioni a mano della pagina, dentro la matrice dei conflitti.

Cancellare, annullare e le azioni di Tools toccano gli stessi artefatti
dei job, ma gui/execution/conflicts.py ragiona fra PASSI. La pagina
sintetizza quindi un passo fittizio che dichiara di modificare
l'artefatto, e lo passa alla stessa funzione: una regola sola, su una
superficie in piu'.
"""
from collections import namedtuple

from gui.execution.conflicts import conflict
from gui.progetti import identita_workspace, stesso_workspace

PassoFittizio = namedtuple("PassoFittizio", ["name", "consumes", "produces", "modifies"])

ARTEFATTI = {"src": "faceset_src", "dst": "faceset_dst"}


def artefatto_di(cartella, dataset):
    """L'artefatto del workflow che quella cartella rappresenta.

    Grossolano di proposito: aligned, aligned_trash e aligned_debug dello
    stesso dataset sono lo stesso artefatto ai fini del conflitto. Un
    falso positivo costa un tasto grigio, un falso negativo costa un
    dataset rovinato scoperto ore dopo.
    """
    return ARTEFATTI.get(dataset, "faceset_src")


def passo_di_mutazione(cartella, dataset):
    return PassoFittizio("manual edit", (), (), (artefatto_di(cartella, dataset),))


def chi_occupa(job_manager, workspace, cartella, dataset):
    """(nome del passo, artefatto) del job che tiene occupata la cartella, o None.

    Un job la cui identita' non e' ancora nota (None -- un job costruito
    prima che il suo workspace si risolvesse, o un doppio di test) non va
    scartato in silenzio: fra un falso positivo (un tasto grigio di troppo)
    e un falso negativo (due scritture sullo stesso faceset) qui si sceglie
    sempre il primo, quindi quel job resta candidato al conflitto invece di
    essere confrontato con stesso_workspace, che non saprebbe che farsene
    di un None.
    """
    mio = passo_di_mutazione(cartella, dataset)
    identita = identita_workspace(workspace)
    for job in job_manager.active_jobs():
        if job.identita is not None and not stesso_workspace(identita, job.identita):
            continue
        artefatto = conflict(mio, job.step)
        if artefatto is not None:
            return (job.step.name, artefatto)
    return None
