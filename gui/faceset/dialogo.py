"""Il form del catalogo dentro un dialogo modale.

Nessun form nuovo: StepForm e' lo stesso della vista-passo, con la sua
semantica _touched -- solo i campi toccati vengono spediti, che e'
l'invariante nata dalla voce 3.14 (il form che riportava batch_size da 12
a 8 al primo Start).

L'area scorrevole non e' un vezzo: e' la lezione del ciclo v4, dove un
form da 36 campi superava i 1900 px a scala xlarge e lasciava il bottone
di conferma fuori schermo. Qui i form sono piu' corti, ma la classe di
difetto e' la stessa e costa una riga evitarla.
"""
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QScrollArea, QVBoxLayout

from gui.forms import StepForm


def serve_il_dialogo(passo):
    """Se questo passo ha qualcosa da chiedere.

    Le tre righe che StepForm puo' costruire sono i campi, il selettore di
    file di `passthrough` e il nome del modello di `needs_model_name`:
    senza nessuna delle tre il dialogo nasce con dentro solo Ok e Cancel,
    una finestra che non chiede niente e che l'utente deve confermare per
    far partire cio' che aveva gia' cliccato. Cinque dei diciassette passi
    di questa famiglia sono cosi' (unpack e recover original filename, coi
    gemelli dst).

    Si guarda il DATO del passo, non `process`: un PROCESS_PROMPT senza
    campi resterebbe comunque un dialogo vuoto, ed e' il form a decidere
    cosa disegna.
    """
    return bool(passo.fields or passo.passthrough or passo.needs_model_name)


class DialogoOperazione(QDialog):
    def __init__(self, passo, parent=None):
        super().__init__(parent)
        self.setWindowTitle(passo.name)
        self.form = StepForm(passo)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.form)
        bottoni = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bottoni.accepted.connect(self.accept)
        bottoni.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(area, 1)
        layout.addWidget(bottoni)

    def risposte(self):
        return self.form.answers()
