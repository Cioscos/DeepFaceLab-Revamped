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
