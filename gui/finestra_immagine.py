"""La finestra che mostra un'anteprima a dimensione naturale.

E' uno **scatto**: costruita una volta, con l'immagine e il titolo di quel
momento, non si aggancia in nessun modo a chi l'ha aperta -- ne' al
pannello, ne' alle anteprime che arrivano dopo. Per questo non ha bisogno
di nessun canale ne' di nessun aggiornamento: quello che mostra e' quello
che c'era quando e' stata aperta, e il titolo lo dice esplicitamente
(l'iterazione), proprio perche' non cambia mai.

Non modale apposta: puo' restare aperta insieme al pannello che l'ha
generata, e insieme a piu' copie di se stessa su celle diverse -- guardare
due volti in grande fianco a fianco e' l'uso previsto, non un caso limite.
"""
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QScrollArea, QVBoxLayout

#Quanto dello schermo puo' prendersi al massimo una finestra appena aperta:
#un'anteprima intera di SAEHD a 512 e' 2560x2048 pixel, e una finestra piu'
#grande dello schermo nasce con gli angoli fuori e i bordi irraggiungibili.
#Oltre questa soglia decidono le barre di scorrimento, che ci sono gia'.
_QUOTA_SCHERMO = 0.9


class FinestraImmagine(QDialog):
    """Una QDialog non modale con l'immagine a 1:1 dentro una QScrollArea.

    `Esc` la chiude: e' il comportamento di default di QDialog, e non c'e'
    bisogno di aggiungerlo a mano.

    Si distrugge alla chiusura (`WA_DeleteOnClose`), e non e' un dettaglio:
    e' figlia del pannello, che vive per tutta la durata del training, e
    senza quell'attributo ogni doppio click lascerebbe dietro di se' un
    figlio vivo con dentro una `QImage` intera -- una cella di un'anteprima
    a 512 sono megabyte, e i doppi click in una sessione di ore sono tanti.
    Chi la apre non la tiene: `apri_a_dimensione_naturale` la ritorna per
    poterla mostrare e basta.
    """

    def __init__(self, immagine, titolo, parent=None):
        super().__init__(parent)
        self._immagine = immagine
        self.setWindowTitle(titolo)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        contenuto = QLabel()
        contenuto.setPixmap(QPixmap.fromImage(immagine))

        area = QScrollArea()
        area.setWidget(contenuto)
        area.setWidgetResizable(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(area)
        self.resize(self._dimensione_voluta())

    def _dimensione_voluta(self):
        """L'immagine intera, se ci sta nello schermo; altrimenti la quota
        di schermo qui sopra.

        Senza, la dimensione la sceglieva l'euristica di QDialog e non
        c'entrava niente con l'immagine: una cella da 128 px si apriva in
        una finestra piu' grande di lei, con un mare di sfondo attorno.
        """
        voluta = self._immagine.size()
        schermo = QApplication.primaryScreen()
        if schermo is None:
            return voluta
        disponibile = schermo.availableGeometry()
        return QSize(min(voluta.width(), int(disponibile.width() * _QUOTA_SCHERMO)),
                     min(voluta.height(), int(disponibile.height() * _QUOTA_SCHERMO)))

    def immagine(self):
        """La QImage conservata, cosi' com'era al momento dell'apertura."""
        return self._immagine
