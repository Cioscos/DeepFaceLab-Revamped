"""La maschera XSeg come pellicola colorata.

Una maschera e' un raster in scala di grigi. Disegnata com'e' sopra un
fotogramma a colori si legge come una toppa grigia, non come una
segmentazione: qui il bianco diventa opaco e il nero trasparente, con un
colore fisso, cosi' il volto sotto resta visibile e il bordo della
maschera si vede dove sta davvero.

La conversione si fa UNA VOLTA, quando i dati arrivano, mai dentro un
paintEvent: un fotogramma con tre volti la pagherebbe a ogni ridisegno.

`setAlphaChannel` e' segnata come deprecata da Qt 5.15 ma e' presente e
corretta in questa installazione (verificato eseguendo: una scala di
grigi 0/85/170/255 da' alfa 0/85/170/255). Se un aggiornamento di PyQt5
la togliesse, la sostituta e' reinterpretare i byte della grigia come un
Format_Alpha8 e comporre con CompositionMode_DestinationIn -- misurata
identica, ma con un buffer Python da tenere vivo, che e' la ragione per
cui non e' la prima scelta.
"""
from PyQt5.QtGui import QImage


def pellicola_colorata(immagine, colore):
    """Un QImage in cui il COLORE e' uniforme e l'ALFA e' la luminanza
    della maschera. None per un'immagine assente o nulla: da qui si va
    dentro un paintEvent, quindi non si solleva mai.

    Il risultato e' PREMOLTIPLICATO: ad alfa 0 il colore letto e'
    (0, 0, 0) e non `colore`. Chi lo verifica deve guardare l'alfa, non
    l'RGB -- tranne ad alfa 255, dove la premoltiplicazione e'
    l'identita'.
    """
    if immagine is None or immagine.isNull():
        return None
    fuori = QImage(immagine.size(), QImage.Format_ARGB32)
    fuori.fill(colore)
    fuori.setAlphaChannel(immagine)
    return fuori
