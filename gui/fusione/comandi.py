"""Le scorciatoie della pagina di fusione: le stesse lettere della finestra
`cv2` che questa pagina sostituisce (help_merger_masked.jpg), piu' quelle
che la pagina aggiunge (Ctrl+S, K, Ctrl+K).

La tabella E' la spiegazione: la descrizione che sta qui e' quella che
finisce sulla QAction, quindi non c'e' una seconda lista da tenere
allineata. Come in gui/estrazione/comandi.py il contesto e'
`Qt.WidgetWithChildrenShortcut` e le azioni appartengono alla PAGINA --
l'appartenenza, non la costruzione, e' cio' che rende una scorciatoia
raggiungibile da tastiera.
"""

# chiave -> (testo del tasto per QKeySequence, descrizione inglese)
COMANDI = {
    "precedente": (",", "Previous frame"),
    "primo": ("Shift+,", "First frame"),
    "precedente_propaga": ("M", "Previous frame, copying this frame's settings onto it"),
    "primo_propaga": ("Shift+M", "Back to the first frame, copying the settings all the way"),
    "successivo": (".", "Next frame"),
    "batch": ("Shift+.", "Process all remaining frames (again to stop)"),
    "successivo_propaga": ("/", "Next frame, copying this frame's settings onto it"),
    "ultimo_propaga": ("Shift+/", "Copy the settings to every following frame"),
    "zoom_meno": ("-", "Zoom out"),
    "zoom_piu": ("=", "Zoom in"),
    "vista_maschera": ("V", "Toggle the mask view"),
    "vista_originale": ("O", "Toggle the original frame"),
    "salva_sessione": ("Ctrl+S", "Save the session"),
    "keyframe": ("K", "Set or clear a keyframe on this frame"),
    "piano": ("Ctrl+K", "Apply the keyframe plan to every frame"),
}
CHIAVI_INSTRADATE = frozenset(COMANDI)
