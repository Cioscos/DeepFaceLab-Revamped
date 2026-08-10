"""Il taglio della griglia che il modello ha composto.

Il descrittore arriva dal figlio dentro l'evento `preview` e dice che forma
dovrebbe avere l'immagine; l'immagine dice che forma ha davvero. Dove le due
non coincidono vince l'immagine: un jpg dello storico puo' essere stato
scritto quando il batch size era un altro, e quindi avere meno righe di
quante il modello ne dichiari adesso.

Un descrittore incoerente vale come un descrittore assente -- l'immagine
intera senza etichette e' sempre meglio di etichette sbagliate, perche' il
secondo caso mente esattamente su cio' che l'utente sta guardando.
"""

CAMPI = ("righe", "colonne", "celle")


def normalizza(voce):
    """Il descrittore validato dentro `voce`, o None se manca o non regge."""
    if voce is None:
        return None
    try:
        if any(c not in voce for c in CAMPI):
            return None
    except TypeError:
        # voce non supporta l'operatore 'in' (es. intero, float, bool)
        return None
    righe, colonne, celle_ = voce["righe"], voce["colonne"], voce["celle"]
    if not isinstance(righe, int) or not isinstance(colonne, int):
        return None
    if righe < 1 or colonne < 1:
        return None
    if not isinstance(celle_, list) or len(celle_) != righe:
        return None
    if any(not isinstance(r, list) or len(r) != colonne for r in celle_):
        return None
    risultato = voce.get("risultato")
    if risultato is not None:
        if not isinstance(risultato, (list, tuple)) or len(risultato) != 2:
            risultato = None
        #`risultato` e' una **coordinata**, e va guardata come tale: il tipo
        #prima dell'intervallo. Un `[0.0, 4.0]` passa il controllo di
        #intervallo senza fatica -- e' vero che `0 <= 0.0 < righe` -- e poi
        #indicizza la griglia, dove un float solleva `TypeError` dentro il
        #disegno. E' la stessa forma del difetto delle iterazioni float,
        #trovata cercandola qui apposta: dove un numero che arriva dal canale
        #diventa un indice, l'intervallo non e' la domanda giusta.
        elif any(isinstance(c, bool) or not isinstance(c, int) for c in risultato):
            risultato = None
        elif not (0 <= risultato[0] < righe) or not (0 <= risultato[1] < colonne):
            risultato = None
    #Le etichette si **convertono**, non si rifiutano. Sono l'unico campo
    #puramente cosmetico del descrittore: un `1` al posto di `"1"` e'
    #mostrabile cosi' com'e', mentre buttare il descrittore intero per colpa
    #sua farebbe perdere anche la cella del risultato, cioe' la sola cosa che
    #decide *quale* volto va in grande. Convertire qui e' anche cio' che
    #tiene la promessa del tipo verso chi le mette a schermo: `"\n".join` e
    #`QLabel(...)` vogliono stringhe, e sono raggiunti da slot senza rete.
    return {"righe": righe, "colonne": colonne,
            "celle": [[str(c) for c in r] for r in celle_],
            "risultato": list(risultato) if risultato is not None else None,
            "righe_sono_campioni": bool(voce.get("righe_sono_campioni", False))}


def righe_effettive(immagine, colonne):
    """Quante righe ci sono davvero in questa immagine.

    Il lato della cella si ricava dalla larghezza, che e' l'unica dimensione
    stabile per modello e risoluzione; l'altezza e' quella che cambia col
    batch size della corsa che ha scritto l'immagine.
    """
    if colonne < 1:
        return 1
    lato = immagine.width() // colonne
    if lato < 1:
        return 1
    return max(1, immagine.height() // lato)


def celle(immagine, colonne, righe):
    """La griglia tagliata, riga per riga, come lista di liste di QImage."""
    lato = immagine.width() // colonne
    return [[immagine.copy(c * lato, r * lato, lato, lato)
             for c in range(colonne)]
            for r in range(righe)]


def etichetta(descrittore, riga, colonna):
    """L'etichetta della cella, stringa vuota se il descrittore non la copre."""
    if descrittore is None:
        return ""
    celle_ = descrittore["celle"]
    if not (0 <= riga < len(celle_)) or not (0 <= colonna < len(celle_[riga])):
        return ""
    return celle_[riga][colonna]
