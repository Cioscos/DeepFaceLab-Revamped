"""Chi viene dallo stesso frame: l'inversione di una colonna gia' in memoria.

`source_filename` nei metadati DFL e' il NOME del file di frame -- lo
scrivono i due soli chiamanti di `ExtractorLib.salva_volto`
(`filepath.name` in Extractor.py, `percorso.name` in ExtractManual.py) --
e `FacesetIndex.descrivi` lo porta nell'indice sotto la chiave `src`, che
`gui/faceset/indice.py` legge nel campo `source` della `Voce`. Da li' in
poi due volti sono fratelli se e solo se quella stringa coincide.

Nessuna apertura di file e nessuno `stat()`: si lavora su `_abbinati`, che
la pagina ha gia' costruito in `ricalcola()`. E' la ragione per cui questa
funzione costa quanto un giro di dizionario e non quanto una passata su
50 000 testate DFL.

Un volto senza voce d'indice, o con `source` assente, non entra in nessun
gruppo e non ne forma uno a se': un gruppo di orfani non e' un frame, e
mostrarli come fratelli sarebbe la peggiore bugia disponibile qui.

**La stringa non si normalizza.** Viene dal file dell'utente ed e' per noi
una chiave opaca: abbassarne le maiuscole o togliere l'estensione farebbe
fratelli due volti che il resto del sistema tiene distinti.
"""


def nome_frame_di(abbinati, percorso):
    """Il frame da cui viene questo volto, o None se non si sa."""
    voce = abbinati.get(percorso)
    if voce is None:
        return None
    nome = voce.source
    return nome if isinstance(nome, str) and nome else None


def mappa_per_frame(abbinati):
    """{nome_frame: [percorsi]}.

    L'ordine dei percorsi dentro un gruppo e' quello dei percorsi ordinati,
    non quello di iterazione del dizionario: il contatore «2 of 4» non deve
    cambiare fra due ricalcoli della stessa cartella.
    """
    mappa = {}
    for percorso in sorted(abbinati, key=str):
        nome = nome_frame_di(abbinati, percorso)
        if nome is not None:
            mappa.setdefault(nome, []).append(percorso)
    return mappa


def percorsi_del_frame(mappa, nome_frame):
    """Tutti i volti di quel frame, il volto stesso compreso.

    E' la forma che il filtro consuma, e l'ingresso dall'altra pagina
    arriva col solo nome del frame: li' non c'e' nessun volto da escludere.
    """
    return list(mappa.get(nome_frame) or ())


def fratelli_di(mappa, abbinati, percorso):
    """Gli ALTRI volti dello stesso frame, escluso `percorso`."""
    nome = nome_frame_di(abbinati, percorso)
    if nome is None:
        return []
    return [p for p in mappa.get(nome, ()) if p != percorso]
