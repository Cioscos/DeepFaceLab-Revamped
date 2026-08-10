"""La riga che racconta un training mentre gira, scritta in un posto solo.

Due superfici la mostrano: la striscia sotto la vista del passo e la scheda
del training, sotto il grafico. E' lo stesso fatto detto due volte, non due
fatti, e finche' erano due funzioni gemelle bastava toccarne una perche' i
due punti dello schermo dicessero la stessa cosa in modi diversi.

Qui non c'e' nessun widget e nessun Qt: sono stringhe, piu' la piccola
memoria che serve a calcolare il ritmo fra un evento e il successivo. Chi le
mette a schermo decide dove, e la scheda aggiunge la VRAM perche' e' l'unica
delle due a ricevere gli eventi che la portano.
"""
#`MASSIMA_ITERAZIONE` e `iterazione_utilizzabile` stavano qui e sono in
#`gui.numeri` dal momento in cui il terzo consumatore non e' stato piu' una
#superficie di testo: la storia della loss deve validare la colonna `iter`
#del CSV con la stessa regola, e farglielo importare *dalla riga di stato*
#sarebbe stata la dipendenza sbagliata. Si continuano a leggere anche da
#qui, che e' la porta che le due superfici gia' usavano: una definizione
#sola, due nomi per raggiungerla.
from gui.numeri import MASSIMA_ITERAZIONE, iterazione_utilizzabile, numero_finito


def obiettivo_valido(valore):
    """Il `target_iter` annunciato, o 0 quando non e' un'iterazione.

    0 significa gia' "nessun obiettivo" per chi compone la riga, ed e'
    esattamente quello che si sa di una corsa il cui obiettivo e' arrivato
    illeggibile. L'alternativa -- ricordarlo com'e' -- e' il difetto: viene
    confrontato con l'iterazione a ogni evento, quindi un `target_iter`
    storto non rompe l'evento che lo porta, rompe tutti quelli buoni che
    vengono dopo, e solo dal secondo in poi (nel primo il ritmo non si sa
    ancora e il confronto non si raggiunge).
    """
    return valore if iterazione_utilizzabile(valore) else 0


class RitmoIterazioni:
    """Il ritmo fra due eventi `iter`, con la memoria che gli serve.

    **Valida prima di ricordare**, ed e' tutto il punto di questa classe.
    L'ordine opposto -- ricordare e poi usare -- e' quello che rende
    permanente un guasto momentaneo: basta un'iterazione inutilizzabile
    perche' resti nello stato, e da li' in poi ogni evento buono si rompe
    sul confronto con lei, per tutta la corsa, senza che niente lo dica.
    Il difetto e' esistito, in due copie: era scritto due volte, una per
    superficie, e nessuna delle due copie validava.
    """

    def __init__(self):
        self.iterazione = None
        self.istante = None

    def aggiorna(self, iterazione, istante):
        """Ricorda il punto e torna il ritmo dal precedente, o None.

        None quando il ritmo non si sa: al primo evento, con un orologio
        fermo o tornato indietro, o con un ingresso inutilizzabile -- e in
        quest'ultimo caso **non ricorda niente**, cosi' l'evento buono che
        viene dopo trova la memoria com'era.

        I due ingressi non sono la stessa cosa e non passano dallo stesso
        controllo: l'iterazione e' un contatore, e ha anche un tetto di
        plausibilita'; l'istante e' un orologio, e gli si chiede solo di
        essere un numero finito. Un `inf` ricordato come istante sarebbe il
        guasto piu' silenzioso di tutti: da li' in poi `istante <= quando`
        e' sempre vero e il ritmo non torna **mai piu'**.
        """
        if not iterazione_utilizzabile(iterazione) or not numero_finito(istante):
            return None
        precedente, quando = self.iterazione, self.istante
        self.iterazione, self.istante = iterazione, istante
        if precedente is None or istante <= quando:
            return None
        return (iterazione - precedente) / (istante - quando)


def formatta_eta(secondi):
    """Un conto alla rovescia come H:MM:SS, o M:SS sotto l'ora."""
    secondi = max(0, int(secondi))
    ore, resto = divmod(secondi, 3600)
    minuti, secondi = divmod(resto, 60)
    if ore:
        return "%d:%02d:%02d" % (ore, minuti, secondi)
    return "%d:%02d" % (minuti, secondi)


def pezzi_di_stato(iterazione, losses, ritmo, obiettivo,
                   vram_usata=None, vram_totale=None):
    """I pezzi della riga in ordine fisso; cio' che non si sa non compare.

    `ritmo` e' in iterazioni al secondo e vale None finche' non ci sono due
    eventi da confrontare -- al primo, un'ETA sarebbe inventata. `obiettivo`
    e' il `target_iter` del modello, 0 quando la corsa non ne ha uno.

    Un pezzo assente sparisce invece di diventare un trattino: chi le legge
    unisce con " | ", e una riga che va e viene di lunghezza e' preferibile
    a una che dichiara di sapere qualcosa che non sa.

    Le loss non numeriche non fanno sparire il pezzo, pero': diventano un
    "?" al loro posto. Sono quattro numeri in fila e portano informazione
    anche a uno solo mancante -- src che diverge mentre dst scende e'
    proprio cio' che si vuole leggere -- mentre `"%.4f" % None` avrebbe
    tolto la riga *intera*, iterazione e ritmo ed ETA compresi.
    """
    pezzi = ["iter %d" % iterazione]
    if losses:
        pezzi.append("loss " + ", ".join(
            "%.4f" % v if numero_finito(v) else "?" for v in losses))
    if ritmo is not None:
        pezzi.append("%.2f it/s" % ritmo)
        if obiettivo and obiettivo > iterazione and ritmo > 0:
            pezzi.append("ETA %s" % formatta_eta((obiettivo - iterazione) / ritmo))
    if numero_finito(vram_usata) and numero_finito(vram_totale):
        pezzi.append("VRAM %.1f/%.1f GiB" % (vram_usata, vram_totale))
    return pezzi
