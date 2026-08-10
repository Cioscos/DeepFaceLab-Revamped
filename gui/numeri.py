"""I predicati sui numeri che arrivano da fuori.

Nel pacchetto i numeri entrano da due porte, e nessuna delle due e'
sorvegliata a monte: gli eventi del canale sono JSON scritto da un altro
processo (`json.dumps` scrive `NaN` senza chiedere niente e `json.loads` lo
rilegge intatto), e il CSV della loss e' un file di testo dove `"%.6f" % nan`
ha lasciato la stringa `nan`. Chi li riceve non ha lo stesso metro -- un
contatore di iterazioni ha anche un tetto di plausibilita', un orologio no,
una loss nemmeno -- ma la domanda sotto e' una sola, ed e' questa.

Sta in un modulo suo, e non accanto a uno dei tre consumatori, proprio
perche' sono tre: la riga di stato, la storia della loss e il grafico. Una
copia per superficie e' il modo in cui questa famiglia di difetti e' nata la
prima volta.
"""
import math


def numero_finito(valore):
    """True se con `valore` si puo' fare aritmetica.

    Fuori restano: cio' che non e' un numero; il `bool`, che per Python e'
    un `int` e passerebbe da solo, ma al posto di una loss o di
    un'iterazione viene da un canale rotto; NaN e infiniti; e gli interi
    troppo grandi per entrare in un `float` (vedi il ramo qui sotto).

    **Non solleva mai**: la risposta e' sempre `True` o `False`, ed e' la
    parte del contratto piu' facile da rompere senza accorgersene -- chi
    chiama sta gia' proteggendo qualcosa, e un predicato che solleva sposta
    il guasto dentro di lui invece di fermarlo.

    NaN e infiniti sono i piu' insidiosi perche' non si fermano dove
    entrano: sopravvivono a somme, medie e massimi senza sollevare, e
    muoiono molto piu' tardi, dentro l'`int()` di un `paintEvent`. Un
    `paintEvent` e' un metodo virtuale chiamato da Qt: nessuna cattura del
    chiamante lo copre, PyQt5 chiama `qFatal` e il processo se ne va di
    colpo -- con dentro ogni altro training aperto, perche' la finestra e'
    una sola.
    """
    if isinstance(valore, bool) or not isinstance(valore, (int, float)):
        return False
    try:
        return math.isfinite(valore)
    except OverflowError:
        #Un `int` di Python non ha limite di grandezza, un `float` si': per
        #decidere se e' finito `math.isfinite` prova a convertirlo, e sopra
        #~1,8e308 non ci riesce. Non e' un infinito, ma con lui non si fa
        #aritmetica insieme ai float, che e' esattamente cio' che questa
        #funzione promette -- quindi la risposta e' "no", non un'eccezione.
        #Sollevare qui sposterebbe soltanto il guasto dentro chi chiama, e
        #chi chiama sono la lettura del CSV e il disegno, cioe' i due posti
        #dove nessuno raccoglie.
        return False


#Il tetto di plausibilita' di un contatore di iterazioni. Un miliardo e'
#oltre quattro anni di addestramento ininterrotto al ritmo piu' alto mai
#misurato su questo progetto (~7 it/s su una RTX 4080), quindi nessuna corsa
#vera lo raggiunge. Il numero preciso conta meno dell'asimmetria fra i due
#errori possibili, che e' la ragione per cui un tetto c'e': rifiutare per
#sbaglio un'iterazione vera costa un punto vivo, lo dice a schermo e il
#salvataggio successivo lo rimette a posto rileggendo il CSV; accettare per
#sbaglio un numero assurdo alza per sempre l'ultimo punto della storia
#della loss, e da li' in poi ogni punto vero viene scartato in silenzio,
#per tutto il resto della corsa. Il primo errore e' visibile e si ripara,
#il secondo no.
MASSIMA_ITERAZIONE = 10 ** 9


def iterazione_utilizzabile(valore):
    """True se `valore` puo' essere il numero di un'iterazione: un `int`.

    Serve perche' gli eventi sono JSON scritto da un altro processo e
    nessuno li valida: `"iter": "cinque"`, una loss `null`, e perfino
    `Infinity` -- che `json.loads` accetta senza chiedere niente -- arrivano
    intatti fin qui. Il CSV su disco e' la seconda porta, e la sua colonna
    `iter` passa da un `int()` che accetta qualunque grandezza.

    Restano fuori: cio' che non e' un numero; il `bool`, che per Python e'
    un `int` ma come iterazione viene da un canale rotto; **i float, compresi
    quelli tondi come `12.0`**; NaN e infiniti, che sono float e quindi
    escono di li'; i negativi; e i numeri finiti ma implausibili (vedi
    MASSIMA_ITERAZIONE).

    Il float e' l'ultimo arrivato di questa lista e il piu' difficile da
    vedere, perche' `12.0` *sembra* un'iterazione. Non tutti i consumatori ci
    fanno aritmetica: una parte la usa come **coordinata** -- indice di
    colonna nel grafico, chiave del nome di file nello storico
    (`"%07d"`), taglio della finestra mostrata -- e un indice float solleva
    `TypeError` dentro un `paintEvent`, cioe' dove nessuna cattura arriva.
    Il danno non si ferma nemmeno al punto storto: `LossPlot.finestra` prende
    gli estremi dalla lista, quindi un solo float in coda rende float lo span
    e da li' l'indice diventa float **anche per le iterazioni sane**.

    Un predicato stretto per quelli e uno lasco per gli altri rimetterebbe in
    piedi *due regole per lo stesso valore*, che e' la forma di difetto che
    questo modulo esiste per non avere -- una sola regola per le iterazioni,
    da qualunque porta entrino. E l'asimmetria vale identica al tetto qui
    sopra: rifiutare un float costa un punto vivo, lo dice a schermo e la
    rilettura del CSV lo rimette a posto; accettarlo si porta via tutti i
    training aperti. Nessun produttore vero manda un float -- il trainer
    conta con `int`, il CSV passa da `int()` -- quindi un float sul canale
    e' per costruzione un canale rotto, non un'iterazione da salvare.

    Il NaN merita ancora una riga sua, anche se adesso e' escluso dal tipo
    prima che dal valore, perche' la ragione per escluderlo e' spesso detta
    male: non e' che non sollevi mai -- `"%d" % nan` solleva -- e' che ogni
    suo *confronto* e' falso, `nan <= 10` come `10 <= nan`, e su quei
    confronti si regge chi decide se un punto e' nuovo. Entra in silenzio, e
    quello e' il guasto che non si vede.
    """
    if isinstance(valore, bool) or not isinstance(valore, int):
        return False
    #Nessun `math.isfinite` qui: un `int` e' sempre finito, e su uno da 400
    #cifre la conversione a float che quella funzione fa solleverebbe
    #`OverflowError`, mentre il confronto col tetto risponde "no" da solo.
    return 0 <= valore <= MASSIMA_ITERAZIONE
