"""L'indice come lo vede l'interfaccia: voci, chiavi, riconciliazione.

La chiave primaria e' (nome, dimensione, mtime); il fallback e'
(dimensione, mtime), consultato solo quando il nome non si trova e solo se
quella coppia e' unica. Le due meta' servono a due cose diverse:

  il fallback     rende la cache immune ai sort -- un sort rinomina 50 000
                  file e dimensione e mtime sopravvivono;
  il nome         evita il guasto silenzioso su un filesystem che tiene
                  l'mtime a 2 secondi (exFAT, un disco di rete), dove
                  un'estrazione scrive migliaia di volti nello stesso
                  istante nominale e due file diversi collidono. Nel caso
                  ambiguo si reindicizza, non si indovina: attribuire la
                  posa di un volto a un altro non farebbe rumore.

Il file dell'indice e' append-only: una faccia riscritta
riceve una nuova riga, la vecchia resta sul disco. Fra due righe con la
stessa chiave vince l'ultima -- e' la riconciliazione con la storia.
"""
import json
import os
from collections import namedtuple
from pathlib import Path

from gui.faceset import fratelli

Voce = namedtuple("Voce", [
    "nome", "dimensione", "mtime", "yaw", "pitch", "roll",
    "face_type", "source", "mask_off", "mask_len",
])

NOME_INDICE = "index.ndjson"
NOME_BLOB = "masks.bin"

STATO_ASSENTE = "assente"
STATO_PARZIALE = "parziale"
STATO_COMPLETO = "completo"


def _stringa_valida(x):
    return isinstance(x, str)


def _intero_valido(x):
    # bool e' una sottoclasse di int in Python, ma non e' un tipo
    # plausibile per una dimensione o un mtime -- escluso esplicitamente.
    return isinstance(x, int) and not isinstance(x, bool)


def leggi(cache_dir):
    percorso = Path(cache_dir) / NOME_INDICE
    try:
        testo = percorso.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    voci = []
    for riga in testo.splitlines():
        if not riga:
            continue
        try:
            d = json.loads(riga)
        except ValueError:
            continue
        try:
            nome, dimensione, mtime = d["n"], d["s"], d["m"]
        except (KeyError, TypeError):
            continue
        mask_off, mask_len = d.get("mo", 0), d.get("ml", 0)
        # Un tipo sbagliato vale come una chiave mancante: la riga si
        # scarta, quel volto resta semplicemente senza voce d'indice --
        # mai una forma capace di far sollevare __init__ o maschera().
        if not (_stringa_valida(nome) and _intero_valido(dimensione)
                and _intero_valido(mtime) and _intero_valido(mask_off)
                and _intero_valido(mask_len)):
            continue
        voci.append(Voce(nome, dimensione, mtime,
                         d.get("yaw"), d.get("pitch"), d.get("roll"),
                         d.get("ft"), d.get("src"),
                         mask_off, mask_len))
    return voci


def maschera(cache_dir, voce):
    if not voce.mask_len:
        return None
    try:
        with open(Path(cache_dir) / NOME_BLOB, "rb") as f:
            f.seek(voce.mask_off)
            return f.read(voce.mask_len)
    except (OSError, TypeError):
        return None


def elenca(cartella, estensioni):
    """[(percorso, dimensione, mtime_ns), ...] con UNA lettura di directory.

    Una `DirEntry` porta il tipo (dal `d_type` del `readdir`) e i campi
    della chiave dalla stessa lettura: elencare cosi' non costa nessuno
    `stat()` per filtrare, e uno solo -- gia' quello della chiave -- per
    conoscerne i campi. La forma di prima ne pagava DUE per file, una in
    `Path.is_file()` e una in `Path.stat()`, ed e' quello che rendeva
    l'apertura di una cartella indicizzata 1,28 s a 400 volti su drvfs
    (~160 s proiettati a 50 000, contro i 0,21 s di adesso).

    I due campi restano `st_size` e `st_mtime_ns`, gli stessi che
    l'indicizzatore scrive: cambiarli farebbe smettere di abbinare ogni
    cache gia' sul disco, e il guasto si vedrebbe solo come una
    re-indicizzazione completa.

    Un file sparito fra la lettura della directory e quella dei suoi campi
    torna con i campi a None -- `abbina_letti` lo conta fra i mancanti,
    esattamente come faceva lo `stat()` che sollevava.

    **Lo `stat()` sta DENTRO il ciclo dello `scandir`, e non e' uno stile.**
    Misurato su drvfs, 400 volti: statare mentre l'iteratore e' ancora
    aperto costa 204 ms, materializzare prima l'elenco con un `list()` e
    statare dopo ne costa 633 -- tre volte tanto, e quanto l'intera forma
    che questa sostituisce. Un `list(os.scandir(...))` innocuo in cima al
    ciclo si riprende tutto il guadagno senza cambiare un risultato: e' un
    peggioramento che nessun test vedrebbe.
    """
    try:
        return _elenca_o_solleva(cartella, estensioni)
    except OSError:
        # Voce 3.23: la cartella c'e' ma non si elenca. Si mostra il fatto,
        # non si solleva dentro un evento Qt.
        return []


def _elenca_o_solleva(cartella, estensioni):
    """Come `elenca`, ma SOLLEVA `OSError` invece di inghiottirla.

    Serve a chi -- fuori dal thread di Qt -- deve distinguere una cartella
    VUOTA (letta, zero file) da una IRRAGGIUNGIBILE (inesistente, senza
    permessi): `mappa_per_fotogramma` la usa proprio per questo, perche'
    confondere le due cose azzererebbe un rapporto vero scambiandolo per
    un fotogramma senza volti."""
    letti = []
    with os.scandir(str(cartella)) as voci:
        for voce in voci:
            try:
                if not voce.is_file():
                    continue
            except OSError:
                continue
            if os.path.splitext(voce.name)[1].lower() not in estensioni:
                continue
            try:
                st = voce.stat()
            except OSError:
                letti.append((Path(voce.path), None, None))
                continue
            letti.append((Path(voce.path), st.st_size, st.st_mtime_ns))
    letti.sort(key=lambda t: t[0])
    return letti


def stato(abbinati, mancanti):
    if not abbinati:
        return STATO_ASSENTE
    return STATO_PARZIALE if mancanti else STATO_COMPLETO


def _con_stat(percorsi):
    for p in percorsi:
        try:
            st = p.stat()
        except OSError:
            yield (p, None, None)
        else:
            yield (p, st.st_size, st.st_mtime_ns)


class Indice:
    def __init__(self, voci):
        self._per_chiave = {(v.nome, v.dimensione, v.mtime): v for v in voci}
        per_coppia = {}
        for v in voci:
            per_coppia.setdefault((v.dimensione, v.mtime), []).append(v)
        # Solo le coppie uniche: una coppia condivisa da due voci non
        # identifica niente, e va reindicizzata invece che indovinata.
        self._per_coppia = {k: vs[0] for k, vs in per_coppia.items() if len(vs) == 1}

    def abbina(self, percorsi):
        """La firma storica: i campi della chiave li legge lui, uno `stat()`
        per percorso. E' la forma giusta per chi ha in mano soltanto dei
        percorsi -- oggi nessuno in produzione, solo i test: la pagina ha
        gia' letto la cartella con `elenca()` e usa `abbina_letti`, che non
        paga niente. Detto qui invece di lasciarlo scoprire."""
        return self.abbina_letti(_con_stat(percorsi))

    def abbina_letti(self, letti):
        """`letti`: (percorso, dimensione, mtime_ns) gia' noti.

        Dimensione a None significa "i suoi campi non si sono potuti
        leggere" -- un file sparito fra la scansione e ora: mancante, come
        lo era quando lo `stat()` sollevava qui dentro.
        """
        abbinati, mancanti = {}, []
        for p, dimensione, mtime in letti:
            if dimensione is None:
                mancanti.append(p)
                continue
            v = self._per_chiave.get((p.name, dimensione, mtime))
            if v is None:
                v = self._per_coppia.get((dimensione, mtime))
            if v is None:
                mancanti.append(p)
            else:
                abbinati[p] = v
        return abbinati, mancanti


def mappa_per_fotogramma(cache_dir, cartella, estensioni=(".jpg",)):
    """({nome_fotogramma: [percorsi]}, [percorsi senza voce]).

    Prima viveva in mainscripts/ExtractIndex.py::percorsi_di_un_frame come
    glob su `<stelo>_*.jpg`, e un sort che rinomina i volti la rendeva
    muta: il rettangolo restava e i landmark sparivano. Qui non guarda i
    nomi dei file: enumera la cartella e la abbina all'indice, poi delega
    l'inversione a `fratelli.mappa_per_frame`, che e' la sola sede della
    regola «due volti sono fratelli se e solo se `source` coincide».

    Non apre nessun JPEG: `src` sta gia' nell'indice, e l'abbinamento
    disco-indice regge la rinomina da se'. Chi non e' nell'indice esce fra
    i `mancanti` -- non e' uno scarto, e' il segnale che fa partire
    l'indicizzazione.

    SOLLEVA `OSError` se `cartella` non si puo' enumerare -- a differenza
    di `elenca`, che la inghiotte per i chiamanti dentro un evento Qt:
    questa funzione gira fuori dal thread di Qt (`_LavoroMappa.run`), e chi
    la chiama deve poter distinguere «cartella vuota» da «cartella
    irraggiungibile», o rischia di riconciliare il rapporto su una mappa
    che non ha mai visto il disco davvero.
    """
    letti = _elenca_o_solleva(cartella, estensioni)
    abbinati, mancanti = Indice(leggi(cache_dir)).abbina_letti(letti)
    return fratelli.mappa_per_frame(abbinati), mancanti
