"""Il lettore del rapporto per frame, lato GUI.

Seconda implementazione del formato, deliberata: gui/ non importa
mainscripts, ed e' cio' che tiene la suite GUI a ~28 s invece che a ~140.
Le due implementazioni sono inchiodate da un test che confronta le
costanti: se una cambia, quel test diventa rosso.

Ogni predicato deve reggere una voce con campi assurdi: il rapporto e'
scritto da un processo che puo' essere stato ucciso a meta'.
"""
import json
from pathlib import Path

NOME_RAPPORTO = "frames.ndjson"
FORMATO = 1

# Misurata su 654 volti reali (whole_face, 512 px): lato minimo osservato
# 231 px, quindi la soglia qui sotto non scatta su quel materiale -- non
# perche' sia sbagliata, ma perche' quel dataset non ha volti piccoli.
SOGLIA_LATO_PICCOLO = 96

# Misurata sullo stesso dataset: |yaw| p50=15.5 gradi, p90=27.5, p95=28.9,
# p99=30.7, max=35.1. A 0.9 rad (51.6 gradi) il filtro seleziona ZERO volti
# su quel materiale, ed e' l'esito voluto: quel dataset e' pulito (654
# volti rilevati su 655 frame), e un filtro che cerca i frame problematici
# non deve scattare quando problemi non ce ne sono. Abbassarla
# scambierebbe normali rotazioni della testa per posa estrema. I volti
# davvero di profilo non escono con yaw alto: falliscono il rilevamento e
# arrivano con n_volti=0, quindi li prende "senza-volto", non questo.
SOGLIA_YAW_ESTREMO = 0.9

# NON misurata: nessun dataset con frame scuri sotto mano. Indovinata.
SOGLIA_LUMINANZA_SCURA = 40.0


def leggi(cache_dir):
    """Ultima riga vince. Una riga non decodificabile -- il troncamento che
    uno Stop lascia dietro -- si salta senza sollevare."""
    percorso = Path(cache_dir) / NOME_RAPPORTO
    if not percorso.exists():
        return []
    per_nome = {}
    with open(str(percorso), "r", encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if not riga:
                continue
            try:
                v = json.loads(riga)
            except ValueError:
                continue
            if isinstance(v, dict) and isinstance(v.get("nome"), str):
                per_nome[v["nome"]] = v
    return list(per_nome.values())


def scrivi_voce(cache_dir, voce):
    """Appende una riga al rapporto. Ultima riga vince: riscrivere un
    fotogramma vuol dire appenderlo di nuovo, mai riscrivere il file."""
    percorso = Path(cache_dir)
    percorso.mkdir(parents=True, exist_ok=True)
    with open(str(percorso / NOME_RAPPORTO), "a", encoding="utf-8") as f:
        f.write(json.dumps(voce) + "\n")
        f.flush()


def motore_di(v):
    """La coppia rilevatore+allineatore che ha prodotto la voce, o None.

    Seconda implementazione deliberata, come tutto questo modulo: gui/ non
    importa mainscripts.
    """
    m = v.get("motore")
    return m if isinstance(m, str) and m else None


def _n_volti(v):
    n = v.get("n_volti")
    return n if isinstance(n, int) else 0


def _volti(v):
    volti = v.get("volti")
    return volti if isinstance(volti, list) else []


def _senza_volto(v):
    return _n_volti(v) == 0


def _piu_volti(v):
    return _n_volti(v) > 1


def _volto_piccolo(v):
    lati = [f.get("lato") for f in _volti(v) if isinstance(f, dict)]
    lati = [l for l in lati if isinstance(l, (int, float))]
    return bool(lati) and min(lati) < SOGLIA_LATO_PICCOLO


def _posa_estrema(v):
    for f in _volti(v):
        if not isinstance(f, dict):
            continue
        posa = f.get("posa")
        if isinstance(posa, list) and len(posa) == 3 \
                and isinstance(posa[1], (int, float)) \
                and abs(posa[1]) >= SOGLIA_YAW_ESTREMO:
            return True
    return False


def _scuro(v):
    # None (il ripiego di `main.py extracttool index`, che ricostruisce da
    # una cartella gia' estratta e non ha i frame da cui calcolare la
    # luminanza) non e' "scuro": e' di luminosita' sconosciuta. isinstance
    # esclude None senza bisogno di controllarlo a mano.
    l = v.get("luminanza")
    return isinstance(l, (int, float)) and l < SOGLIA_LUMINANZA_SCURA


# Le etichette sono testo visibile, quindi in inglese, e stanno accanto al
# predicato che descrivono invece che in gui/testi.py -- stessa scelta di
# gui/estrazione/azioni.py::OPERAZIONI e gui/faceset/azioni.py, e stessa
# conseguenza: la rete AST di tests_gui/test_testi_soltanto_da_un_posto.py
# non le vede, perche' raggiungono il widget come attributo e non come
# letterale. L'ORDINE non e' libero: e' quello dei bottoni sulla riga.
FILTRI = (
    ("tutti",         "All",             lambda v: True),
    ("senza-volto",   "No face",         _senza_volto),
    ("piu-volti",     "Several faces",   _piu_volti),
    ("volto-piccolo", "Small face",      _volto_piccolo),
    ("posa-estrema",  "Extreme pose",    _posa_estrema),
    ("scuro",         "Dark frame",      _scuro),
)


def conta(voci):
    return dict((chiave, sum(1 for v in voci if predicato(v)))
                for chiave, _, predicato in FILTRI)
