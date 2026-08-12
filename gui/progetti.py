"""Progetti: piu' workspace nello stesso pacchetto, uno per progetto.

Un progetto e' una sottocartella della radice che contiene project.json,
accanto alle tre sottocartelle standard che gui/workspace.py conosce. Questo
modulo non importa Qt a livello di modulo -- stessa regola di
gui/workspace.py, cosi' e' esercitabile senza una QApplication.
"""
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gui.model_lock_status import busy_holder
from gui.workspace import STANDARD_SUBDIRS, create_workspace


def identita_workspace(path):
    """L'identita' di una cartella: (chiave del filesystem, percorso normalizzato).

    La chiave viene da os.stat -- la coppia su cui os.path.samefile e'
    costruito, e che funziona anche su Windows (indice del file piu' numero
    di serie del volume). Chiedere al sistema operativo copre i casi che una
    normalizzazione di stringhe non copre: maiuscole diverse, un
    collegamento, una junction, un `..` nel mezzo, una cartella cancellata e
    ricreata. NON copre in modo garantito un percorso UNC contro l'unita' di
    rete mappata che lo raggiunge -- rivendicato in un primo momento senza
    poterlo verificare (nessun mount di rete vero disponibile), corretto
    dopo: in alcune configurazioni Windows GetFileInformationByHandle non
    normalizza il volume, e sia la chiave sia il percorso normalizzato
    (sotto) divergono fra i due modi di nominare la stessa cartella. Resta
    un rischio noto, non un caso risolto.

    Torna anche il percorso normalizzato perche' i due criteri servono
    entrambi -- vedi stesso_workspace.

    st_ino == 0 va scartato: alcuni filesystem di Windows lo riportano cosi'
    per ogni voce, e preso per buono farebbe coincidere ogni cartella con
    ogni altra.
    """
    testo = os.path.normcase(os.path.realpath(str(path)))
    chiave = None
    try:
        st = os.stat(path)
    except OSError:
        return None, testo
    if st.st_ino:
        chiave = (st.st_dev, st.st_ino)
    return chiave, testo


def stesso_workspace(a, b):
    """Vero se le due identita' nominano lo stesso workspace.

    Il predicato e' deliberatamente sovrabbondante -- identita' OPPURE
    percorso -- perche' i due errori non costano uguale. Un falso positivo
    costa un messaggio "occupato" che non serviva; un falso negativo costa
    due processi che scrivono lo stesso faceset, e non si manifesta come un
    errore ma come un dataset rovinato scoperto ore dopo.

    Nessuno dei due criteri basta da solo: l'identita' non riconosce una
    cartella cancellata e ricreata sotto un processo che ci sta ancora
    scrivendo, e il percorso non riconosce i due nomi diversi della stessa
    cartella.
    """
    chiave_a, testo_a = a
    chiave_b, testo_b = b
    if chiave_a is not None and chiave_a == chiave_b:
        return True
    return testo_a == testo_b


FILE_PROGETTO = "project.json"
FILE_PUNTATORE = ".progetto-attivo"
VERSIONE = 1

_NON_AMMESSI = re.compile(r"[^a-z0-9._-]+")

# Nomi di device riservati da Windows -- una cartella con uno di questi nomi
# non si puo' creare, indipendentemente da maiuscole o da un'estensione dopo
# il primo punto (Windows valuta solo il pezzo prima del primo punto).
_NOMI_RISERVATI_WINDOWS = frozenset(
    ["con", "prn", "aux", "nul"]
    + ["com%d" % n for n in range(1, 10)]
    + ["lpt%d" % n for n in range(1, 10)]
)

# Ben sotto i 255 caratteri per componente di NTFS: margine per il suffisso
# che il conflitto riservato aggiunge qui sotto e per il -2/-3 di
# slug_libero, oltre a restare leggibile in Esplora risorse.
_LUNGHEZZA_MASSIMA = 60


def slug(nome):
    """Il nome della cartella derivato dal nome leggibile.

    Ristretto ad [a-z0-9._-] perche' deve attraversare setenv.bat, dove uno
    spazio o un carattere accentato in una set /p e' un guaio silenzioso.
    Deterministico: stesso nome in ingresso, stesso slug in uscita, sempre --
    la risoluzione delle collisioni e' compito di slug_libero, non di questa
    funzione.
    """
    base = unicodedata.normalize("NFKD", str(nome))
    base = base.encode("ascii", "ignore").decode("ascii").lower()
    base = _NON_AMMESSI.sub("-", base).strip("-._")
    base = base[:_LUNGHEZZA_MASSIMA].strip("-._") or "progetto"
    prefisso, punto, resto = base.partition(".")
    if prefisso in _NOMI_RISERVATI_WINDOWS:
        base = prefisso + "-x" + punto + resto
    return base


def slug_libero(radice, nome):
    """Lo slug di nome, con un suffisso numerico se la cartella esiste gia'."""
    base = slug(nome)
    radice = Path(radice)
    if not (radice / base).exists():
        return base
    n = 2
    while (radice / ("%s-%d" % (base, n))).exists():
        n += 1
    return "%s-%d" % (base, n)


def adesso():
    """L'ora corrente in ISO 8601, senza microsecondi."""
    return datetime.now().replace(microsecond=0).isoformat()


@dataclass
class Progetto:
    cartella: Path
    nome: str
    creato: str
    usato: str
    memoria: dict = field(default_factory=dict)


def leggi_progetto(cartella):
    """Il Progetto in cartella, o None se non ce n'e' uno leggibile.

    Il file e' il marcatore: una cartella e' un progetto se e solo se
    contiene project.json. La regola "ha data_src" sarebbe piu' permissiva
    e prenderebbe per progetto qualunque cartella capitata nella radice.
    """
    cartella = Path(cartella)
    try:
        dati = json.loads((cartella / FILE_PROGETTO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(dati, dict):
        return None
    memoria = dati.get("memory")
    return Progetto(
        cartella=cartella,
        nome=str(dati.get("name") or cartella.name),
        creato=str(dati.get("created") or ""),
        usato=str(dati.get("last_used") or ""),
        memoria=memoria if isinstance(memoria, dict) else {},
    )


def scrivi_progetto(progetto):
    """Scrive project.json in modo atomico: .tmp accanto e poi os.replace."""
    dati = {
        "version": VERSIONE,
        "name": progetto.nome,
        "created": progetto.creato,
        "last_used": progetto.usato,
        "memory": progetto.memoria,
    }
    destinazione = progetto.cartella / FILE_PROGETTO
    temporaneo = progetto.cartella / (FILE_PROGETTO + ".tmp")
    temporaneo.write_text(json.dumps(dati, indent=2), encoding="utf-8")
    os.replace(temporaneo, destinazione)


def _nome_esce_dalla_radice(nome):
    """Vero se `nome` non e' il nome di una sottocartella diretta della radice.

    Controllo diretto, non un confronto con slug(nome): slug() taglia a
    _LUNGHEZZA_MASSIMA caratteri, quindi e' senza perdita solo sotto quel
    limite -- un nome di cartella scelto da slug_libero() (che aggiunge
    "-2", "-3"... in caso di collisione) puo' superarlo e non tornare
    uguale a se stesso, pur restando dentro la radice. Rifiuta: la stringa
    vuota, un separatore di percorso -- "/" e "\\" incondizionatamente,
    perche' il puntatore puo' essere scritto su un sistema e riletto su un
    altro -- "." e "..", un percorso assoluto, un nome che inizia con un
    punto. E' la stessa regola che setenv.sh applica col `case
    "" | */* | *\\* | .*` e che setenv.bat applica con quattro `if` in
    sequenza sul candidato letto dal puntatore (barra normale, barra
    rovesciata, i due punti di una lettera di unita', punto iniziale) --
    sintassi diversa per necessita' di ogni shell, stessa esclusione. slug()
    comunque non produce mai un nome che inizia per punto, li toglie con
    .strip("-._").
    """
    if not nome:
        return True
    if "/" in nome or "\\" in nome:
        return True
    if nome in (".", ".."):
        return True
    if nome.startswith("."):
        return True
    if Path(nome).is_absolute():
        return True
    return False


def radice_progetti(dfl_root):
    """La radice dei progetti: <pacchetto>/workspace.

    E' la stessa cartella che gui/workspace.py::default_workspace nomina, e
    resta dov'e': cio' che cambia e' che ora puo' contenere progetti invece
    di essere essa stessa il workspace.
    """
    return Path(dfl_root).parent.parent / "workspace"


def lucchetto_vivo(cartella):
    """Vero se qualcuno sta scrivendo un modello in questo progetto.

    Il glob '*_*.lock' che busy_holder costruisce con model_name='*'
    intercetta ogni file di lucchetto, compreso quello a nome fisso che il
    training XSeg scrive. Serve perche' un training lanciato da un terminale
    non compare fra i job dell'interfaccia, ma il lucchetto lo prende
    comunque.
    """
    return busy_holder(Path(cartella) / "model", "*") is not None


class DuplicazioneIncompleta(Exception):
    """Un annullamento di ArchivioProgetti.duplica non e' riuscito a
    ripulire la destinazione parziale.

    Distinta da un ritorno None (annullamento pulito, niente sul disco):
    qui e' rimasta una cartella parziale che chi chiama deve dire
    all'utente di controllare -- non e' mai un progetto valido (project.json
    non ci finisce mai dentro), ma non e' nemmeno sparita da sola.
    """

    def __init__(self, destinazione):
        super().__init__(str(destinazione))
        self.destinazione = destinazione


# Le sottocartelle che ogni voce di `cosa` (in ArchivioProgetti.duplica)
# porta con se'. "video" non ci sta: i video stanno alla radice del
# progetto, non in una sottocartella standard, e li' li cerca
# _file_da_copiare per conto suo.
CARTELLE_PER_COSA = {
    "modello": ("model",),
    "dataset": ("data_src", "data_dst"),
}


def dimensione(cartella):
    """I byte occupati sotto `cartella`. Un file illeggibile vale zero.

    Usata dalla conferma di elimina -- la dimensione in GB comunica il peso
    di un'operazione irreversibile meglio di un nome da ridigitare.
    """
    totale = 0
    for radice, _dirs, files in os.walk(str(cartella)):
        for nome in files:
            try:
                totale += os.path.getsize(os.path.join(radice, nome))
            except OSError:
                continue
    return totale


class ArchivioProgetti:
    """I progetti sotto una radice, piu' il puntatore a quello attivo.

    Il puntatore e' un file di una riga col NOME DELLA CARTELLA, non un
    percorso assoluto, cosi' il pacchetto resta spostabile. Lo leggono anche
    setenv.bat e setenv.sh, che di questo modulo non sanno niente: il
    formato e' deliberatamente il piu' povero che una shell sappia leggere.
    """

    def __init__(self, radice):
        self.radice = Path(radice)

    def elenca(self):
        """I progetti della radice, l'ultimo usato per primo."""
        progetti = []
        try:
            voci = sorted(self.radice.iterdir())
        except OSError:
            return []
        for voce in voci:
            if not voce.is_dir():
                continue
            progetto = leggi_progetto(voce)
            if progetto is not None:
                progetti.append(progetto)
        progetti.sort(key=lambda p: p.usato, reverse=True)
        return progetti

    def crea(self, nome):
        cartella = self.radice / slug_libero(self.radice, nome)
        create_workspace(cartella)
        ora = adesso()
        progetto = Progetto(cartella=cartella, nome=str(nome), creato=ora, usato=ora)
        scrivi_progetto(progetto)
        return progetto

    def apri(self, cartella):
        progetto = leggi_progetto(cartella)
        if progetto is None:
            raise ValueError("%s non e' un progetto" % cartella)
        progetto.usato = adesso()
        scrivi_progetto(progetto)
        return progetto

    def attivo(self):
        """Il progetto che il puntatore nomina, o None.

        None copre ogni forma di puntatore inservibile -- assente,
        illeggibile, vuoto, con un nome che esce dalla radice, o che nomina
        una cartella che non e' (piu') un progetto -- perche' per chi
        chiama sono tutti lo stesso caso: non c'e' un progetto attivo.
        """
        # ValueError accanto a OSError, come in leggi_progetto: un puntatore
        # scritto a mano puo' non essere UTF-8 affatto (PowerShell 5.1 e il
        # "Salva con nome > Unicode" del Blocco note scrivono UTF-16LE), e
        # UnicodeDecodeError e' una sottoclasse di ValueError, non di OSError.
        # Questo metodo viene chiamato anche dal costruttore della finestra,
        # fuori da qualunque rete: senza questa cattura l'interfaccia non si
        # apre affatto, e l'utente non ha modo di sapere quale file togliere.
        try:
            nome = (self.radice / FILE_PUNTATORE).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
        if _nome_esce_dalla_radice(nome):
            return None
        return leggi_progetto(self.radice / nome)

    def imposta_attivo(self, progetto):
        """Scrive il puntatore, in modo atomico e con terminatore unix.

        Il terminatore non e' un dettaglio: il file lo rilegge anche
        setenv.bat con una `set /p`.
        """
        destinazione = self.radice / FILE_PUNTATORE
        temporaneo = self.radice / (FILE_PUNTATORE + ".tmp")
        with open(temporaneo, "w", encoding="utf-8", newline="\n") as f:
            f.write(progetto.cartella.name + "\n")
        os.replace(temporaneo, destinazione)

    def pulisci_attivo(self):
        """Rimuove il puntatore, se c'e'.

        Chiamato quando la GUI passa a una cartella che non e' (o non e'
        piu') un progetto -- la radice stessa, un'installazione non ancora
        migrata: senza questo (I1, revisione finale) la riga di comando
        continuerebbe a lavorare sul progetto precedente mentre la finestra
        mostra tutt'altro, "progetto attivo" resterebbe il nome di una
        cartella che la GUI non guarda piu'. Stessa idea di elimina(), sotto,
        che ripulisce il puntatore quando il progetto che nominava sparisce.
        """
        (self.radice / FILE_PUNTATORE).unlink(missing_ok=True)

    def rinomina(self, progetto, nome):
        """Cambia il nome leggibile. La cartella non si tocca -- vedi riconcilia."""
        progetto.nome = str(nome)
        progetto.usato = adesso()
        scrivi_progetto(progetto)
        return progetto

    def _pronta_a_muoversi(self, progetto):
        """Le tre condizioni di riconciliabilita' che non richiedono sapere
        quanti job dell'interfaccia stanno scrivendo nel progetto -- quello
        lo sa solo chi chiama (vedi riconciliabile, sotto, e il commento di
        riconcilia su perche' lui la richiama da se'). Condivisa dai due
        punti apposta, cosi' le due liste di condizioni non possono
        divergere silenziosamente in una modifica futura.
        """
        atteso = slug(progetto.nome)
        if atteso == progetto.cartella.name:
            return False
        if (self.radice / atteso).exists():
            return False
        return not lucchetto_vivo(progetto.cartella)

    def riconciliabile(self, progetto, job_attivi=0):
        """Vero se la cartella puo' essere riallineata al nome, adesso.

        Quattro condizioni, tutte necessarie: nessun job dell'interfaccia
        sul progetto, nessun lucchetto vivo (un training da terminale non e'
        un job dell'interfaccia), la destinazione libera, e uno slug che
        davvero non corrisponde piu'. Le ultime tre sono _pronta_a_muoversi;
        il conteggio dei job dell'interfaccia lo sa solo chi chiama, quindi
        resta qui.
        """
        if job_attivi:
            return False
        return self._pronta_a_muoversi(progetto)

    def riconcilia(self, progetto):
        """Rinomina la cartella e aggiorna il puntatore. None se non ci riesce.

        Fra il "si puo'" di riconciliabile (nel chiamante) e questa
        chiamata c'e' spesso una conferma dell'utente in mezzo -- una
        decisione umana, di durata potenzialmente illimitata: nel
        frattempo un training lanciato da un terminale puo' prendere il
        lucchetto, o la destinazione puo' essersi occupata. Per questo qui
        si ricontrolla da se' tutto cio' che si puo' vedere senza l'aiuto
        del chiamante -- non il conteggio dei job dell'interfaccia, che
        tocca a lui ricontrollare appena prima di richiamare questo metodo
        -- e si rifiuta di spostare (`None`, come per l'OSError sotto) se
        la situazione e' cambiata. Resta comunque una finestra fra questo
        controllo e os.replace: e' voluta, non una svista -- senza un
        lucchetto a livello di sistema operativo non si chiude del tutto,
        e il punto di questo ricontrollo e' restringerla dalla durata di
        una decisione umana a quella di una chiamata di sistema, non
        annullarla.

        Un OSError qui non e' un guasto da riferire nemmeno lui: su
        Windows una rename fallisce se qualcuno tiene un handle aperto
        sulla cartella, e una finestra di Esplora risorse basta. Si
        riprova alla prossima apertura.
        """
        if not self._pronta_a_muoversi(progetto):
            return None
        vecchia = progetto.cartella
        nuova = self.radice / slug(progetto.nome)
        attivo = self.attivo()
        try:
            os.replace(vecchia, nuova)
        except OSError:
            return None
        progetto.cartella = nuova
        scrivi_progetto(progetto)
        if attivo is not None and attivo.cartella == vecchia:
            self.imposta_attivo(progetto)
        return nuova

    def elimina(self, progetto):
        """Rimuove la cartella del progetto e tutto cio' che contiene.

        L'unica operazione davvero irreversibile del ciclo. Chi chiama deve
        aver gia' verificato che nessun job stia scrivendo nel progetto --
        questo metodo non lo ricontrolla da se', esattamente come crea() non
        ricontrolla che il nome sia libero: quella verifica vuole sapere dei
        job dell'interfaccia, che solo chi chiama conosce (vedi
        gui.main_window._job_attivi_su).

        Se il progetto eliminato era quello attivo, il puntatore viene
        ripulito insieme alla cartella: lasciarlo nominare una cartella che
        non c'e' piu' e' un debito silenzioso -- nessun lettore ne soffre
        oggi (attivo(), setenv.sh e setenv.bat verificano tutti e tre
        l'esistenza della cartella prima di fidarsi del puntatore), ma un
        puntatore che nomina il nulla resta un debito da pagare per chi lo
        legge dopo. Letto PRIMA di rimuovere la cartella: dopo, attivo()
        tornerebbe None comunque (leggi_progetto non trova piu' project.json
        sotto una cartella cancellata), il che non direbbe se il puntatore
        nominava davvero questo progetto.
        """
        era_attivo = self.attivo()
        shutil.rmtree(progetto.cartella)
        if era_attivo is not None and era_attivo.cartella == progetto.cartella:
            self.pulisci_attivo()

    def duplica(self, progetto, nome, cosa, avanzamento=None, annullato=None):
        """Copia in un progetto nuovo cio' che `cosa` nomina.

        `cosa` e' un sottoinsieme di {"modello", "dataset", "video"}.
        project.json e' scritto per ULTIMO, e questa non e' una rifinitura:
        e' cio' che impedisce a una copia interrotta di essere scambiata
        per un progetto valido -- il file e' il marcatore (vedi
        leggi_progetto), quindi finche' non c'e' la cartella non e' un
        progetto, per quanti file ci siano gia' dentro.

        Torna None se `annullato()` dice di fermarsi in un punto qualsiasi
        della copia e la destinazione parziale viene rimossa senza intoppi:
        un annullamento pulito. Solleva `DuplicazioneIncompleta` se invece
        la rimozione stessa fallisce (un permesso negato, un handle aperto):
        chi chiama -- il dialogo di avanzamento -- deve poter distinguere
        "annullato, niente da ripulire" da "annullato, ma e' rimasta della
        roba sul disco che l'utente deve sapere di dover togliere a mano".
        In nessuno dei due casi project.json viene scritto: la destinazione
        non e' mai un progetto valido, per definizione del marcatore sopra.
        """
        destinazione = self.radice / slug_libero(self.radice, nome)
        create_workspace(destinazione)
        sorgenti = self._file_da_copiare(progetto, cosa)
        totale = len(sorgenti)
        for fatti, (origine, relativo) in enumerate(sorgenti, start=1):
            if annullato is not None and annullato():
                try:
                    shutil.rmtree(destinazione)
                except OSError:
                    raise DuplicazioneIncompleta(destinazione)
                return None
            arrivo = destinazione / relativo
            arrivo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origine, arrivo)
            if avanzamento is not None:
                avanzamento(fatti, totale)
        if avanzamento is not None and totale == 0:
            avanzamento(0, 0)
        ora = adesso()
        nuovo = Progetto(cartella=destinazione, nome=str(nome), creato=ora, usato=ora)
        scrivi_progetto(nuovo)
        return nuovo

    def migra(self, nome):
        """Sposta il workspace unico di prima dei progetti dentro un progetto.

        os.replace per sottocartella e' istantaneo sullo stesso volume, che e'
        il caso normale: la radice e il progetto sono entrambi sotto il
        pacchetto. La ricaduta su shutil.move copre il caso in cui non lo
        siano (una radice che l'utente ha collegato altrove).

        Il project.json lo scrive `crea`, prima dello spostamento: qui non e'
        un marcatore di completezza come nella duplicazione, perche' una
        migrazione interrotta lascia comunque un progetto valido con dentro
        una parte dei dati -- e l'altra parte ancora nella radice, dove
        l'utente la vede.
        """
        progetto = self.crea(nome)
        for sotto in STANDARD_SUBDIRS:
            origine = self.radice / sotto
            if not origine.is_dir():
                continue
            arrivo = progetto.cartella / sotto
            if arrivo.exists():
                arrivo.rmdir()          # crea() l'ha appena fatta, ed e' vuota
            try:
                os.replace(origine, arrivo)
            except OSError:
                shutil.move(str(origine), str(arrivo))
        for video in sorted(self.radice.glob("*.mp4")):
            try:
                os.replace(video, progetto.cartella / video.name)
            except OSError:
                shutil.move(str(video), str(progetto.cartella / video.name))
        self.imposta_attivo(progetto)
        return progetto

    def _file_da_copiare(self, progetto, cosa):
        """Le coppie (origine, percorso relativo alla radice del progetto)
        che `duplica` deve copiare per il sottoinsieme `cosa` -- calcolate
        tutte in anticipo cosi' l'avanzamento riportato a chi chiama ha un
        totale vero fin dal primo passo, non una stima.
        """
        coppie = []
        for chiave, cartelle in CARTELLE_PER_COSA.items():
            if chiave not in cosa:
                continue
            for sotto in cartelle:
                base = progetto.cartella / sotto
                for radice, _dirs, files in os.walk(str(base)):
                    for nome in files:
                        origine = Path(radice) / nome
                        coppie.append((origine, origine.relative_to(progetto.cartella)))
        if "video" in cosa:
            for video in sorted(progetto.cartella.glob("*.mp4")):
                coppie.append((video, Path(video.name)))
        return coppie


def serve_migrazione(radice):
    """Vero se la radice e' ancora il workspace unico di prima dei progetti.

    Non basta che le tre cartelle ci siano: se un progetto esiste gia', la
    radice le puo' avere per qualunque ragione e spostarle sarebbe una
    sorpresa. E non si propone mai mentre un lucchetto e' vivo: significa
    che un training da terminale sta lavorando proprio li' dentro, e
    spostargli le cartelle sotto i piedi e' la cosa peggiore possibile.
    """
    radice = Path(radice)
    if not any((radice / sotto).is_dir() for sotto in STANDARD_SUBDIRS):
        return False
    if ArchivioProgetti(radice).elenca():
        return False
    return not lucchetto_vivo(radice)
