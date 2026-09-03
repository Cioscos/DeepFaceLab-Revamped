"""Lo stdin del protocollo dei servizi senza schermo -- fusione
(`MergeSession`), estrazione manuale (`ExtractManual`), dettaglio del
faceset (`FacesetDetail`) -- come va letto: in un descrittore NOSTRO, e su
Windows con I/O overlapped vera (voce 3.92).

`sys.stdin` non e' nostro. Mentre un modello si carica, `ModelBase` chiama
`io.input_skip_pending()`, che avvia un processo tenendo il descrittore 0,
lo uccide, e poi **ri-avvolge** `sys.stdin` (`sys.stdin =
os.fdopen(sys.stdin.fileno())`, `core/interact`): da quel punto l'oggetto
che un lettore userebbe non e' piu' quello con cui il servizio e' nato, e il
descrittore ha due proprietari. Un `dup` preso PRIMA di tutto il resto
toglie la classe intera: e' lo stesso gesto che `MergeSession._canale_del_protocollo`
fa per lo stdout, e per la stessa ragione -- un canale del protocollo non
puo' dipendere da un oggetto che il resto del programma si sente libero di
sostituire.

Su Windows il descrittore duplicato non basta: l'handle sotto e' overlapped
e il `read` del CRT ci si sveglia con un EOF finto al primo processo figlio
(`righe_overlapped`, che spiega e misura). Il canale e' quindi, li', un
generatore di righe e non un file: i tre `servi` lo percorrono allo stesso
modo (`for riga in entrata`).
"""
import os
import sys


def _piattaforma():
    return sys.platform


def _handle_del_descrittore(fd):
    import msvcrt
    return msvcrt.get_osfhandle(fd)


def righe_overlapped(handle, winapi=None):
    """Le righe di una pipe letta con I/O overlapped VERA (Windows).

    Lo stdin che la GUI ci da' e' il capo client di una named pipe che Qt
    apre con FILE_FLAG_OVERLAPPED. Su un handle cosi' una ReadFile
    sincrona (quella del CRT, quindi di ogni `read` di Python) aspetta
    l'evento del file object, non il proprio completamento: se
    un'altra I/O completa sullo stesso oggetto, si sveglia e riporta
    «0 byte» -- un EOF che non c'e', documentato da Microsoft come
    «la funzione puo' riportare erroneamente che la lettura e'
    completa». E qualcun altro quell'oggetto lo tocca: Windows 8+
    propaga gli std handle a ogni figlio anche con bInheritHandles
    falso, e un Python appena nato interroga il suo stdin all'avvio.
    Misurato con il Python del pacchetto sotto un QProcess: un
    `subprocess.Popen([python, "-c", "pass"])` nel servizio manda a
    EOF il lettore 20 ms dopo, un `PeekNamedPipe` in-process pure,
    un figlio con gli std handle su NUL no, e questo lettore no (la
    riproduzione eseguibile, con i suoi esiti, sta nei documenti del
    pacchetto, voce 3.92). Con 8 client
    di compositing da avviare la sessione di fusione moriva a ogni Start;
    gli altri due servizi erano immuni solo perche' leggono sul thread
    principale, fra un comando e l'altro -- finche' un figlio non fosse
    ancora in avvio al ritorno alla lettura.

    Una lettura overlapped, con la sua OVERLAPPED e il suo evento,
    aspetta il completamento della PROPRIA richiesta: e' immune per
    costruzione, ed e' come `multiprocessing.connection.PipeConnection`
    legge le sue pipe -- stesso `_winapi`, stessa sequenza. La fine del
    tubo e' SOLO ERROR_BROKEN_PIPE, cioe' la GUI che chiude il suo
    capo; 0 byte (una scrittura vuota) non lo e'.
    """
    if winapi is None:
        import _winapi
        winapi = _winapi
    resto = b""
    while True:
        try:
            ov, err = winapi.ReadFile(handle, 8192, overlapped=True)
            try:
                letti, err = ov.GetOverlappedResult(True)
            except BaseException:
                ov.cancel()
                raise
        except OSError as e:
            if getattr(e, "winerror", None) == winapi.ERROR_BROKEN_PIPE:
                break
            raise
        if letti == 0:
            continue
        resto += ov.getbuffer()[:letti]
        while b"\n" in resto:
            riga, resto = resto.split(b"\n", 1)
            yield riga.decode("utf-8", "replace") + "\n"
    if resto:
        yield resto.decode("utf-8", "replace")


def apri():
    """Il canale dei comandi: un iterabile di righe, da chiamare PRIMA di
    ogni altra cosa (dopo, il caricamento di un modello passa da codice
    che ri-avvolge `sys.stdin`)."""
    fd = os.dup(0)
    if _piattaforma() == "win32":
        return righe_overlapped(_handle_del_descrittore(fd))
    return os.fdopen(fd, "r", encoding="utf-8", errors="replace")
