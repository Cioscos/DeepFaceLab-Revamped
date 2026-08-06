import io
import pickle


class _Numpy1Pickler(pickle._Pickler):
    """
    Il pickler standard, ma i moduli di numpy sono citati coi nomi di numpy 1.x.

    numpy 2 ha rinominato numpy.core in numpy._core, e serializza i suoi array
    attraverso numpy._core.multiarray._reconstruct. Un'installazione ferma a
    numpy 1 -- ogni build di DeepFaceLab precedente alla porta -- quel modulo non
    ce l'ha, e rifiuta il file intero con ModuleNotFoundError. Al contrario numpy
    2 i nomi vecchi li legge da se', tramite il proprio strato di compatibilita':
    scriverli e' quindi leggibile in entrambe le direzioni, non leggibile solo
    all'indietro.

    Serve la versione Python del pickler perche' quella in C non passa da
    save_global, e non c'e' altro aggancio: il nome del modulo lo decide
    __module__ della funzione di ricostruzione, che appartiene a numpy.
    """

    def save_global(self, obj, name=None):
        if name is None:
            name = getattr(obj, "__qualname__", None) or obj.__name__

        module_name = pickle.whichmodule(obj, name)
        if module_name.startswith("numpy._core"):
            self.save("numpy.core" + module_name[len("numpy._core"):])
            self.save(name)
            self.write(pickle.STACK_GLOBAL)
            self.memoize(obj)
            return

        return super().save_global(obj, name)


def dumps(obj):
    """
    Serializza per il disco: protocollo 4, nomi di modulo di numpy 1.

    Il 4 e' il protocollo con cui e' stato scritto ogni faceset gia' esistente,
    ed e' il piu' alto che l'interprete 3.6.8 di quelle build sappia leggere. E'
    congelato dal formato su disco, non scelto: il protocollo di default
    dell'interprete e' libero di salire, questo no.
    """
    f = io.BytesIO()
    _Numpy1Pickler(f, 4).dump(obj)
    return f.getvalue()
