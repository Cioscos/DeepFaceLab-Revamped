import os
import time
from pathlib import Path
from os import scandir

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

#Quante volte, e a che distanza, riprovare a mettere il file al suo posto
#quando Windows lo tiene occupato: l'antivirus in tempo reale apre un file
#appena scritto per esaminarlo, e finche' lo tiene `os.replace` risponde
#PermissionError. Su Linux non succede e il primo tentativo basta.
TENTATIVI_SOSTITUZIONE = 10
ATTESA_SOSTITUZIONE_SEC = 0.5


def write_bytes_safe(p, bytes_data):
    """
    writes to .tmp first and then rename to target filename
    """
    scrivi_al_sicuro(p, lambda f: f.write(bytes_data))


def scrivi_al_sicuro(p, scrivi):
    """
    `scrivi(f)` riempie `<p>.tmp`, che poi prende il posto di `p` con un
    solo `os.replace`: atomico su Linux e su Windows, e il file precedente
    resta intero finche' il nuovo non e' completo. Riprova la sostituzione
    se il sistema tiene il file occupato (vedi TENTATIVI_SOSTITUZIONE).
    Se non ce la fa -- il disco pieno e' il caso vero -- il `.tmp` non
    resta in giro: e' proprio il mezzo file che si voleva evitare.
    """
    p = Path(p)
    p_tmp = p.parent / (p.name + '.tmp')
    try:
        with open(p_tmp, 'wb') as f:
            scrivi(f)
        for tentativo in range(TENTATIVI_SOSTITUZIONE):
            try:
                os.replace(p_tmp, p)
                return
            except PermissionError:
                if tentativo == TENTATIVI_SOSTITUZIONE - 1:
                    raise
                time.sleep(ATTESA_SOSTITUZIONE_SEC)
    except BaseException:
        try:
            p_tmp.unlink()
        except OSError:
            pass    # il guasto da riferire e' l'altro
        raise

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry

def get_image_paths(dir_path, image_extensions=image_extensions, subdirs=False, return_Path_class=False):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():

        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = scandir(str(dir_path))

        for x in list(gen):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append( x.path if not return_Path_class else Path(x.path) )
    return sorted(result)

def get_image_unique_filestem_paths(dir_path, verbose_print_func=None):
    result = get_image_paths(dir_path)
    result_dup = set()

    for f in result[:]:
        f_stem = Path(f).stem
        if f_stem in result_dup:
            result.remove(f)
            if verbose_print_func is not None:
                verbose_print_func ("Duplicate filenames are not allowed, skipping: %s" % Path(f).name )
            continue
        result_dup.add(f_stem)

    return sorted(result)

def get_paths(dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) ]) ]
    else:
        return []
        
def get_file_paths(dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) if x.is_file() ]) ]
    else:
        return []

def get_all_dir_names (dir_path):
    dir_path = Path (dir_path)

    if dir_path.exists():
        return sorted([ x.name for x in list(scandir(str(dir_path))) if x.is_dir() ])
    else:
        return []

def get_all_dir_names_startswith (dir_path, startswith):
    dir_path = Path (dir_path)
    startswith = startswith.lower()

    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if x.name.lower().startswith(startswith):
                result.append ( x.name[len(startswith):] )
    return sorted(result)

def get_first_file_by_stem (dir_path, stem, exts=None):
    dir_path = Path (dir_path)
    stem = stem.lower()

    if dir_path.exists():
        for x in sorted(list(scandir(str(dir_path))), key=lambda x: x.name):
            if not x.is_file():
                continue
            xp = Path(x.path)
            if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
                return xp

    return None

def move_all_files (src_dir_path, dst_dir_path):
    paths = get_file_paths(src_dir_path)
    for p in paths:
        p = Path(p)
        p.rename ( Path(dst_dir_path) / p.name )

def delete_all_files (dir_path):
    paths = get_file_paths(dir_path)
    for p in paths:
        p = Path(p)
        p.unlink()
