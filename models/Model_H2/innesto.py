"""Dal .npy di un SAEHD liae-udt ai file di H2: encoder tal quale, decoder con
res0/res1 rinominati pim0/pim1 e le chiavi nuove del PIM lasciate
all'inizializzazione (out1x1 a zero: il decoder innestato E' quello di
SAEHD finche' il training non muove il ramo modulato). Niente identita',
niente ottimizzatore: partono da zero, come gli inter della baseline del
banco. L'inter invece si puo' copiare tal quale con `inter_dal_sorgente`: e'
lecito perche' l'inter di H2 (`Inter(in_ch, ae_dims, ae_dims*2)`) ha le
stesse chiavi e le stesse forme dell'inter_B di un liae-udt -- le chiavi non
portano il nome della rete.

Due vie: `innesta` scrive su disco (usata dal banco, cartella dedicata alla
corsa, svuotata a ogni chiamata); `copia_pesi` innesta in memoria sulle reti
gia' costruite, senza toccare il disco (usata dal primo avvio di `train`,
dove il model dir e' dell'utente e ha dentro altri modelli)."""
import pathlib
import pickle
import shutil

import torch

from core import pickleex
from core.interact import interact as io
from core.leras import nn
from core.leras.weight_io import read_weights_file, torch_name_to_disk_key

CHIAVI_RINOMINATE = {'res0/': 'pim0/', 'res1/': 'pim1/'}      # le chiavi non portano il nome della rete
CLASSI_SORGENTE = ('SAEHD', 'SAEHDX')
OPZIONI_EREDITATE = ('resolution', 'face_type', 'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims',
                     'models_opt_on_gpu', 'masked_training', 'eyes_mouth_prio', 'uniform_yaw',
                     'blur_out_mask', 'adabelief', 'lr_dropout', 'random_warp', 'random_hsv_power',
                     'ct_mode', 'clipgrad', 'random_src_flip', 'random_dst_flip', 'batch_size',
                     'autobackup_hour', 'write_preview_history', 'target_iter', 'cudnn_benchmark')
# Le opzioni che i pesi del sorgente fissano: con un innesto non si chiedono.
OPZIONI_FISSATE_DAI_PESI = ('resolution', 'face_type', 'ae_dims', 'e_dims', 'd_dims', 'd_mask_dims')
# Chiavi proprie di H2 (Model.py::on_initialize_options): il sorgente SAEHDX
# non le porta, quindi non sono nel .dat da cui si innesta.
OPZIONI_H2 = ('identita', 'id_power', 'ifsr_power', 'dino_power', 'dino_ogni', 'ffl_power',
              'bleed_power', 'bleed_campione', 'cuda_graph', 'torch_compile', 'maschera_tronco',
              'innesto', 'innesto_inter')
E_DIM = 512
# Le opzioni che l'innesto su disco scrive esplicite quando non richieste:
# i default di Model.py sono la ricetta raccomandata all'utente, non il punto
# zero di una misura.
NEUTRE = {'id_power': 0.0, 'ifsr_power': 0.0, 'dino_power': 0.0, 'dino_ogni': 1, 'ffl_power': 0.0,
          'bleed_power': 0.0, 'bleed_campione': False, 'maschera_tronco': False}


def _rinomina(chiave):
    for vecchio, nuovo in CHIAVI_RINOMINATE.items():
        if chiave.startswith(vecchio):
            return nuovo + chiave[len(vecchio):]
    return chiave


def decoder_da_saehd(decoder_h2, dizionario):
    """Copia nel DecoderH2 (gia' costruito) ogni chiave del decoder SAEHD che
    esiste anche qui, rinominando res0/res1; ritorna quante ne ha copiate."""
    decoder_h2._ensure_built()
    per_chiave = {torch_name_to_disk_key(decoder_h2, path): (path, p) for path, p in decoder_h2.named_parameters()}
    copiate = 0
    with torch.no_grad():
        for chiave, valore in dizionario.items():
            nuova = _rinomina(chiave)
            if nuova not in per_chiave:
                continue
            path, p = per_chiave[nuova]
            p.copy_(torch.as_tensor(decoder_h2._owner_of(path).weight_from_disk(path, valore), dtype=p.dtype))
            copiate += 1
    return copiate


def copia_decoder(decoder_h2, percorso_decoder_saehd):
    """Il decoder SAEHD nel DecoderH2 gia' costruito; ogni chiave del sorgente
    deve trovare posto, o e' un errore. La parte comune delle due vie."""
    dizionario = read_weights_file(percorso_decoder_saehd)
    copiate = decoder_da_saehd(decoder_h2, dizionario)
    if copiate != len(dizionario):
        raise ValueError(f"innesto: copiate {copiate} chiavi su {len(dizionario)} del decoder sorgente")
    return copiate


def sorgenti(model_dir):
    """{nome: (classe, options)} dei modelli liae-udt di classe SAEHD/SAEHDX in
    model_dir, letti dai <nome>_<classe>_data.dat. A parita' di nome vince
    SAEHDX; un .dat che non si legge non e' un sorgente."""
    trovati = {}
    for classe in CLASSI_SORGENTE:
        suffisso = f"_{classe}_data.dat"
        for p in sorted(pathlib.Path(model_dir).glob(f"*{suffisso}")):
            try:
                o = pickle.loads(p.read_bytes()).get("options", {})
            except Exception:
                continue
            if o.get("archi") == "liae-udt":
                trovati[p.name[:-len(suffisso)]] = (classe, o)
    return trovati


def copia_pesi(nets, model_dir, nome, inter=False):
    """L'innesto in memoria, sulle reti gia' costruite: encoder tal quale,
    decoder con copia_decoder, inter (l'inter_B del sorgente) solo se richiesto.
    Non scrive ne' cancella nulla: e' la via del primo avvio di `train`, dove
    la cartella e' dell'utente e ha dentro altri modelli. Ritorna le chiavi
    copiate nel decoder."""
    trovati = sorgenti(model_dir)
    if nome not in trovati:
        raise FileNotFoundError(f"innesto: {nome!r} non e' un modello liae-udt SAEHD/SAEHDX in {model_dir}")
    prefisso = str(pathlib.Path(model_dir) / f"{nome}_{trovati[nome][0]}_")
    if not nets['encoder'].load_weights(prefisso + "encoder.npy"):
        raise FileNotFoundError(prefisso + "encoder.npy")
    copiate = copia_decoder(nets['decoder'], prefisso + "decoder.npy")
    if inter and not nets['inter'].load_weights(prefisso + "inter_B.npy"):
        raise FileNotFoundError(prefisso + "inter_B.npy")
    return copiate


def innesta(dir_sorgente, nome_sorgente, classe_sorgente, dest, nome, opzioni, inter_dal_sorgente=False):
    dir_sorgente, dest = pathlib.Path(dir_sorgente), pathlib.Path(dest)
    stato = pickle.loads((dir_sorgente / f"{nome_sorgente}_{classe_sorgente}_data.dat").read_bytes())
    o = stato["options"]
    if o.get("archi") != "liae-udt":
        raise ValueError(f"l'innesto vuole un modello liae-udt, questo e' {o.get('archi')!r}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in list(dest.glob("*.npy")) + list(dest.glob("*.dat")):
        f.unlink()
    prefisso = f"{nome_sorgente}_{classe_sorgente}_"
    shutil.copy2(dir_sorgente / f"{prefisso}encoder.npy", dest / f"{nome}_H2_encoder.npy")
    if inter_dal_sorgente:
        shutil.copy2(dir_sorgente / f"{prefisso}inter_B.npy", dest / f"{nome}_H2_inter.npy")

    decoder = nn.DecoderH2(in_ch=int(o['ae_dims']) * 4, d_ch=int(o['d_dims']), d_mask_ch=int(o['d_mask_dims']),
                           e_dim=E_DIM, name='decoder', maschera_tronco=bool(opzioni.get('maschera_tronco', False)))
    decoder.build(); decoder.init_weights()
    copia_decoder(decoder, dir_sorgente / f"{prefisso}decoder.npy")
    decoder.save_weights(dest / f"{nome}_H2_decoder.npy")

    chiavi_ammesse = OPZIONI_EREDITATE + OPZIONI_H2
    nuove = {k: o[k] for k in OPZIONI_EREDITATE if k in o}
    ignorate = sorted(set(opzioni) - set(chiavi_ammesse))
    if ignorate:
        io.log_info(f"innesto: ignorate le opzioni del sorgente che H2 non conosce: {', '.join(ignorate)}")
    nuove.update({k: v for k, v in opzioni.items() if k in chiavi_ammesse})
    nuove.setdefault('identita', 'learned')
    # Il banco misura solo cio' che chiede: un braccio senza potenze e' il
    # tronco, non la ricetta che Model.py propone a chi parte dalla GUI.
    # Misurato: una «sonda del tronco» a default impliciti addestrava con
    # id 2 / ifsr 0,08 / bleed 1 (204 ms/it invece di 115).
    for k, v in NEUTRE.items():
        nuove.setdefault(k, v)
    (dest / f"{nome}_H2_data.dat").write_bytes(pickleex.dumps(
        {"iter": 1, "options": nuove, "loss_history": [[0.0, 0.0]]}))
    (dest / "H2_default_options.dat").write_bytes(pickleex.dumps(nuove))
    return dest
