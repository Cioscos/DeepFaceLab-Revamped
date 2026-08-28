"""I supervisori congelati come mixin: H1 li somma alla loss di SAEHDX, H2 li
usa anche per calcolare i vettori d'identita' dei due faceset. Il mixin va
messo PRIMA di SAEHDXModel nelle basi: onTrainOneIter, _esegui_il_passo e
_motivo_per_non_catturare chiamano super()."""
import pathlib

import cv2
import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn
from DFLIMG import DFLJPG
from facelib.supervisori import crop, pesi, termini
from models.Model_SAEHD.Model import saehd_mask_blur


class Supervisori:
    # nome del termine -> nome dell'opzione
    POTENZE = {'id': 'id_power', 'ifsr': 'ifsr_power', 'dino': 'dino_power', 'ffl': 'ffl_power', 'bleed': 'bleed_power'}
    # i primi 2000 nomi in ordine: basta per un set di 604, con set piu' grandi il riferimento e' parziale (verbale §6)
    MASSIMO_VOLTI_RIFERIMENTO = 2000
    LOTTO_RIFERIMENTO = 32

    def _spegni_le_leve_incompatibili(self):
        """cuda_graph non convive coi supervisori: le reti congelate e l'hook
        non stanno nel grafo catturato. torch_compile invece si e' misurato
        con l'hook acceso (h2-misure.md par. 10) e resta alla scelta dell'utente."""
        if not self._potenze_accese():
            return
        if self.options['cuda_graph']:
            io.log_info("H1: cuda_graph spento -- i supervisori non stanno nel grafo catturato")
            self.options['cuda_graph'] = False

    def _potenze_accese(self):
        return {opzione: float(self.options.get(opzione, 0.0))
                for opzione in self.POTENZE.values() if float(self.options.get(opzione, 0.0)) > 0}

    def _richiedi_cuda(self):
        if nn.device is None or nn.device.type != "cuda":
            raise Exception("i supervisori richiedono una GPU CUDA")

    def _carica_supervisori(self, serve_adaface, serve_dino):
        self._adaface = self._congela(pesi.carica_adaface_rete()) if serve_adaface else None
        self._dinov2 = self._congela(pesi.carica_dinov2()) if serve_dino else None

    @staticmethod
    def _congela(rete):
        return rete.to(nn.device).eval().requires_grad_(False).to(memory_format=torch.channels_last)

    def _volti(self, percorso, lato, con_immagini=True):
        """5 punti (in pixel a `lato`) dei volti allineati in `percorso` e,
        solo se `con_immagini`, le immagini (lato x lato, BGR [0,1]): sono gia'
        allineati, i landmark del DFLJPG stanno nel frame allineato e si
        scalano con l'immagine.

        Senza AdaFace nessuno guarda i pixel -- di questi volti serve solo
        theta -- e la decodifica e' tutto il costo dell'avvio: `get_shape`
        legge l'intestazione SOF, `get_img` decodifica il JPEG intero."""
        percorsi = sorted(pathlib.Path(percorso).glob("*.jpg"))[:self.MASSIMO_VOLTI_RIFERIMENTO]
        immagini, punti = [], []
        for p in percorsi:
            j = DFLJPG.load(str(p))
            if j is None or j.get_landmarks() is None or j.get_shape() is None:
                continue
            scala = lato / j.get_shape()[0]
            if con_immagini:
                immagini.append(cv2.resize(j.get_img(), (lato, lato), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
            punti.append(crop.cinque_punti(j.get_landmarks()) * scala)
        if not punti:
            raise Exception(f"nessun volto con landmark in {percorso}")
        return immagini, np.stack(punti)

    def _riferimento(self, percorso):
        """theta del crop fisso, residuo (medio, massimo) in px dei 5 punti di
        ogni volto dai 5 medi, embedding medio normalizzato, coseno medio dei
        volti a quel riferimento e su quanti volti tutto questo e' calcolato.

        Senza AdaFace l'embedding e il coseno non li usa nessuno: sono zeri, e
        i pixel non vengono nemmeno letti."""
        rete = self._adaface
        immagini, punti = self._volti(percorso, self.resolution, rete is not None)
        medi = punti.mean(0)
        distanze = np.linalg.norm(punti - medi, axis=2).mean(1)
        theta = crop.matrice_fissa(medi, self.resolution)
        if rete is None:
            return theta, float(distanze.mean()), float(distanze.max()), torch.zeros(512), 0.0, len(punti)

        dev = next(rete.parameters()).device
        embedding = []
        with torch.no_grad():
            for i in range(0, len(immagini), self.LOTTO_RIFERIMENTO):
                x = torch.from_numpy(np.stack(immagini[i:i + self.LOTTO_RIFERIMENTO])).permute(0, 3, 1, 2).to(dev)
                x112 = crop.crop_112_torch(x, theta)
                embedding.append(torch.nn.functional.normalize(rete(x112 * 2 - 1).float(), dim=1).cpu())
        e = torch.cat(embedding)
        e_src = torch.nn.functional.normalize(e.mean(0), dim=0)
        return (theta, float(distanze.mean()), float(distanze.max()),
                e_src.to(dev), float((e * e_src).sum(1).mean()), len(immagini))

    def _riferimento_src(self):
        return self._riferimento(self.training_data_src_path)

    def _riferimento_dst(self):
        """Come _riferimento_src ma sul faceset dst: serve a bleed_power quando
        respinge lo swap dal riferimento medio (bleed_campione=False). Con
        bleed_campione=True il riferimento e' per campione e questo non serve."""
        return self._riferimento(self.training_data_dst_path)

    def _serve_e_dst_medio(self, potenze):
        """bleed_power vuole il riferimento medio del dst solo se non lavora
        per campione: con bleed_campione=True lo calcola nel loss_extra dal
        dst del proprio campione, e caricare il medio sarebbe un costo
        d'avvio sprecato."""
        return 'bleed_power' in potenze and not self.options.get('bleed_campione', False)

    def _costruisci_loss_extra(self):
        potenze = self._potenze_accese()
        risoluzione = self.resolution
        theta, e_src = self._theta, self._e_src
        bleed_campione = 'bleed_power' in potenze and bool(self.options.get('bleed_campione', False))
        e_dst = self._e_dst if self._serve_e_dst_medio(potenze) else None
        ada, vit_ = self._adaface, self._dinov2
        opzione = self.POTENZE
        # dino_ogni=1 (default) applica il termine a ogni iterazione com'era:
        # non e' una "potenza" perche' non moltiplica nulla nella loss, sceglie
        # solo quando il forward DINOv2 gira.
        dino_ogni = max(1, int(self.options.get('dino_ogni', 1)))

        def loss_extra(p):
            t = {}
            if 'id_power' in potenze or 'ifsr_power' in potenze or 'bleed_power' in potenze:
                swap112 = crop.crop_112_torch(p['pred_src_dst'], theta)
            e_swap = None
            if 'id_power' in potenze or 'bleed_power' in potenze:
                # un solo forward AdaFace sullo swap, condiviso da id e bleed.
                e_swap = termini.embedding_swap(ada, swap112)
            if 'id_power' in potenze:
                t['id'] = termini.id_loss_da_embedding(e_swap, e_src)
            dst112 = None
            if 'ifsr_power' in potenze or bleed_campione:
                # il dst e' il bersaglio: crop e, per bleed_campione, anche
                # l'embedding restano fuori dal grafo.
                with torch.no_grad():
                    dst112 = crop.crop_112_torch(p['target_dst'], theta)
            if 'ifsr_power' in potenze:
                t['ifsr'] = termini.ifsr(ada, swap112, dst112)
            if 'bleed_power' in potenze:
                if bleed_campione:
                    with torch.no_grad():
                        e_dst_campione = termini.embedding_swap(ada, dst112)
                    t['bleed'] = termini.bleed_da_embedding(e_swap, e_dst_campione)
                else:
                    t['bleed'] = termini.bleed_da_embedding(e_swap, e_dst)
            # dino_ogni salta il forward DINOv2 sulle iterazioni non multiple di
            # k: il termine calcolato va moltiplicato per k, cosi' il gradiente
            # medio su k iterazioni resta quello di dino_power (lazy regularization).
            applica_dino = 'dino_power' in potenze and self.iter % dino_ogni == 0
            if applica_dino or 'ffl_power' in potenze:
                m_src = p['target_srcm_blur']; m_dst = saehd_mask_blur(p['target_dstm'], risoluzione)
                ss, ts = p['pred_src_src'] * m_src, p['target_src'] * m_src
                dd, td = p['pred_dst_dst'] * m_dst, p['target_dst'] * m_dst
            if applica_dino:
                t['dino'] = dino_ogni * (termini.dino_perceptual(vit_, ss, ts) + termini.dino_perceptual(vit_, dd, td))
            if 'ffl_power' in potenze:
                t['ffl'] = termini.focal_frequency(ss, ts) + termini.focal_frequency(dd, td)
            self._termini_h1 = {k: v.detach() for k, v in t.items()}
            return sum(potenze[opzione[k]] * v for k, v in t.items())
        return loss_extra

    #override
    def _motivo_per_non_catturare(self):
        if self._potenze_accese():
            return "H1: i supervisori congelati non stanno nel grafo catturato"
        return super()._motivo_per_non_catturare()

    #override
    def _esegui_il_passo(self, passo, tensori):
        with torch.autocast("cuda", dtype=self.DTYPE_AUTOCAST):
            return passo(self.nets, self.src_dst_opt, self.src_dst_trainable_weights,
                         tensori, self._train_cfg, 1, loss_extra=self._loss_extra)

    #override
    def onTrainOneIter(self):
        out = super().onTrainOneIter()
        it = self.get_iter()
        if self._termini_h1 and (it % 100 == 0 or (it <= 100 and it % 10 == 0)):
            potenze = self._potenze_accese()
            pezzi = " ".join(f"{k}={float(v.float().mean()):.4f}(x{potenze[self.POTENZE[k]]:g})"
                             for k, v in self._termini_h1.items())
            io.log_info(f"[H1 it {it}] {pezzi}")
        return out
