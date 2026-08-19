from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn

NUM_LMS = 68
NUM_NB = 10


def _blocchi_resnet18():
    """(canali interni, ripetizioni, stride) dei quattro stadi. Sta fuori
    dalle classi perche' la usano sia la rete sia il convertitore dei pesi,
    che deve camminare gli stessi nomi nello stesso ordine."""
    return ((64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2))


def _classi_rete():
    class BasicBlock(nn.ModelBase):
        """Il blocco a due convoluzioni di ResNet-18. Nessuna espansione:
        `out_ch == planes`, a differenza del Bottleneck di ResNet-50."""

        def __init__(self, in_ch, planes, strides=1, downsample=False, **kwargs):
            self.in_ch, self.planes = in_ch, planes
            self.strides, self.usa_downsample = strides, downsample
            super().__init__(**kwargs)

        def on_build(self):
            self.conv1 = nn.Conv2D(self.in_ch, self.planes, kernel_size=3,
                                   strides=self.strides, padding='SAME', use_bias=False)
            self.bn1 = nn.BatchNorm2D(self.planes)
            self.conv2 = nn.Conv2D(self.planes, self.planes, kernel_size=3,
                                   strides=1, padding='SAME', use_bias=False)
            self.bn2 = nn.BatchNorm2D(self.planes)
            if self.usa_downsample:
                self.downsample_conv = nn.Conv2D(self.in_ch, self.planes, kernel_size=1,
                                                 strides=self.strides, padding='VALID',
                                                 use_bias=False)
                self.downsample_bn = nn.BatchNorm2D(self.planes)

        def forward(self, x):
            identita = x
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            if self.usa_downsample:
                identita = self.downsample_bn(self.downsample_conv(x))
            return F.relu(y + identita)

    class ResNet18(nn.ModelBase):
        def __init__(self, **kwargs):
            super().__init__(name='ResNet18', **kwargs)

        def on_build(self):
            self.conv1 = nn.Conv2D(3, 64, kernel_size=7, strides=2, padding=3,
                                   use_bias=False)
            self.bn1 = nn.BatchNorm2D(64)
            in_ch = 64
            for stadio, (planes, ripetizioni, strides) in enumerate(_blocchi_resnet18(), start=1):
                blocchi = []
                for i in range(ripetizioni):
                    primo = i == 0
                    blocchi.append(BasicBlock(
                        in_ch, planes,
                        strides=strides if primo else 1,
                        downsample=primo and (strides != 1 or in_ch != planes)))
                    in_ch = planes
                # i blocchi vanno registrati come attributi, non tenuti in una
                # lista Python: named_parameters() non vedrebbe una lista, e
                # save_weights scriverebbe un file senza di loro.
                for i, b in enumerate(blocchi):
                    setattr(self, f"layer{stadio}_{i}", b)
                setattr(self, f"_conteggio_layer{stadio}", ripetizioni)

        def _stadio(self, x, stadio):
            for i in range(getattr(self, f"_conteggio_layer{stadio}")):
                x = getattr(self, f"layer{stadio}_{i}")(x)
            return x

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self._stadio(x, 1)
            x = self._stadio(x, 2)
            x = self._stadio(x, 3)
            x = self._stadio(x, 4)
            return x

    class PIPNet(nn.ModelBase):
        def __init__(self, **kwargs):
            super().__init__(name='PIPNet', **kwargs)

        def on_build(self):
            self.body = ResNet18()
            self.cls_layer = nn.Conv2D(512, NUM_LMS, kernel_size=1, strides=1, padding='VALID')
            self.x_layer = nn.Conv2D(512, NUM_LMS, kernel_size=1, strides=1, padding='VALID')
            self.y_layer = nn.Conv2D(512, NUM_LMS, kernel_size=1, strides=1, padding='VALID')
            self.nb_x_layer = nn.Conv2D(512, NUM_LMS * NUM_NB, kernel_size=1, strides=1, padding='VALID')
            self.nb_y_layer = nn.Conv2D(512, NUM_LMS * NUM_NB, kernel_size=1, strides=1, padding='VALID')
            # I 10 vicini di ogni landmark, calcolati una volta dal meanface e
            # portati dentro il .npy: un Parameter non addestrabile, non un
            # register_buffer, per lo stesso motivo di BatchNorm2D.running_mean
            # -- Saveable salva solo i Parameter (vedi il suo docstring), quindi
            # un buffer sparirebbe dal file in silenzio.
            self.indice_vicini = torch.nn.Parameter(
                torch.zeros((NUM_LMS, NUM_NB), dtype=nn.floatx), requires_grad=False)

        def forward(self, x):
            x = self.body(x)
            cls = self.cls_layer(x)
            off_x = self.x_layer(x)
            off_y = self.y_layer(x)
            nb_x = self.nb_x_layer(x)
            nb_y = self.nb_y_layer(x)
            return cls, off_x, off_y, nb_x, nb_y

    return {"BasicBlock": BasicBlock, "ResNet18": ResNet18, "PIPNet": PIPNet}


def costruisci_pipnet_per_prova():
    return _classi_rete()["PIPNet"]()


class PipNetExtractor(object):
    """L'allineatore -- ResNet-18 e cinque teste 1x1, porting di PIPNet-68
    (jhb86253817/PIPNet, licenza MIT, pesi in facelib/PIPNet68.npy).

    Nessuna seconda passata di rifinitura (quella che FANExtractor fa con
    `second_pass_extractor`): e' una scelta di progetto, non un buco -- il
    confronto misurato fra i motori mostra che PIPNet vince comunque su
    materiale difficile. Il terzo argomento di `extract` resta per firma
    identica a FANExtractor.extract, che `Extractor.py::landmarks_stage`
    passa sempre.
    """
    INPUT_SIZE = 256
    # Il rettangolo si allarga del 10% per lato prima del ritaglio (fattore
    # 1.2 sull'intero lato): la stessa convenzione del demo.py originale di
    # PIPNet, det_box_scale=1.2.
    SCALA_BOX = 1.2
    # Sotto questo lato il ritaglio non porta nessuna informazione utile alla
    # rete (upscaling estremo di pochi pixel): trattarlo come un fallimento
    # esplicito -- None, non un crash silenzioso ne' un landmark spazzatura.
    LATO_MIN_RITAGLIO = 8
    MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    SCARTO = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    def __init__(self, place_model_on_cpu=False):
        nn.initialize()

        model_path = Path(__file__).parent / "PIPNet68.npy"
        if not model_path.exists():
            raise Exception("Unable to load PIPNet68.npy")

        # Stessa scelta di S3FDExtractor/FANExtractor/RetinaFaceExtractor: la
        # rete gira per intero su `device`, pesi inclusi.
        self.device = torch.device('cpu') if place_model_on_cpu else nn.device

        self.model = _classi_rete()["PIPNet"]()
        self.model.build()
        self.model.to(self.device)
        self.model.load_weights(model_path)

        indice_vicini = self.model.indice_vicini.detach().cpu().numpy()
        self._idx1, self._idx2, self._max_len = self._indice_inverso(indice_vicini)

    @staticmethod
    def _indice_inverso(indice_vicini):
        """Inverte `indice_vicini` (NUM_LMS, NUM_NB): per ogni landmark j, chi
        lo predice (quale landmark i lo ha fra i propri NUM_NB piu' vicini) e
        a quale slot k. E' l'algoritmo dell'autore (get_meanface/gen_data di
        PIPNet), riprodotto fedelmente: le liste piu' corte della piu' lunga
        (max_len, il numero massimo di predittori su tutti i 68 landmark)
        vengono allungate **ripetendo se stesse**, non azzerate ne'
        troncate -- e' cosi' che il riferimento calcola la media finale
        (denominatore fisso 1+max_len per ogni landmark), e un padding
        diverso sposterebbe il risultato rispetto al riferimento."""
        num_lms, num_nb = indice_vicini.shape
        grezzo = [[] for _ in range(num_lms)]
        for i in range(num_lms):
            for k in range(num_nb):
                j = int(round(float(indice_vicini[i, k])))
                grezzo[j].append((i, k))
        max_len = max(len(g) for g in grezzo)
        idx1 = np.zeros((num_lms, max_len), dtype=np.int64)
        idx2 = np.zeros((num_lms, max_len), dtype=np.int64)
        for j in range(num_lms):
            g = grezzo[j]
            ripetuto = (g * (max_len // len(g) + 1))[:max_len]
            idx1[j] = [p[0] for p in ripetuto]
            idx2[j] = [p[1] for p in ripetuto]
        return idx1, idx2, max_len

    def _net_run(self, x):
        """NCHW numpy in (gia' normalizzato), le cinque uscite della rete
        come numpy, asse di batch tenuto."""
        t = torch.as_tensor(np.ascontiguousarray(x)).to(self.device, nn.floatx)
        with torch.no_grad():
            cls, off_x, off_y, nb_x, nb_y = self.model(t)
        return (cls.cpu().numpy(), off_x.cpu().numpy(), off_y.cpu().numpy(),
                nb_x.cpu().numpy(), nb_y.cpu().numpy())

    def _decodifica(self, cls, off_x, off_y, nb_x, nb_y):
        """forward_pip dell'autore: la cella a punteggio massimo per ogni
        landmark, l'offset dentro quella cella, poi la media con le
        predizioni che i dieci vicini fanno dello stesso punto.

        La media coi vicini non e' una rifinitura opzionale: e' il modo in
        cui PIPNet ottiene i suoi numeri, e toglierla peggiora i landmark
        senza che nessuna forma cambi -- cioe' in silenzio.

        cls/off_x/off_y: (NUM_LMS, gh, gw); nb_x/nb_y: (NUM_LMS*NUM_NB, gh, gw)
        -- il canale i*NUM_NB+k e' la predizione che il landmark i fa del suo
        k-esimo vicino, letta sulla STESSA cella dove i e' stato rilevato (la
        rete non conosce la cella del vicino in questo punto del calcolo).
        Ritorna (NUM_LMS, 2) in [0,1], frazione del ritaglio -- extract() la
        riporta nello spazio dell'immagine.
        """
        c, gh, gw = cls.shape
        cls_flat = cls.reshape(c, -1)
        max_ids = cls_flat.argmax(axis=1)
        riga = (max_ids // gw).astype(np.float32)
        colonna = (max_ids % gw).astype(np.float32)

        idx = np.arange(c)
        ox = off_x.reshape(c, -1)[idx, max_ids]
        oy = off_y.reshape(c, -1)[idx, max_ids]
        x = (colonna + ox) / gw
        y = (riga + oy) / gh

        max_ids_rip = np.repeat(max_ids, NUM_NB)
        idx_nb = np.arange(c * NUM_NB)
        nb_ox = nb_x.reshape(c * NUM_NB, -1)[idx_nb, max_ids_rip].reshape(c, NUM_NB)
        nb_oy = nb_y.reshape(c * NUM_NB, -1)[idx_nb, max_ids_rip].reshape(c, NUM_NB)
        colonna_rip = np.repeat(colonna, NUM_NB).reshape(c, NUM_NB)
        riga_rip = np.repeat(riga, NUM_NB).reshape(c, NUM_NB)
        nb_x_val = (colonna_rip + nb_ox) / gw
        nb_y_val = (riga_rip + nb_oy) / gh

        vicini_x = nb_x_val[self._idx1, self._idx2]  # (NUM_LMS, max_len)
        vicini_y = nb_y_val[self._idx1, self._idx2]
        divisore = 1 + self._max_len
        x_media = (x + vicini_x.sum(axis=1)) / divisore
        y_media = (y + vicini_y.sum(axis=1)) / divisore

        return np.stack([x_media, y_media], axis=1).astype(np.float32)

    def extract(self, input_image, rects, second_pass_extractor=None, is_bgr=True):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        h_img, w_img = input_image.shape[:2]

        landmarks = []
        for (left, top, right, bottom) in rects:
            try:
                w, h = right - left, bottom - top
                margine_x = w * (self.SCALA_BOX - 1) / 2
                margine_y = h * (self.SCALA_BOX - 1) / 2
                l, r = int(left - margine_x), int(right + margine_x)
                t, b = int(top - margine_y), int(bottom + margine_y)
                l, t = max(l, 0), max(t, 0)
                r, b = min(r, w_img - 1), min(b, h_img - 1)
                det_w, det_h = r - l, b - t
                if det_w < self.LATO_MIN_RITAGLIO or det_h < self.LATO_MIN_RITAGLIO:
                    raise ValueError("ritaglio troppo piccolo per PIPNet")

                crop = cv2.resize(input_image[t:b, l:r], (self.INPUT_SIZE, self.INPUT_SIZE),
                                   interpolation=cv2.INTER_LINEAR)

                x = (crop.astype(np.float32) - self.MEDIA) / self.SCARTO
                x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

                cls, off_x, off_y, nb_x, nb_y = self._net_run(x)
                frazioni = self._decodifica(cls[0], off_x[0], off_y[0], nb_x[0], nb_y[0])

                pts = np.empty_like(frazioni)
                pts[:, 0] = frazioni[:, 0] * det_w + l
                pts[:, 1] = frazioni[:, 1] * det_h + t
                landmarks.append(pts.astype(np.float32))
            except Exception:
                landmarks.append(None)

        return landmarks
