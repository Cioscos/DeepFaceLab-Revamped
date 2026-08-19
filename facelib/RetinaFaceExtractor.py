import operator
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn
from mainscripts import MotoriCatalog


def _blocchi_resnet50():
    """(canali interni, ripetizioni, stride) dei quattro stadi. Sta fuori
    dalle classi perche' la usano sia la rete sia il convertitore dei pesi,
    che deve camminare gli stessi nomi nello stesso ordine."""
    return ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2))


def _classi_rete():
    class Bottleneck(nn.ModelBase):
        """Il blocco a tre convoluzioni di ResNet-50. `expansion = 4` non e'
        un parametro: e' l'architettura, e i pesi originali la assumono."""

        def __init__(self, in_ch, planes, strides=1, downsample=False, **kwargs):
            self.in_ch, self.planes = in_ch, planes
            self.strides, self.usa_downsample = strides, downsample
            super().__init__(**kwargs)

        def on_build(self):
            out_ch = self.planes * 4
            self.conv1 = nn.Conv2D(self.in_ch, self.planes, kernel_size=1, strides=1,
                                   padding='VALID', use_bias=False)
            self.bn1 = nn.BatchNorm2D(self.planes)
            self.conv2 = nn.Conv2D(self.planes, self.planes, kernel_size=3,
                                   strides=self.strides, padding='SAME', use_bias=False)
            self.bn2 = nn.BatchNorm2D(self.planes)
            self.conv3 = nn.Conv2D(self.planes, out_ch, kernel_size=1, strides=1,
                                   padding='VALID', use_bias=False)
            self.bn3 = nn.BatchNorm2D(out_ch)
            if self.usa_downsample:
                self.downsample_conv = nn.Conv2D(self.in_ch, out_ch, kernel_size=1,
                                                 strides=self.strides, padding='VALID',
                                                 use_bias=False)
                self.downsample_bn = nn.BatchNorm2D(out_ch)

        def forward(self, x):
            identita = x
            y = F.relu(self.bn1(self.conv1(x)))
            y = F.relu(self.bn2(self.conv2(y)))
            y = self.bn3(self.conv3(y))
            if self.usa_downsample:
                identita = self.downsample_bn(self.downsample_conv(x))
            return F.relu(y + identita)

    class ResNet50(nn.ModelBase):
        def __init__(self, **kwargs):
            super().__init__(name='ResNet50', **kwargs)

        def on_build(self):
            self.conv1 = nn.Conv2D(3, 64, kernel_size=7, strides=2, padding=3,
                                   use_bias=False)
            self.bn1 = nn.BatchNorm2D(64)
            in_ch = 64
            for stadio, (planes, ripetizioni, strides) in enumerate(_blocchi_resnet50(), start=1):
                blocchi = []
                for i in range(ripetizioni):
                    primo = i == 0
                    blocchi.append(Bottleneck(
                        in_ch, planes,
                        strides=strides if primo else 1,
                        downsample=primo))
                    in_ch = planes * 4
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
            c2 = self._stadio(x, 2)
            c3 = self._stadio(c2, 3)
            c4 = self._stadio(c3, 4)
            return c2, c3, c4

    class ConvBN(nn.ModelBase):
        def __init__(self, in_ch, out_ch, strides=1, kernel_size=3, leaky=0.0,
                     con_relu=True, **kwargs):
            self.in_ch, self.out_ch = in_ch, out_ch
            self.strides, self.kernel_size = strides, kernel_size
            self.leaky, self.con_relu = leaky, con_relu
            super().__init__(**kwargs)

        def on_build(self):
            padding = 'SAME' if self.kernel_size == 3 else 'VALID'
            self.conv = nn.Conv2D(self.in_ch, self.out_ch, kernel_size=self.kernel_size,
                                  strides=self.strides, padding=padding, use_bias=False)
            self.bn = nn.BatchNorm2D(self.out_ch)

        def forward(self, x):
            x = self.bn(self.conv(x))
            return F.leaky_relu(x, self.leaky) if self.con_relu else x

    class SSH(nn.ModelBase):
        """Tre rami a campo recettivo crescente, concatenati: 3x3 sulla meta'
        dei canali, 5x5 e 7x7 (ottenuti impilando 3x3) su un quarto ciascuno."""

        def __init__(self, in_ch, out_ch, **kwargs):
            self.in_ch, self.out_ch = in_ch, out_ch
            super().__init__(**kwargs)

        def on_build(self):
            leaky = 0.1 if self.out_ch <= 64 else 0.0
            self.conv3X3 = ConvBN(self.in_ch, self.out_ch // 2, con_relu=False)
            self.conv5X5_1 = ConvBN(self.in_ch, self.out_ch // 4, leaky=leaky)
            self.conv5X5_2 = ConvBN(self.out_ch // 4, self.out_ch // 4, con_relu=False)
            self.conv7X7_2 = ConvBN(self.out_ch // 4, self.out_ch // 4, leaky=leaky)
            self.conv7x7_3 = ConvBN(self.out_ch // 4, self.out_ch // 4, con_relu=False)

        def forward(self, x):
            a = self.conv3X3(x)
            b1 = self.conv5X5_1(x)
            b = self.conv5X5_2(b1)
            c = self.conv7x7_3(self.conv7X7_2(b1))
            return F.relu(torch.cat([a, b, c], dim=1))

    class RetinaFace(nn.ModelBase):
        NUM_ANCHOR = 2

        def __init__(self, **kwargs):
            super().__init__(name='RetinaFace', **kwargs)

        def on_build(self):
            self.body = ResNet50()
            leaky = 0.0
            self.output1 = ConvBN(512, 256, kernel_size=1, leaky=leaky)
            self.output2 = ConvBN(1024, 256, kernel_size=1, leaky=leaky)
            self.output3 = ConvBN(2048, 256, kernel_size=1, leaky=leaky)
            self.merge1 = ConvBN(256, 256, leaky=leaky)
            self.merge2 = ConvBN(256, 256, leaky=leaky)
            for i in (1, 2, 3):
                setattr(self, f"ssh{i}", SSH(256, 256))
                setattr(self, f"class_head{i}",
                        nn.Conv2D(256, self.NUM_ANCHOR * 2, kernel_size=1,
                                  strides=1, padding='VALID'))
                setattr(self, f"bbox_head{i}",
                        nn.Conv2D(256, self.NUM_ANCHOR * 4, kernel_size=1,
                                  strides=1, padding='VALID'))

        def forward(self, x):
            c2, c3, c4 = self.body(x)
            p1, p2, p3 = self.output1(c2), self.output2(c3), self.output3(c4)
            # interpolate su size=p2.shape[2:], non scale_factor=2: con un lato
            # dispari il raddoppio non ricade sulla dimensione giusta e la
            # somma esplode con uno shape mismatch su frame veri.
            p2 = self.merge2(p2 + F.interpolate(p3, size=p2.shape[2:], mode='nearest'))
            p1 = self.merge1(p1 + F.interpolate(p2, size=p1.shape[2:], mode='nearest'))

            loc, conf = [], []
            for i, p in enumerate((p1, p2, p3), start=1):
                f = getattr(self, f"ssh{i}")(p)
                b = getattr(self, f"bbox_head{i}")(f)
                c = getattr(self, f"class_head{i}")(f)
                loc.append(b.permute(0, 2, 3, 1).contiguous().view(b.shape[0], -1, 4))
                conf.append(c.permute(0, 2, 3, 1).contiguous().view(c.shape[0], -1, 2))
            return torch.cat(loc, dim=1), torch.cat(conf, dim=1)

    return {"Bottleneck": Bottleneck, "ResNet50": ResNet50,
            "ConvBN": ConvBN, "SSH": SSH, "RetinaFace": RetinaFace}


def costruisci_rete_per_prova():
    return _classi_rete()["RetinaFace"]()


class RetinaFaceExtractor(object):
    """Il rilevatore MIT che recupera i volti di profilo che S3FD manca.

    `lato_rete` ha lo stesso ruolo che ha in S3FDExtractor -- il lato lungo a
    cui il frame arriva alla rete -- ma NON la stessa regola: qui e' un
    tetto puro (nessun dimezzamento sotto soglia), perche' e' cosi' che il
    modello e' stato addestrato e misurato. 960 e' il valore dell'autore, ed
    e' anche il valore con cui si sono misurati 103 volti utilizzabili sui
    311 fotogrammi difficili; la variante senza tetto ne fa 93 ed e' stata
    esclusa.
    """
    MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    SCARTO = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    VARIANZE = (0.1, 0.2)
    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    STEPS = (8, 16, 32)

    def __init__(self, place_model_on_cpu=False, lato_rete=960,
                 lato_min=MotoriCatalog.LATO_MIN_PREDEFINITO,
                 confidenza=0.7, nms=0.4):
        self.lato_rete = lato_rete
        self.lato_min = lato_min
        self.confidenza = confidenza
        self.nms = nms
        nn.initialize()

        model_path = Path(__file__).parent / "RetinaFaceR50.npy"
        if not model_path.exists():
            raise Exception("Unable to load RetinaFaceR50.npy")

        # Stessa scelta di S3FDExtractor: la rete gira per intero su
        # `device`, pesi inclusi -- niente split CPU/GPU alla TF.
        self.device = torch.device('cpu') if place_model_on_cpu else nn.device

        self.model = _classi_rete()["RetinaFace"]()
        self.model.build()
        self.model.to(self.device)
        self.model.load_weights(model_path)

    def _priorbox(self, h, w):
        """Gli anchor, nello stesso ordine in cui le teste concatenano i
        livelli: livello esterno, poi riga, poi colonna, poi `min_size`. Un
        ordine diverso produce box plausibili e sbagliati."""
        ancore = []
        for k, step in enumerate(self.STEPS):
            fh, fw = int(np.ceil(h / step)), int(np.ceil(w / step))
            for i in range(fh):
                for j in range(fw):
                    for min_size in self.MIN_SIZES[k]:
                        ancore.append((
                            (j + 0.5) * step / w,
                            (i + 0.5) * step / h,
                            min_size / w,
                            min_size / h))
        return torch.tensor(ancore, dtype=nn.floatx)

    def _decodifica(self, loc, prior):
        cxcy = prior[:, :2] + loc[:, :2] * self.VARIANZE[0] * prior[:, 2:]
        wh = prior[:, 2:] * torch.exp(loc[:, 2:] * self.VARIANZE[1])
        box = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=1)
        return box

    def _net_run(self, x):
        """NCHW numpy in (gia' normalizzato), (loc, conf) numpy out, asse di
        batch tenuto -- la rete li restituisce gia' come (N, priors, 4) e
        (N, priors, 2)."""
        t = torch.as_tensor(np.ascontiguousarray(x)).to(self.device, nn.floatx)
        with torch.no_grad():
            loc, conf = self.model(t)
        return loc.cpu().numpy(), conf.cpu().numpy()

    def _rettangoli_grezzi(self, img):
        """RGB in ingresso (extract ha gia' fatto BGR->RGB). Ridimensiona
        col tetto, passa alla rete, decodifica via priorbox, soglia,
        NMS -- e riporta i rettangoli nello spazio di coordinate di `img`.
        Separata da `extract` perche' e' la parte che i test sostituiscono
        per isolare la post-elaborazione (le tre cose di S3FD) dal modello.
        """
        h, w = img.shape[:2]
        d = max(w, h)
        # Tetto puro: sotto lato_rete l'immagine non viene toccata, sopra si
        # scala fino a farci stare esattamente (vedi il docstring di classe).
        scale_to = min(float(d), float(self.lato_rete)) if self.lato_rete is not None else float(d)
        input_scale = d / scale_to
        rw, rh = max(1, int(w / input_scale)), max(1, int(h / input_scale))
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)

        x = (resized.astype(np.float32) - self.MEDIA) / self.SCARTO
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        loc, conf = self._net_run(x)
        loc = torch.from_numpy(loc[0])
        # indice 1 = volto, indice 0 = sfondo -- la stessa convenzione dei
        # pesi originali, che il porting riproduce bit per bit (verificato
        # per confronto diretto coi tensori del riferimento originale).
        scores = torch.softmax(torch.from_numpy(conf[0]), dim=1)[:, 1]

        priors = self._priorbox(rh, rw)
        boxes = self._decodifica(loc, priors)

        tieni = scores >= self.confidenza
        boxes = boxes[tieni].numpy()
        scores = scores[tieni].numpy()
        if len(boxes) == 0:
            return []

        # cv2.dnn.NMSBoxes vuole (x, y, larghezza, altezza), non (x1,y1,x2,y2).
        bboxes_xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                       for (x1, y1, x2, y2) in boxes]
        indici = cv2.dnn.NMSBoxes(bboxes_xywh, scores.tolist(), 0.0, self.nms)
        indici = np.array(indici).reshape(-1)

        rettangoli = []
        for i in indici:
            x1, y1, x2, y2 = boxes[i]
            # de-normalizza (frazioni 0..1 del frame ridotto) e torna nello
            # spazio dell'immagine originale.
            rettangoli.append((x1 * rw * input_scale, y1 * rh * input_scale,
                                x2 * rw * input_scale, y2 * rh * input_scale))
        return rettangoli

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        if is_bgr:
            input_image = input_image[:, :, ::-1]
            is_bgr = False

        detected_faces = []
        for (l, t, r, b) in self._rettangoli_grezzi(input_image):
            bt = b - t
            if min(r - l, bt) < self.lato_min:  # filtering small faces by any side
                continue
            b += bt * 0.1  # enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append([int(x) for x in (l, t, r, b)])

        # sort by largest area first
        detected_faces = [[(l, t, r, b), (r - l) * (b - t)] for (l, t, r, b) in detected_faces]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True)
        detected_faces = [x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range(len(detected_faces) - 1, 0, -1):
                l1, t1, r1, b1 = detected_faces[i]
                l0, t0, r0, b0 = detected_faces[i - 1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx >= 0) and (dy >= 0):
                    detected_faces.pop(i)

        return detected_faces
