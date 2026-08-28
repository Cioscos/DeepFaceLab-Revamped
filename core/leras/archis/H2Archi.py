"""I tre blocchi di H2: il PIM (Partial Identity Modulation, CanonSwap) che
legge un vettore d'identita' dentro un residual block, il vettore stesso
(Identita) e il decoder di SAEHD `-t` con i due residui bassi avvolti nel
PIM. Il disegno, l'invariante dell'innesto e il perche' della fusione
additiva stanno nella documentazione del ciclo H2 sotto docs/."""
import torch
import torch.nn.functional as F

from core.leras import nn


def _zeri(t):
    torch.nn.init.zeros_(t)


def _piccoli(t):
    torch.nn.init.normal_(t, 0.0, 0.02)


class PIM(nn.ModelBase):
    """y = act(x + conv2(act(conv1(x))) + m * out1x1(modconv(h, s))).

    conv1/conv2 sono il ResidualBlock di SAEHD con gli stessi nomi (l'innesto
    li copia); il ramo modulato parte da zero (out1x1) cosi' all'inizio il
    blocco E' il residuo di SAEHD per qualunque vettore; la fusione e'
    additiva e non (1-m)*std + m*mod perche' con m ~ 0.5 quell'invariante
    non reggerebbe."""
    def on_build(self, ch, e_dim, kernel_size=3):
        self.conv1   = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME')
        self.conv2   = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME')
        self.affine  = nn.Dense(e_dim, ch, use_bias=False, kernel_initializer=_piccoli)
        self.modconv = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME', use_bias=False)
        self.out1x1  = nn.Conv2D(ch, ch, kernel_size=1, padding='SAME', kernel_initializer=_zeri)
        self.convm   = nn.Conv2D(ch, 1, kernel_size=kernel_size, padding='SAME')

    def _modulata(self, h, s):
        """Conv modulata con demodulazione (StyleGAN2) nella forma non fusa:
        si scala l'ingresso per canale, si convolve col peso condiviso e si
        divide per la norma del peso scalato -- una sola Conv nel grafo, e
        nessun `groups=N`. demod^2[n, out] = sum_in s[n, in]^2 * sum_k w[out, in, k]^2."""
        w2 = self.modconv.weight.pow(2).sum((2, 3))                       # (out, in)
        demod = torch.rsqrt(s.pow(2) @ w2.t() + 1e-8)                    # (N, out)
        return self.modconv(h * s[:, :, None, None]) * demod[:, :, None, None]

    def forward(self, x, e):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h_std = self.conv2(h)
        s = 1.0 + self.affine(e)
        h_mod = self.out1x1(self._modulata(h, s))
        m = torch.sigmoid(self.convm(h))
        return F.leaky_relu(x + h_std + m * h_mod, 0.2)


class Identita(nn.ModelBase):
    """I due vettori d'identita', un file solo (identita.npy). Normalizzati
    all'uso: la scala del Parameter non e' un grado di liberta'."""
    def on_build(self, e_dim, allenabile=True):
        self.allenabile = allenabile
        self.e_src = torch.nn.Parameter(torch.zeros(e_dim, dtype=nn.floatx), requires_grad=allenabile)
        self.e_dst = torch.nn.Parameter(torch.zeros(e_dim, dtype=nn.floatx), requires_grad=allenabile)

    #override
    def init_weights(self):
        super().init_weights()
        with torch.no_grad():
            for p in (self.e_src, self.e_dst):
                p.copy_(F.normalize(torch.randn_like(p), dim=0))

    def imposta(self, e_src, e_dst):
        with torch.no_grad():
            self.e_src.copy_(e_src.to(self.e_src)); self.e_dst.copy_(e_dst.to(self.e_dst))

    def interpola(self, m):
        # .to() e non torch.as_tensor() quando m e' gia' un tensore: as_tensor
        # su un tensore avverte ("copy construct from a tensor") e stacca il
        # nodo dal grafo sotto tracciamento.
        m = m.to(dtype=self.e_src.dtype, device=self.e_src.device) if torch.is_tensor(m) \
            else torch.as_tensor(m, dtype=self.e_src.dtype, device=self.e_src.device)
        m = m.reshape(())
        return F.normalize((1.0 - m) * self.e_dst + m * self.e_src, dim=0)

    def forward(self, quale, n):
        if quale not in ('src', 'dst'):
            raise ValueError(f"quale deve essere 'src' o 'dst', non {quale!r}")
        e = self.e_src if quale == 'src' else self.e_dst
        return F.normalize(e, dim=0)[None].expand(n, -1)


class DecoderH2(nn.ModelBase):
    """Il Decoder `-t` con uscita `d` di DeepFakeArchi, con res0/res1
    sostituiti da pim0/pim1 e gli stessi nomi per tutto il resto: le chiavi
    su disco del RTM si copiano tal quali (innesto). Con `maschera_tronco`
    un ponte 1x1 a zero (`ponte_m`) somma il tronco modulato al ramo
    maschera, cosi' la maschera puo' leggere il vettore d'identita' senza
    perdere l'invariante a ponte nullo (registro 3.75)."""
    def on_build(self, in_ch, d_ch, d_mask_ch, e_dim, maschera_tronco=False):
        archi = nn.DeepFakeArchi(64, opts='udt')       # la risoluzione non entra nei blocchi usati qui
        Upscale, ResidualBlock = archi.Upscale, archi.ResidualBlock
        self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
        self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
        self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
        self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
        self.pim0 = PIM(d_ch*8, e_dim, kernel_size=3)
        self.pim1 = PIM(d_ch*8, e_dim, kernel_size=3)
        self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
        self.res3 = ResidualBlock(d_ch*2, kernel_size=3)

        self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
        self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
        self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
        self.out_conv  = nn.Conv2D(d_ch*2, 3, kernel_size=1, padding='SAME')
        self.out_conv1 = nn.Conv2D(d_ch*2, 3, kernel_size=3, padding='SAME')
        self.out_conv2 = nn.Conv2D(d_ch*2, 3, kernel_size=3, padding='SAME')
        self.out_conv3 = nn.Conv2D(d_ch*2, 3, kernel_size=3, padding='SAME')
        self.out_convm = nn.Conv2D(d_mask_ch*1, 1, kernel_size=1, padding='SAME')

        # Il ponte (registro 3.75): la maschera legge il tronco modulato, e
        # quindi il vettore. Nasce a zero, cosi' all'innesto la maschera E'
        # quella del RTM; la chiave ponte_m/ non esiste nel sorgente e resta
        # all'inizializzazione.
        self.maschera_tronco = maschera_tronco
        if maschera_tronco:
            self.ponte_m = nn.Conv2D(d_ch*8, d_mask_ch*8, kernel_size=1, padding='SAME', kernel_initializer=_zeri)

    def forward(self, z, e):
        x = self.pim0(self.upscale0(z), e)
        x = self.pim1(self.upscale1(x), e)
        m = self.upscalem1(self.upscalem0(z))
        if self.maschera_tronco:
            m = m + self.ponte_m(x)
        x = self.res2(self.upscale2(x))
        x = self.res3(self.upscale3(x))
        x = torch.sigmoid(nn.depth_to_space(torch.cat((self.out_conv(x), self.out_conv1(x),
                                                       self.out_conv2(x), self.out_conv3(x)), nn.conv2d_ch_axis), 2))
        m = self.upscalem4(self.upscalem3(self.upscalem2(m)))
        m = torch.sigmoid(self.out_convm(m))
        return x, m


nn.PIM = PIM
nn.Identita = Identita
nn.DecoderH2 = DecoderH2
