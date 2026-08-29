"""H1, «SAEHD sorvegliato»: SAEHDX piu' i supervisori congelati nella
loss -- identita' (AdaFace), IFSR e la repulsione dal dst (bleed) sullo swap,
DINOv2 e focal frequency sulle ricostruzioni. Stessa rete, stessi file di
pesi, stesso export di SAEHDX: chi spegne tutte le potenze ha esattamente
SAEHDX. Il disegno di H1 e le ragioni delle potenze stanno nella
documentazione del ciclo, sotto docs/."""
import numpy as np

from core.interact import interact as io
from models.Model_H1.supervisione import Supervisori
from models.Model_SAEHDX.Model import SAEHDXModel


class H1Model(Supervisori, SAEHDXModel):
    #override
    def on_initialize_options(self):
        super().on_initialize_options()

        default_id_power    = self.options['id_power']    = self.load_or_def_option('id_power', 0.0)
        default_ifsr_power  = self.options['ifsr_power']  = self.load_or_def_option('ifsr_power', 0.0)
        default_dino_power  = self.options['dino_power']  = self.load_or_def_option('dino_power', 0.0)
        default_dino_ogni   = self.options['dino_ogni']   = self.load_or_def_option('dino_ogni', 1)
        default_ffl_power   = self.options['ffl_power']   = self.load_or_def_option('ffl_power', 0.0)
        default_bleed_power = self.options['bleed_power'] = self.load_or_def_option('bleed_power', 0.0)
        default_bleed_campione = self.options['bleed_campione'] = self.load_or_def_option('bleed_campione', False)

        if self.is_first_run() or self.override_richiesto:     # pesi della loss: si cambiano fra una sessione e l'altra
            self.options['id_power']   = np.clip(io.input_number("Identity power", default_id_power, add_info="0.0 .. 10.0",
                help_message="Cosine loss between the AdaFace embedding of the swapped face and the mean embedding of the src faceset. 0 disables it."), 0.0, 10.0)
            self.options['ifsr_power'] = np.clip(io.input_number("IFSR power", default_ifsr_power, add_info="0.0 .. 10.0",
                help_message="L1 between AdaFace intermediate features of the swapped face and of the dst face: keeps pose, lighting and occlusions of dst. 0 disables it."), 0.0, 10.0)
            self.options['dino_power'] = np.clip(io.input_number("DINOv2 perceptual power", default_dino_power, add_info="0.0 .. 10.0",
                help_message="L1 between DINOv2-S tokens of the masked reconstructions and their targets. 0 disables it."), 0.0, 10.0)
            self.options['dino_ogni'] = int(np.clip(io.input_int("DINOv2 stride", default_dino_ogni, add_info="1 .. 100",
                help_message="Applies the DINOv2 term every N iterations instead of every one, scaled by N so the average gradient is unchanged (lazy regularization). 1 disables the stride."), 1, 100))
            self.options['ffl_power']  = np.clip(io.input_number("Focal frequency power", default_ffl_power, add_info="0.0 .. 10.0",
                help_message="Focal Frequency Loss on the masked reconstructions. 0 disables it."), 0.0, 10.0)
            self.options['bleed_power'] = np.clip(io.input_number("Bleed power", default_bleed_power, add_info="0.0 .. 10.0",
                help_message="Penalizes the swap when its AdaFace embedding drifts toward the mean embedding of the dst faceset beyond a fixed cosine margin. 0 disables it."), 0.0, 10.0)
            self.options['bleed_campione'] = io.input_bool("Bleed per sample", default_bleed_campione,
                help_message="Bleed repels the swap from its own sample's dst embedding instead of the dst faceset mean. Only effective when bleed power > 0.")

        self._spegni_le_leve_incompatibili()

    #override
    def on_initialize(self):
        super().on_initialize()
        self._loss_extra = None
        self._termini_h1 = {}
        potenze = self._potenze_accese()
        if not potenze or not self.is_training:
            return
        self._richiedi_cuda()
        self._carica_supervisori('id_power' in potenze or 'ifsr_power' in potenze or 'bleed_power' in potenze,
                                  'dino_power' in potenze)
        self._theta, residuo_medio, residuo_max, self._e_src, coseno_medio, n_volti = self._riferimento_src()
        io.log_info(f"H1 crop fisso: residuo medio per volto {residuo_medio:.2f} px, massimo {residuo_max:.2f} px")
        io.log_info(f"H1 e_src da {n_volti} volti, coseno medio dei volti al riferimento {coseno_medio:.3f}")
        if self._serve_e_dst_medio(potenze):
            *_, self._e_dst, coseno_dst, n_dst = self._riferimento_dst()
            io.log_info(f"H1 e_dst da {n_dst} volti, coseno medio dei volti al riferimento {coseno_dst:.3f}")
        self._loss_extra = self._costruisci_loss_extra()


Model = H1Model
