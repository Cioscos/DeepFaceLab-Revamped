"""Il frontend cv2 della fusione: la finestra, i tasti e il pool.

Lo stato (i frame, il cursore, la cfg per frame, la propagazione, il .dat)
sta tutto in merger/SessioneMerge.py: qui restano la finestra, la tabella
dei tasti e i ganci del Subprocessor. `Frame` e `ProcessingFrame` sono
alias del nucleo, cosi' un .dat vecchio -- che le pickla col qualname
annidato di questo modulo -- si ricarica ancora.
"""
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from merger import MergeFaceAvatar, MergeMasked, MergerConfig, SessioneMerge
from merger.SessioneMerge import Frame, ProcessingFrame

from .MergerScreen import Screen, ScreenManager

MERGER_DEBUG = False

# I tasti di configurazione: chr -> funzione(cfg, shift). Trascrizione di
# gfx/help_merger_masked.jpg, riga per riga (scheda fusione.md).
TASTI_CONFIG = {
    '`' : lambda cfg,shift: cfg.set_mode(0),
    '1' : lambda cfg,shift: cfg.set_mode(1),
    '2' : lambda cfg,shift: cfg.set_mode(2),
    '3' : lambda cfg,shift: cfg.set_mode(3),
    '4' : lambda cfg,shift: cfg.set_mode(4),
    '5' : lambda cfg,shift: cfg.set_mode(5),
    '6' : lambda cfg,shift: cfg.set_mode(6),
    'q' : lambda cfg,shift: cfg.add_hist_match_threshold(1 if not shift else 5),
    'a' : lambda cfg,shift: cfg.add_hist_match_threshold(-1 if not shift else -5),
    'w' : lambda cfg,shift: cfg.add_erode_mask_modifier(1 if not shift else 5),
    's' : lambda cfg,shift: cfg.add_erode_mask_modifier(-1 if not shift else -5),
    'e' : lambda cfg,shift: cfg.add_blur_mask_modifier(1 if not shift else 5),
    'd' : lambda cfg,shift: cfg.add_blur_mask_modifier(-1 if not shift else -5),
    'r' : lambda cfg,shift: cfg.add_motion_blur_power(1 if not shift else 5),
    'f' : lambda cfg,shift: cfg.add_motion_blur_power(-1 if not shift else -5),
    't' : lambda cfg,shift: cfg.add_super_resolution_power(1 if not shift else 5),
    'g' : lambda cfg,shift: cfg.add_super_resolution_power(-1 if not shift else -5),
    'y' : lambda cfg,shift: cfg.add_blursharpen_amount(1 if not shift else 5),
    'h' : lambda cfg,shift: cfg.add_blursharpen_amount(-1 if not shift else -5),
    'u' : lambda cfg,shift: cfg.add_output_face_scale(1 if not shift else 5),
    'j' : lambda cfg,shift: cfg.add_output_face_scale(-1 if not shift else -5),
    'i' : lambda cfg,shift: cfg.add_image_denoise_power(1 if not shift else 5),
    'k' : lambda cfg,shift: cfg.add_image_denoise_power(-1 if not shift else -5),
    'o' : lambda cfg,shift: cfg.add_bicubic_degrade_power(1 if not shift else 5),
    'l' : lambda cfg,shift: cfg.add_bicubic_degrade_power(-1 if not shift else -5),
    'p' : lambda cfg,shift: cfg.add_color_degrade_power(1 if not shift else 5),
    ';' : lambda cfg,shift: cfg.add_color_degrade_power(-1),
    ':' : lambda cfg,shift: cfg.add_color_degrade_power(-5),
    'z' : lambda cfg,shift: cfg.toggle_masked_hist_match(),
    'x' : lambda cfg,shift: cfg.toggle_mask_mode(),
    'c' : lambda cfg,shift: cfg.toggle_color_transfer_mode(),
    'n' : lambda cfg,shift: cfg.toggle_sharpen_mode(),
}


def applica_tasto(sessione, key, chr_key, shift):
    """Un tasto della finestra -> il nucleo. Torna "esci", "tab", "scala-",
    "scala+", "scacchiera", "cfg" (la cfg e' cambiata), "nav" (il cursore
    puo' essersi mosso) o None. Nessun disegno qui: e' la parte provabile."""
    if key == 9:
        return "tab"
    if key == 27:
        sessione.in_chiusura = True
        return "esci"
    if chr_key in TASTI_CONFIG:
        # i tasti della tabella sono quelli della configurazione mascherata:
        # su una sessione face avatar -- o su un frame che porta una cfg di
        # altro tipo -- non hanno alcun effetto, come ieri
        cur = sessione.frame_corrente()
        if sessione.merger_config.type != MergerConfig.TYPE_MASKED or \
           (cur.cfg is not None and cur.cfg.type != MergerConfig.TYPE_MASKED):
            return None
        funzione = TASTI_CONFIG[chr_key]
        if not sessione.modifica_cfg(lambda cfg: funzione(cfg, shift)):
            return None                       # niente e' cambiato: niente da dire
        return "cfg"
    if chr_key == ',':
        sessione.elabora_restanti = False
        sessione.precedente(fino_al_primo=shift)
        return "nav"
    if chr_key == 'm':
        sessione.elabora_restanti = False
        sessione.precedente(propaga=True, fino_al_primo=shift)
        return "nav"
    if chr_key == '.':
        if shift:
            sessione.commuta_batch()          # voce 1.20: interruttore vero
        else:
            sessione.elabora_restanti = False
            sessione.successivo()
        return "nav"
    if chr_key == '/':
        sessione.elabora_restanti = False
        sessione.successivo(propaga=True, fino_all_ultimo=shift)
        return "nav"
    if chr_key == '-':
        return "scala-"
    if chr_key == '=':
        return "scala+"
    if chr_key == 'v':
        return "scacchiera"
    return None


class InteractiveMergerSubprocessor(Subprocessor):
    Frame = Frame
    ProcessingFrame = ProcessingFrame

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info ('Running on %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.face_enhancer_func = client_dict['face_enhancer_func']
            self.xseg_256_extract_func = client_dict['xseg_256_extract_func']


            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            return None

        #override
        def process_data(self, pf): #pf=ProcessingFrame
            cfg = pf.cfg.copy()

            frame_info = pf.frame_info
            filepath = frame_info.filepath

            if len(frame_info.landmarks_list) == 0:
                
                if cfg.mode == 'raw-predict':        
                    h,w,c = self.predictor_input_shape
                    img_bgr = np.zeros( (h,w,3), dtype=np.uint8)
                    img_mask = np.zeros( (h,w,1), dtype=np.uint8)               
                else:                
                    self.log_info (f'no faces found for {filepath.name}, copying without faces')
                    img_bgr = cv2_imread(filepath)
                    imagelib.normalize_channels(img_bgr, 3)                    
                    h,w,c = img_bgr.shape
                    img_mask = np.zeros( (h,w,1), dtype=img_bgr.dtype)
                    
                self._scrivi (pf, img_bgr, img_mask)

                if pf.need_return_image:
                    pf.image = np.concatenate ([img_bgr, img_mask], axis=-1)

            else:
                if cfg.type == MergerConfig.TYPE_MASKED:
                    try:
                        final_img = MergeMasked (self.predictor_func, self.predictor_input_shape,
                                                 face_enhancer_func=self.face_enhancer_func,
                                                 xseg_256_extract_func=self.xseg_256_extract_func,
                                                 cfg=cfg,
                                                 frame_info=frame_info)
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception( f'Error while merging file [{filepath}]: {e_str}' )

                elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
                    final_img = MergeFaceAvatar (self.predictor_func, self.predictor_input_shape,
                                                   cfg, pf.prev_temporal_frame_infos,
                                                        pf.frame_info,
                                                        pf.next_temporal_frame_infos )

                self._scrivi (pf, final_img[...,0:3], final_img[...,3:4] )

                if pf.need_return_image:
                    pf.image = final_img

            return pf

        def _scrivi (self, pf, img_bgr, img_mask):
            """Un fotogramma che non arriva sul disco non e' fatto.

            Sollevare fa uccidere questo client e passare il pf a
            `on_data_return` -> `su_ritorno`, che lo rimette «da fare»:
            e' la stessa strada di un errore di compositing. Prima
            l'errore di `cv2_imwrite` era ingoiato, il frame tornava
            «fatto» e sul disco restava un PNG da 0 byte."""
            if not cv2_imwrite (pf.output_filepath, img_bgr) or \
               not cv2_imwrite (pf.output_mask_filepath, img_mask):
                raise Exception( f'Error while writing file [{pf.output_filepath}]: the disk refused the write (is it full?)' )

        #overridable
        def get_data_name (self, pf):
            #return string identificator of your data
            return pf.frame_info.filepath
    #override
    def __init__(self, is_interactive, merger_session_filepath, predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, merger_config, frames, frames_root_path, output_path, output_mask_path, model_iter, subprocess_count=4, pulisci_subito=True):
        super().__init__('Merger', InteractiveMergerSubprocessor.Cli, io_loop_sleep_time=0.001)

        self.is_interactive = is_interactive
        self.merger_session_filepath = Path(merger_session_filepath)
        self.merger_config = merger_config

        self.predictor_func = predictor_func
        self.predictor_input_shape = predictor_input_shape

        self.face_enhancer_func = face_enhancer_func
        self.xseg_256_extract_func = xseg_256_extract_func

        self.frames_root_path = frames_root_path
        self.output_path = output_path
        self.output_mask_path = output_mask_path
        self.model_iter = model_iter

        self.process_count = subprocess_count

        self.sessione = SessioneMerge.SessioneMerge(frames, merger_config, prefetch=subprocess_count)
        self.sessione.assegna_output(self.output_path, self.output_mask_path)

        ripresa = SessioneMerge.RIPRESA_NESSUNA
        if self.is_interactive and self.merger_session_filepath.exists():
            io.input_skip_pending()
            if io.input_bool ("Use saved session?", True):
                ripresa = self.sessione.carica_sessione(self.merger_session_filepath, self.model_iter)
                if ripresa in (SessioneMerge.RIPRESA_OK, SessioneMerge.RIPRESA_AZZERATA):
                    io.log_info ('Using saved session from ' + '/'.join (self.merger_session_filepath.parts[-2:]) )

        if pulisci_subito:
            self.pulisci_uscita(ripresa)
        self._avanzamento_visto = self.sessione.corrente()

    def pulisci_uscita(self, ripresa):
        """Sessione assente o non corrispondente: si riparte da zero, e i PNG
        gia' scritti se ne vanno (il nucleo non tocca il filesystem).

        Metodo e non righe del costruttore perche' i due frontend caricano il
        .dat in due momenti diversi: la finestra cv2 dentro il costruttore
        (`pulisci_subito`), il servizio per la GUI subito dopo, che percio'
        lo chiama da se'. Pulire prima di sapere l'esito della ripresa
        cancellerebbe proprio i frame da non rifare; non pulire affatto
        lascia in `merged` i fotogrammi della fusione precedente, e sono
        quelli che finiscono nel video di «merged to mp4»."""
        if ripresa in (SessioneMerge.RIPRESA_OK, SessioneMerge.RIPRESA_AZZERATA):
            return
        for filename in pathex.get_image_paths(self.output_path):
            Path(filename).unlink()

        for filename in pathex.get_image_paths(self.output_mask_path):
            Path(filename).unlink()

        self.sessione._riavvolgi_sui_mancanti()

    #override
    def process_info_generator(self):
        r = [0] if MERGER_DEBUG else range(self.process_count)

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_func': self.predictor_func,
                                      'predictor_input_shape' : self.predictor_input_shape,
                                      'face_enhancer_func': self.face_enhancer_func,
                                      'xseg_256_extract_func' : self.xseg_256_extract_func,
                                      'stdin_fd': sys.stdin.fileno() if MERGER_DEBUG else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        s = self.sessione
        io.progress_bar ("Merging", len(s.frames), initial=s.corrente() )

        s.elabora_restanti = not self.is_interactive
        s.in_chiusura = not self.is_interactive

        if self.is_interactive:
            help_images = {
                    MergerConfig.TYPE_MASKED :      cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_masked.jpg') ),
                    MergerConfig.TYPE_FACE_AVATAR : cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_face_avatar.jpg') ),
                }

            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.merger_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager( "Merger", [self.main_screen, self.help_screen], capture_keys=True )
            self.screen_manager.set_current (self.help_screen)
            self.screen_manager.show_current()

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.is_interactive:
            self.screen_manager.finalize()

            self.sessione.salva_sessione(self.merger_session_filepath, self.model_iter)

            io.log_info ("Session is saved to " + '/'.join (self.merger_session_filepath.parts[-2:]) )

    def _mostra_corrente(self):
        """Il frame corrente sullo schermo principale, letto dal PNG se il
        pool non ha lasciato l'immagine (ripresa da .dat)."""
        s = self.sessione
        cur = s.frame_corrente()

        screen_image = None if s.elabora_restanti else self.main_screen.get_image()
        self.main_screen.set_waiting_icon( s.elabora_restanti or s.in_chiusura )

        if not s.in_chiusura and not s.elabora_restanti:
            if cur.is_done:
                if not cur.is_shown:
                    if cur.image is None:
                        image      = cv2_imread (cur.output_filepath, verbose=False)
                        image_mask = cv2_imread (cur.output_mask_filepath, verbose=False)
                        if image is None or image_mask is None:
                            # illeggibile? si ricalcola
                            cur.is_done = False
                        else:
                            image = imagelib.normalize_channels(image, 3)
                            image_mask = imagelib.normalize_channels(image_mask, 1)
                            cur.image = np.concatenate([image, image_mask], -1)

                    if cur.is_done:
                        io.log_info (cur.cfg.to_string( cur.frame_info.filepath.name) )
                        cur.is_shown = True
                        screen_image = cur.image
            else:
                self.main_screen.set_waiting_icon(True)

        self.main_screen.set_image(screen_image)
        self.screen_manager.show_current()

    def _avanzamento(self):
        """Quanto segna la barra: i frame dietro al cursore piu' il corrente
        se e' fatto. Sul solo cursore l'ultimo frame non si conterebbe mai --
        non c'e' un frame dopo di lui -- e la barra chiuderebbe a N-1 su N."""
        s = self.sessione
        return s.corrente() + (1 if s.frame_corrente().is_done else 0)

    #override
    def on_tick(self):
        io.process_messages()
        s = self.sessione

        if self.is_interactive:
            self._mostra_corrente()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if self.screen_manager.get_current() is self.main_screen or key in (9, 27):
                esito = applica_tasto(s, key, chr_key, shift_pressed)

                if esito == "tab":
                    self.screen_manager.switch_screens()
                elif esito == "cfg":
                    cur = s.frame_corrente()
                    io.log_info ( cur.cfg.to_string(cur.frame_info.filepath.name) )
                elif esito == "scala-":
                    self.screen_manager.get_current().diff_scale(-0.1)
                elif esito == "scala+":
                    self.screen_manager.get_current().diff_scale(0.1)
                elif esito == "scacchiera":
                    self.screen_manager.get_current().toggle_show_checker_board()

        s.avanza_batch()

        avanzamento = self._avanzamento()
        delta = avanzamento - self._avanzamento_visto
        if delta:
            io.progress_bar_inc(delta)
            self._avanzamento_visto = avanzamento

        if self.is_interactive:
            return s.in_chiusura
        return s.elabora_restanti == False and all(f.is_done for f in s.frames)

    #override
    def on_data_return (self, host_dict, pf):
        self.sessione.su_ritorno(pf.idx)

    #override
    def on_result (self, host_dict, pf_sent, pf_result):
        self.sessione.su_risultato(pf_result.idx, pf_result.cfg, pf_result.image)

    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.sessione.in_chiusura:
            return None
        return self.sessione.da_elaborare()

    #override
    def get_result(self):
        return 0
