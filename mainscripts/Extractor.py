import traceback
import multiprocessing
import operator
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from mainscripts import ExtractorLib, ExtractReport, MotoriCatalog
from core import imagelib
from facelib import FaceType, LandmarksProcessor, motori
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from core import pathex
from core.cv2ex import *
from DFLIMG import *

DEBUG = False

class ExtractSubprocessor(Subprocessor):
    class Data(object):
        def __init__(self, filepath=None, rects=None, landmarks = None, landmarks_accurate=True, manual=False, force_output_path=None, final_output_files = None, luminanza=0.0):
            self.filepath = filepath
            self.rects = rects or []
            self.rects_rotation = 0
            self.landmarks_accurate = landmarks_accurate
            self.manual = manual
            self.landmarks = landmarks or []
            self.force_output_path = force_output_path
            self.final_output_files = final_output_files or []
            self.faces_detected = 0
            self.luminanza = luminanza
            #Gli indici dei rilevamenti che final_stage ha davvero scritto su
            #disco. None finche' quello stadio non e' passato di qui: e' la
            #differenza fra "non ne so niente" e "nessuno". Serve al rapporto
            #per frame, che deve contare i volti SCRITTI e non i rilevamenti
            #(ExtractorLib.voce_da_data) -- salva_volto ne scarta una parte, e
            #`faces_detected`, il numero che la riga di comando stampa, conta
            #gia' quelli scritti.
            self.indici_salvati = None

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.type                 = client_dict['type']
            self.image_size           = client_dict['image_size']
            self.jpeg_quality         = client_dict['jpeg_quality']
            self.face_type            = client_dict['face_type']
            self.max_faces_from_image = client_dict['max_faces_from_image']
            self.device_idx           = client_dict['device_idx']
            self.cpu_only             = client_dict['device_type'] == 'CPU'
            self.final_output_path    = client_dict['final_output_path']
            self.output_debug_path    = client_dict['output_debug_path']

            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None and DEBUG:
                sys.stdin = os.fdopen(stdin_fd)

            if self.cpu_only:
                device_config = nn.DeviceConfig.CPU()
                place_model_on_cpu = True
            else:
                device_config = nn.DeviceConfig.GPUIndexes ([self.device_idx])
                place_model_on_cpu = device_config.devices[0].total_mem_gb < 4

            if self.type == 'all' or 'rects' in self.type or 'landmarks' in self.type:
                nn.initialize (device_config)

            self.log_info (f"Running on {client_dict['device_name'] }")

            if self.type == 'all' or self.type == 'rects-s3fd' or 'landmarks' in self.type:
                self.rects_extractor = motori.costruisci_rilevatore(
                    client_dict['rilevatore'], place_model_on_cpu=place_model_on_cpu)
                # Parametro della corsa, non del motore: applicarlo qui evita
                # una voce di catalogo per ogni valore possibile.
                if client_dict.get('min_face_size') is not None:
                    self.rects_extractor.lato_min = client_dict['min_face_size']

            if self.type == 'all' or 'landmarks' in self.type:
                # per il face type 'head', costruisci_allineatore alza da
                # solo a landmark 3D: e' il pavimento storico, non una scelta
                # qui.
                self.landmarks_extractor = motori.costruisci_allineatore(
                    client_dict['allineatore'], self.face_type,
                    place_model_on_cpu=place_model_on_cpu)

            self.cached_image = (None, None)

        #override
        def process_data(self, data):
            if 'landmarks' in self.type and len(data.rects) == 0:
                return data

            filepath = data.filepath
            cached_filepath, image = self.cached_image
            if cached_filepath != filepath:
                image = cv2_imread( filepath )
                if image is None:
                    self.log_err (f'Failed to open {filepath}, reason: cv2_imread() fail.')
                    return data
                image = imagelib.normalize_channels(image, 3)
                image = imagelib.cut_odd_image(image)
                self.cached_image = ( filepath, image )

            h, w, c = image.shape

            # Punto di strozzatura: l'immagine e' gia' decodificata e non lo
            # sara' mai piu' senza pagare una seconda decodifica dell'intero
            # video. Un float, quindi picklabile attraverso lo spawn.
            data.luminanza = float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

            if 'rects' in self.type or self.type == 'all':
                data = ExtractSubprocessor.Cli.rects_stage (data=data,
                                                            image=image,
                                                            max_faces_from_image=self.max_faces_from_image,
                                                            rects_extractor=self.rects_extractor,
                                                            )

            if 'landmarks' in self.type or self.type == 'all':
                data = ExtractSubprocessor.Cli.landmarks_stage (data=data,
                                                                image=image,
                                                                landmarks_extractor=self.landmarks_extractor,
                                                                rects_extractor=self.rects_extractor,
                                                                )

            if self.type == 'final' or self.type == 'all':
                data = ExtractSubprocessor.Cli.final_stage(data=data,
                                                           image=image,
                                                           face_type=self.face_type,
                                                           image_size=self.image_size,
                                                           jpeg_quality=self.jpeg_quality,
                                                           output_debug_path=self.output_debug_path,
                                                           final_output_path=self.final_output_path,
                                                           )
            return data

        @staticmethod
        def rects_stage(data,
                        image,
                        max_faces_from_image,
                        rects_extractor,
                        ):
            h,w,c = image.shape
            if min(h,w) < 128:
                # Image is too small
                data.rects = []
            else:
                for rot in ([0, 90, 270, 180]):
                    if rot == 0:
                        rotated_image = image
                    elif rot == 90:
                        rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
                    elif rot == 180:
                        rotated_image = image[::-1,::-1,:]
                    elif rot == 270:
                        rotated_image = image.swapaxes( 0,1 )[::-1,:,:]
                    rects = data.rects = rects_extractor.extract (rotated_image, is_bgr=True)
                    if len(rects) != 0:
                        data.rects_rotation = rot
                        break
                if max_faces_from_image is not None and \
                   max_faces_from_image > 0 and \
                   len(data.rects) > 0:
                    data.rects = data.rects[0:max_faces_from_image]
            return data


        @staticmethod
        def landmarks_stage(data,
                            image,
                            landmarks_extractor,
                            rects_extractor,
                            ):
            h, w, ch = image.shape

            if data.rects_rotation == 0:
                rotated_image = image
            elif data.rects_rotation == 90:
                rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
            elif data.rects_rotation == 180:
                rotated_image = image[::-1,::-1,:]
            elif data.rects_rotation == 270:
                rotated_image = image.swapaxes( 0,1 )[::-1,:,:]

            data.landmarks = landmarks_extractor.extract (rotated_image, data.rects, rects_extractor if (data.landmarks_accurate) else None, is_bgr=True)
            if data.rects_rotation != 0:
                for i, (rect, lmrks) in enumerate(zip(data.rects, data.landmarks)):
                    new_rect, new_lmrks = rect, lmrks
                    (l,t,r,b) = rect
                    if data.rects_rotation == 90:
                        new_rect = ( t, h-l, b, h-r)
                        if lmrks is not None:
                            new_lmrks = lmrks[:,::-1].copy()
                            new_lmrks[:,1] = h - new_lmrks[:,1]
                    elif data.rects_rotation == 180:
                        if lmrks is not None:
                            new_rect = ( w-l, h-t, w-r, h-b)
                            new_lmrks = lmrks.copy()
                            new_lmrks[:,0] = w - new_lmrks[:,0]
                            new_lmrks[:,1] = h - new_lmrks[:,1]
                    elif data.rects_rotation == 270:
                        new_rect = ( w-b, l, w-t, r )
                        if lmrks is not None:
                            new_lmrks = lmrks[:,::-1].copy()
                            new_lmrks[:,0] = w - new_lmrks[:,0]
                    data.rects[i], data.landmarks[i] = new_rect, new_lmrks

            return data

        @staticmethod
        def final_stage(data,
                        image,
                        face_type,
                        image_size,
                        jpeg_quality,
                        output_debug_path=None,
                        final_output_path=None,
                        ):
            data.final_output_files = []
            data.indici_salvati = []
            filepath = data.filepath
            rects = data.rects
            landmarks = data.landmarks

            if output_debug_path is not None:
                debug_image = image.copy()

            face_idx = 0
            for indice_rilevamento, (rect, image_landmarks) in enumerate(zip( rects, landmarks )):
                if image_landmarks is None:
                    continue

                output_path = final_output_path
                if data.force_output_path is not None:
                    output_path = data.force_output_path

                output_filepath = ExtractorLib.salva_volto(
                    immagine=image, rect=rect, image_landmarks=image_landmarks,
                    face_type=face_type, image_size=image_size,
                    jpeg_quality=jpeg_quality,
                    output_filepath=output_path / f"{filepath.stem}_{face_idx}.jpg",
                    source_filename=filepath.name, manuale=data.manual)
                if output_filepath is None:
                    continue

                if output_debug_path is not None and face_type != FaceType.MARK_ONLY:
                    LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, face_type, image_size, transparent_mask=True)

                data.final_output_files.append (output_filepath)
                data.indici_salvati.append (indice_rilevamento)
                face_idx += 1
            data.faces_detected = face_idx

            if output_debug_path is not None:
                cv2_imwrite( output_debug_path / (filepath.stem+'.jpg'), debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

            return data

        #overridable
        def get_data_name (self, data):
            #return string identificator of your data
            return data.filepath

    @staticmethod
    def get_devices_for_config (type, device_config):
        devices = device_config.devices
        cpu_only = len(devices) == 0

        if 'rects'     in type or \
           'landmarks' in type or \
           'all'       in type:

            if not cpu_only:
                if type == 'landmarks-manual':
                    devices = [devices.get_best_device()]

                result = []

                for device in devices:
                    count = 1

                    if count == 1:
                        result += [ (device.index, 'GPU', device.name, device.total_mem_gb) ]
                    else:
                        for i in range(count):
                            result += [ (device.index, 'GPU', f"{device.name} #{i}", device.total_mem_gb) ]

                return result
            else:
                if type == 'landmarks-manual':
                    return [ (0, 'CPU', 'CPU', 0 ) ]
                else:
                    return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range( min(8, multiprocessing.cpu_count() // 2) ) ]

        elif type == 'final':
            return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in (range(min(8, multiprocessing.cpu_count())) if not DEBUG else [0]) ]

    def __init__(self, input_data, type, image_size=None, jpeg_quality=None, face_type=None, output_debug_path=None, manual_window_size=0, max_faces_from_image=0, final_output_path=None, rilevatore=None, allineatore=None, min_face_size=None, device_config=None):
        if type == 'landmarks-manual':
            for x in input_data:
                x.manual = True

        self.input_data = input_data

        self.type = type
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality
        self.face_type = face_type
        self.output_debug_path = output_debug_path
        self.final_output_path = final_output_path
        self.manual_window_size = manual_window_size
        self.max_faces_from_image = max_faces_from_image
        self.rilevatore    = rilevatore or MotoriCatalog.DEFAULT_RILEVATORE
        self.allineatore   = allineatore or MotoriCatalog.DEFAULT_ALLINEATORE
        self.min_face_size = min_face_size
        self.result = []

        self.devices = ExtractSubprocessor.get_devices_for_config(self.type, device_config)

        super().__init__('Extractor', ExtractSubprocessor.Cli,
                             999999 if type == 'landmarks-manual' or DEBUG else 120)

    #override
    def on_clients_initialized(self):
        if self.type == 'landmarks-manual':
            self.wnd_name = 'Manual pass'
            io.named_window(self.wnd_name)
            io.capture_mouse(self.wnd_name)
            io.capture_keys(self.wnd_name)

            self.cache_original_image = (None, None)
            self.cache_image = (None, None)
            self.cache_text_lines_img = (None, None)
            self.hide_help = False
            self.landmarks_accurate = True
            self.force_landmarks = False

            self.landmarks = None
            self.x = 0
            self.y = 0
            self.rect_size = 100
            self.rect_locked = False
            self.extract_needed = True

            self.image = None
            self.image_filepath = None

        io.progress_bar (None, len (self.input_data))

    #override
    def on_clients_finalized(self):
        if self.type == 'landmarks-manual':
            io.destroy_all_windows()

        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type,
                     'image_size': self.image_size,
                     'jpeg_quality' : self.jpeg_quality,
                     'face_type': self.face_type,
                     'max_faces_from_image':self.max_faces_from_image,
                     'output_debug_path': self.output_debug_path,
                     'final_output_path': self.final_output_path,
                     'rilevatore': self.rilevatore,
                     'allineatore': self.allineatore,
                     'min_face_size': self.min_face_size,
                     'stdin_fd': sys.stdin.fileno() }


        for (device_idx, device_type, device_name, device_total_vram_gb) in self.devices:
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):
        if self.type == 'landmarks-manual':
            need_remark_face = False
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filepath, data_rects, data_landmarks = data.filepath, data.rects, data.landmarks
                is_frame_done = False

                if self.image_filepath != filepath:
                    self.image_filepath = filepath
                    if self.cache_original_image[0] == filepath:
                        self.original_image = self.cache_original_image[1]
                    else:
                        self.original_image = imagelib.normalize_channels( cv2_imread( filepath ), 3 )

                        self.cache_original_image = (filepath, self.original_image )

                    (h,w,c) = self.original_image.shape
                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / ( h * (16.0/9.0) )

                    if self.cache_image[0] == (h,w,c) + (self.view_scale,filepath):
                        self.image = self.cache_image[1]
                    else:
                        self.image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)
                        self.cache_image = ( (h,w,c) + (self.view_scale,filepath), self.image )

                    (h,w,c) = self.image.shape

                    sh = (0,0, w, min(100, h) )
                    if self.cache_text_lines_img[0] == sh:
                        self.text_lines_img = self.cache_text_lines_img[1]
                    else:
                        self.text_lines_img = (imagelib.get_draw_text_lines ( self.image, sh,
                                                        [   '[L Mouse click] - lock/unlock selection. [Mouse wheel] - change rect',
                                                            '[R Mouse Click] - manual face rectangle',
                                                            '[Enter] / [Space] - confirm / skip frame',
                                                            '[,] [.]- prev frame, next frame. [Q] - skip remaining frames',
                                                            '[a] - accuracy on/off (more fps)',
                                                            '[h] - hide this help'
                                                        ], (1, 1, 1) )*255).astype(np.uint8)

                        self.cache_text_lines_img = (sh, self.text_lines_img)

                if need_remark_face: # need remark image from input data that already has a marked face?
                    need_remark_face = False
                    if len(data_rects) != 0: # If there was already a face then lock the rectangle to it until the mouse is clicked
                        self.rect = data_rects.pop()
                        self.landmarks = data_landmarks.pop()
                        data_rects.clear()
                        data_landmarks.clear()

                        self.rect_locked = True
                        self.rect_size = ( self.rect[2] - self.rect[0] ) / 2
                        self.x = ( self.rect[0] + self.rect[2] ) / 2
                        self.y = ( self.rect[1] + self.rect[3] ) / 2
                        self.redraw()

                if len(data_rects) == 0:
                    (h,w,c) = self.image.shape
                    while True:
                        io.process_messages(0.0001)

                        if not self.force_landmarks:
                            new_x = self.x
                            new_y = self.y

                        new_rect_size = self.rect_size

                        mouse_events = io.get_mouse_events(self.wnd_name)
                        for ev in mouse_events:
                            (x, y, ev, flags) = ev
                            if ev == io.EVENT_MOUSEWHEEL and not self.rect_locked:
                                mod = 1 if flags > 0 else -1
                                diff = 1 if new_rect_size <= 40 else np.clip(new_rect_size / 10, 1, 10)
                                new_rect_size = max (5, new_rect_size + diff*mod)
                            elif ev == io.EVENT_LBUTTONDOWN:
                                if self.force_landmarks:
                                    self.x = new_x
                                    self.y = new_y
                                    self.force_landmarks = False
                                    self.rect_locked = True
                                    self.redraw()
                                else:
                                    self.rect_locked = not self.rect_locked
                                    self.extract_needed = True
                            elif ev == io.EVENT_RBUTTONDOWN:
                                self.force_landmarks = not self.force_landmarks
                                if self.force_landmarks:
                                    self.rect_locked = False
                            elif not self.rect_locked:
                                new_x = np.clip (x, 0, w-1) / self.view_scale
                                new_y = np.clip (y, 0, h-1) / self.view_scale

                        key_events = io.get_key_events(self.wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

                        if key == ord('\r') or key == ord('\n'):
                            #confirm frame
                            is_frame_done = True
                            data_rects.append (self.rect)
                            data_landmarks.append (self.landmarks)
                            break
                        elif key == ord(' '):
                            #confirm skip frame
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            #go prev frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)


                            self.input_data.insert(0, self.result.pop() )
                            io.progress_bar_inc(-1)
                            need_remark_face = True

                            break
                        elif key == ord('.'):
                            #go next frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            need_remark_face = True
                            is_frame_done = True
                            break
                        elif key == ord('q'):
                            #skip remaining

                            if self.rect_locked:
                                self.rect_locked = False
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            while len(self.input_data) > 0:
                                self.result.append( self.input_data.pop(0) )
                                io.progress_bar_inc(1)

                            break

                        elif key == ord('h'):
                            self.hide_help = not self.hide_help
                            break
                        elif key == ord('a'):
                            self.landmarks_accurate = not self.landmarks_accurate
                            break

                        if self.force_landmarks:
                            self.rect, lmrks = ExtractorLib.landmarks_da_vettore(
                                (self.x, self.y), (new_x, new_y))
                            self.rect_size = npla.norm(
                                np.float32([new_x, new_y]) - np.float32([self.x, self.y]))
                            if lmrks is not None:
                                self.landmarks = lmrks

                            self.redraw()

                        elif self.x != new_x or \
                           self.y != new_y or \
                           self.rect_size != new_rect_size or \
                           self.extract_needed:
                            self.x = new_x
                            self.y = new_y
                            self.rect_size = new_rect_size
                            self.rect = ( int(self.x-self.rect_size),
                                          int(self.y-self.rect_size),
                                          int(self.x+self.rect_size),
                                          int(self.y+self.rect_size) )

                            return ExtractSubprocessor.Data (filepath, rects=[self.rect], landmarks_accurate=self.landmarks_accurate)

                else:
                    is_frame_done = True

                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    io.progress_bar_inc(1)
                    self.extract_needed = True
                    self.rect_locked = False
        else:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        if not self.type != 'landmarks-manual':
            self.input_data.insert(0, data)

    def redraw(self):
        (h,w,c) = self.image.shape

        if not self.hide_help:
            image = cv2.addWeighted (self.image,1.0,self.text_lines_img,1.0,0)
        else:
            image = self.image.copy()

        # np.int e' stato rimosso in numpy 2; era solo un alias del builtin,
        # quindi il comportamento e' identico. Queste due righe stanno sul
        # manual fix, che `extract` propone di default quando il detector
        # manca facce: il primo fotogramma disegnato sollevava AttributeError.
        view_rect = (np.array(self.rect) * self.view_scale).astype(int).tolist()
        view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(int).tolist()

        if self.rect_size <= 40:
            scaled_rect_size = h // 3 if w > h else w // 3

            p1 = (self.x - self.rect_size, self.y - self.rect_size)
            p2 = (self.x + self.rect_size, self.y - self.rect_size)
            p3 = (self.x - self.rect_size, self.y + self.rect_size)

            wh = h if h < w else w
            np1 = (w / 2 - wh / 4, h / 2 - wh / 4)
            np2 = (w / 2 + wh / 4, h / 2 - wh / 4)
            np3 = (w / 2 - wh / 4, h / 2 + wh / 4)

            mat = cv2.getAffineTransform( np.float32([p1,p2,p3])*self.view_scale, np.float32([np1,np2,np3]) )
            image = cv2.warpAffine(image, mat,(w,h) )
            view_landmarks = LandmarksProcessor.transform_points (view_landmarks, mat)

        landmarks_color = (255,255,0) if self.rect_locked else (0,255,0)
        LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.face_type, self.image_size, landmarks_color=landmarks_color)
        self.extract_needed = False

        io.show_image (self.wnd_name, image)


    #override
    def on_result (self, host_dict, data, result):
        if self.type == 'landmarks-manual':
            filepath, landmarks = result.filepath, result.landmarks

            if len(landmarks) != 0 and landmarks[0] is not None:
                self.landmarks = landmarks[0]

            self.redraw()
        else:
            self.result.append ( result )
            io.progress_bar_inc(1)



    #override
    def get_result(self):
        return self.result


class DeletedFilesSearcherSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.debug_paths_stems = client_dict['debug_paths_stems']
            return None

        #override
        def process_data(self, data):
            input_path_stem = Path(data[0]).stem
            return any ( [ input_path_stem == d_stem for d_stem in self.debug_paths_stems] )

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_paths, debug_paths ):
        self.input_paths = input_paths
        self.debug_paths_stems = [ Path(d).stem for d in debug_paths]
        self.result = []
        super().__init__('DeletedFilesSearcherSubprocessor', DeletedFilesSearcherSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'debug_paths_stems' : self.debug_paths_stems}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Searching deleted files", len (self.input_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.input_paths) > 0:
            return [self.input_paths.pop(0)]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_paths.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result == False:
            self.result.append( data[0] )
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def _scrivi_rapporto(report_dir, data, motore=None):
    """Una voce per ogni Data che e' passato dallo stadio 'final': con
    report_dir=None non si scrive nulla, ed e' cio' che tiene identico il
    comportamento da riga di comando.

    `motore` e' la coppia rilevatore+allineatore che ha prodotto queste
    voci ("manual" per i due rami tracciati a mano); None -- il ripiego --
    vuol dire sconosciuto."""
    if report_dir is None:
        return
    with ExtractReport.Scrittore(report_dir) as scrittore:
        for d in data:
            scrittore.scrivi(ExtractorLib.voce_da_data(
                d, getattr(d, 'luminanza', 0.0), motore=motore))


def main(detector=None,
         landmarker=None,
         input_path=None,
         output_path=None,
         output_debug=None,
         manual_fix=False,
         manual_output_debug_fix=False,
         manual_window_size=1368,
         face_type='full_face',
         max_faces_from_image=None,
         image_size=None,
         jpeg_quality=None,
         cpu_only = False,
         force_gpu_idxs = None,
         report_dir=None,
         min_face_size=None,
         ):

    if not input_path.exists():
        io.log_err ('Input directory not found. Please ensure it exists.')
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if face_type is not None:
        face_type = FaceType.fromString(face_type)

    if face_type is None:
        if manual_output_debug_fix:
            files = pathex.get_image_paths(output_path)
            if len(files) != 0:
                dflimg = DFLIMG.load(Path(files[0]))
                if dflimg is not None and dflimg.has_data():
                     face_type = FaceType.fromString ( dflimg.get_face_type() )

    input_image_paths = pathex.get_image_unique_filestem_paths(input_path, verbose_print_func=io.log_info)
    output_images_paths = pathex.get_image_paths(output_path)
    output_debug_path = output_path.parent / (output_path.name + '_debug')

    continue_extraction = False
    if not manual_output_debug_fix and len(output_images_paths) > 0:
        if len(output_images_paths) > 128:
            continue_extraction = io.input_bool ("Continue extraction?", True, help_message="Extraction can be continued, but you must specify the same options again.")

        if len(output_images_paths) > 128 and continue_extraction:
            try:
                input_image_paths = input_image_paths[ [ Path(x).stem for x in input_image_paths ].index ( Path(output_images_paths[-128]).stem.split('_')[0] ) : ]
            except:
                io.log_err("Error in fetching the last index. Extraction cannot be continued.")
                return
        elif input_path != output_path:
                io.input(f"\n WARNING !!! \n {output_path} contains files! \n They will be deleted. \n Press enter to continue.\n")
                for filename in output_images_paths:
                    Path(filename).unlink()

    device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(choose_only_one=detector=='manual', suggest_all_gpu=True) ) \
                    if not cpu_only else nn.DeviceConfig.CPU()

    if face_type is None:
        face_type = io.input_str ("Face type", 'wf', ['f','wf','head'], help_message="Full face / whole face / head. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()
        face_type = {'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]

    if max_faces_from_image is None:
        max_faces_from_image = io.input_int(f"Max number of faces from image", 0, help_message="If you extract a src faceset that has frames with a large number of faces, it is advisable to set max faces to 3 to speed up extraction. 0 - unlimited")

    if image_size is None:
        image_size = io.input_int(f"Image size", 512 if face_type < FaceType.HEAD else 768, valid_range=[256,2048], help_message="Output image size. The higher image size, the worse face-enhancer works. Use higher than 512 value only if the source image is sharp enough and the face does not need to be enhanced.")

    if jpeg_quality is None:
        jpeg_quality = io.input_int(f"Jpeg quality", 90, valid_range=[1,100], help_message="Jpeg quality. The higher jpeg quality the larger the output file size.")

    if detector is None:
        scelte = list(MotoriCatalog.RILEVATORI)
        io.log_info ("Choose detector type.")
        for m in scelte:
            io.log_info (f"[{m.key}] {m.label}: {m.help}")
        io.log_info ("[manual] manual")
        # Testo non vuoto e distinto da quello dell'allineatore qui sotto:
        # con "" (il prompt di prima) le due chiavi collidevano in
        # prompt_key e nessuna risposta di DFL_ANSWERS_FILE poteva
        # raggiungere l'una o l'altra. input_str, non input_int: le chiavi
        # del catalogo GUI (gui/catalog/extraction.py) sono stringhe, non
        # indici.
        detector = io.input_str ("Detector", MotoriCatalog.DEFAULT_RILEVATORE,
                                  list(MotoriCatalog.CHIAVI_RILEVATORI) + ['manual'])

    if landmarker is None and detector != 'manual':
        io.log_info ("Choose landmark model.")
        for m in MotoriCatalog.ALLINEATORI:
            io.log_info (f"[{m.key}] {m.label}: {m.help}")
        landmarker = io.input_str ("Landmarker", MotoriCatalog.DEFAULT_ALLINEATORE,
                                    list(MotoriCatalog.CHIAVI_ALLINEATORI))

    if min_face_size is None and detector != 'manual':
        # Il default e' quello del rilevatore, letto dalla stessa costante
        # che S3FDExtractor usa nella propria firma: qui non se ne scrive un
        # secondo. Senza questo prompt il parametro esisteva solo per chi
        # digitava --min-face-size: il guadagno sui volti sotto i 40 px
        # deve arrivare anche a chi sceglie dall'interfaccia.
        min_face_size = io.input_int ("Minimum face size", MotoriCatalog.LATO_MIN_PREDEFINITO,
                                       valid_range=[1,1024],
                                       help_message="Detections whose shorter side is below this many pixels of the source frame are discarded. Lower it to keep small or distant faces the default throws away; it costs nothing in speed.")


    if output_debug is None:
        output_debug = io.input_bool (f"Write debug images to {output_debug_path.name}?", False)

    if output_debug:
        output_debug_path.mkdir(parents=True, exist_ok=True)

    if manual_output_debug_fix:
        if not output_debug_path.exists():
            io.log_err(f'{output_debug_path} not found. Re-extract faces with "Write debug images" option.')
            return
        else:
            detector = 'manual'
            io.log_info('Performing re-extract frames which were deleted from _debug directory.')

            input_image_paths = DeletedFilesSearcherSubprocessor (input_image_paths, pathex.get_image_paths(output_debug_path) ).run()
            input_image_paths = sorted (input_image_paths)
            io.log_info('Found %d images.' % (len(input_image_paths)))
    else:
        if not continue_extraction and output_debug_path.exists():
            for filename in pathex.get_image_paths(output_debug_path):
                Path(filename).unlink()

    images_found = len(input_image_paths)
    faces_detected = 0
    if images_found != 0:
        if detector == 'manual':
            io.log_info ('Performing manual extract...')
            # Anche la passata a mano costruisce i due estrattori (il tipo
            # contiene 'landmarks'): il rilevatore le serve da
            # second_pass_extractor e l'allineatore mette i 68 punti dentro
            # il rettangolo tracciato. Senza queste chiavi userebbe i motori
            # di default mentre l'utente ne ha scelti altri -- qui
            # `rilevatore` e' None per costruzione (detector == 'manual'),
            # ma `landmarker` e `min_face_size` arrivano da --landmarker e
            # --min-face-size e vanno inoltrati.
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ], 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size,
                                         allineatore=landmarker,
                                         min_face_size=min_face_size,
                                         device_config=device_config).run()

            io.log_info ('Performing 3rd pass...')
            data = ExtractSubprocessor (data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()
            _scrivi_rapporto(report_dir, data, motore=ExtractReport.MOTORE_MANUALE)

        else:
            io.log_info ('Extracting faces...')
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ],
                                         'all',
                                         image_size,
                                         jpeg_quality,
                                         face_type,
                                         output_debug_path if output_debug else None,
                                         max_faces_from_image=max_faces_from_image,
                                         final_output_path=output_path,
                                         rilevatore=detector if detector != 'manual' else None,
                                         allineatore=landmarker,
                                         min_face_size=min_face_size,
                                         device_config=device_config).run()
            _scrivi_rapporto(report_dir, data, motore=f"{detector}+{landmarker}")

        faces_detected += sum([d.faces_detected for d in data])

        if manual_fix:
            if all ( np.array ( [ d.faces_detected > 0 for d in data] ) == True ):
                io.log_info ('All faces are detected, manual fix not needed.')
            else:
                fix_data = [ ExtractSubprocessor.Data(d.filepath) for d in data if d.faces_detected == 0 ]
                io.log_info ('Performing manual fix for %d images...' % (len(fix_data)) )
                # Le stesse tre chiavi della passata automatica qui sopra: i
                # frame recuperati a
                # mano finiscono nello STESSO aligned/ di quelli automatici,
                # quindi devono passare per lo stesso allineatore. Con
                # 3DFAN scelto nel dialogo, ometterle scriveva landmark
                # 3DFAN nella prima passata e 2DFAN nella correzione --
                # due convenzioni di inquadratura nella stessa cartella,
                # invisibili fino al training.
                fix_data = ExtractSubprocessor (fix_data, 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size,
                                                rilevatore=detector if detector != 'manual' else None,
                                                allineatore=landmarker,
                                                min_face_size=min_face_size,
                                                device_config=device_config).run()
                fix_data = ExtractSubprocessor (fix_data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()
                _scrivi_rapporto(report_dir, fix_data, motore=ExtractReport.MOTORE_MANUALE)
                faces_detected += sum([d.faces_detected for d in fix_data])


    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')
