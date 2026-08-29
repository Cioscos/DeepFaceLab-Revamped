import multiprocessing
import operator
import pickle
import traceback
from pathlib import Path

import numpy as np

import samplelib.PackedFaceset
import samplelib.PosaCampioni
from core import pathex
from core.mplib import MPSharedList
from core.interact import interact as io
from core.joblib import Subprocessor
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor

from .Sample import Sample, SampleType


class SampleLoader:
    samples_cache = dict()
    posa_cache = dict()       # str(samples_path) -> ndarray [N, 3], stessa vita di samples_cache

    @staticmethod
    def posa(samples_path):
        """La matrice [N, 3] delle pose dei campioni FACE gia' caricati da
        `samples_path`, o None se quel percorso non e' mai passato da `load`."""
        return SampleLoader.posa_cache.get(str(samples_path))

    @staticmethod
    def posa_di(samples_path, samples):
        """Come `posa`, ma con un ripiego: se il percorso non e' passato da
        `load` -- un test che consegna una lista sua -- la matrice si
        costruisce dai campioni, come faceva il ciclo originale."""
        posa = SampleLoader.posa(samples_path)
        if posa is None or len(posa) != len(samples):
            posa = np.array([s.get_pitch_yaw_roll() for s in samples], dtype=np.float64).reshape(len(samples), 3)
        return posa
    @staticmethod
    def get_person_id_max_count(samples_path):
        samples = None
        try:
            samples = samplelib.PackedFaceset.load(samples_path)
        except:
            io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_path)}, {traceback.format_exc()}")

        if samples is None:
            raise ValueError("packed faceset not found.")
        persons_name_idxs = {}
        for sample in samples:
            persons_name_idxs[sample.person_name] = 0
        return len(list(persons_name_idxs.keys()))

    @staticmethod
    def load(sample_type, samples_path, subdirs=False):
        """
        Return MPSharedList of samples
        """
        samples_cache = SampleLoader.samples_cache

        if str(samples_path) not in samples_cache.keys():
            samples_cache[str(samples_path)] = [None]*SampleType.QTY

        samples = samples_cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  samples[sample_type] is None:
                samples[sample_type] = [ Sample(filename=filename) for filename in io.progress_bar_generator( pathex.get_image_paths(samples_path, subdirs=subdirs), "Loading") ]

        elif          sample_type == SampleType.FACE:
            if  samples[sample_type] is None:
                try:
                    result = samplelib.PackedFaceset.load(samples_path)
                except:
                    io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")

                if result is not None:
                    io.log_info (f"Loaded {len(result)} packed faces from {samples_path}")

                if result is None:
                    result = SampleLoader.load_face_samples( pathex.get_image_paths(samples_path, subdirs=subdirs) )
                    pak = None
                else:
                    pak = Path(samples_path) / samplelib.PackedFaceset.NOME_FILE

                #Prima dell'impacchettamento in memoria condivisa, cosi' ogni
                #figlio eredita la posa gia' fatta e il generatore la legge
                #dalla matrice invece di de-picklare N campioni.
                SampleLoader.posa_cache[str(samples_path)] = samplelib.PosaCampioni.stima_posa(result, pak=pak)
                samples[sample_type] = MPSharedList(result)
        elif          sample_type == SampleType.FACE_TEMPORAL_SORTED:
                result = SampleLoader.load (SampleType.FACE, samples_path)
                result = SampleLoader.upgradeToFaceTemporalSortedSamples(result)
                samples[sample_type] = MPSharedList(result)

        return samples[sample_type]

    @staticmethod
    def load_face_samples ( image_paths):
        result = FaceSamplesLoaderSubprocessor(image_paths).run()
        sample_list = []

        for filename, data in result:
            if data is None:
                continue
            ( face_type,
              shape,
              landmarks,
              seg_ie_polys,
              xseg_mask_compressed,
              eyebrows_expand_mod,
              source_filename,
              pitch_yaw_roll ) = data
              
            sample_list.append( Sample(filename=filename,
                                        sample_type=SampleType.FACE,
                                        face_type=FaceType.fromString (face_type),
                                        shape=shape,
                                        landmarks=landmarks,
                                        seg_ie_polys=seg_ie_polys,
                                        xseg_mask_compressed=xseg_mask_compressed,
                                        eyebrows_expand_mod=eyebrows_expand_mod,
                                        source_filename=source_filename,
                                        pitch_yaw_roll=pitch_yaw_roll,
                                    ))
        return sample_list

    @staticmethod
    def upgradeToFaceTemporalSortedSamples( samples ):
        new_s = [ (s, s.source_filename) for s in samples]
        new_s = sorted(new_s, key=operator.itemgetter(1))

        return [ s[0] for s in new_s]


class FaceSamplesLoaderSubprocessor(Subprocessor):
    #override
    def __init__(self, image_paths ):
        self.image_paths = image_paths
        self.image_paths_len = len(image_paths)
        self.idxs = [*range(self.image_paths_len)]
        self.result = [None]*self.image_paths_len
        super().__init__('FaceSamplesLoader', FaceSamplesLoaderSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Loading samples", len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {}

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            idx = self.idxs.pop(0)
            return idx, self.image_paths[idx]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, dflimg = result
        self.result[idx] = (self.image_paths[idx], dflimg)
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, filename = data
            dflimg = DFLIMG.load (Path(filename))

            if dflimg is None or not dflimg.has_data():
                self.log_err (f"FaceSamplesLoader: {filename} is not a dfl image file.")
                data = None
            else:
                landmarks = dflimg.get_landmarks()
                shape = dflimg.get_shape()
                #La posa qui, dove il lavoro e' gia' distribuito sui figli:
                #costa quanto un solvePnP per volto e risparmia al caricatore
                #di rifarla in serie, e a `pack` di scriverla vuota nel .pak.
                data = (dflimg.get_face_type(),
                        shape,
                        landmarks,
                        dflimg.get_seg_ie_polys(),
                        dflimg.get_xseg_mask_compressed(),
                        dflimg.get_eyebrows_expand_mod(),
                        dflimg.get_source_filename(),
                        samplelib.PosaCampioni._una(landmarks, shape[1]) )

            return idx, data

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[1]
