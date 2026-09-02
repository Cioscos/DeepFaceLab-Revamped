"""La raccolta degli allineamenti e i vettori di moto, da Merger.main.

Stessa logica riga per riga; la sola differenza e' che gli avvisi
("multiple faces detected", i frame senza volto) TORNANO come dati:
Merger.main li stampa come prima, il servizio per la GUI li tiene per il
rapporto. `avvisi["multipli"]` porta una terna per gruppo (nome del frame,
nomi degli allineati, nomi delle rispettive sorgenti) perche' due
allineati con lo stesso stem possono avere un `source_filename` diverso
(estensione, maiuscole): la sorgente stampata e' quella del singolo
record, non quella del primo del gruppo.
"""
import math
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.interact import interact as io
from DFLIMG import DFLIMG
from facelib import FaceType, LandmarksProcessor
from merger.FrameInfo import FrameInfo
from merger.SessioneMerge import Frame


def raccogli_frames(input_path, aligned_path):
    input_path_image_paths = pathex.get_image_paths(input_path)

    packed_samples = None
    try:
        packed_samples = samplelib.PackedFaceset.load(aligned_path)
    except:
        io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(aligned_path)}, {traceback.format_exc()}")

    if packed_samples is not None:
        io.log_info ("Using packed faceset.")
        def generator():
            for sample in io.progress_bar_generator( packed_samples, "Collecting alignments"):
                filepath = Path(sample.filename)
                yield filepath, DFLIMG.load(filepath, loader_func=lambda x: sample.read_raw_file()  )
    else:
        def generator():
            for filepath in io.progress_bar_generator( pathex.get_image_paths(aligned_path), "Collecting alignments"):
                filepath = Path(filepath)
                yield filepath, DFLIMG.load(filepath)

    alignments = {}
    multiple_faces_detected = False

    for filepath, dflimg in generator():
        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} is not a dfl image file")
            continue

        source_filename = dflimg.get_source_filename()
        if source_filename is None:
            continue

        source_filepath = Path(source_filename)
        source_filename_stem = source_filepath.stem

        if source_filename_stem not in alignments.keys():
            alignments[ source_filename_stem ] = []

        alignments_ar = alignments[ source_filename_stem ]
        alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath ) )

        if len(alignments_ar) > 1:
            multiple_faces_detected = True

    multipli = []
    for a_key in list(alignments.keys()):
        a_ar = alignments[a_key]
        if len(a_ar) > 1:
            multipli.append((a_ar[0][2].name,
                              [filepath.name for _, filepath, _ in a_ar],
                              [source_filepath.name for _, _, source_filepath in a_ar]))
        alignments[a_key] = [ a[0] for a in a_ar]

    frames = [ Frame( frame_info=FrameInfo(filepath=Path(p),
                                           landmarks_list=alignments.get(Path(p).stem, None)))
               for p in input_path_image_paths ]

    senza_volto = [f.frame_info.filepath.name for f in frames if len(f.frame_info.landmarks_list) == 0]

    if not multiple_faces_detected:
        s = 256
        local_pts = [ (s//2-1, s//2-1), (s//2-1,0) ] #center+up
        frames_len = len(frames)
        for i in io.progress_bar_generator( range(len(frames)) , "Computing motion vectors"):
            fi_prev = frames[max(0, i-1)].frame_info
            fi      = frames[i].frame_info
            fi_next = frames[min(i+1, frames_len-1)].frame_info
            if len(fi_prev.landmarks_list) == 0 or \
               len(fi.landmarks_list) == 0 or \
               len(fi_next.landmarks_list) == 0:
                    continue

            mat_prev = LandmarksProcessor.get_transform_mat ( fi_prev.landmarks_list[0], s, face_type=FaceType.FULL)
            mat      = LandmarksProcessor.get_transform_mat ( fi.landmarks_list[0]     , s, face_type=FaceType.FULL)
            mat_next = LandmarksProcessor.get_transform_mat ( fi_next.landmarks_list[0], s, face_type=FaceType.FULL)

            pts_prev = LandmarksProcessor.transform_points (local_pts, mat_prev, True)
            pts      = LandmarksProcessor.transform_points (local_pts, mat, True)
            pts_next = LandmarksProcessor.transform_points (local_pts, mat_next, True)

            motion_vector = pts_next[0] - pts_prev[0]
            fi.motion_power = npla.norm(motion_vector)

            motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0,0],dtype=np.float32)

            fi.motion_deg = -math.atan2(motion_vector[1],motion_vector[0])*180 / math.pi

    return frames, {"multipli": multipli, "senza_volto": senza_volto}


def stampa_avvisi(avvisi):
    """Le stesse righe che Merger.main stampava, nello stesso ordine."""
    multipli = avvisi["multipli"]
    if multipli:
        io.log_info ("")
        io.log_info ("Warning: multiple faces detected. Only one alignment file should refer one source file.")
        io.log_info ("")
        for _, allineati, nomi_sorgente in multipli:
            for nome, sorgente in zip(allineati, nomi_sorgente):
                io.log_info (f"alignment {nome} refers to {sorgente} ")
            io.log_info ("")
        io.log_info ("It is strongly recommended to process the faces separatelly.")
        io.log_info ("Use 'recover original filename' to determine the exact duplicates.")
        io.log_info ("")
        io.log_info ("Warning: multiple faces detected. Motion blur will not be used.")
        io.log_info ("")
