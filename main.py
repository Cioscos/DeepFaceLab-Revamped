if __name__ == "__main__":
    # Fix for linux
    import multiprocessing
    multiprocessing.set_start_method("spawn")

    from core.leras import nn
    nn.initialize_main_env()
    import os
    import sys
    import time
    import argparse

    from core import pathex
    from core import osex
    from pathlib import Path
    from core.interact import interact as io

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    def positive_interval_min(value):
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("--save-interval-min must be an integer, got %r" % value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("--save-interval-min must be >= 1, got %d (0 or negative never becomes due)" % ivalue)
        return ivalue

    exit_code = 0
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def process_extract(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.main( detector                = arguments.detector,
                        landmarker              = arguments.landmarker,
                        min_face_size           = arguments.min_face_size,
                        input_path              = Path(arguments.input_dir),
                        output_path             = Path(arguments.output_dir),
                        output_debug            = arguments.output_debug,
                        manual_fix              = arguments.manual_fix,
                        manual_output_debug_fix = arguments.manual_output_debug_fix,
                        manual_window_size      = arguments.manual_window_size,
                        face_type               = arguments.face_type,
                        max_faces_from_image    = arguments.max_faces_from_image,
                        image_size              = arguments.image_size,
                        jpeg_quality            = arguments.jpeg_quality,
                        cpu_only                = arguments.cpu_only,
                        force_gpu_idxs          = [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                        report_dir              = Path(arguments.report_dir) if arguments.report_dir is not None else None,
                      )

    # import qui e non in cima al file: nn.initialize_main_env() deve
    # girare prima di ogni altra cosa, e MotoriCatalog e' dati puri quindi
    # importarlo qui non costa nulla.
    from mainscripts import MotoriCatalog

    p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    p.add_argument('--detector', dest="detector",
                   choices=MotoriCatalog.CHIAVI_RILEVATORI + ('manual',),
                   default=None, help="Type of detector.")
    p.add_argument('--landmarker', dest="landmarker",
                   choices=MotoriCatalog.CHIAVI_ALLINEATORI,
                   default=None, help="Landmark model. Asked interactively when omitted.")
    p.add_argument('--min-face-size', type=int, dest="min_face_size", default=None,
                   help=f"Discard detections whose shorter side is below this, in pixels of the source frame. Asked interactively when omitted, with {MotoriCatalog.LATO_MIN_PREDEFINITO} as the default.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="Writes debug images to <output-dir>_debug\\ directory.")
    p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="Don't writes debug images to <output-dir>_debug\\ directory.")
    p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None)
    p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None, help="Max faces from image.")    
    p.add_argument('--image-size', type=int, dest="image_size", default=None, help="Output image size.")
    p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None, help="Jpeg quality.")    
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU..")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    # Assente per default, e con il default il comportamento non cambia di
    # un byte (report_dir=None e' gia' il default di Extractor.main): il
    # canale per cui gui/estrazione/pagina.py legge il rapporto per frame
    # (frames.ndjson, mainscripts/ExtractReport.py) che l'estrazione scrive
    # gia' mentre gira, ma che senza questo flag nessuno chiedeva mai.
    p.add_argument('--report-dir', action=fixPathAction, dest="report_dir", default=None, help="Where the per-frame report (frames.ndjson) is written, for the GUI's extraction page. Nothing is written when omitted.")

    p.set_defaults (func=process_extract)

    def process_sort(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Sorter
        par = Sorter.Parametri(target_count=arguments.target_count,
                               ref_dir=arguments.ref_dir,
                               similar=arguments.similar,
                               threshold=arguments.threshold)
        Sorter.main (input_path=Path(arguments.input_dir),
                     sort_by_method=arguments.sort_by_method,
                     par=par)

    from mainscripts.SorterCatalog import CHIAVI as SORT_CHIAVI

    p = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', dest="sort_by_method", default=None, choices=SORT_CHIAVI, help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    p.add_argument('--target-count', type=int, default=None, dest="target_count", help="How many faces to keep, for the methods that cut the faceset down. Asked interactively when omitted.")
    p.add_argument('--ref-dir', action=fixPathAction, default=None, dest="ref_dir", help="Reference faceset for 'match-dst'. Defaults to the sibling data_dst/aligned when omitted.")
    p.add_argument('--threshold', type=int, default=None, dest="threshold", help="Distance threshold for 'dedup'. Asked interactively when omitted.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--similar', dest="similar", action='store_true', default=None, help="For 'absdiff' and 'absdiff-fast': group similar faces together.")
    grp.add_argument('--dissimilar', dest="similar", action='store_false', default=None, help="For 'absdiff' and 'absdiff-fast': push the most different faces to the front.")
    p.set_defaults (func=process_sort)

    def process_util(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Util

        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)

        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename (input_path=arguments.input_dir)

        if arguments.save_faceset_metadata:
            Util.save_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.restore_faceset_metadata:
            Util.restore_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.pack_faceset:
            io.log_info ("Performing faceset packing...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.pack( Path(arguments.input_dir) )

        if arguments.unpack_faceset:
            io.log_info ("Performing faceset unpacking...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.unpack( Path(arguments.input_dir) )

    p = subparsers.add_parser( "util", help="Utilities.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="Add landmarks debug image for aligned faces.")
    p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False, help="Recover original aligned filename.")
    p.add_argument('--save-faceset-metadata', action="store_true", dest="save_faceset_metadata", default=False, help="Save faceset metadata to file.")
    p.add_argument('--restore-faceset-metadata', action="store_true", dest="restore_faceset_metadata", default=False, help="Restore faceset metadata to file. Image filenames must be the same as used with save.")
    p.add_argument('--pack-faceset', action="store_true", dest="pack_faceset", default=False, help="")
    p.add_argument('--unpack-faceset', action="store_true", dest="unpack_faceset", default=False, help="")

    p.set_defaults (func=process_util)

    def process_train(arguments):
        osex.set_process_lowest_prio()


        kwargs = {'model_class_name'         : arguments.model_name,
                  'saved_models_path'        : Path(arguments.model_dir),
                  'training_data_src_path'   : Path(arguments.training_data_src_dir),
                  'training_data_dst_path'   : Path(arguments.training_data_dst_dir),
                  'pretraining_data_path'    : Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
                  'no_preview'               : arguments.no_preview,
                  'force_model_name'         : arguments.force_model_name,
                  'force_gpu_idxs'           : [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                  'cpu_only'                 : arguments.cpu_only,
                  'silent_start'             : arguments.silent_start,
                  'debug'                    : arguments.debug,
                  'save_interval_min'        : arguments.save_interval_min,
                  }
        from mainscripts import Trainer
        Trainer.main(**kwargs)

    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of extracted SRC faceset.")
    p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of extracted DST faceset.")
    p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="Optional dir of extracted faceset that will be used in pretraining mode.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
    p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="Disable preview window.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.add_argument('--silent-start', action="store_true", dest="silent_start", default=False, help="Silent start. Automatically chooses Best GPU and last used model.")
    p.add_argument('--save-interval-min', type=positive_interval_min, dest="save_interval_min", default=25, help="How often the model is saved, in minutes. Must be >= 1.")

    p.set_defaults (func=process_train)
    
    def process_exportdfm(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import ExportDFM
        ExportDFM.main(model_class_name = arguments.model_name, saved_models_path = Path(arguments.model_dir), force_model_name = arguments.force_model_name)

    p = subparsers.add_parser( "exportdfm", help="Export model to use in DeepFaceLive.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.set_defaults (func=process_exportdfm)

    def process_gui(arguments):
        from gui.app import run
        exit(run())

    p = subparsers.add_parser( "gui", help="Start the graphical interface.")
    p.set_defaults (func=process_gui)

    def process_merge(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Merger
        Merger.main ( model_class_name       = arguments.model_name,
                      saved_models_path      = Path(arguments.model_dir),
                      force_model_name       = arguments.force_model_name,
                      input_path             = Path(arguments.input_dir),
                      output_path            = Path(arguments.output_dir),
                      output_mask_path       = Path(arguments.output_mask_dir),
                      aligned_path           = Path(arguments.aligned_dir) if arguments.aligned_dir is not None else None,
                      force_gpu_idxs         = [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                      cpu_only               = arguments.cpu_only)

    p = subparsers.add_parser( "merge", help="Merger")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the merged files will be stored.")
    p.add_argument('--output-mask-dir', required=True, action=fixPathAction, dest="output_mask_dir", help="Output mask directory. This is where the mask files will be stored.")
    p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", default=None, help="Aligned directory. This is where the extracted of dst faces stored.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Merge on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.set_defaults(func=process_merge)

    videoed_parser = subparsers.add_parser( "videoed", help="Video processing.").add_subparsers()

    def process_videoed_extract_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.extract_video (arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="Extract images from video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted images will be stored.")
    p.add_argument('--output-ext', dest="output_ext", default=None, help="Image format (extension) of output files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. 0 - full fps.")
    p.set_defaults(func=process_videoed_extract_video)

    def process_videoed_cut_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.cut_video (arguments.input_file,
                           arguments.from_time,
                           arguments.to_time,
                           arguments.audio_track_id,
                           arguments.bitrate)
    p = videoed_parser.add_parser( "cut-video", help="Cut video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--from-time', dest="from_time", default=None, help="From time, for example 00:00:00.000")
    p.add_argument('--to-time', dest="to_time", default=None, help="To time, for example 00:00:00.000")
    p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None, help="Specify audio track id.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.set_defaults(func=process_videoed_cut_video)

    def process_videoed_denoise_image_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.denoise_image_sequence (arguments.input_dir, arguments.factor)
    p = videoed_parser.add_parser( "denoise-image-sequence", help="Denoise sequence of images, keeping sharp edges. Helps to remove pixel shake from the predicted face.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory to be processed.")
    p.add_argument('--factor', type=int, dest="factor", default=None, help="Denoise factor (1-20).")
    p.set_defaults(func=process_videoed_denoise_image_sequence)

    def process_videoed_video_from_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.video_from_sequence (input_dir      = arguments.input_dir,
                                     output_file    = arguments.output_file,
                                     reference_file = arguments.reference_file,
                                     ext      = arguments.ext,
                                     fps      = arguments.fps,
                                     bitrate  = arguments.bitrate,
                                     include_audio = arguments.include_audio,
                                     lossless = arguments.lossless)

    p = videoed_parser.add_parser( "video-from-sequence", help="Make video from image sequence.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--reference-file', action=fixPathAction, dest="reference_file", help="Reference file used to determine proper FPS and transfer audio from it. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default='png', help="Image format (extension) of input files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="FPS of output file. Overwritten by reference-file.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.add_argument('--include-audio', action="store_true", dest="include_audio", default=False, help="Include audio from reference file.")
    p.add_argument('--lossless', action="store_true", dest="lossless", default=False, help="PNG codec.")

    p.set_defaults(func=process_videoed_video_from_sequence)

    facesettool_parser = subparsers.add_parser( "facesettool", help="Faceset tools.").add_subparsers()

    def process_faceset_enhancer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetEnhancer
        FacesetEnhancer.process_folder ( Path(arguments.input_dir),
                                         cpu_only=arguments.cpu_only,
                                         force_gpu_idxs=[ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None
                                       )

    p = facesettool_parser.add_parser ("enhance", help="Enhance details in DFL faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Process on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

    p.set_defaults(func=process_faceset_enhancer)
    
    
    p = facesettool_parser.add_parser ("resize", help="Resize DFL faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")

    def process_faceset_resizer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetResizer
        FacesetResizer.process_folder ( Path(arguments.input_dir) )
    p.set_defaults(func=process_faceset_resizer)


    p = facesettool_parser.add_parser ("index", help="Index a faceset folder for the graphical interface.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--cache-dir', required=True, action=fixPathAction, dest="cache_dir", help="Where the index and the mask blob are written.")
    p.add_argument('--only-missing', action="store_true", dest="only_missing", default=False, help="Index only the faces the cache does not already hold.")

    def process_faceset_index(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetIndex
        FacesetIndex.indicizza ( Path(arguments.input_dir),
                                  Path(arguments.cache_dir),
                                  only_missing=arguments.only_missing )
    p.set_defaults(func=process_faceset_index)


    p = facesettool_parser.add_parser ("detail", help="Serve one face at a time to the graphical interface.")
    p.add_argument('--workdir', required=True, action=fixPathAction, dest="workdir", help="Where the raster answers are written.")

    def process_faceset_detail(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetDetail
        FacesetDetail.main ( Path(arguments.workdir) )
    p.set_defaults(func=process_faceset_detail)


    extracttool_parser = subparsers.add_parser( "extracttool", help="Extraction tools.").add_subparsers()

    p = extracttool_parser.add_parser ("index", help="Rebuild the per-frame report for an already extracted folder.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of frames.")
    p.add_argument('--aligned-dir', required=True, action=fixPathAction, dest="aligned_dir", help="Directory of aligned faces already extracted.")
    p.add_argument('--cache-dir', required=True, action=fixPathAction, dest="cache_dir", help="Where the per-frame report is written.")

    def process_extract_index(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import ExtractIndex
        ExtractIndex.ricostruisci ( Path(arguments.input_dir),
                                    Path(arguments.aligned_dir),
                                    Path(arguments.cache_dir) )
    p.set_defaults(func=process_extract_index)

    p = extracttool_parser.add_parser ("manual", help="Serve one frame at a time to the graphical interface.")
    p.add_argument('--workdir', required=True, action=fixPathAction, dest="workdir", help="Where the raster answers are written.")

    def process_extract_manual(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import ExtractManual
        ExtractManual.main ( Path(arguments.workdir) )
    p.set_defaults(func=process_extract_manual)

    def process_dev_test(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.dev_test( arguments.input_dir )

    p = subparsers.add_parser( "dev_test", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_dev_test)
    
    # ========== XSeg
    xseg_parser = subparsers.add_parser( "xseg", help="XSeg tools.").add_subparsers()
    
    p = xseg_parser.add_parser( "editor", help="XSeg editor.")

    def process_xsegeditor(arguments):
        osex.set_process_lowest_prio()
        from XSegEditor import XSegEditor
        global exit_code
        exit_code = XSegEditor.start (Path(arguments.input_dir))
        
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")

    p.set_defaults (func=process_xsegeditor)
  
    p = xseg_parser.add_parser( "apply", help="Apply trained XSeg model to the extracted faces.")

    def process_xsegapply(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.apply_xseg (Path(arguments.input_dir), Path(arguments.model_dir))
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir")
    p.set_defaults (func=process_xsegapply)
    
    
    p = xseg_parser.add_parser( "remove", help="Remove applied XSeg masks from the extracted faces.")
    def process_xsegremove(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremove)
    
    
    p = xseg_parser.add_parser( "remove_labels", help="Remove XSeg labels from the extracted faces.")
    def process_xsegremovelabels(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg_labels (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremovelabels)
    
    
    p = xseg_parser.add_parser( "fetch", help="Copies faces containing XSeg polygons in <input_dir>_xseg dir.")

    def process_xsegfetch(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.fetch_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegfetch)
    
    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    arguments.func(arguments)

    if exit_code == 0:
        print ("Done.")
        
    exit(exit_code)
    
'''
import code
code.interact(local=dict(globals(), **locals()))
'''
