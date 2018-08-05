import sys

from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.video_worker.text_detection import TextDetection
from AccessMath.util.deep_model_loader import CaffeModelLoader

mdl = CaffeModelLoader(
    model_def=Parameters.Model_TextDetection+'deploy.prototxt',
    model_weights=Parameters.Model_TextDetection+'VGG_text_longer_conv_300x300_iter_10000.caffemodel')

def get_worker(process):
    worker = TextDetection(net=mdl.getModel(),
                           detection_threshold=0.6)
    worker.set_debug_mode(False, 0, 1e9, out_dir=None, video_name=process.current_lecture.title)
    return worker

def get_results(worker):
    return (worker.get_results(),)

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, Parameters.Output_TextDetection)
    if not process.initialize():
        return

    # fps = Parameters.Sampling_FPS
    # fps = 1
    # process.start_video_processing(fps, get_worker, get_results, 0, True)
    process.start_image_list_preprocessing(src_dir=Parameters.Output_FrameExport,
                                           get_worker_function=get_worker,
                                           get_results_function=get_results,
                                           frames_limit=0,
                                           verbose=True)
    print("finished")

if __name__ == "__main__":
    main()
