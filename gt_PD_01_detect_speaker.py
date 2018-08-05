import sys

from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.preprocessing.video_worker.person_detection import PersonDetection
from AccessMath.util.deep_model_loader import TorchModelLoader

from ssd import build_ssd

model_path = Parameters.Model_PersonDetection + 'ssd_300_VOC0712.pth'
mdl = TorchModelLoader(net=build_ssd('Test', 300, 21), model_path=model_path)

def get_worker(process):
    worker = PersonDetection(net=mdl.getModel(),
                             detection_threshold=0.4)
    worker.set_debug_mode(False, 0, 1e9, out_dir=None, video_name=process.current_lecture.title)
    return worker

def get_results(worker):
    return (worker.get_results(),)

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, Parameters.Output_PersonDetection)
    if not process.initialize():
        return

    fps = 1
    process.start_video_processing(fps, get_worker, get_results, 0, True)
    # process.start_image_list_preprocessing(src_dir=Parameters.Output_FrameExport,
    #                                        get_worker_function=get_worker,
    #                                        get_results_function=get_results,
    #                                        frames_limit=0,
    #                                        verbose=True)
    print("finished")

if __name__ == "__main__":
    main()
