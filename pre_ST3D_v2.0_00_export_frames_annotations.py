import sys

from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
from AccessMath.annotation.text_annotation_exporter import TextAnnotationExporter
from AccessMath.preprocessing.config.parameters import Parameters

def get_worker(process):
    export_dir = Parameters.Output_FrameExport + process.current_lecture.title
    # TODO: this should have a default in Parameters class + a parameter to override at run-time
    export_mode = Parameters.Output_FrameExport_Mode
    text_exporter = TextAnnotationExporter.FromAnnotationXML(export_mode, process.database, process.current_lecture,
                                                             export_dir, export_images=False)

    print(" -> Total Text-regions found: {0:d}".format(len(text_exporter.text_objects)))

    if text_exporter.speaker is None:
        print(" -> Speaker object not found")
    else:
        print(" -> Speaker object found")

    return text_exporter


def get_results():
    return None


def main():
    # usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, None)

    if not process.initialize():
        return

    fps = Parameters.Sampling_FPS # 1.0
    process.start_video_processing(fps, get_worker, get_results, 0, True)

    print("Finished")


if __name__ == "__main__":
    main()