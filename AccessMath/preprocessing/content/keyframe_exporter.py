
import os
import cv2
import math
import numpy as np

from AccessMath.preprocessing.content.helper import Helper
from AccessMath.util.misc_helper import MiscHelper

from concurrent.futures import ThreadPoolExecutor

class KeyframeExporter:
    @staticmethod
    def GenerateVideoSegmentsXML(idx_intervals, time_intervals):
        xml_string = "  <VideoSegments>\n"
        for idx, (idx_frame_start, idx_frame_end) in enumerate(idx_intervals):
            time_start, time_end = time_intervals[idx]

            xml_string += "    <VideoSegment>\n"
            xml_string += "        <Start>" + str(idx_frame_start) + "</Start>\n"
            xml_string += "        <End>" + str(idx_frame_end) + "</End>\n"
            xml_string += "        <AbsTimeStart>" + str(time_start) + "</AbsTimeStart>\n"
            xml_string += "        <AbsTimeEnd>" + str(time_end) + "</AbsTimeEnd>\n"
            xml_string += "    </VideoSegment>\n"
        xml_string += "  </VideoSegments>\n"

        return xml_string

    @staticmethod
    def GenerateKeyframesXML(summary_indices, summary_times):
        xml_string = "  <VideoKeyFrames>\n"
        for idx, frame_time in enumerate(summary_times):
            xml_string += "    <VideoKeyFrame>\n"
            xml_string += "       <Index>" + str(summary_indices[idx]) + "</Index>\n"
            xml_string += "       <AbsTime>" + str(frame_time) + "</AbsTime>\n"
            xml_string += "       <VideoObjects>\n"
            xml_string += "       </VideoObjects>\n"
            xml_string += "    </VideoKeyFrame>\n"
        xml_string += "  </VideoKeyFrames>\n"

        return xml_string

    @staticmethod
    def GenerateMetadataHeaderXML(output_filename, database, lecture):
        xml_string = "  <Database>" + database.name + "</Database>\n"
        xml_string += "  <Lecture>" + lecture.title + "</Lecture>\n"
        xml_string += "  <Filename>" + output_filename + "</Filename>\n"
        xml_string += "  <VideoFiles>\n"
        for video_data in lecture.main_videos:
            xml_string += "  <VideoFile>" + video_data['path'] + "</VideoFile>\n"
        xml_string += "  </VideoFiles>\n"

        return xml_string

    @staticmethod
    def GenerateExportXML(filename, database, lecture, idx_intervals, time_intervals, summary_indices, summary_times):
        xml_string = "<Annotations>\n"

        # general meta-data
        xml_string += KeyframeExporter.GenerateMetadataHeaderXML(filename, database, lecture)

        # segments
        xml_string += KeyframeExporter.GenerateVideoSegmentsXML(idx_intervals, time_intervals)

        # key frames with object data ...
        xml_string += KeyframeExporter.GenerateKeyframesXML(summary_indices, summary_times)

        xml_string += "</Annotations>\n"

        return xml_string

    @staticmethod
    def Export(main_path, database, lecture, idx_intervals, time_intervals, kf_indices, kf_times, kf_images):
        # check if output directory exists
        if not os.path.exists(main_path):
            os.mkdir(main_path)

        # check if keyframes sub-directory exists
        keyframes_path = main_path + "/keyframes"
        if not os.path.exists(keyframes_path):
            os.mkdir(keyframes_path)

        # save images for key-frames ....
        frame_times = []
        for idx, keyframe_idx in enumerate(kf_indices):
            # save image to file ...
            cv2.imwrite(keyframes_path + "/" + str(keyframe_idx) + ".png", kf_images[idx])

        # save XML string to output file
        filename = main_path + "/segments.xml"
        xml_data = KeyframeExporter.GenerateExportXML(filename, database, lecture, idx_intervals, time_intervals,
                                                      kf_indices, kf_times)

        out_file = open(filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("Metadata Saved to: " + filename)

    @staticmethod
    def GenerateKeyframeGUIContentXML(keyframe_ccs):
        xml_string = ""

        for abs_time, min_x, max_x, min_y, max_y in keyframe_ccs:
            xml_string += "\t\t<content>\n"
            xml_string += "\t\t\t<minX>" + str(min_x) + "</minX>\n"
            xml_string += "\t\t\t<maxX>" + str(max_x) + "</maxX>\n"
            xml_string += "\t\t\t<minY>" + str(min_y) + "</minY>\n"
            xml_string += "\t\t\t<maxY>" + str(max_y) + "</maxY>\n"
            xml_string += "\t\t\t<jump>" + str(abs_time) + "</jump>\n"
            xml_string += "\t\t</content>\n"

        return xml_string

    @staticmethod
    def GenerateGUIExportXML(cc_group_times):
        xml_string = "<lecture_info>\n"

        # Generate Key-frame click locations with jumps ...
        for idx, keyframe_ccs in enumerate(cc_group_times):
            xml_string += "\t<keyframe>\n"
            xml_string += KeyframeExporter.GenerateKeyframeGUIContentXML(keyframe_ccs)
            xml_string += "\t</keyframe>\n"

        xml_string += "</lecture_info>\n"

        return xml_string

    @staticmethod
    def ExportGUIInfo(main_path, cc_group_times):
        # check if output directory exists
        if not os.path.exists(main_path):
            raise Exception("Must export key-frame data before exporting GUI data")

        # check if keyframes sub-directory exists
        keyframes_path = main_path + "/keyframes"
        if not os.path.exists(keyframes_path):
            raise Exception("Must export key-frame data before exporting GUI data")

        # save XML string to output file
        filename = main_path + "/gui_export.xml"
        xml_data = KeyframeExporter.GenerateGUIExportXML(cc_group_times)

        out_file = open(filename, "w")
        out_file.write(xml_data)
        out_file.close()

        print("GUI Metadata Saved to: " + filename)

    @staticmethod
    def FromUniformSample(database, lecture, step, sample_name, binary_source):
        # load output from pipeline ...
        lecture_suffix = str(lecture.id) + ".dat"

        # load binary images
        tempo_binary_filename = database.output_temporal + "/" + binary_source + lecture_suffix
        binary_data = MiscHelper.dump_load(tempo_binary_filename)
        original_frame_times, frame_indices, frame_compressed = binary_data

        # take a sample
        frame_times = [time for time in original_frame_times[::step]]
        frame_indices = [idx for idx in frame_indices[::step]]
        frame_compressed = [frame for frame in frame_compressed[::step]]

        print("Expanding loaded frames .... ")
        binary_frames = Helper.decompress_binary_images(frame_compressed)

        # segments ....
        output_prefix = database.output_summaries + "/" + sample_name + "_" + database.name + "_" + lecture.title.lower()
        print("Saving data to: " + output_prefix)

        # in abs frame indices ...
        intervals = []
        abs_intervals = []
        for idx, comp_frame in enumerate(binary_frames):
            if idx == 0:
                curr_start = int(frame_indices[idx] / 2)
                abs_start = frame_times[idx] / 2.0
            else:
                curr_start = int((frame_indices[idx - 1] + frame_indices[idx]) / 2)
                abs_start = (frame_times[idx - 1] + frame_times[idx]) / 2.0

            if idx + 1 < len(frame_indices):
                curr_end = int((frame_indices[idx + 1] + frame_indices[idx]) / 2)
                abs_end = (frame_times[idx + 1] + frame_times[idx]) / 2.0
            else:
                curr_end = frame_indices[idx]
                abs_end = frame_times[idx]

            # invert binarization ...
            binary_frames[idx] = 255 - comp_frame

            intervals.append((curr_start, curr_end))
            abs_intervals.append((abs_start, abs_end))

        KeyframeExporter.Export(output_prefix, database, lecture, intervals, abs_intervals, frame_indices, frame_times,
                                binary_frames)

    @staticmethod
    def SaveInterpolatedTemp(binary_frames, idx, mid_frames, temporary_prefix):
        out_frame_idx = idx
        curr_image = binary_frames[idx].astype(np.float64)

        if mid_frames > 0 and idx > 0:
            out_frame_idx += mid_frames * (idx - 1)

            # generate intermediate frames ...
            prev_image = binary_frames[idx - 1].astype(np.float64)

            for mid in range(mid_frames):
                prc = (mid + 1) / (mid_frames + 1)

                mid_img = prev_image * (1.0 - prc) + curr_image * prc
                mid_img = mid_img.astype(np.uint8)

                cv2.imwrite(temporary_prefix + str(out_frame_idx) + ".png", cv2.cvtColor(mid_img,cv2.COLOR_GRAY2RGB))
                out_frame_idx += 1

        # save the current frame ...
        curr_image = binary_frames[idx]
        cv2.imwrite(temporary_prefix + str(out_frame_idx) + ".png", cv2.cvtColor(curr_image,cv2.COLOR_GRAY2RGB))

    @staticmethod
    def ExpandGenerateSaveTemp(file_prefix, compressed_frames, frame_indices, invert_binary, frame_start, n_frames, interp_skip=0):
        # first, find start
        start_idx = 0
        end_idx = len(frame_indices)
        while start_idx < end_idx:
            mid_idx = int((start_idx + end_idx) / 2)
            if frame_indices[mid_idx] == frame_start:
                end_idx = mid_idx
                break
            elif frame_indices[mid_idx] < frame_start:
                start_idx = mid_idx + 1
            else:
                end_idx = mid_idx

        if end_idx == 0:
            prev_img = None
        else:
            prev_img = cv2.imdecode(compressed_frames[end_idx - 1], cv2.IMREAD_GRAYSCALE)
            if invert_binary:
                prev_img = 255 - prev_img

            prev_as_f = prev_img.astype(np.float64)

        if end_idx < len(frame_indices):
            next_img = cv2.imdecode(compressed_frames[end_idx], cv2.IMREAD_GRAYSCALE)
            if invert_binary:
                next_img = 255 - next_img

            next_as_f = next_img.astype(np.float64)
        else:
            next_img = None

        last_img = None

        next_to_interpolate = 0
        for frame_idx in range(frame_start, frame_start + n_frames):
            if end_idx < len(frame_indices):
                if frame_idx == frame_indices[end_idx]:
                    # use exact frame, no interpolation
                    curr_img = next_img
                    # move to the next
                    end_idx += 1
                    prev_img = next_img
                    prev_as_f = next_as_f
                    if end_idx < len(frame_indices):
                        next_img = cv2.imdecode(compressed_frames[end_idx], cv2.IMREAD_GRAYSCALE)
                        if invert_binary:
                            next_img = 255 - next_img

                        next_as_f = next_img.astype(np.float64)
                    else:
                        next_img = None

                    next_to_interpolate = interp_skip + 1
                elif end_idx == 0:
                    # ... has no preview, use next, no interpolation
                    curr_img = next_img
                else:
                    # ... check if interpolate ...
                    next_to_interpolate -= 1
                    if next_to_interpolate <= 0:
                        next_to_interpolate = interp_skip + 1

                        prc = (frame_idx - frame_indices[end_idx - 1]) / (frame_indices[end_idx] - frame_indices[end_idx - 1])
                        curr_img = (prev_as_f * (1.0 - prc) + next_as_f * prc).astype(np.uint8)
                    else:
                        curr_img = last_img
            else:
                # after the last frame ...
                curr_img = prev_img

            cv2.imwrite(file_prefix + str(frame_idx) + ".png", cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB))
            last_img = curr_img

    @staticmethod
    def ExportVideo(database, lecture, binary_source, video_prefix, invert_binary, skip_interpolation=0, workers=7, block_size=100):
        # .... paths ...
        lecture_sufix = str(lecture.id) + ".dat"
        tempo_binary_filename = database.output_temporal + "/" + binary_source + lecture_sufix
        lecture_str = video_prefix + "_" + database.name + "_" + lecture.title.lower()
        temporary_prefix = database.output_images + "/" + lecture_str + "_"
        first_video_filename = lecture.main_videos[0]["path"]

        # load binary images
        binary_data = MiscHelper.dump_load(tempo_binary_filename)
        original_frame_times, frame_indices, frame_compressed = binary_data

        print("Generating Temporary Files")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            n_blocks = int(math.ceil(frame_indices[-1] / block_size))

            pref_list = [temporary_prefix] * n_blocks
            frame_list = [frame_compressed] * n_blocks
            idx_list = [frame_indices] * n_blocks
            inv_list = [invert_binary] * n_blocks
            start_list = [idx * block_size for idx in range(n_blocks)]
            block_list = [block_size] * n_blocks
            skip_list = [skip_interpolation] * n_blocks

            total_frames = n_blocks * block_size

            for idx, _ in enumerate(executor.map(KeyframeExporter.ExpandGenerateSaveTemp, pref_list, frame_list,
                                                 idx_list, inv_list, start_list, block_list, skip_list)):
                prc_progress = ((idx + 1) * 100) / n_blocks
                print("-> Exporting: {0:.4f}% (Block {1:d} of {2:d})".format(prc_progress, idx + 1, n_blocks), end="\r", flush=True)
            
            print("", flush=True)

        # find source sampling frames per second
        capture = cv2.VideoCapture(first_video_filename)
        video_fps = capture.get(cv2.CAP_PROP_FPS)

        source_videos_str = " ".join(["-i " + video["path"] for video in lecture.main_videos])
        audio_filter_complex = " ".join(["[{0:d}:a:0]".format(idx + 1) for idx in range(len(lecture.main_videos))])
        audio_filter_complex += " concat=n={0:d}:v=0:a=1 [audio]".format(len(lecture.main_videos))
        video_output = database.output_videos + "/" + lecture_str + ".mp4"

        input_framerate = video_fps
        output_framerate = video_fps

        video_inputs = "-hwaccel dxva2 -framerate {0:.2f} -start_number 0 -i {1:s}%d.png".format(input_framerate,
                                                                                                 temporary_prefix)
        audio_inputs = "{0:s} -filter_complex \"{1:s}\"".format(source_videos_str, audio_filter_complex)
        output_flags = "-pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
        output_flags += "-s:v 1920x1080 -codec:v mpeg4 -c:v libx264 -r {0:s} -shortest".format(str(output_framerate))

        export_command = "ffmpeg -y {0:s} {1:s} -map 0:0 -map \"[audio]\" {2:s} {3:s}"
        export_command = export_command.format(video_inputs, audio_inputs, output_flags, video_output)

        # generate video using ffmpeg ....
        print("Saving data to: " + video_output)
        print(export_command)
        os.system(export_command)

        # delete temporary images
        print("Deleting Temporary Files")
        for idx in range(total_frames):
            os.remove(temporary_prefix + str(idx) + ".png")
