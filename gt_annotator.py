
import sys
import pygame
import traceback
import json

from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.annotation.gt_content_annotator import GTContentAnnotator

def get_video_files(argvs):
    try:
        database = MetaDataDB.from_file(argvs[1])
    except:
        print("Invalid database file")
        return None, None, None

    # now search for specified lecture
    lecture_name = argvs[2].lower()

    current_lecture = None
    for lecture in database.lectures:
        if lecture.title.lower() == lecture_name:
            current_lecture = lecture
            break

    if current_lecture is None:
        print("Lecture not found in database")
        print("Available lectures:")
        for lecture in database.lectures:
            print(lecture.title)
        return None, None, None

    m_videos = [video["path"] for video in current_lecture.main_videos]

    return m_videos, database, current_lecture

def main():
    if len(sys.argv) < 3:
        print("Usage: python gt_annotator.py database lecture [metrics]")
        print("Where")
        print("\tdatabase\t= Database metadata file")
        print("\tlecture\t\t= Lecture video to process")
        print("\tmetrics\t\t= Optional, use pre-computed video metrics")
        print("")
        return

    m_videos, database, current_lecture = get_video_files(sys.argv)
    if m_videos is None:
        return

    if len(sys.argv) >= 4:
        metrics_filename = sys.argv[3]
        with open(metrics_filename, "r") as in_file:
            video_metrics = json.load(in_file)

        if current_lecture.title in video_metrics:
            lecture_metrics = video_metrics[current_lecture.title]
        else:
            print("The video metrics file does not contain information about current lecture")
            return
    else:
        lecture_metrics = None

    output_prefix = database.output_annotations + "/" + database.name + "_" + current_lecture.title.lower()

    pygame.init()
    pygame.display.set_caption('Access Math - Ground Truth Annotation Tool - ' + database.name + "/" + current_lecture.title)
    screen_w = 1500
    screen_h = 900
    window = pygame.display.set_mode((screen_w, screen_h))
    background = pygame.Surface(window.get_size())
    background = background.convert()

    if "forced_width" in current_lecture.parameters:
        forced_res = (current_lecture.parameters["forced_width"], current_lecture.parameters["forced_height"])
        print("Video Resolution will be forced to : " + str(forced_res))
    else:
        forced_res = None

    main_menu = GTContentAnnotator(window.get_size(), m_videos, database.name, current_lecture.title, output_prefix, forced_res)

    if lecture_metrics is not None:
        main_menu.player.video_player.update_video_metrics(lecture_metrics)

    current_screen = main_menu
    current_screen.prepare_screen()
    prev_screen = None

    while not current_screen is None:
        #detect when the screen changes...
        if current_screen != prev_screen:
            #remember last screen...
            prev_screen = current_screen

        #capture events...
        current_events = pygame.event.get()
        try:
            current_screen = current_screen.handle_events(current_events)
        except Exception as e:
            print("An exception ocurred")
            print(e)
            traceback.print_exc()

        if current_screen != prev_screen:
            if current_screen != None:
                #prepare the screen for new display ...
                current_screen.prepare_screen()

        #draw....
        background.fill((0, 0, 0))

        if not current_screen is None:
            current_screen.render(background)

        window.blit(background, (0, 0))
        pygame.display.flip()


if __name__ == "__main__":
    main()