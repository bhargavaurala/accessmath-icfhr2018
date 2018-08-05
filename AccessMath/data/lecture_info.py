import xml.etree.ElementTree as ET

class LectureInfo:
    Namespace = ''

    def __init__(self, id, title):
        self.id = id
        self.title = title

        self.parameters = {}

        self.main_videos = []
        self.aux_videos = []

        self.main_audio = []
        self.aux_audio = []

    @staticmethod
    def metadata_from_XML(root):
        matadata = {}

        for child in root:
            matadata[child.tag.lower()] = child.text

        return matadata

    @staticmethod
    def from_XML_node(root):
        # general data
        # (ID doesn't need to be an integer now)
        id = root.find(LectureInfo.Namespace + "Id").text
        title = root.find(LectureInfo.Namespace + "Title").text

        parameters = root.find(LectureInfo.Namespace + "Parameters")

        # create object
        info = LectureInfo(id, title)

        if parameters is not None:
            # sync window
            node_sync_window = parameters.find(LectureInfo.Namespace + "SyncWindow")
            if node_sync_window is not None:
                try:
                    sync_window = float(node_sync_window.text)
                    info.parameters["sync_window"] = sync_window
                except:
                    print("Invalid Sync Window parameter found")

            # force video resolution?
            force_resolution = parameters.find(LectureInfo.Namespace + "ForceResolution")
            if force_resolution is not None:
                try:
                    forced_width = int(force_resolution.find(LectureInfo.Namespace + "Width").text)
                    forced_height = int(force_resolution.find(LectureInfo.Namespace + "Height").text)

                    info.parameters["forced_width"] = forced_width
                    info.parameters["forced_height"] = forced_height
                except Exception as e:
                    print(e)
                    print("Invalid forced resolution parameter found")

            # custom binarization method
            node_binarization = parameters.find(LectureInfo.Namespace + "Binarization")
            if node_binarization is not None:
                try:
                    binarization = int(node_binarization.text)
                    info.parameters["binarization"] = binarization
                except:
                    print("Invalid binarization parameter found")

        # now, read video meta-data
        videos = root.find(LectureInfo.Namespace + "Videos")
        main_videos = videos.find(LectureInfo.Namespace + "Main")
        auxiliary_videos = videos.find(LectureInfo.Namespace + "Auxiliary")

        # add the main videos (mandatory)
        for video_root in main_videos:
            video = LectureInfo.metadata_from_XML(video_root)
            info.main_videos.append(video)

        # add the auxiliary videos (optional)
        if not  auxiliary_videos is None:
            for video_root in auxiliary_videos:
                video = LectureInfo.metadata_from_XML(video_root)
                info.aux_videos.append(video)

        # finally, read the audio meta-data
        audios = root.find(LectureInfo.Namespace + "AudioStreams")
        if audios is not None:
            main_audios = audios.find(LectureInfo.Namespace + "Main")
            auxiliary_audios = audios.find(LectureInfo.Namespace + "Auxiliary")
        else:
            main_audios = None
            auxiliary_audios = None

        # add the main audio (optional)
        if main_audios is not None:
            for audio_root in main_audios:
                audio = LectureInfo.metadata_from_XML(audio_root)
                info.main_audio.append(audio)

        # add the auxiliary audio (optional)
        if auxiliary_audios is not None:
            for audio_root in auxiliary_audios:
                audio = LectureInfo.metadata_from_XML(audio_root)
                info.aux_audio.append(audio)

        return info