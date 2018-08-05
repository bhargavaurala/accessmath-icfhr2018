
import xml.etree.ElementTree as ET
from .lecture_info import LectureInfo
from .indexing_info import IndexingInfo

class MetaDataDB:
    Namespace = ''

    def __init__(self, name):
        self.name = name

        self.output_temporal = ""
        self.output_preprocessed = ""
        self.output_indices = ""
        self.output_images = ""
        self.output_videos = ""
        self.output_annotations = ""
        self.output_summaries = ""
        self.output_search_results = ""

        self.lectures = []
        self.datasets = {}

        self.indexing = None


    @staticmethod
    def from_XML_node(root):
        # read the most general metadata
        data = root.find(MetaDataDB.Namespace + 'DataBase')
        name = data.find(MetaDataDB.Namespace + 'Name').text

        outputs = data.find(MetaDataDB.Namespace + 'OutputPaths')
        output_temporal = outputs.find(MetaDataDB.Namespace + 'Temporal').text
        output_preprocessed = outputs.find(MetaDataDB.Namespace + 'Preprocessed').text
        output_indices = outputs.find(MetaDataDB.Namespace + 'Indices').text
        output_images = outputs.find(MetaDataDB.Namespace + 'Images').text
        output_videos = outputs.find(MetaDataDB.Namespace + 'Videos').text
        output_annotations = outputs.find(MetaDataDB.Namespace + 'Annotations').text
        output_summaries = outputs.find(MetaDataDB.Namespace + 'Summaries').text
        output_search_results = outputs.find(MetaDataDB.Namespace + 'SearchResults').text

        # create DB object
        db = MetaDataDB(name)
        db.output_temporal = output_temporal
        db.output_preprocessed = output_preprocessed
        db.output_indices = output_indices
        db.output_images = output_images
        db.output_videos = output_videos
        db.output_annotations = output_annotations
        db.output_summaries = output_summaries
        db.output_search_results = output_search_results

        # now, read every lecture info in Database
        lectures = data.find(MetaDataDB.Namespace + "Lectures")
        for lecture_node in lectures.findall(MetaDataDB.Namespace + "Lecture"):
            # ... read XML
            lecture = LectureInfo.from_XML_node(lecture_node)
            # .... add
            db.lectures.append(lecture)

        # check if datasets are defined ...
        datasets = data.find(MetaDataDB.Namespace + 'Datasets')

        for node in datasets:
            dataset_name = node.tag.lower()

            ds_lectures = []
            lecture_titles_xml = node.findall(MetaDataDB.Namespace + 'LectureTitle')
            for xml_title in lecture_titles_xml:
                lecture = db.get_lecture(xml_title.text)
                ds_lectures.append(lecture)

            db.datasets[dataset_name] = ds_lectures

        # check for indexing information ...
        indexing_root = data.find(MetaDataDB.Namespace + 'LectureIndexing')
        if indexing_root:
            db.indexing = IndexingInfo.from_XML_node(indexing_root)

        return db

    def get_lecture(self, title):
        title = title.lower()
        current_lecture = None
        for lecture in self.lectures:
            if lecture.title.lower() == title:
                current_lecture = lecture
                break

        return current_lecture

    def get_dataset(self, name):
        key = name.lower()
        if key in self.datasets:
            return self.datasets[key]
        else:
            return None

    @staticmethod
    def from_file(filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        return MetaDataDB.from_XML_node(root)

    @staticmethod
    def load_database_lecture(database_filename, lecture_name):
        try:
            database = MetaDataDB.from_file(database_filename)
        except:
            print("Invalid database file")
            return None, None

        current_lecture = database.get_lecture(lecture_name)

        if current_lecture is None:
            print("Lecture not found in database")
            print("Available lectures:")
            for lecture in database.lectures:
                print(lecture.title)
            return None, None

        return database, current_lecture