
from recognizer.preprocessing.inkml_loader import INKMLLoader

class LectureClassMapping:
    def __init__(self, lecture_mapping, unique_classes, classes_per_lecture):
        self.lecture_mapping = lecture_mapping
        self.unique_classes = unique_classes
        self.classes_per_lecture = classes_per_lecture

        self.mapping_sim_shape = None
        self.mapping_am_to_rec = None
        self.mapping_am_ignore = None
        self.mapping_rec_to_latex = None

    @staticmethod
    def load_ordinal_labels(ord_lbls_filename):
        # read normal raw labels ...
        raw_labels = INKMLLoader.load_label_replacement(ord_lbls_filename)

        # convert integer strings to int keys for usage with ord function (when possible)
        ord_labels = {}
        for key in raw_labels:
            try:
                # store key as ordinal (integer)
                new_key = int(key)
            except:
                # keep using as string
                new_key = key

            ord_labels[new_key] = raw_labels[key]

        return ord_labels