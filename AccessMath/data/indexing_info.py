
import xml.etree.ElementTree as ET

class IndexingInfo:
    Namespace = ''

    def __init__(self, sym_rec_config_filename, symbolic_mir_config_filename, image_mir_config_filename):
        self.recognizer_config_filename = sym_rec_config_filename
        self.symbolic_mir_config_filename = symbolic_mir_config_filename
        self.image_mir_config_filename = image_mir_config_filename

        self.use_explict_junk = None
        self.implicit_junk_threshold = None

        self.mapping_sim_shape_filename = None
        self.mapping_notes_to_rec_filename = None
        self.mapping_to_ignore_notes_filename = None
        self.mapping_rec_to_latex_filename = None

        self.named_hw_classifiers_filename = None
        self.named_latex_classifiers_filename = None

        self.allow_default_recognizer = None

        self.hw_image_path = None
        self.hw_tanv_config_path = None
        self.latex_image_path = None
        self.latex_tanv_config_path = None

        self.visualization_server = None


    @staticmethod
    def from_XML_node(root):
        # general data

        # recognition-related parameters
        recognition_root = root.find(IndexingInfo.Namespace + "SymbolRecognition")

        # Find symbol recognizer name
        recognizer_filename = recognition_root.find(IndexingInfo.Namespace + "Recognizer").text

        explicit_junk_value = int(recognition_root.find(IndexingInfo.Namespace + "UseExplicitJunk").text)
        implicit_junk_threshold = float(recognition_root.find(IndexingInfo.Namespace + "ImplicitJunk").text)

        # existing mappings
        mappings_root = recognition_root.find(IndexingInfo.Namespace + "LabelCorrections")

        mapping_similar_shape = mappings_root.find(IndexingInfo.Namespace + "SimilarShape").text
        mapping_rec_to_latex = mappings_root.find(IndexingInfo.Namespace + "RecToLatex").text
        mapping_notes_to_rec = mappings_root.find(IndexingInfo.Namespace + "NotesToRec").text
        mapping_to_ignore_notes = mappings_root.find(IndexingInfo.Namespace + "NotesIgnore").text

        named_classifers_root = recognition_root.find(IndexingInfo.Namespace + "PerLecture")
        named_hw_classifiers = named_classifers_root.find(IndexingInfo.Namespace + "IndexHW").text
        named_latex_classifiers = named_classifers_root.find(IndexingInfo.Namespace + "IndexLaTeX").text

        # MIR parameters ...
        mir_root = root.find(IndexingInfo.Namespace + "MathInformationRetrieval")

        # find Tangent-S config file name
        symbolic_filename = mir_root.find(IndexingInfo.Namespace + "Symbolic").text
        # ... Tangent-V config file name
        image_based_filename = mir_root.find(IndexingInfo.Namespace + "ImageBased").text

        # Export to Tangent-V parameters ....
        export_root = root.find(IndexingInfo.Namespace + "ExportInfo")

        hw_image_path = export_root.find(IndexingInfo.Namespace + "ImagePathHW").text
        latex_image_path = export_root.find(IndexingInfo.Namespace + "ImagePathLaTeX").text
        hw_tanv_config_path = export_root.find(IndexingInfo.Namespace + "TangentVConfigHW").text
        latex_tanv_config_path = export_root.find(IndexingInfo.Namespace + "TangentVConfigLaTeX").text
        allow_default_recognizer = int(export_root.find(IndexingInfo.Namespace + "AllowDefaultRec").text) > 0

        visualization_server = root.find(IndexingInfo.Namespace + "VisualizationServer").text

        # create final object ...
        index_info = IndexingInfo(recognizer_filename, symbolic_filename, image_based_filename)
        # ... other recognition thresholds ....
        index_info.use_explict_junk = explicit_junk_value
        index_info.implicit_junk_threshold = implicit_junk_threshold
        # ... named classifiers ...
        index_info.named_hw_classifiers_filename = named_hw_classifiers
        index_info.named_latex_classifiers_filename = named_latex_classifiers
        # ... mappings ...
        index_info.mapping_sim_shape_filename = mapping_similar_shape
        index_info.mapping_rec_to_latex_filename = mapping_rec_to_latex
        index_info.mapping_notes_to_rec_filename = mapping_notes_to_rec
        index_info.mapping_to_ignore_notes_filename = mapping_to_ignore_notes
        # ... final export info ....
        index_info.allow_default_recognizer = allow_default_recognizer
        index_info.hw_image_path = hw_image_path
        index_info.hw_tanv_config_path = hw_tanv_config_path
        index_info.latex_image_path = latex_image_path
        index_info.latex_tanv_config_path = latex_tanv_config_path

        # ... for HTML visualization (navigation)
        index_info.visualization_server = visualization_server

        return index_info



