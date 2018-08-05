
import xml.etree.ElementTree as ET

from AccessMath.data.meta_data_DB import MetaDataDB

# interface with Tangent Formula Retrieval system
from TangentS.utility.control import Control
from TangentS.math.math_document import MathDocument
from TangentS.math.math_extractor import MathExtractor
from TangentS.math.symbol_tree import SymbolTree
from TangentS.math.layout_symbol import LayoutSymbol

class TangentS_Helper:
    @staticmethod
    def collect_tree_tags(root):
        tags = [root.tag]

        for rel, node in root.active_children():
            tags += TangentS_Helper.collect_tree_tags(node)

        return tags

    @staticmethod
    def get_unique_symbols(slts_strs, rec_labels, accessmath_labels, ignore_labels):
        # step 1: collect all tags from all nodes from all trees in a list (from a single document)
        unique_tags = []
        for slt_str in slts_strs:
            slt = SymbolTree.parse_from_slt(slt_str)

            all_tags = TangentS_Helper.collect_tree_tags(slt.root)
            unique_tags += all_tags

        # step 2: filter unique tags
        unique_tags = list(set(unique_tags))

        # Step 3: map each unique tag to a list of symbols
        all_symbols = []
        for tag in unique_tags:
            if "!" in tag and len(tag) >= 2:
                # check for special cases ...
                if tag[0] == "T":
                    # text ... treat each character as a symbol ..
                    all_symbols += [val for val in tag[2:]]
                elif tag[0] == "M":
                    # matrices ...
                    # restore special symbols ...
                    tag = tag[2:]
                    tag = tag.replace("&lsqb;", "[")
                    tag = tag.replace("&rsqb;", "]")

                    # find and ignore size ...
                    # note that we don't know how many digits wil be in # row and # of columns
                    size_middle = tag.find("x")
                    size_start = size_middle - 1
                    while size_start >= 1 and "0" <= tag[size_start - 1] <= "9":
                        size_start -= 1

                    # only if brackets were specified .. .
                    if size_start > 0:
                        brackets = tag[:size_start]

                        # add brackets ...
                        all_symbols.append(brackets[0])
                        all_symbols.append(brackets[1])
                elif tag[0] == "N":
                    # separate by digits
                    all_symbols += [val for val in tag[2:]]
                elif tag[0] == "W":
                    # whitespace ... ignore!
                    pass
                elif tag[0] == "V":
                    # variable names, separate by characters ...
                    all_symbols += [val for val in tag[2:]]
                else:
                    all_symbols.append(tag[2:])
            else:
                # operators in SLT's (not yet mapped to a Subtype (O! or U!)
                # split per element ...
                all_symbols += [val for val in tag]

            if "" in all_symbols:
                print("FOUND!")
                print(tag)
                print(all_symbols)
                raise Exception("Empty node tag found!")

        # Step 4: get unique (Again) ....
        all_symbols = list(set(all_symbols))

        # Step 5: check for unexpected symbol classes, map those known to be different
        accepted_symbols = []
        rejected_symbols = []
        for idx, symbol in enumerate(all_symbols):
            # check if label should be ignored ... (avoid showing classes that are known to be absent from CROHME data)
            if len(symbol) == 1 and ord(symbol) in ignore_labels:
                continue
            elif symbol in ignore_labels:
                continue

            # check if on mapping ...
            if len(symbol) == 1 and ord(symbol) in accessmath_labels:
                # map from unicode ordinal
                symbol = accessmath_labels[ord(symbol)]
            elif symbol in accessmath_labels:
                # map from string
                symbol = accessmath_labels[symbol]

            if symbol not in rec_labels:
                rejected_symbols.append(symbol)
            else:
                accepted_symbols.append(symbol)

        # Step 6: after mapping again, some classes might be repeated ... make unique (one more time)
        accepted_symbols = list(set(accepted_symbols))

        print("-> Not recognizable symbols found: {0:d}".format(len(rejected_symbols)))
        #print(rejected_symbols)
        for symbol in rejected_symbols:
            print("-->" + str((ord(symbol[0]), symbol)))

        print("-> Final unique symbols: {0:d}".format(len(accepted_symbols)))

        return accepted_symbols, rejected_symbols

    @staticmethod
    def map_lectures_to_doc_ids(database, tan_control):
        math_doc = MathDocument(tan_control)

        assert isinstance(database, MetaDataDB)
        assert isinstance(math_doc, MathDocument)

        map_file = open(math_doc.doc_list, encoding='utf-8')
        all_transcript_files = map_file.readlines()
        map_file.close()

        mapping = {}
        for idx, transcript_filename in enumerate(all_transcript_files):
            parts = transcript_filename.strip().split("/")
            # assume they are stored using "lecture-title/lecture-title.html"
            lecture_title = parts[-2]

            db_lecture = database.get_lecture(lecture_title)
            if db_lecture is not None:
                mapping[idx] = db_lecture
            else:
                print("Could not find lecture " + lecture_title + " in database")

        return mapping

    @staticmethod
    def load_document_latex_expressions(tan_control, doc_id):
        math_doc = MathDocument(tan_control)

        ext, content = math_doc.read_doc_file(math_doc.find_doc_file(doc_id))
        maths = MathExtractor.math_tokens(content)

        all_latex = []
        total_errors = 0
        for mathml in maths:
            try:
                root = ET.fromstring(mathml)
                latex_str = root.attrib['alttext']
            except:
                print("\nError parsing:")
                print(mathml)
                print("")
                total_errors += 1

                latex_str = "{~}"

            all_latex.append(latex_str)

        return all_latex, total_errors

    @staticmethod
    def load_index_expressions(tan_control, index_dir="db-index"):
        isinstance(tan_control, Control)

        exp_index = {}
        index_ids = tan_control.read("index_fileids")[1:-1].split(",")
        db_name = tan_control.read("database")

        for index_id in index_ids:
            index_filename = index_dir + "/" + db_name + "_i_" + index_id.strip() + ".tsv"
            with open(index_filename, "r", encoding="utf-8") as index_file:
                index_lines = index_file.readlines()

                last_doc_id = None
                for line in index_lines:
                    parts = line.strip().split("\t")
                    if len(parts) == 2 and parts[0] == "D":
                        last_doc_id = int(parts[1])
                        exp_index[last_doc_id] = []
                    elif len(parts) == 3 and parts[0] == "E":
                        new_exp = parts[1]
                        exp_index[last_doc_id].append(new_exp)

        return exp_index
