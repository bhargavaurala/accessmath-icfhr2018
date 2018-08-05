
import xml.etree.ElementTree as ET
from .video_object import VideoObject

class FormulaCCs:
    def __init__(self, start_group, latex_tag=""):
        if isinstance(start_group, list):
            self.groups_refs = start_group
        else:
            self.groups_refs = [start_group]

        self.latex_tag = latex_tag

        self.first_frame = None
        self.last_frame = None
        self.first_visible = None
        self.last_visible = None

        self.__update_frames()

    def strID(self):
        return "/".join([group.strID() for group in self.groups_refs])

    def add_group(self, new_group):
        if not new_group in self.groups_refs:
            self.groups_refs.append(new_group)
            self.__update_frames()

    def remove_group(self, to_remove):
        if to_remove in self.groups_refs:
            self.groups_refs.remove(to_remove)
            self.__update_frames()

    def visible_at(self, frame):
        return self.first_visible <= frame <= self.last_visible

    def getBoundingBox(self):
        all_l, all_r, all_b, all_t = [], [], [], []

        for group in self.groups_refs:
            for cc in group.cc_refs:
                all_l.append(cc.min_x)
                all_r.append(cc.max_x)
                all_t.append(cc.min_y)
                all_b.append(cc.max_y)

        min_x = min(all_l)
        max_x = max(all_r)
        min_y = min(all_t)
        max_y = max(all_b)

        return min_x, max_x, min_y, max_y


    def __eq__(self, other):
        if not isinstance(other, FormulaCCs):
            return False
        else:
            return self.groups_refs == other.groups_refs

    def __lt__(self, other):
        if isinstance(other, FormulaCCs):
            if self.first_visible < other.first_visible:
                return True
            elif self.first_visible > other.first_visible:
                return False
            else:
                l_min_x, l_max_x, l_min_y, l_max_y = self.getBoundingBox()
                o_min_x, o_max_x, o_min_y, o_max_y = other.getBoundingBox()

                if l_min_y < o_min_y:
                    return True
                elif l_min_y > o_max_y:
                    return False
                else:
                    return l_min_x < o_min_x
        else:
            raise Exception("Cannot compare FormulaCC to " + str(type(other)))

    def __update_frames(self):
        all_firsts = [group.start_frame for group in self.groups_refs]
        all_lasts = [group.lastFrame() for group in self.groups_refs]

        self.first_frame = min(all_firsts)
        self.last_frame = max(all_lasts)

        first_visible = max(all_firsts)
        last_visible = min(all_lasts)

        if first_visible > last_visible:
            self.first_visible = None
            self.last_visible = None
        else:
            self.first_visible = first_visible
            self.last_visible = last_visible

    @staticmethod
    def GenerateFormulaXML(formulas):
        xml_str = "<FormulaCCS>\n"

        # add the basic information (assume Frame and Group data is stored elsewhere)
        for formula in formulas:
            assert isinstance(formula, FormulaCCs)

            xml_str += "    <Formula>\n"
            xml_str += "        <LatexTag>" + formula.latex_tag + "</LatexTag>\n"
            xml_str += "        <FirstFrame>" + str(formula.first_frame) + "</FirstFrame>\n"
            xml_str += "        <LastFrame>" + str(formula.last_frame) + "</LastFrame>\n"
            xml_str += "        <FirstVisible>" + str(formula.first_visible) + "</FirstVisible>\n"
            xml_str += "        <LastVisible>" + str(formula.last_visible) + "</LastVisible>\n"
            xml_str += "        <CCGroups>\n"
            for group in formula.groups_refs:
                xml_str += "            <CCGroup>" + group.strID() + "</CCGroup>\n"
            xml_str += "        </CCGroups>\n"
            xml_str += "    </Formula>\n"

        xml_str += "</FormulaCCS>\n"

        return xml_str

    @staticmethod
    def FormulasFromXML(unique_groups, xml_filename):
        # returns the formulas
        # requires the set of groups of representing UniqueCC

        groups_by_id = {group.strID(): group for group in unique_groups}

        tree = ET.parse(xml_filename)
        root = tree.getroot()

        # load file!
        namespace = VideoObject.XMLNamespace
        formulas_xml_roots = root.findall(namespace + 'Formula')

        loaded_formulas = []
        for fr_idx, xml_formula in enumerate(formulas_xml_roots):
            latex_tag = xml_formula.find(namespace + "LatexTag").text.strip()

            groups_root = xml_formula.find(namespace + 'CCGroups')
            group_roots = groups_root.findall(namespace + "CCGroup")

            current_groups = []
            for group_root in group_roots:
                groupID = group_root.text.strip()
                if not groupID in groups_by_id:
                    print("Warning: Could not load equation #{0:d}. Groups have changed".format(fr_idx + 1))
                    continue
                else:
                    current_groups.append(groups_by_id[groupID])

            loaded_formulas.append(FormulaCCs(current_groups, latex_tag))

        return sorted(loaded_formulas)

