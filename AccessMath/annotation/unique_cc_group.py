
import xml.etree.ElementTree as ET
from .video_object import VideoObject

class UniqueCCGroup:
    def __init__(self, start_cc, start_frame):
        # ending frame can be computed by adding length of cc_refs to start minus 1
        # as all CCs must appear in contiguous frames.
        self.cc_refs = [start_cc]
        self.start_frame = start_frame

    def lastFrame(self):
        return self.start_frame + len(self.cc_refs) - 1

    def strID(self):
        return str(self.start_frame) + "-" + self.cc_refs[0].strID()

    def __eq__(self, other):
        if not isinstance(other, UniqueCCGroup):
            return False
        else:
            return self.cc_refs == other.cc_refs

    @staticmethod
    def GroupsFromXML(keyframes, xml_filename):
        # returns the groups and the inverted index ...
        # also, input set of key-frames for consistency validation!

        # Initially, build inverted indexes for groups and key-frames CCs
        unique_groups = []
        cc_group = []
        cc_index = []
        for keyframe in keyframes:
            group_dict = {}
            index_dict = {}
            for cc in keyframe.binary_cc:
                cc_id = cc.strID()
                group_dict[cc_id] = None
                index_dict[cc_id] = cc

            cc_group.append(group_dict)
            cc_index.append(index_dict)

        ids_added = [[] for keyframe in keyframes]
        ids_removed = [[] for keyframe in keyframes]

        tree = ET.parse(xml_filename)
        root = tree.getroot()

        # load file!
        namespace = VideoObject.XMLNamespace
        keyframes_root = root.find(namespace + 'KeyFrames')
        keyframes_xml_roots = keyframes_root.findall(namespace + 'KeyFrame')

        ids_file = [{} for keyframe in keyframes]
        for kf_idx, xml_keyframe in enumerate(keyframes_xml_roots):
            ccs_root = xml_keyframe.find(namespace + 'CCs')
            cc_roots = ccs_root.findall(namespace + 'CC')

            for xml_cc in cc_roots:
                cc_id = xml_cc.text.strip()
                ids_file[kf_idx][cc_id] = True

                if not cc_id in cc_index[kf_idx]:
                    print("Key-frame # " + str(keyframes[kf_idx].idx) + ", missing CC {" + cc_id + "}")
                    ids_removed[kf_idx].append(cc_id)

            # now, check new CC's ...
            for kf_cc_ids in cc_index[kf_idx]:
                if not kf_cc_ids in ids_file[kf_idx]:
                    print("Key-frame # " + str(keyframes[kf_idx].idx) + ", Added CC {" + kf_cc_ids + "}")
                    ids_added[kf_idx].append(kf_cc_ids)

        print("Total Missing: " + str(sum([len(ccs_missing) for ccs_missing in ids_removed])))
        print("Total Added: " + str(sum([len(ccs_added) for ccs_added in ids_added])))

        groups_root = root.find(namespace + 'CCGroups')
        groups_xml_roots = groups_root.findall(namespace + 'CCGroup')
        for group_xml in groups_xml_roots:
            group_start = int(group_xml.find(namespace + "Start").text.strip())

            group_ccs_root = group_xml.find(namespace + "CCs")
            group_cc_xml_roots = group_ccs_root.findall(namespace + "CC")
            valid_group_ids = []
            for kf_offset, group_cc_xml in enumerate(group_cc_xml_roots):
                cc_id = group_cc_xml.text.strip()

                if cc_id in cc_group[group_start + kf_offset]:
                    valid_group_ids.append(cc_id)
                else:
                    # mismatch found, stop loading group
                    break

            if len(valid_group_ids) > 0:
                # create group and link with valid members ...
                first_id = valid_group_ids[0]
                new_group = UniqueCCGroup(cc_index[group_start][first_id], group_start)
                # first member
                cc_group[group_start][first_id] = new_group

                # The rest of the members
                for kf_offset in range(1, len(valid_group_ids)):
                    # add to the group
                    new_group.cc_refs.append(cc_index[group_start + kf_offset][valid_group_ids[kf_offset]])
                    # link to the group
                    cc_group[group_start + kf_offset][valid_group_ids[kf_offset]] = new_group

                # add to the general set
                unique_groups.append(new_group)

        # find CC without groups
        for kf_idx in range(len(keyframes)):
            for cc_id in cc_group[kf_idx]:
                if cc_group[kf_idx][cc_id] is None:
                    print("Will create group for new CC {" + cc_id + "} on Keyframe # " + str(keyframes[kf_idx].idx))

                    # creating ...
                    new_group = UniqueCCGroup(cc_index[kf_idx][cc_id], kf_idx)
                    # add link ...
                    cc_group[kf_idx][cc_id] = new_group
                    # add group
                    unique_groups.append(new_group)

        print("Loaded: " + str(len(unique_groups)) + " CC groups (Unique CC)")

        return cc_group, unique_groups

    @staticmethod
    def GenerateGroupsXML(keyframes, groups):
        xml_str = "<UniqueCCS>\n"

        # first add the complete set of ccs per key-frames, using ID
        xml_str += "  <KeyFrames>\n"
        for keyframe in keyframes:
            kf_xml = "    <KeyFrame>\n"
            kf_xml += "      <CCs>\n"
            for cc in keyframe.binary_cc:
                kf_xml += "         <CC>" + cc.strID() + "</CC>\n"
            kf_xml += "      </CCs>\n"
            kf_xml += "    </KeyFrame>\n"
            xml_str += kf_xml
        xml_str += "  </KeyFrames>\n"

        # Then, add the group information
        xml_str += "  <CCGroups>\n"
        for group in groups:
            xml_str += "    <CCGroup>\n"
            xml_str += "        <Start>" + str(group.start_frame) + "</Start>\n"
            xml_str += "        <End>" + str(group.start_frame + len(group.cc_refs) - 1) + "</End>\n"
            xml_str += "        <CCs>\n"
            for cc in group.cc_refs:
                xml_str += "          <CC>" + cc.strID() + "</CC>\n"
            xml_str += "        </CCs>\n"
            xml_str += "    </CCGroup>\n"
        xml_str += "  </CCGroups>\n"

        xml_str += "</UniqueCCS>\n"

        return xml_str

    @staticmethod
    def Copy(original):
        return UniqueCCGroup(list(original.cc_refs), original.start_frame)

    @staticmethod
    def Split(original, split_frame):

        offset_split = split_frame - original.start_frame

        if offset_split <= 0 or offset_split >= len(original.cc_refs):
            # nothing to split ...
            return None
        else:
            # create new Unique CC group ...
            new_group = UniqueCCGroup(original.cc_refs[offset_split], split_frame)
            original_len = len(original.cc_refs)
            del original.cc_refs[offset_split]

            # add copy remaining elements ...
            for offset_idx in range(offset_split + 1, original_len):
                new_group.cc_refs.append(original.cc_refs[offset_split])
                del original.cc_refs[offset_split]

            return new_group

