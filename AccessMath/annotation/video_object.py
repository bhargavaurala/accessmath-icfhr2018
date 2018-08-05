
from .video_object_location import VideoObjectLocation


class VideoObject:
    XMLNamespace = ''

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.locations = []

    def find_location_idx(self, frame):
        loc_min = 0
        loc_max = len(self.locations) - 1
        while loc_min <= loc_max:
            loc_mid = int((loc_min + loc_max) / 2.0)

            if self.locations[loc_mid].frame == frame:
                return loc_mid
            elif self.locations[loc_mid].frame < frame:
                loc_min = loc_mid + 1
            else:
                if loc_max == loc_mid:
                    break
                else:
                    loc_max = loc_mid

        return loc_min

    def set_location_at(self, frame, abs_time, visible, x, y, w, h):
        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (self.locations[loc_idx].frame != frame):
            # does not exist, create
            location = VideoObjectLocation(visible, frame, abs_time, x, y, w, h)
            # insert at desired idx ...
            self.locations.insert(loc_idx, location)

            # Key-frame was added
            return True
        else:
            # udpate existing ...
            location = self.locations[loc_idx]
            location.visible = visible
            location.x = x
            location.y = y
            location.w = w
            location.h = h

            # an existing Key-frame was updated
            return False

    def del_location_at(self, frame):
        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (self.locations[loc_idx].frame != frame):
            # does not exist
            return False
        else:
            # exists
            del self.locations[loc_idx]
            return True

    def get_location_at(self, frame, out_range):
        # out of range -> None if not extrapolate else
        if len(self.locations) == 0:
            raise Exception("Cannot estimate out of range, no existing locations")

        loc_idx = self.find_location_idx(frame)

        if (loc_idx >= len(self.locations)) or (loc_idx == 0 and self.locations[loc_idx].frame != frame):
            if not out_range:
                # out of range
                return None
            else:
                if loc_idx == 0:
                    # use the first
                    return self.locations[0]
                else:
                    # use the last
                    return self.locations[-1]
        else:
            # check if exact ...
            if self.locations[loc_idx].frame == frame:
                # exact match, no interpolation required ...
                return self.locations[loc_idx]
            else:
                # interpolate ...
                return VideoObjectLocation.interpolate(self.locations[loc_idx - 1], self.locations[loc_idx], frame)

    def toXML(self):
        result = "  <VideoObject>\n"
        result += "    <Id>" + self.id + "</Id>\n"
        result += "    <Name>" + self.name + "</Name>\n"
        result += "    <VideoObjectLocations>\n"
        for location in self.locations:
            result += location.toXML()
        result += "    </VideoObjectLocations>\n"

        result += "  </VideoObject>\n"

        return result

    @staticmethod
    def fromXML(root):
        # general properties
        object_id = root.find(VideoObject.XMLNamespace + 'Id').text
        object_name = root.find(VideoObject.XMLNamespace + 'Name').text
        video_object = VideoObject(object_id, object_name)

        # locations
        locations_root = root.find(VideoObject.XMLNamespace + 'VideoObjectLocations')
        locations_xml = locations_root.findall(VideoObject.XMLNamespace + 'VideoObjectLocation')
        
        for location_xml in locations_xml:
            video_object.locations.append(VideoObjectLocation.fromXML(location_xml))

        return video_object

