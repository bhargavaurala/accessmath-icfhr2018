

class VideoObjectLocation:
    XMLNamespace = ''

    def __init__(self, visible,frame,abs_time, x=None, y=None, w=None, h=None):
        self.visible = visible
        self.frame = frame
        self.abs_time = abs_time
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return "({0},{1},{2},{3})".format(self.x, self.y, self.w, self.h)

    def intersects(self, other):
        return ((self.x < other.x + other.w and other.x < self.x + self.w) and
                (self.y < other.y + other.h and other.y < self.y + self.h))

    def area(self):
        return self.w * self.h

    def intersection_area(self, other):
        int_min_x = max(self.x, other.x)
        int_max_x = min(self.x + self.w, other.x + other.w)

        int_min_y = max(self.y, other.y)
        int_max_y = min(self.y + self.h, other.y + other.h)

        int_w = int_max_x - int_min_x + 1
        int_h = int_max_y - int_min_y + 1

        if int_w <= 0.0 or int_h <= 0.0:
            return 0.0
        else:
            return int_w * int_h

    def intersection_percentage(self, other):
        local_area = self.area()
        int_area = self.intersection_area(other)

        return int_area / local_area

    def IOU(self, other):
        local_area = self.area()
        other_area = other.area()
        int_area = self.intersection_area(other)
        union_area = local_area + other_area - int_area

        return int_area / union_area


    def get_XYXY_box(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    @staticmethod
    def fromLocation(original):
        return VideoObjectLocation(original.visible, original.frame, original.abs_time,
                                   original.x, original.y, original.w, original.h)

    @staticmethod
    def interpolate(location1, location2, frame):
        assert location1.frame < location2.frame

        if frame <= location1.frame:
            return location1

        if frame >= location2.frame:
            return location2

        # interpolation weight ...
        interval = location2.frame - location1.frame
        w = (frame - location1.frame) / float(interval)
        new_abs_time = location1.abs_time * (1.0 - w) + location2.abs_time * w

        result = VideoObjectLocation(location1.visible, frame, new_abs_time)

        # interpolate the location ...
        result.x = location1.x * (1.0 - w) + location2.x * w
        result.y = location1.y * (1.0 - w) + location2.y * w
        result.w = location1.w * (1.0 - w) + location2.w * w
        result.h = location1.h * (1.0 - w) + location2.h * w

        return result

    def toXML(self):
        result = "<VideoObjectLocation>\n"
        result += "  <Visible>" + ("1" if self.visible else "0") + "</Visible>\n"
        result += "  <Frame>" + str(self.frame) + "</Frame>\n"
        result += "  <AbsTime>" + str(self.abs_time) + "</AbsTime>\n"
        result += "  <X>" + str(self.x) + "</X>\n"
        result += "  <Y>" + str(self.y) + "</Y>\n"
        result += "  <W>" + str(self.w) + "</W>\n"
        result += "  <H>" + str(self.h) + "</H>\n"
        result += "</VideoObjectLocation>\n"

        return result

    @staticmethod
    def fromXML(root):
        visible = int(root.find(VideoObjectLocation.XMLNamespace + 'Visible').text) > 0
        frame = int(root.find(VideoObjectLocation.XMLNamespace + 'Frame').text)
        abs_time = float(root.find(VideoObjectLocation.XMLNamespace + 'AbsTime').text)
        x = float(root.find(VideoObjectLocation.XMLNamespace + 'X').text)
        y = float(root.find(VideoObjectLocation.XMLNamespace + 'Y').text)
        w = float(root.find(VideoObjectLocation.XMLNamespace + 'W').text)
        h = float(root.find(VideoObjectLocation.XMLNamespace + 'H').text)

        return VideoObjectLocation(visible, frame, abs_time, x, y, w, h)
