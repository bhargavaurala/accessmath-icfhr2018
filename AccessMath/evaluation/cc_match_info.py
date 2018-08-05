
class CCMatchInfo:
    def __init__(self, matches1=None, matches2=None):
        self.frame1_ccs_refs = []
        if matches1 is not None:
            if isinstance(matches1, list):
                self.frame1_ccs_refs += matches1
            else:
                self.frame1_ccs_refs.append(matches1)

        self.frame2_ccs_refs = []
        if matches2 is not None:
            if isinstance(matches2, list):
                self.frame2_ccs_refs += matches2
            else:
                self.frame2_ccs_refs.append(matches2)

    @staticmethod
    def Merge(match1, match2):
        frame1_ccs_refs = list(set.union(set(match1.frame1_ccs_refs), set(match2.frame1_ccs_refs)))
        frame2_ccs_refs = list(set.union(set(match1.frame2_ccs_refs), set(match2.frame2_ccs_refs)))

        return CCMatchInfo(frame1_ccs_refs, frame2_ccs_refs)


