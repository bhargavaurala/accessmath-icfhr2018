import itertools

class Interval:
    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.data = data

    def __eq__(self, other):
        return (self.start == other.start) and (self.end == other.end) and (self.data == other.data)

    def __str__(self):
        return "<Interval(" + str(self.start) + "," + str(self.end) + "," + str(self.data) + ">"

class IntervalIndex:
    def __init__(self, only_data=False):
        self.intervals = {}
        self.only_data = only_data

    def add(self, start, end, data):
        if self.only_data:
            to_add = data
        else:
            to_add = Interval(start, end, data)

        while len(self.intervals) < end + 1:
            self.intervals[len(self.intervals)] = {}

        if end in self.intervals[start]:
            self.intervals[start][end].append(to_add)
        else:
            self.intervals[start][end] = [to_add]

    def remove(self, start, end, data):
        if self.only_data:
            to_remove = data
        else:
            to_remove = Interval(start, end, data)

        self.intervals[start][end].remove(to_remove)

    def find_matches(self, other):
        matches = []
        local_valid = {}
        other_valid = {}
        for local_pos in range(len(self.intervals)):
            # update valid, remove intervals which do not cover pos ...
            if local_pos in local_valid:
                del local_valid[local_pos]
            #print str((len(local_valid), len(other_valid))),

            # do the same for the other ...
            if local_pos in other_valid:
                del other_valid[local_pos]

            # now add new intervals at this position...
            local_new = self.intervals[local_pos]

            # now the new intervals a this position ...
            if local_pos < len(other.intervals):
                other_new = other.intervals[local_pos]
            else:
                break

            # Add new matches ....,

            # 0) create iterators ...
            local_valid_it = itertools.chain(*local_valid.values())
            other_valid_it = itertools.chain(*other_valid.values())
            local_new_it = list(itertools.chain(*local_new.values()))
            other_new_it = list(itertools.chain(*other_new.values()))

            # 1) the old local with the new from second
            matches += list(itertools.product(local_valid_it, other_new_it))
            # 2) the old second with the new from local
            matches += list(itertools.product(local_new_it, other_valid_it))
            # 3) the new local with the new from second
            matches += list(itertools.product(local_new_it, other_new_it))

            # Merge ...
            for local_end in local_new:
                if not local_end in local_valid:
                    local_valid[local_end] = []

                for local_int in local_new[local_end]:
                    local_valid[local_end].append(local_int)
                #local_valid[local_end] += local_new[local_end]

            for other_end in other_new:
                if not other_end in other_valid:
                    other_valid[other_end] = []

                for other_int in other_new[other_end]:
                    other_valid[other_end].append(other_int)
                #other_valid[other_end] += other_new[other_end]

            #print(str((local_pos, local_valid)))

        return matches