
#===================================================================
# Miscelaneous Rutines
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013, 2015
#===================================================================

import pickle

class MiscHelper:

    #=====================================================
    #  Miscellaneous functions
    #=====================================================
    @staticmethod
    def optional_parameters(params, offset):
        pos = offset
        result = {}

        while pos < len(params):
            if params[pos][0] == "-":
                key = params[pos][1:]

                if pos + 1 < len(params):
                    parts = params[pos + 1].split()

                    if len(parts) == 1:
                        result[key] = params[pos + 1]
                    else:
                        result[key] = parts
                else:
                    print("No value specified for parameter: " + key)

                pos += 2
            else:
                print("Unexpected parameter: " + params[pos])
                pos += 1

        return result


    #=============================================================
    #   Auxiliar function that finds intervals on boolean arrays
    #=============================================================
    @staticmethod
    def findBooleanIntervals( boolean_list, value):
        intervals = []
        on_interval = False
        int_ini = -1
        int_end = -1

        for idx, element in enumerate(boolean_list):
            if element == value:
                if not on_interval:
                    on_interval = True
                    int_ini = idx
                int_end = idx
            else:
                if on_interval:
                    on_interval = False
                    intervals.append( (int_ini, int_end) )

        if on_interval:
            intervals.append( (int_ini, int_end) )

        return intervals

    #==========================================================
    # Auxiliary function that finds the mid points in an array
    # of intervals
    #==========================================================
    @staticmethod
    def intervalMidPoints( intervals ):
        midpoints = []
        for init, end in intervals:
            m = int((end + init) / 2.0)

            midpoints.append( m )

        return midpoints

    #=========================================================
    # Auxiliar function that scales a list of values
    #=========================================================
    @staticmethod
    def scaleValues( values, cur_min, cur_max, new_min, new_max ):
        new_values = []
        for value in values:
            percent = (value - cur_min) / float(cur_max - cur_min)

            if percent < 0.0:
                percent = 0.0
            if percent > 1.0:
                percent = 1.0

            new_val = percent * (new_max - new_min) + new_min

            new_values.append( new_val )

        return new_values

    #=========================================================
    # Finds the average boundaries for a given list of boxes
    #=========================================================
    @staticmethod
    def averageBoxes(box_list):
        total_min_x = 0.0
        total_max_x = 0.0
        total_min_y = 0.0
        total_max_y = 0.0

        for box in box_list:
            #unpack...
            min_x, max_x, min_y, max_y = box
            #now add...
            total_min_x += min_x
            total_max_x += max_x
            total_min_y += min_y
            total_max_y += max_y

        total_min_x /= float(len(box_list))
        total_max_x /= float(len(box_list))
        total_min_y /= float(len(box_list))
        total_max_y /= float(len(box_list))

        return (total_min_x, total_max_x, total_min_y, total_max_y)

    #=================================================================
    # Gets a set of values of the given size in the given range
    #=================================================================
    @staticmethod
    def distribute_values( n, init, end):
        length = end - init + 1

        if n >= length:
            #base (full)
            return [ x for x in range(init, end + 1) ]
        elif n == 1:
            #base (put in the middle)
            m = int((init + end) / 2.0)
            return [ m ]
        else:
            #do recursion....
            half1 = int( n / 2 )
            m = int((init + end) / 2.0)

            part1 = MiscHelper.distribute_values( half1, init, m )
            part2 = MiscHelper.distribute_values( n - half1, m + 1, end)

            return part1 + part2

    #==============================================
    # Dumps an objet to a file...
    #==============================================
    @staticmethod
    def dump_save( to_dump, file_name):
        file_debug = open(file_name, 'wb')
        pickle.dump(to_dump , file_debug, pickle.HIGHEST_PROTOCOL)
        file_debug.close()

        print( "-> SAVED <" + file_name + ">" )

    #==============================================
    # Recovers an object from a file...
    #==============================================
    @staticmethod
    def dump_load(file_name):
        file_debug = open(file_name, 'rb')

        try:
            loaded = pickle.load(file_debug)
        except:
            print("-> Warning: default ASCII encoding failed. Trying latin1")

            # close, and re-open the file ...
            file_debug.close()
            file_debug = open(file_name, 'rb')

            # try latin-1 encoding
            loaded = pickle.load(file_debug, encoding="latin1")

        file_debug.close()

        print("-> LOADED <" + file_name + ">")

        return loaded

    @staticmethod
    def print_histogram(edges, values, add_CDF=False):
        total_sum = values.sum()

        n_bins = values.shape[0]
        current_sum = 0.0
        for current_bin in range(n_bins):
            output = str(edges[current_bin]) + "\t" + str(edges[current_bin+ 1]) + "\t" + str(values[current_bin])
            current_sum += values[current_bin]
            if add_CDF:
                output += "\t" + str(current_sum / total_sum)
            print(output)

