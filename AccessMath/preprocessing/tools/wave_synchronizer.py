import sys
import struct
import wave

#==============================================================
# CLASS USED TO SYNCHRONIZE TWO AUDIO STREAMS
#
# By: Kenny Davila
#     Rochester Institute of Technology
#     2013
#==============================================================
class WaveSynchronizer:
    PEAKS_PER_SECOND = 10

    def __init__(self):
        pass

    @staticmethod
    def getWavesPeaks(filename, time_window, intervals_per_second ):
        try:
            f = wave.open( filename, 'rb' )
        except:
            raise Exception( "INVALID FILE: <" + filename + "> " )

        sample_rate = f.getframerate()
        channels= f.getnchannels()
        w = f.getsampwidth()
        n_frames = f.getnframes()

        total_frames = min(time_window * 2 * sample_rate, n_frames )

        data = f.readframes(total_frames)
        values = []
        current_max = 0.0

        interval_size = int(sample_rate / intervals_per_second)

        unpacked = struct.unpack_from("<" + "h" * total_frames, data)

        count = 0
        u = 0
        while u < len(unpacked):
            #use channel 0 only....
            #take max on current interval of channel 0
            current_max = max( unpacked[u:u+interval_size*channels:channels] )
            #add peak
            values.append( current_max )
            #jump to next interval
            u += interval_size * channels

        base = min( values )
        values = [ x - base for x in values ]

        return values

    @staticmethod
    def computePeaksDifference(peaks1, peaks2, offset):
        if offset < 0:
            #swap....
            temp = peaks1
            peaks1 = peaks2
            peaks2 = temp

            offset = -offset

        size = min( len(peaks1) - offset, len(peaks2))

        diff = 0.0
        for x in range(size):
            diff += abs(peaks1[x + offset] - peaks2[x]) ** 2

        diff /= size

        return diff

    @staticmethod
    def synchronize(file_name1, file_name2, window):
        try:
            window = int(window)
        except:
            print("INVALID TIME WINDOW")
            return None

        peaks_per_second = WaveSynchronizer.PEAKS_PER_SECOND

        peaks1 = WaveSynchronizer.getWavesPeaks( file_name1, window, peaks_per_second )
        peaks2 = WaveSynchronizer.getWavesPeaks( file_name2, window, peaks_per_second )
        return 0.0
        diff = WaveSynchronizer.computePeaksDifference( peaks1, peaks2, 0)
        min_diff = diff
        min_offset = 0

        total_positions = window * peaks_per_second
        for i in range(-total_positions, total_positions):
            diff = WaveSynchronizer.computePeaksDifference( peaks1, peaks2, i)

            if diff < min_diff:
                min_diff = diff
                min_offset = i

        return min_offset / float(peaks_per_second)

if __name__== '__main__':
    if len( sys.argv )!= 4:
        print "Usage: synchronize <file1> <file2> <Window>"
    else:
        offset = WaveSynchronizer.synchronize( sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3 ] )
        print( "OFFSET : " + str(offset) + "s" )
