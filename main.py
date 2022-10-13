import math
from typing import Callable, Any

from PIL import Image
import cv2
import timeit

FRAME_COUNT = 40
COLOUR_DISTANCE = 10
MAX_PACK_SIZE = 1000
MIN_PACK_SIZE = 50
MAX_PACK_PROPORTIONS_RATIO = 2
MIN_CIRCLE_PERCENTAGE = 0.7


def angle(point1, point2, point3):
    vector1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vector2 = [point2[0] - point3[0], point2[1] - point3[1]]
    top = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    len1 = math.hypot(vector1[0], vector1[1])
    len2 = math.hypot(vector2[0], vector2[1])
    if len1 != 0 and len2 != 0 and point1 != point2 and point2 != point3 and point1 != point3:
        return math.degrees(math.acos(round(top / (len1 * len2), 4)))
    else:
        return 9999


def get_frames(path, frame_count):
    frames = []
    sec = 0
    count = 0
    success = True
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = video_frame_count / fps
    frame_rate = duration / frame_count
    while success:
        count += 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, image = vidcap.read()
        success = bool(has_frames)
        if success:
            frames.append(Image.fromarray(image).resize((720, 1080), resample=Image.BILINEAR))

    return frames


class BallTracker:
    def __init__(self, path: str, frame_count=FRAME_COUNT, colour_distance=COLOUR_DISTANCE) -> None:
        self.path: str = path
        self.ball_packs: [[(int,)]] = []
        self.ball_pack_sizes: [int] = []
        self.ball_locations: [(int,)] = []
        self.ball_frame_numbers: [int] = []
        self.ball_path: [(int,)] = []
        self.frame_number: int = 1
        self.frames: [Image] = get_frames(path, frame_count)
        self.width: int = self.frames[0].width
        self.height: int = self.frames[0].height
        self.colour_distance: int = colour_distance

    def analise_video(self) -> None:
        for frame_number in range(1, len(self.frames)):
            self.frame_number = frame_number
            print(frame_number)
            self.analise_frame(self.get_moving_pixels())

        frame = self.frames[0].copy()
        for ball in self.ball_packs:
            for pos in ball:
                frame.putpixel(pos, (255, 0, 0))

        frame.save("coordinates.png")

        #print("new way", timeit.timeit(self.new_way, number=1))

        print("old way", timeit.timeit(self.get_ball_path, number=1))

        # self.predict(self.get_ball_path())

    def get_moving_pixels(self) -> [(int,)]:
        moving_pixels: [(int,)] = []
        small_width = int(self.width / 4)
        small_height = int(self.height / 4)

        current_frame_data = self.frames[self.frame_number].load()
        last_frame_data = self.frames[self.frame_number - 1].load()
        current_small_frame = self.frames[self.frame_number - 1].resize((small_width, small_height)).load()
        last_small_frame = self.frames[self.frame_number].resize((small_width, small_height)).load()

        for y in range(small_height):
            for x in range(small_width):
                if math.dist(current_small_frame[x, y], last_small_frame[x, y]) > 30:
                    for i in range(4):
                        for ii in range(4):
                            if math.dist(current_frame_data[x * 4 - i, y * 4 - ii],
                                last_frame_data[x * 4 - i, y * 4 - ii]) > self.colour_distance:
                                moving_pixels.append((x * 4 - i, y * 4 - ii))

        return moving_pixels

    def analise_frame(self, moving_pixels: [(int,)]) -> None:
        searched_pixels = {}
        moved_pixels = {}
        for pixel in moving_pixels:
            moved_pixels[pixel] = True

        directions: [(int,)] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def get_pack(x: int, y: int, valid: callable) -> [(int,)]:
            pack_pixels: [(int,)] = [(x, y)]
            to_search: [(int,)] = [(x, y)]
            while to_search:
                searched_pixels[(x, y)] = True
                x, y = to_search.pop(0)
                for a, b in directions:
                    if valid(x + b, y + a):
                        if 0 < x < self.width and 0 < y < self.height and not (x + b, y + a) in to_search:
                            to_search.append((x + b, y + a))
                            pack_pixels.append((x + b, y + a))

            return pack_pixels

        def assess_pack(pack_pixels):
            if len(pack_pixels) < MIN_PACK_SIZE or len(pack_pixels) > MAX_PACK_SIZE:
                return

            min_x: int = min(pack_pixels)[0]
            max_x: int = max(pack_pixels)[0]
            min_y: int = min(pack_pixels, key=lambda t: t[1])[1]
            max_y: int = max(pack_pixels, key=lambda t: t[1])[1]
            pack_width: int = max_x - min_x
            pack_height: int = max_y - min_y
            if pack_height and (1 / MAX_PACK_PROPORTIONS_RATIO) < pack_width / pack_height < MAX_PACK_PROPORTIONS_RATIO:

                radius = int(round(pack_width / 2))
                incorrect_pixels: int = 0
                correct_pixels: int = 0
                circle_pixels = {}
                for y in range(min_y, radius * 2 + 1 + min_y):
                    for x in range(min_x, radius * 2 + 1 + min_x):
                        if x < self.width and y < self.height:
                            if pow(abs(x - min_x - radius), 2) + pow(abs(y - radius - min_y), 2) < pow(radius, 2):
                                if (x, y) in pack_pixels:
                                    correct_pixels += 1
                                else:
                                    incorrect_pixels += 1
                                circle_pixels[(x, y)] = True

                ball_average_position: [int] = [0, 0]
                for x, y in pack_pixels:
                    ball_average_position[0] += x
                    ball_average_position[1] += y
                    if (x, y) in circle_pixels:
                        correct_pixels += 1
                    else:
                        incorrect_pixels += 1

                ball_average_position[0] /= len(pack_pixels)
                ball_average_position[1] /= len(pack_pixels)
                circle_percentage = correct_pixels / (correct_pixels + incorrect_pixels)
                if circle_percentage > MIN_CIRCLE_PERCENTAGE:
                    self.ball_packs.append(pack_pixels)
                    self.ball_locations.append(ball_average_position)
                    self.ball_frame_numbers.append(self.frame_number)
                    self.ball_pack_sizes.append(len(pack_pixels))

        def valid(x, y):
            if searched_pixels.get((x, y)):
                return False
            elif moved_pixels.get((x, y)):
                return True
            else:
                return False

        for x, y in moving_pixels:
            if valid(x, y):
                assess_pack(get_pack(x, y, valid))

    def new_way(self):
        acceptdistance = 4

        def valid_chain(a,b,c):
            if angle(self.getball(a), self.getball(b), self.getball(c)) < 10:
                if 0.3 < self.ball_pack_sizes[a]/self.ball_pack_sizes[b] < 3:
                    if 0.3 < self.ball_pack_sizes[b]/self.ball_pack_sizes[c] < 3:
                        if .1 < math.dist(self.getball(i) + [0], self.getball(ii) + [0]) / math.dist(self.getball(ii) + [0], self.getball(iii) + [0]) < 10:
                            return True
            return False

        """ Creates the chains of three"""

        three_chains = []
        i = 0
        while i < len(self.ball_frame_numbers) - 1:
            ii = i
            while self.ball_frame_numbers[ii] == self.ball_frame_numbers[i] and ii < len(self.ball_frame_numbers) - 1:
                ii += 1
            while ii < len(self.ball_frame_numbers) - 1 and self.ball_frame_numbers[ii] - self.ball_frame_numbers[
                i] <= acceptdistance:
                iii = ii
                while self.ball_frame_numbers[iii] == self.ball_frame_numbers[ii] and iii < len(
                        self.ball_frame_numbers) - 1:
                    iii += 1
                while iii < len(self.ball_frame_numbers) - 1 and self.ball_frame_numbers[iii] - self.ball_frame_numbers[
                    ii] <= acceptdistance:
                    if valid_chain(i, ii, iii):
                        three_chains.append([i, ii, iii])
                    iii += 1
                ii += 1
            i += 1

        """ Removes chains with the same start and end as another chain """

        three_chains.sort(key=lambda x: (x[0], x[2]))
        i = 0
        while i < len(three_chains):
            j = i + 1
            chain = three_chains[i]
            found_dup = False
            while j < len(three_chains) and three_chains[j][0] == chain[0] and three_chains[j][2] == chain[2]:
                if self.ball_frame_numbers[chain[1]] != self.ball_frame_numbers[three_chains[j][1]]:
                    del three_chains[j]
                    found_dup = True
                else:
                    j += 1
            if found_dup:
                del three_chains[i]
            else:
                j += 1
            i = j - 1

        """ Connects chains together """

        three_chains.sort()
        print(len(three_chains))


        """
        def link_up():
            chains_index = 0
            while chains_index < len(chains):
                chain = chains[chains_index]

                def link(a: int, b: int) -> bool:
                    return a[0] == b[-2] and a[1] == b[-1]

                for three_chain in [three_chain for three_chain in three_chains if link(three_chain, chain)]:
                    chains.append(chain + [three_chain[-1]])
                chains_index += 1
        """
        to_do = three_chains[:]
        chains = [[0,0,0]]
        while to_do:
            chain = to_do.pop(0)
            for new_chain in three_chains:
                if chain[-2] == new_chain[0] and chain[-1] == new_chain[1]:
                    chain.append(new_chain[2])
                    to_do.append(chain)

        chains.sort(key=len)
        down_path = chains[-1]

        chains_with_start = [chain for chain in chains if chain[0] == down_path[-1]]
        if chains_with_start:
            up_path = max(chains_with_start, key=len)
        else:
            up_path = []

        self.ball_path = down_path + up_path[1:]

        pitch_map = self.frames[0].copy()
        for i in range(len(self.ball_path)):
            for x, y in self.ball_packs[self.ball_path[i]]:
                pitch_map.putpixel((x, y), (255, 0, 255))
        pitch_map.save("pitchmap.png")

        print(self.ball_path)

    def predict(self, ball_path):
        pass

    def getball(self, i):
        return [self.ball_locations[i][0], self.ball_locations[i][1]]

    def get_ball_path(self) -> [int]:
        acceptdistance = 4

        ball_frame_numbers = self.ball_frame_numbers
        three_chains = []
        i = 0
        while i < len(ball_frame_numbers) - 1:
            ii = i
            while ball_frame_numbers[ii] == ball_frame_numbers[i] and ii < len(ball_frame_numbers) - 1:
                ii += 1
            while ii < len(ball_frame_numbers) - 1 and ball_frame_numbers[ii] - ball_frame_numbers[i] <= acceptdistance:
                iii = ii
                while ball_frame_numbers[iii] == ball_frame_numbers[ii] and iii < len(ball_frame_numbers) - 1:
                    iii += 1
                while iii < len(ball_frame_numbers) - 1 and ball_frame_numbers[iii] - ball_frame_numbers[
                    ii] <= acceptdistance:
                    if angle(self.getball(i), self.getball(ii), self.getball(iii)) < 20 and .25 < math.dist(
                            self.getball(i) + [0], self.getball(ii) + [0]) / math.dist(self.getball(ii) + [0], self.getball(iii) + [0]) < 4:
                        three_chains.append([i, ii, iii])
                    iii += 1
                ii += 1
            i += 1

        three_chains.sort(key=lambda x: (x[0], x[2]))
        i = 0
        while i < len(three_chains):
            j = i + 1
            chain = three_chains[i]
            found_dup = False
            while j < len(three_chains) and three_chains[j][0] == chain[0] and three_chains[j][2] == chain[2]:
                if self.ball_frame_numbers[chain[1]] != self.ball_frame_numbers[three_chains[j][1]]:
                    del three_chains[j]
                    found_dup = True
                else:
                    j += 1
            if found_dup:
                del three_chains[i]
            else:
                j += 1
            i = j - 1

        print(len(three_chains))

        chains = three_chains[:]
        i = 0
        while i < len(chains):
            chain = chains[i]
            for new_chain in three_chains:
                if chain[-2] == new_chain[0] and chain[-1] == new_chain[1]:
                    chains.append(chain + [new_chain[2]])

            i += 1

        print(1)

        chains.sort(key=len)
        down_path = chains[-1]
        up_path = []
        up_path = sorted([chain for chain in chains if chain[0] == down_path[-1]])[-1]

        self.ball_path = down_path + up_path

        pitch_map = self.frames[0].copy()
        for i in range(len(self.ball_path)):
            for x, y in self.ball_packs[self.ball_path[i]]:
                pitch_map.putpixel((x, y), (255, 0, 255))
        pitch_map.save("pitchmap.png")

        
        print(self.ball_path)




import cProfile
import pstats


def main():
    with cProfile.Profile() as pr:
        tracker = BallTracker("video2.mov")
        tracker.analise_video()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__ == '__main__':
    main()
