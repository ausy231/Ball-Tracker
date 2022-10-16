import math
from typing import Callable, Any

from PIL import Image
import cv2

FRAME_COUNT = 40
COLOUR_DISTANCE = 40
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


class Ball:
    def __init__(self, frame_number, pos , size):
        self.pos = pos
        self.frame_number = frame_number
        self.size = size

    def __repr__(self):
        return f"Ball ({self.frame_number}): ({self.pos}), {self.size} pixels"


class BallTracker:
    def __init__(self, path: str, frame_count=FRAME_COUNT, colour_distance=COLOUR_DISTANCE) -> None:
        self.path: str = path
        self.ball_packs: [[(int,)]] = []
        self.balls = []
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
        for i, ball in enumerate(self.ball_packs):
            for pos in ball:
                frame.putpixel(pos, (255, 0, 0))
            if i <= 255:
                frame.putpixel(self.balls[i].pos, (i, self.balls[i].frame_number, 255))


        frame.save("coordinates.png")
        self.get_ball_path()

        # print("old way", timeit.timeit(self.get_ball_path, number=1))

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
                if math.dist(current_small_frame[x, y], last_small_frame[x, y]) > self.colour_distance:
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

                ball_average_position[0] //= len(pack_pixels)
                ball_average_position[1] //= len(pack_pixels)
                circle_percentage = correct_pixels / (correct_pixels + incorrect_pixels)
                if circle_percentage > MIN_CIRCLE_PERCENTAGE:
                    self.balls.append(Ball(self.frame_number, ball_average_position, len(pack_pixels)))
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

    def get_ball_path(self):
        accepted_time_distance = 7
        accepted_size_distance = 3
        accepted_distance = 10
        accepted_angle = 20

        def valid_chain(a, b, c):
            if angle(self.balls[a].pos, self.balls[b].pos, self.balls[c].pos) < accepted_angle:
                return True
                if True or (1/accepted_size_distance) < self.balls[a].size / self.balls[b].size < accepted_size_distance:
                    if True or (1/accepted_size_distance) < self.balls[b].size / self.balls[c].size < accepted_size_distance:
                        if True or (1/accepted_distance) < math.dist(self.balls[i].pos, self.balls[ii].pos) / math.dist(self.balls[ii].pos, self.balls[iii].pos) < accepted_distance:
                            return True
            return False

        """ Creates the chains of three"""

        three_chains = []
        i = 0
        while i < len(self.balls) - 1:
            ii = i
            while self.balls[ii].frame_number == self.balls[i].frame_number and ii < len(self.balls) - 1:
                ii += 1
            while ii < len(self.balls) - 1 and self.balls[ii].frame_number - self.balls[i].frame_number <= accepted_time_distance:
                iii = ii
                while self.balls[iii].frame_number == self.balls[ii].frame_number and iii < len(self.balls) - 1:
                    iii += 1
                while iii < len(self.balls) - 1 and self.balls[iii].frame_number - self.balls[ii].frame_number <= accepted_time_distance:

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
                if self.balls[chain[1]].frame_number != self.balls[three_chains[j][1]].frame_number:
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

        to_do = three_chains[:]
        chains = [[0, 0, 0]]
        while to_do:
            chain = to_do.pop(-1)
            for new_chain in three_chains:
                if chain[-2] == new_chain[0] and chain[-1] == new_chain[1]:
                    chain.append(new_chain[2])
                    to_do.append(chain)
                    chains.append(chain)

        chains.sort(key=lambda chain: (len(chain), chain[-1]))
        print([chain for chain in chains if len(chain) == len(chains[-1])])
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

    def predict(self, ball_path):
        pass


def main():
    tracker = BallTracker("video0708.mov")
    tracker.analise_video()


if __name__ == '__main__':
    main()
