import numpy as np
import cv2
import string
import math
import os
import uuid
import random

##############################################
grad_img = cv2.imread('grad.png')
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            else:
                output[i][j] = image[i][j]
    return output
##############################################

wd, _ = os.path.split(os.path.abspath(__file__))

CAPTCHA_LENGTH = 6
WIDTH = 300     # 120
HEIGHT = 100    # 36

# RGB
# font_colors = {
#     'dark-green':(0, 150, 0),
    # (241, 145, 241)
#     'red':(230, 70, 50),
#     'violet':(135, 80, 250),
#     'light-green':(65, 235, 100)
# }
# BGR
font_colors = {
    'dark-green':(0, 150, 0),
    'red':(50, 70, 230),
    'violet':(250, 80, 135),
    'light-green':(100, 235, 65)
}

class Captcha:
    def __init__(self, width, high, ls=None, lc=CAPTCHA_LENGTH, fs=None,
                 # folder=os.path.join(wd, 'samples'),
                 folder='samples',
                 debug=False):
        """
        :param ls: letter set, all
        :param fs: font set
        :param lc: letter count in one pic
        :param folder: the folder to save img
        :param debug: debug mode
        """

        if fs is None:
            fs = ['FONT_HERSHEY_SIMPLEX', 'FONT_ITALIC']

        self.fs = fs

        if ls is None:
            ls = string.ascii_uppercase + string.digits
        if isinstance(ls, str):
            self.letter = [i for i in ls]
        elif isinstance(ls, list):
            self.letter = ls

        self.lc = lc
        self.width, self.high = width, high
        self.debug = debug
        self.folder = folder
        if not self.debug and folder:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

    def _tilt_img(self, img):
        tmp_img = img.copy()
        tmp_img.fill(255)
        tile_angle = np.random.randint(
            100*-math.pi/6, 0
        ) / 100
        high, width, _ = img.shape
        for y in range(width):
            for x in range(high):
                new_y = int(y + (x-high/2)*math.tanh(tile_angle))
                try:
                    tmp_img[x, new_y, :] = img[x, y, :]
                except IndexError:
                    pass
        img[:, :, :] = tmp_img[:, :, :]

    def _shake_img(self, img, outer_top_left, outer_bottom_right,
                   inner_top_left, inner_bottom_right):
        (x1, y1), (x2, y2) = outer_top_left, outer_bottom_right
        (i1, j1), (i2, j2) = inner_top_left, inner_bottom_right
        delta_x = np.random.randint(x1-i1, x2-i2)
        delta_y = np.random.randint(y1-j1, y2-j2)
        area = img[y1:y2, x1:x2, :]
        area_high, area_width, _ = area.shape
        tmp_area = area.copy()
        tmp_area.fill(255)

        for index_y in range(area_high):
            for index_x in range(area_width):
                new_x, new_y = index_x + delta_x, index_y + delta_y
                if new_x < area_width and new_y < area_high:
                    tmp_area[new_y, new_x, :] = area[index_y, index_x, :]

        area[:, :, :] = tmp_area[:, :, :]

    def _distort_img(self, img):
        high, width, _ = img.shape
        tmp_img = img.copy()
        tmp_img.fill(255)

        coef_vertical = np.random.randint(1, 5)
        coef_horizontal = np.random.choice([2, 3, 4]) * math.pi / width
        scale_biase = np.random.randint(0, 360) * math.pi / 180

        def new_coordinate(x, y):
            return int(x+coef_vertical*math.sin(coef_horizontal*y+scale_biase))

        for y in range(width):
            for x in range(high):
                new_x = new_coordinate(x, y)
                try:
                    tmp_img[x, y, :] = img[new_x, y, :]
                except IndexError:
                    pass

        img[:, :, :] = tmp_img[:, :, :]

    def _draw_basic(self, img, text):
        font_scale = 1.6    # 36 px
        max_width = max_high = 0

        for i in text:
            for _font_face in [getattr(cv2, self.fs[i]) for i in range(len(self.fs))]:
                for _font_thickness in [5, 6]:
                    (width, high), _ = cv2.getTextSize(
                        i, _font_face, font_scale, _font_thickness)
                    max_width, max_high = max(max_width, width), max(max_high, high)

        total_width = max_width * self.lc
        width_delta = np.random.randint(0, self.width - total_width)
        vertical_range = self.high - max_high
        images = list()

        font_color = np.random.choice(a=['dark-green', 'red', 'violet', 'light-green'], p=[0.91, 0.03, 0.03, 0.03])
        font_color = font_colors[font_color]

        delta_high = np.random.randint(
            int(2*vertical_range/5), int(3*vertical_range/5)
        )

        for index, letter in enumerate(text):
            font_face = getattr(cv2, np.random.choice(self.fs))
            font_thickness = np.random.choice([5, 6])
            tmp_img = img.copy()

            bottom_left_coordinate = (
                index*max_width + width_delta,
                self.high - delta_high
            )
            cv2.putText(tmp_img, letter, bottom_left_coordinate, font_face,
                        font_scale, font_color, font_thickness)
            self._tilt_img(tmp_img)

            # cv2.imshow(text, tmp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            images.append(tmp_img)

        high, width, _ = img.shape
        for y in range(width):
            for x in range(high):
                r, g, b = 0, 0, 0
                for tmp_img in images:
                    r += tmp_img[x, y, 0] + 1
                    g += tmp_img[x, y, 1] + 1
                    b += tmp_img[x, y, 2] + 1
                r, g, b = r % 256, g % 256, b % 256
                img[x, y, :] = (r, g, b)

        for y in range(width):
            for x in range(high):
                if (img[x,y,0] + img[x,y,1] + img[x,y,2]) % 256 == 0:
                    img[x,y,0] = img[x,y,1] = img[x,y,2] = 255

    def _draw_line(self, img):
        left_x = np.random.randint(0, self.width//4)
        left_y = np.random.randint(self.high)
        right_x = np.random.randint(self.width*3//4, self.width)
        right_y = np.random.randint(self.high)
        start, end = (left_x, left_y), (right_x, right_y)
        line_color = tuple(int(np.random.choice(range(0, 156)))
                           for _ in range(3))
        line_thickness = np.random.randint(1, 3)
        cv2.line(img, start, end, line_color, line_thickness)

    def _put_noise(self, img):
        for i in range(600):
            x = np.random.randint(self.width)
            y = np.random.randint(self.high)
            dot_color = tuple(int(np.random.choice(range(0, 156)))
                              for _ in range(3))
            img[y, x, :] = dot_color

    def save_img(self, text):
        img = np.zeros((self.high, self.width, 3), np.uint8)
        img.fill(255)

        # img = cv2.imread('grad.png')

        # cv2.imshow(text, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        self._draw_basic(img, text)
        # self._put_noise(img)
        # self._distort_img(img)
        # self._draw_line(img)

        noise_grad_img = sp_noise(grad_img,0.15)
        
        img = cv2.addWeighted(img, 0.5,noise_grad_img,0.5,0)

        # cv2.imshow(text, dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.debug:
            cv2.imshow(text, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            fn = text + ('_'+str(uuid.uuid1())[4: 8])
            cv2.imwrite('{}\\{}.jpg'.format(self.folder, fn), img)




    def batch_create_img(self, number=5):
        exits = set()
        while(len(exits)) < number:
            word = ''.join(np.random.choice(self.letter, self.lc))
            if word not in exits:
                exits.add(word)
                self.save_img(word)
                if not self.debug:
                    if len(exits) % 10 == 0:
                        print('{} generated.'.format(len(exits)))
        if not self.debug:
            print('{} captchas saved into {}.'.format(len(exits), self.folder))

if __name__ == '__main__':
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    c = Captcha(WIDTH, HEIGHT, letters, fs=['FONT_HERSHEY_SIMPLEX', 'FONT_ITALIC'], debug=False)
    c.batch_create_img(19995)

'''
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)
    # make a black image
    img = np.zeros((500, 500))
    # set some text
    text = "Some text in a box!"
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.imshow("A box!", img)
    cv2.waitKey(0)
'''