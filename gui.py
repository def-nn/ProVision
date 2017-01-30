import numpy as np
import pymeanshift as pms
import cv2

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror, showinfo
from PIL import Image, ImageTk
from process import find_edges, Detector, ForegroundObject


MAIN_COLOR = '#2F6883'

params = (
    'Spatial radius',
    'Range radius',
    'Minimum cluster size'
)

filetypes = (
    ('jpeg', '*.jpg'),
    ('jpeg 2000', '*.j2k'),
    ('png', '*.png'),
    ('gif', '*.gif'),
    ('bmp', '*.bmp'),
    ('eps', '*.eps'),
    ('icns', '*.icns'),
    ('im', '*.im'),
    ('msp', '*.msp'),
    ('pcx', '*.pcx'),
    ('ppm', '*.ppm'),
    ('xbm', '*.xbm'),
    ('spider', '*.spi'),
    ('tiff', '*.tiff'),
    ('webp', '*.webp')
)

SEGMENT = 0
DETECT = 1
FIND_EDGES = 2


class MainWindow(Tk):

    def __init__(self):
        Tk.__init__(self)

        self.title('ProVision')
        self.init_size()
        self.original_img = None
        self.pack_widgets()

    def init_size(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        self.geometry('{0}x{1}'.format(int(screen_width * 0.7), int(screen_height * 0.7)))
        self.resizable(width=False, height=False)

    def pack_widgets(self):
        self.img_container = ImageContainer(self)
        self.img_container.pack(side=RIGHT, expand=YES, fill=BOTH)

        self.act_panel = ActionPanel(self)
        self.act_panel.pack(side=LEFT, fill=Y)


class ImageContainer(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        self.img = None
        self.widget = Label(self, text="Please choose image", font=('helvetica', 20), fg='#ffffff', bg=MAIN_COLOR)
        self.widget.pack(expand=YES, fill=BOTH)

        self.widget_width, self.widget_height = 0, 0
        self.canvas = None

        self.draw_mode = False
        self.markers = []

    def reset_markers(self):
        self.markers = []

    def pack_image(self, file=None, img_matrix=None):
        self.draw_mode = False

        if file:
            try:
                self.img = Image.open(file)
            except IOError:
                raise IOError("Can't open image {}".format(file))
        elif img_matrix != None:
            self.img = Image.fromarray(img_matrix)
        else:
            raise ValueError(
                """
                Method ImageContainer.pack_image shoud get one additional argument -
                path to the image or image as PIL.Image object
                """
            )

        self.widget_width, self.widget_height = self.winfo_width(), self.winfo_height()
        img_width, img_height = self.img.size

        if img_width > self.widget_width or img_height > self.widget_height:
            self.img.thumbnail((self.widget_width - 100, self.widget_height - 120), Image.ANTIALIAS)
            img_width, img_height = self.img.size

        self.widget.pack_forget()
        self.widget = Label(self)
        self.widget.pack(expand=YES, fill=BOTH)

        canvas = Canvas(self.widget, width=self.widget_width, height=self.widget_height,
                                                            bg=MAIN_COLOR, highlightthickness=0)
        canvas.pack(expand=YES, fill=BOTH)

        self.canvas = canvas

        photo = ImageTk.PhotoImage(self.img)
        canvas.create_image(self.widget_width // 2, self.widget_height // 2, image=photo)
        canvas.image = photo
        canvas.img_width, canvas.img_height = img_width, img_height

        params = self.widget_width, self.widget_height, img_width, img_height
        canvas.bind('<ButtonPress-1>', lambda event, params=params: self.onClick(event, *params))

    def activate_draw_mode(self, panel):
        self.draw_mode = True

        self.canvas.create_text(
            self.widget_width // 2,
            (self.widget_height - self.canvas.img_height) // 4,
            text='Add markers to the image',
            font=('helvetica', 20),
            fill='#ffffff'
        )

        btn = Button(
            master=self.canvas,
            padx=50,
            pady=5,
            text='DETECT',
            state=DISABLED,
            command=lambda: panel.detect(self.markers)
        )
        btn.pack(side=BOTTOM, pady=(self.widget_height - self.canvas.img_height) // 8)
        self.canvas.btn = btn

    def onClick(self, event, widget_width, widget_height, img_width, img_height):
        if self.draw_mode:
            _x = event.x - (widget_width - img_width) // 2
            _y = event.y - (widget_height - img_height) // 2

            if 0 < _x < img_width and 0 < _y < img_height:
                self.canvas.btn.configure(state=NORMAL)
                self.canvas.create_oval(
                    event.x - 5, event.y - 5, event.x + 5, event.y + 5, width=2, fill='#0000ff', outline='')
                self.markers.append((_y, _x))

class ActionPanel(Frame):

    def __init__(self, master):
        Frame.__init__(self, master=master)
        self.parent = master

        self.options = {
            'Segment'   : lambda: self.image_process(SEGMENT),
            'Detect'    : lambda: self.image_process(DETECT),
            'Find edges': lambda: self.image_process(FIND_EDGES)
        }

        self.is_settings_enabled = IntVar()

        self.pack_buttons()
        self.add_load_img_btn()
        self.pack_params_panel()

    def pack_buttons(self):
        btn_widget = Frame(self, padx=12, pady=10)
        for label in sorted(self.options.keys()):
            btn = Button(
                master=btn_widget,
                command=self.options[label],
                padx=50,
                pady=5,
                text=label,
                state=DISABLED
            )
            btn.pack(side=TOP, fill=X, pady=4)
        btn_widget.pack(side=TOP, fill=X)

    def pack_params_panel(self):
        params_container = Frame(self, padx=12, pady=120)
        params_container.pack()

        params_panel = Frame(params_container)
        custom_params = Checkbutton(
            master=params_container,
            text='Custom settings',
            variable=self.is_settings_enabled,
            command=lambda widget=params_panel: self.onStateChanged(params_panel)
        )

        self.params_list = []
        for param in params:
            field = Frame(params_panel)
            field.pack(side=TOP, fill=X, pady=5)

            label = Label(field, text=param, anchor=W, state=DISABLED)
            entry = Entry(field, state=DISABLED)
            label.pack(fill=BOTH)
            entry.pack(side=TOP, fill=X)

            self.params_list.append(entry)

        custom_params.pack(fill=Y, anchor=W)
        params_panel.pack(fill=X)

    def add_load_img_btn(self):
        load_btn = Button(
            master=self,
            command=self.load_image,
            text='Load image',
            padx=50,
            pady=5,
        )
        load_btn.pack(side=BOTTOM, fill=X, padx=12, pady=15)

    def set_buttons_enabled(self):
        for button in self.winfo_children()[0].winfo_children():
            button.configure(state=NORMAL)

    def onStateChanged(self, params_panel):
        state = NORMAL if self.is_settings_enabled.get() else DISABLED

        for field in params_panel.winfo_children():
            for child in field.winfo_children():
                child.configure(state=state)

    def load_image(self):
        file = askopenfilename(title='Choose image', filetypes=filetypes)
        if file:
            self.file = file
            self.master.img_container.pack_image(file=self.file)
            self.set_buttons_enabled()

    def get_settings(self):
        if self.is_settings_enabled.get():
            settings = []
            for param in self.params_list:
                try:
                    settings.append(int(param.get()))
                except ValueError:
                    showerror('Invalid parameters',
                              'Parameters must be positve integer value')
                    return None
        else: settings = (8, 10, 20)

        return settings

    def image_process(self, action_key):
        if action_key == DETECT:
            self.master.img_container.pack_image(file=self.file)
            self.master.img_container.activate_draw_mode(self)
            return

        action = {
            SEGMENT   : self.segment,
            FIND_EDGES: self.find_edges
        }[action_key]

        settings = self.get_settings()
        if not settings: return

        original_img = Image.open(self.file)

        self.config(cursor='wait')

        res = action(original_img, settings)
        self.master.img_container.pack_image(img_matrix=res)

        self.config(cursor='')

    def segment(self, original_img, settings):
        spatial_radius, range_radius, min_density = settings
        segmented_img, labels_img, number_regions = pms.segment(original_img,
                                                            spatial_radius, range_radius, min_density)

        return segmented_img

    def find_edges(self, original_img, settings):
        spatial_radius, range_radius, min_density = settings
        segmented_img, labels_img, number_regions = pms.segment(original_img,
                                                            spatial_radius, range_radius, min_density)
        res = find_edges(original_img, labels_img)

        return res

    def detect(self, markers):
        self.master.img_container.pack_image(file=self.file)

        settings = self.get_settings()
        if not settings: return

        original_img = Image.open(self.file)

        spatial_radius, range_radius, min_density = settings
        segmented_img, labels_img, number_regions = pms.segment(original_img,
                                                            spatial_radius, range_radius, min_density)
        detector = Detector(original_img, segmented_img, labels_img)

        img = np.copy(original_img)

        for row, col in markers:
            f_obj = ForegroundObject((row, col), labels_img[row, col])
            detector.f_object = f_obj

            (x1, y1), (x2, y2) = detector.detect_object()
            if not(x1 == 0 and y1 == 0 and x2 == labels_img.shape[0] and y2 == labels_img.shape[1]):
                x1 = labels_img.shape[0] * 0.05
                x1 = labels_img.shape[1] * 0.05
                x1 *= 0.9
                x1 *= 0.9

            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.master.img_container.reset_markers()
        self.master.img_container.pack_image(img_matrix=img)
