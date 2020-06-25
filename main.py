from tkinter import *
from tkinter import filedialog
from histogramas import adjust_intensity, equalize_intensity
from convolution_filtering import filter_image, medianFilter, gaussianFilter, highBoost
from PIL import ImageTk, Image
from skimage import io, img_as_ubyte, img_as_float
import matplotlib.pyplot as plt


# diccionario de funciones, parametros segun los argumentos requeridos con "signature" len(sig.parameters)

def hello():
    print("hola")


class Va(Frame):
    dict_metodos = {
        'ard': ("Alteración de Rango dinámico", adjust_intensity, ['inrange', 'outrange']),
        'eqh': ("Ecualización de histograma", equalize_intensity, ['nbins']),
        'fti': ("Filtro espacial de imagen", filter_image, ['kernel']),
        'mdf': ("Filtro de medianas", medianFilter, ['filtersize']),
        'gaf': ("Filtro gausiano", gaussianFilter, ['sigma']),
        'hbf': ("Filtro high boost", highBoost, ['A', 'method', 'param'])}
    ruta_imagen = "C:/Users/diego/Pictures/test/descar.png"
    raw_inimage = None
    raw_outimage = None
    inimage = None
    inimagecopy = None
    outimage = None
    outimagecopy = None
    inphoto = None
    outphoto = None
    labelimagen = None
    labelimagen2 = None
    frame_parametros = None
    frame_imagen1 = None
    frame_imagen2 = None
    panel_lateral = None

    botones_metodos = []
    entradas_parametros = []
    labels_parametros = []

    activefunction = None
    active_method = 'ard'

    def __init__(self, master):
        self.v = StringVar(value='ard')
        self.master = master
        super().__init__()
        self.init_ui()

    def _resize_image1(self, event):
        new_width = int(3 * self.master.winfo_width() / 7)
        new_height = int(6 * event.height / 7)

        self.inimage = self.inimagecopy.resize((new_width, new_height))

        self.inphoto = ImageTk.PhotoImage(self.inimage)
        self.labelimagen.configure(image=self.inphoto)

    def _resize_image2(self, event):
        new_width = int(3 * self.master.winfo_width() / 7)
        new_height = int(6 * event.height / 7)

        self.outimage = self.outimagecopy.resize((new_width, new_height))

        self.outphoto = ImageTk.PhotoImage(self.outimage)
        self.labelimagen2.configure(image=self.outphoto)

    def init_ui(self):
        self.master.title("Simple menu")

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        self.frame_imagen1 = Frame()
        self.frame_imagen1.pack(side="left", fill="both", expand="yes")

        self.labelimagen = Label(self.frame_imagen1, text="imagen entrada")
        self.labelimagen.pack(fill="both", expand="yes")
        self.labelimagen.bind('<Configure>', self._resize_image1)
        self.raw_inimage = io.imread(self.ruta_imagen, as_gray=True)
        self.inimage = Image.fromarray(img_as_ubyte(self.raw_inimage), 'L')
        self.inimagecopy = self.inimage.copy()
        self.inphoto = ImageTk.PhotoImage(self.inimage)
        self.labelimagen.config(image=self.inphoto)

        self.frame_imagen2 = Frame()
        self.frame_imagen2.pack(side="left", fill="both", expand="yes")

        self.labelimagen2 = Label(self.frame_imagen2, text="imagen salida")
        self.labelimagen2.pack(fill="both", expand="yes")
        self.labelimagen2.bind('<Configure>', self._resize_image2)

        menu_archivo = Menu(menubar)
        menu_archivo.add_command(label="Abrir imágen", command=self.on_open_file)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.on_exit)
        menubar.add_cascade(label="Archivo", menu=menu_archivo)

        self.panel_lateral = Frame()
        self.panel_lateral.pack(side="left", fill=Y, expand="yes")

        frame_metodos = LabelFrame(self.panel_lateral, text="Métodos")
        self.frame_parametros = LabelFrame(self.panel_lateral, text="Parámetros")
        boton_procesar = Button(self.panel_lateral, text="Procesar", command=self.on_process)
        boton_histogramas = Button(self.panel_lateral, text="Comparar histogramas", command=self.comparar_histogramas)

        for m in self.dict_metodos:
            self.botones_metodos.append(Radiobutton(frame_metodos,
                                                    text=self.dict_metodos[m][0],
                                                    variable=self.v, value=m,
                                                    indicatoron=False,
                                                    command=lambda metodo=m: self.handleparametros(metodo)))

        self.handleparametros(self.active_method)  # parametros por defecto

        frame_metodos.pack(padx=10, pady=10)
        self.frame_parametros.pack(padx=10, pady=10)
        boton_procesar.pack(padx=5, pady=10)
        boton_histogramas.pack(padx=5, pady=10)

        for c in self.botones_metodos:
            c.pack(side="top", fill="both", expand="yes")

    def handleparametros(self, metodo):
        self.active_method = metodo
        for e in self.entradas_parametros:
            e.destroy()
        self.entradas_parametros = []
        for l in self.labels_parametros:
            l.destroy()
        self.labels_parametros = []
        for p in self.dict_metodos[metodo][2]:
            entrada = Entry(self.frame_parametros)
            self.entradas_parametros.append(entrada)

            label_parametros = Label(self.frame_parametros, text=p)
            self.labels_parametros.append(label_parametros)

            label_parametros.pack()
            entrada.pack()

    def on_exit(self):
        self.quit()

    def comparar_histogramas(self):

        plt.subplot(1, 4, 1, title="Histograma Imagen Previa")
        hist1, bins1, patches1 = plt.hist(img_as_ubyte(self.raw_inimage).ravel(), range(257))

        plt.subplot(1, 4, 2, title="Histograma Acumulado Imagen Previa")
        plt.plot(hist1.cumsum())

        if self.raw_outimage is not None:
            plt.subplot(1, 4, 3, title="Histograma Imagen Salida")
            hist2, bin2, patches2 = plt.hist(img_as_ubyte(self.raw_outimage).ravel(), range(257))
            plt.subplot(1, 4, 4, title="Histograma Acumulado Imagen Salida")
            plt.plot(hist2.cumsum())

        plt.draw()
        plt.pause(0.001)

    def on_open_file(self):
        self.ruta_imagen = filedialog.askopenfilename(initialdir="C:/Users/diego/Pictures",
                                                      filetypes=[("All files", "*.*")],
                                                      title="Choose a file.")
        self.raw_inimage = io.imread(self.ruta_imagen, as_gray=True)
        self.inimage = Image.fromarray(img_as_ubyte(self.raw_inimage), 'L')
        self.inimagecopy = self.inimage.copy()
        self.inphoto = ImageTk.PhotoImage(self.inimage)

        self.labelimagen.config(image=self.inphoto)
        self.outphoto = None
        self.raw_outimage = None
        self.outimage = None
        self.labelimagen.imagen = self.inphoto

    def on_process(self):
        args = dict(inimage=(io.imread(self.ruta_imagen, as_gray=True)))
        i = 0
        for e in self.entradas_parametros:
            entrada = e.get()
            arg = entrada.split()
            if arg:
                if len(arg) == 1:
                    arg = arg[0]
                args[self.dict_metodos[self.active_method][2][i]] = arg
            i += 1
        self.raw_outimage = self.dict_metodos[self.active_method][1](**args)
        # print(outimage)
        self.outimage = Image.fromarray(img_as_ubyte(self.raw_outimage), 'L')
        self.outimagecopy = self.outimage.copy()
        self.outphoto = ImageTk.PhotoImage(self.outimage)

        self.labelimagen2.config(image=self.outphoto)
        self.labelimagen2.imagen = self.outphoto


def main():
    root = Tk()
    root.geometry("1366x768+100+50")
    gui = Va(root)
    root.minsize(1366, 768)
    root.mainloop()


main()
