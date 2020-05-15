import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from load_and_other import distance as dst
from load_and_other.func import findThreshold, detectFace, loadVerify
from load_and_other import Load
root = tk.Tk()
root.title("Age prediction")
root.anchor('n')
#img1 = tk.PhotoImage(file='./img/icon1.png')
#root.iconphoto(False, img1)


def open_File():
    global path
    path = filedialog.askopenfilename(initialdir="/", title="Select File"
                                          , filetypes=(
            ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    loadIMG = Image.open(path)
    loadIMG = loadIMG.resize((224, 224), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(loadIMG)
    img = tk.Label(image=render)
    img.image = render
    img.place(relx=0.23, rely=0.40, anchor='nw')


def start_AP():
    age, gender, emo, race = Load.analyzeImage(path)
    if age == -1:
        messagebox.showinfo(title='Thông báo',
                            message="Không tìm thấy gương mặt trong ảnh bạn đưa vào.")
    if age == -2:
        messagebox.showinfo(title='Thông báo',
                            message="Đường dẫn đưa vào phải là type string !!!")
    if age == -3:
        messagebox.showinfo(title='Thông báo',
                            message="Đường dẫn không tồn tại.")
    if age == -4:
        messagebox.showinfo(title='Thông báo',
                            message="Định dạng ảnh không hợp lệ.")
    if age > 0:
        messagebox.showinfo(title='Kết quả', message='Tuổi dự đoán là: ' + str(int(age)) +
                                                     '\nGiới tính : ' + gender +
                                                     '\nCảm xúc: ' + emo +
                                                     '\nChủng tộc: ' + race)


def open_File1():
    global path1
    path1 = filedialog.askopenfilename(initialdir="/", title="Select File"
                                          , filetypes=(
            ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    loadIMG = Image.open(path1)
    loadIMG = loadIMG.resize((188, 188), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(loadIMG)
    img = tk.Label(image=render)
    img.image = render
    img.place(relx=0.02, rely=0.40, anchor='nw')


def open_File2():
    global path2
    path2 = filedialog.askopenfilename(initialdir="/", title="Select File"
                                          , filetypes=(
            ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    loadIMG = Image.open(path2)
    loadIMG = loadIMG.resize((188, 188), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(loadIMG)
    img = tk.Label(image=render)
    img.image = render
    img.place(relx=0.51, rely=0.40, anchor='nw')


def verify(distance_metric='cosine', enforce_detection=True):

    # ------------------------------
    model = loadVerify()

    # ------------------------------
    # face recognition models have different size of inputs
    input_shape = model.layers[0].input_shape[1:3]

    # ------------------------------

    # tuned thresholds for model and metric pair
    threshold = findThreshold(distance_metric)

    # ------------------------------
    img1 = detectFace(path1, input_shape, enforce_detection=enforce_detection)
    img2 = detectFace(path2, input_shape, enforce_detection=enforce_detection)

    # ----------------------
    # find embeddings

    img1_representation = model.predict(img1)[0, :]
    img2_representation = model.predict(img2)[0, :]

    # ----------------------
    # find distances between embeddings

    if distance_metric == 'cosine':
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                             dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    # ----------------------
    # decision
    if distance <= threshold:
        messagebox.showinfo(title='Thông báo',
                            message="Họ là cùng một người.")
    else:
        messagebox.showinfo(title='Thông báo',
                            message="Họ là hai người khác nhau.")


# anchor = n, ne, e, se, s, sw, w, nw, or center
# window size
canvas = tk.Canvas(root, height=500, width=400)
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=1.0, relheight=1)

#set background
# background_image = tk.PhotoImage(file='./img/cuteBGR.png')
# background_label = tk.Label(root, image=background_image)
# background_label.place(relwidth=1, relheight=1)

#text on the head
text = tk.Label(frame, text='\nAge Prediction',
                bg="white", font="Arial")
text.pack()

#open file diaglog
openFile = tk.Button(frame, text="Open file", padx=20,
                     pady=5, fg="white", bg="#263D42", command=open_File)
openFile.place(relx=0.25, rely=0.09)

#star age preditction
startAP = tk.Button(frame, text="Start", padx=20,
                    pady=5, fg="white", bg="#263D42", command=start_AP)
startAP.place(relx=0.55, rely=0.09)

#text second
text1 = tk.Label(frame, text='\nFace Recognition'
                 ,bg='white', font="Arial")
text1.place(relx=0.33, rely=0.16)

#openfile1 Face Recognition
openFileRE1 = tk.Button(frame, text="Open file 1", padx=20,
                     pady=5, fg="white", bg="#263D42", command=open_File1)
openFileRE1.place(relx=0.07, rely=0.25)

#openfile2 Face Recognition
openFileRE2 = tk.Button(frame, text="Open file 2", padx=20,
                     pady=5, fg="white", bg="#263D42", command=open_File2)
openFileRE2.place(relx=0.37, rely=0.25)

#Start Face Recogntion
start = tk.Button(frame, text="Start", padx=20,
                     pady=5, fg="white", bg="#263D42", command=verify)
start.place(relx=0.67, rely=0.25)

# Set center screen tkinter
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2)
positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2)
root.geometry("+{}+{}".format(positionRight, positionDown))

root.mainloop()