import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
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
    img.place(relx=0.23, rely=0.31, anchor='nw')

def start_AP():
    age = Load.analyzeImage1(path)
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
        messagebox.showinfo(title='Kết quả', message='Tuổi dự đoán là: '  + str(int(age)) )



# anchor = n, ne, e, se, s, sw, w, nw, or center
# window size
canvas = tk.Canvas(root, height=350, width=400)
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
openFile.place(relx=0.25, rely=0.15)

#star age preditction
startAP = tk.Button(frame, text="Start", padx=20,
                    pady=5, fg="white", bg="#263D42", command=start_AP)
startAP.place(relx=0.55, rely=0.15)


# Set center screen tkinter
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2)
positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2)
root.geometry("+{}+{}".format(positionRight, positionDown))

root.mainloop()