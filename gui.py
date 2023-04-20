import os
import tkinter as tk
from PIL import Image, ImageTk

# Create a variable to store the value of the radiobutton
import datetime
import csv



windowopenstate = 0

   
def callGUI(name):
   

    print(name)
    windowopenstate = 1
    # Get the current timestamp
    
    # Set up the main window
    root = tk.Tk()
    root.title("Image Voting")
    root.geometry("1000x500")
    radio_values=  tk.StringVar()

    personsdata={}

    # Create a canvas to hold the images and radio buttons
    canvas = tk.Canvas(root)
    canvas.pack(side="left", fill="both", expand=True)

    # Create a frame to hold the images and radio buttons inside the canvas
    image_frame = tk.Frame(canvas)
    image_frame.pack(side="top", fill="both", expand=True)

    # Add a scrollbar to the canvas
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Set up the canvas scrolling
    canvas.create_window((0, 0), window=image_frame, anchor="nw")

    def on_canvas_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    image_frame.bind("<Configure>", on_canvas_configure)

    # Load the images from the "candidates" folder
    image_dir = "candidates"
    image_files = os.listdir(image_dir)

    # Create a list to hold the radio buttons
    radio_buttons = []

    # Loop through the images and display them
    for i, image_file in enumerate(image_files):
        # Open the image file and resize it to 300x300 pixels
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        image = image.resize((300, 300))

        # Create a Tkinter PhotoImage object from the resized image
        photo = ImageTk.PhotoImage(image)

        # Create a label to display the image
        label = tk.Label(image_frame, image=photo)
        label.image = photo
        label.grid(row=0, column=i, padx=10, pady=10)
        personsdata[i]=image_file.split(".")[0]
        if(i==0):
            radio_values.set(i)

        radio_button = tk.Radiobutton(image_frame, text=image_file.split(".")[0],variable=radio_values, value=i)
        radio_button.grid(row=1, column=i, padx=10, pady=10)
        radio_buttons.append(radio_button)

    # Create a function to handle the vote button click
    def vote():
        print(personsdata)
        print(f'Selected option: {personsdata[int(radio_values.get())] }')
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        print('Current timestamp:', timestamp_str)
         # Open the CSV file in append mode
        with open('votes.csv', 'a+', newline='') as csvfile:
            # Create a writer object
            csvwriterr = csv.writer(csvfile)
            csvwriterr.writerow([timestamp_str, name, personsdata[int(radio_values.get())]])
        windowopenstate=0
        root.destroy()


    cancel_button = tk.Button(root, text="Cancel", command=root.destroy)
    cancel_button.pack(side="bottom", pady=10)

    # Create the vote button
    vote_button = tk.Button(root, text="Vote", command=vote)
    vote_button.pack(side="bottom", pady=10)

    # Start the main event loop
    root.mainloop()

