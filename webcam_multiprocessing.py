from multiprocessing import Process, Queue, Pipe

# Pipe for two-way communication only. Should be faster.
# Queue is a process - and thread - safe implementation with an underlying Pipe

def plot_detections(connection_obj):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    i = 0
    while True:
        try:
            data = connection_obj.recv()
            if i == 0:
                # initialize figure on first message received
                img_size = data
                background = np.ones((img_size[0], img_size[1], img_size[2]), dtype=np.uint8) * 255
                fig = plt.figure()
                disp = plt.imshow(background)
                plt.ion()
            else:
                if isinstance(data, str) and data == "END":
                    print('Last message received')
                    break
                overlay = np.ones((img_size[0], img_size[1], img_size[2]), dtype=np.uint8) * 255
                #print(data)
                [cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,255),2) for (x,y,w,h) in data]
                disp.set_data(overlay)
                plt.pause(0.001)
            i = i + 1
        except EOFError:
            print('Communication end')
            break
    
    print('Closing matplotlib')
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    plt.ioff()
    plt.show()
    plt.close('all')

def cam_output_face_detect(connection_obj, flip=True, haar_frontal='haarcascade_frontalface_default.xml', haar_second=None):
    import cv2
    import matplotlib.pyplot as plt
    # init cam
    cam = cv2.VideoCapture(0)
    # capture first frame
    ret, frame = cam.read()
    h, w, c = frame.shape
    
    # first send the height and width of image
    connection_obj.send([h, w, c])

    # initialize figure
    fig = plt.figure()
    disp = plt.imshow(frame)
    plt.ion()

    frontal_cascade = cv2.CascadeClassifier(haar_frontal)
    if haar_second:
        second_cascade = cv2.CascadeClassifier(haar_second)

    while True:
        ret, frame = cam.read()
        if flip:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = frontal_cascade.detectMultiScale(gray, 1.05, 6)
        connection_obj.send(faces)
        [cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) for (x,y,w,h) in faces]
        # if haar_second:
        #     second_faces = second_cascade.detectMultiScale(gray, 1.05, 6)
        #     [cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) for (x,y,w,h) in second_faces]
        disp.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.001)
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()
    plt.close('all')
    connection_obj.send("END")
    connection_obj.close()

if __name__ == '__main__':
    #haar_profile = 'haarcascade_profileface.xml'
    # create pipe
    recv_conn, send_conn = Pipe(duplex=False)

    # create new processes
    proc_1 = Process(target=cam_output_face_detect, args=(send_conn,))
    proc_2 = Process(target=plot_detections, args=(recv_conn,))

    # run processes
    proc_1.start()
    proc_2.start()

    # wait until processes finish
    proc_1.join()
    proc_2.join()