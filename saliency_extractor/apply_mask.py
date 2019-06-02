'''
TODO document this file
'''
import sys

def iterate_videos(video, mask):

    v_in = cv2.VideoCapture(video)
    m_in = cv2.VideoCapture(mask)

    width = int(v_in.get(3) + m_in.get(3))
    height = int(v_in.get(4))

    output_name = clean_string(video)
    v_out = cv2.VideoWriter('output/masked/' + output_name + '.avi', fourcc, v_in.get(5), (width, height))

    while(v_in.isOpened()):

        f1, v_f = v_in.read()
        f2, m_f = m_in.read()

        if f1:

        else:
            break

        v_in.release()
        m_in.release()
        v_out.release()

def main():
    video = sys.argv[1]
    mask = sys.argv[2]

    iterate_videos(video, mask)

main()
