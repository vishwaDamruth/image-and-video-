#!/usr/bin/env python
# coding: utf-8

# In[12]:


import ffmpeg
print(ffmpeg)


# In[13]:


#help(ffmpeg)
print(dir(ffmpeg))


# In[14]:


probe = ffmpeg.probe("C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/deadpool.mp4")
video_info = None
for stream in probe['streams']:
            if stream['codec_type'] == 'video':
                video_info = stream
                break
        
if video_info is None:
    raise ValueError("No video stream found in the input file.")
        

frame_count = int(video_info['nb_frames'])
frame_rate = eval(video_info['r_frame_rate'])  
width = int(video_info['width'])
height = int(video_info['height'])
duration = float(video_info['duration'])
pix_fmt = video_info.get('pix_fmt', 'unknown')
codec_name = video_info.get('codec_name', 'unknown')
bit_rate = video_info.get('bit_rate', 'unknown')
avg_frame_rate = eval(video_info.get('avg_frame_rate', '0/1'))
time_base = video_info.get('time_base', 'unknown')
color_space = video_info.get('color_space', 'unknown')
color_range = video_info.get('color_range', 'unknown')
profile = video_info.get('profile', 'unknown')
codec_tag_string = video_info.get('codec_tag_string', 'unknown')
level = video_info.get('level', 'unknown')




        
print(f"Frame Count: {frame_count}")
print(f"Frame Rate: {frame_rate} fps")
print(f"Resolution: {width}x{height}")
print(f"Duration: {duration} seconds")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"Pixel Format: {pix_fmt}")
print(f"Codec Name: {codec_name}")
print(f"Bit Rate: {bit_rate}")
print(f"Average Frame Rate: {avg_frame_rate}")
print(f"Time Base: {time_base}")
print(f"Color Space: {color_space}")
print(f"Color Range: {color_range}")
print(f"Profile: {profile}")
print(f"Codec Tag: {codec_tag_string}")
print(f"Level: {level}")


# In[18]:


import ffmpeg
import matplotlib.pyplot as plt

input_file = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/deadpool.mp4'

try:
    probe = ffmpeg.probe(input_file, select_streams='v', show_frames=None, show_entries='frame=pict_type')

    frames = probe['frames']
    frame_count = len(frames)

    i_frames = 0
    p_frames = 0
    b_frames = 0

    for frame in frames:
        if frame['pict_type'] == 'I':
            i_frames += 1
        elif frame['pict_type'] == 'P':
            p_frames += 1
        elif frame['pict_type'] == 'B':
            b_frames += 1

    i_frame_percentage = (i_frames / frame_count) * 100
    p_frame_percentage = (p_frames / frame_count) * 100
    b_frame_percentage = (b_frames / frame_count) * 100

    print(f"I-Frames: {i_frames} ({i_frame_percentage:.2f}%)")
    print(f"P-Frames: {p_frames} ({p_frame_percentage:.2f}%)")
    print(f"B-Frames: {b_frames} ({b_frame_percentage:.2f}%)")

    labels = ['I-Frames', 'P-Frames', 'B-Frames']
    sizes = [i_frame_percentage, p_frame_percentage, b_frame_percentage]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  

    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.axis('equal')  
    plt.title('Distribution of Frame Types')
    plt.show()

except ffmpeg.Error as e:
    print(f"An error occurred: {e.stderr.decode()}")
    
print(f"No of I frames in the video: {i_frames}")
print(f"No of P frames in the video: {p_frames}")
print(f"No of B frames in the video: {b_frames}")


# In[18]:


import glob
from PIL import Image

# Extract I-Frames
ffmpeg.input(input_file).output('I_frame_%03d.png', vf='select=eq(pict_type\\,I)', vsync='vfr').run()

# Extract P-Frames
ffmpeg.input(input_file).output('P_frame_%03d.png', vf='select=eq(pict_type\\,P)', vsync='vfr').run()

# Extract B-Frames
ffmpeg.input(input_file).output('B_frame_%03d.png', vf='select=eq(pict_type\\,B)', vsync='vfr').run()

# Display I-Frames
i_frames = sorted(glob.glob("I_frame_*.png"))
for frame_path in i_frames[:3]:  # Display the first 3 I-frames
    img = Image.open(frame_path)
    plt.figure()
    plt.imshow(img)
    plt.title(f"Displaying: {frame_path}")
    plt.axis('off')
    plt.show()

# Display P-Frames
p_frames = sorted(glob.glob("P_frame_*.png"))
for frame_path in p_frames[:3]:  # Display the first 3 P-frames
    img = Image.open(frame_path)
    plt.figure()
    plt.imshow(img)
    plt.title(f"Displaying: {frame_path}")
    plt.axis('off')
    plt.show()

# Display B-Frames
b_frames = sorted(glob.glob("B_frame_*.png"))
for frame_path in b_frames[:3]:  # Display the first 3 B-frames
    img = Image.open(frame_path)
    plt.figure()
    plt.imshow(img)
    plt.title(f"Displaying: {frame_path}")
    plt.axis('off')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


import ffmpeg
import os

input_file = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/deadpool.mp4'

output_directory = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/images/'

os.makedirs(output_directory, exist_ok=True)

i_frame_pattern = os.path.join(output_directory, 'I_frame_%03d.png')
p_frame_pattern = os.path.join(output_directory, 'P_frame_%03d.png')
b_frame_pattern = os.path.join(output_directory, 'B_frame_%03d.png')

ffmpeg.input(input_file).output(i_frame_pattern, vf="select=eq(pict_type\\,I)", vsync='vfr').run()
print("I frames extracted and saved as images.")

ffmpeg.input(input_file).output(p_frame_pattern, vf="select=eq(pict_type\\,P)", vsync='vfr').run()
print("P frames extracted and saved as images.")

ffmpeg.input(input_file).output(b_frame_pattern, vf="select=eq(pict_type\\,B)", vsync='vfr').run()
print("B frames extracted and saved as images.")


# In[ ]:





# In[ ]:





# In[22]:


import cv2
import os
import glob

output_directory = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/images/'
i_frame_pattern = os.path.join(output_directory, 'I_frame_*.png')
p_frame_pattern = os.path.join(output_directory, 'P_frame_*.png')
b_frame_pattern = os.path.join(output_directory, 'B_frame_*.png')


def display_images(pattern):
    for image_file in glob.glob(pattern):
        img = cv2.imread(image_file)
        if img is not None:
            cv2.imshow(f"Frame: {image_file}", img)
            cv2.waitKey(0) 
            
            
            
            
            
            
            
            
            cv2.destroyAllWindows()


print("Displaying I frames...")
display_images(i_frame_pattern)

print("Displaying P frames...")
display_images(p_frame_pattern)

print("Displaying B frames...")
display_images(b_frame_pattern)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


"""import os

def calculate_frame_sizes(frame_pattern):
    frame_files = sorted(glob.glob(frame_pattern))
    
    sizes = [os.path.getsize(f) for f in frame_files]
    return sizes


i_frame_sizes = calculate_frame_sizes("I_frame_*.png")
p_frame_sizes = calculate_frame_sizes("P_frame_*.png")
b_frame_sizes = calculate_frame_sizes("B_frame_*.png")


def average_size(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

i_frame_avg_size = average_size(i_frame_sizes)
p_frame_avg_size = average_size(p_frame_sizes)
b_frame_avg_size = average_size(b_frame_sizes)

print(f"Average I-Frame Size: {i_frame_avg_size:.2f} bytes")
print(f"Average P-Frame Size: {p_frame_avg_size:.2f} bytes")
print(f"Average B-Frame Size: {b_frame_avg_size:.2f} bytes")
"""


# In[37]:


import os
import glob

def calculate_frame_sizes(frame_pattern):
    frame_files = sorted(glob.glob(frame_pattern))
    sizes = [os.path.getsize(f) for f in frame_files]
    return sizes

def average_size(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

i_frame_sizes = calculate_frame_sizes("I_frame_*.png")
p_frame_sizes = calculate_frame_sizes("P_frame_*.png")
b_frame_sizes = calculate_frame_sizes("B_frame_*.png")

i_frame_avg_size = average_size(i_frame_sizes)
p_frame_avg_size = average_size(p_frame_sizes)
b_frame_avg_size = average_size(b_frame_sizes)

print(f"Average I-Frame Size: {i_frame_avg_size:.2f} bytes")
print(f"Average P-Frame Size: {p_frame_avg_size:.2f} bytes")
print(f"Average B-Frame Size: {b_frame_avg_size:.2f} bytes")

print(i_frame_sizes)
print(p_frame_sizes)
print(b_frame_sizes)


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


import ffmpeg
import os


input_video_path = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/deadpool.mp4'

i_frame_dir = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/i_frames'

os.makedirs(i_frame_dir, exist_ok=True)

ffmpeg.input(input_video_path).output(f'{i_frame_dir}/frame_%04d.jpg', vf='select=eq(pict_type\\,I)').run()


# In[24]:


import cv2
import os

i_frame_dir = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/i_frames'

output_video_path = 'C:/Users/Radcoflex-Purchase/Desktop/COLLEGE/SEM 7/Image and Video Analytics/Lab 2/result.mp4'

frame_rate = 25

frame_files = sorted([f for f in os.listdir(i_frame_dir) if f.endswith('.jpg')])

first_frame = cv2.imread(os.path.join(i_frame_dir, frame_files[0]))
height, width, layers = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

for frame_file in frame_files:
    frame = cv2.imread(os.path.join(i_frame_dir, frame_file))
    video_writer.write(frame)

video_writer.release()

print(f'Reconstructed video saved as {output_video_path}')

