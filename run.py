import os





for i in range (5):
    os.system("python3 classify.py --image_file /home/abel/scratch/output/out_%s.jpg"%i)
#os.system("python3 classify.py --image_file /home/abel/scratch/out.jpg")

