import os

from src.utility import get_project_dir
import matplotlib.pyplot as plt


category_dir = os.path.join(get_project_dir(), "data", "raw", "vinted_shirts")
len_list = []
for x in os.walk(category_dir):
    dir = x[0]
    files = [ f for f in x[2] if f.endswith(".jpg")]

    plt.figure(figsize=(16,16))
    for i,f in enumerate(files):
        plt.subplot(4,5,i+1)
        img = plt.imread(os.path.join(dir,f))
        plt.imshow(img)
        plt.axis("off")
    plt.show()
    while True:
        answer = input()
        if answer == "y":
            plt.close()
            break
        if answer == "n":
            print("everything is gone!")


    len_list.append(len(files))

plt.hist(len_list, bins=20)
plt.show()
print(min(len_list), max(len_list))