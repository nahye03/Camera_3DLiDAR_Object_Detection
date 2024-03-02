import os

with open("train_yolo.txt", "r") as input_file, open("train_cp.txt", "w") as output_file:
    for line in input_file:
        values = line.split('/')

        modified_line = str(values[-1].split(".")[0])
        print(modified_line)
        output_file.write(modified_line + "\n")

print("Conversion complete.")