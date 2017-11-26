while True:
    f = open("scheduled.txt", "r")

    lines = f.readlines()

    comm = lines[0]

    del lines[0]

    f.close()

    import os
    if 0 == os.WEXITSTATUS(os.system(comm)):
        f = open("scheduled.txt","w")

        for line in lines:
            f.write(line)

        f.close()
