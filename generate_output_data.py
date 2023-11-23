if __name__ == '__main__':
    from sys import argv
    from os.path import exists, basename
    from os import scandir
    from time import sleep
    from logger import Logger

    log = Logger(1)

    def usage():
        log(f"Usage:", a=1)
        log(f"\tpython {basename(__file__)} dataset-data-dest", a=1)
        log(f"Example:", a=1)
        log(f"\tpython {basename(__file__)} ./Datasets/x64/dataset/training", a=1)
        log(f"Optional tags '-vv', '-v', '-q' can be used as last argument to specify verbosity.", a=1)

    # Check Args
    try:
        if '-h' in argv:
            log.v = 1
            usage()
            exit()
        if '-vv' == argv[-1]:
            log.v = 2
        elif '-v' == argv[-1]:
            log.v = 1
        elif '-q' == argv[-1]:
            log.v = 0
        path = argv[1]
        if not exists(path): raise FileNotFoundError
        log(f"Initialising in 5 seconds. Press ctrl + c to cancel. Existing input.npy file will be overwritten.", a=0)
        sleep(5)
    except IndexError:
        log("Please provide all arguments.", a=0)
        usage()
        exit()
    except ValueError:
        log("Intiger value passed improperly.", a=0)
        usage()
        exit()
    except FileNotFoundError:
        log("Invalid path. Does it exist?", a=0)
        usage()
        exit()
    except KeyboardInterrupt:
        log("Abort", a=0)
        exit()

    # Import libries
    log(f"Importing libries...", a=1)
    import numpy as np

    log(f"Gathering objects...", a=1)

    names = []
    counts = []
    for file in scandir(f"{path}/data"):
        fname = file.name.split(" ")[0]
        if not fname in names:
            names.append(fname)
            counts.append(1)
        else:
            counts[names.index(fname)] += 1

    output_data = np.zeros((sum(counts), len(counts)))

    count = 0
    for ix, c in enumerate(counts):
        for i in range(c):
            output_data[count][ix] = 1
            count += 1

    key = np.rot90(np.array([names, counts]))

    log(f"Saving data...", a=1)
    np.save(f"{path}/output", output_data)
    np.save(f"{path}/key", key)



    log(f"", a=1)
    log(f"Objects:", *names, a=1)
    log(f"Counts:", *counts, a=1)


else:
    import numpy as np
    import os

    def generate_output_data(path):
        names  = []
        counts = []
        for file in os.scandir(path):
            object_class = file.name.split(" ")[0]

            if not object_class in names:
                names.append(object_class)
                counts.append(1)
            else:
                counts[names.index(object_class)] += 1

        output_data = np.zeros((sum(counts), len(counts)))

        count = 0
        for ix, c in enumerate(counts):
            for i in range(c):
                output_data[count][ix] = 1
                count += 1

        key = np.rot90(np.array([names, counts]))
        return output_data, key
