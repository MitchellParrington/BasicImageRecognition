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
    objects = []
    for file in scandir(f"{path}/data"):
        data = np.load(f"{path}/data/{file.name}")
        data = np.expand_dims(data, 0)
        objects.append(data)
        log(f"loading '{path}/data/{file.name}' -> {data.shape}", a=2)
    input_data = np.vstack(objects)
    log(f"Data shape: {input_data.shape}", a=1)

    log(f"Saving data...", a=1)
    np.save(f"{path}/input", input_data)


else:
    import numpy as np
    import os

    def generate_input_data(path):
        arrays = []
        for file in os.scandir(path):
            data = np.load(file)
            data = np.expand_dims(data, 0)
            arrays.append(data)
        arrays = np.vstack(arrays)
        return arrays
