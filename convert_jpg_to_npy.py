if __name__ == '__main__':
    from sys import argv
    from os.path import exists, basename
    from time import sleep
    from logger import Logger


    log = Logger(1)

    def usage():
        log(f"Usage:", a=1)
        log(f"\tpython {basename(__file__)} dest resolution obj-name id-range", a=1)
        log(f"Example:", a=1)
        log(f"\tpython {basename(__file__)} ./Datasets/x64/dataset/training/data 64 sharpner 12-146", a=1)
        log(f"Optional tags '-vv', '-v', '-q' can be used as last argument to specify verbosity.", a=1)

    # Check Args
    try:
        if '-h' in argv:
            log.v = 1
            usage()
            exit()
        dest = argv[1]
        res = int(argv[2])
        name = argv[3]
        span = argv[4].split("-")
        sl, sh = int(span[0]), int(span[1])
        if '-vv' == argv[-1]:
            log.v = 2
        elif '-v' == argv[-1]:
            log.v = 1
        elif '-q' == argv[-1]:
            log.v = 0
        if not exists(dest): raise FileNotFoundError
        log(f"Initialising in 5 seconds. Press ctrl + c to cancel. All already existing files will be overwritten.", a=0)
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
        log("Invalid destination path. Does it exist?", a=0)
        usage()
        exit()
    except KeyboardInterrupt:
        log("Abort", a=0)
        exit()

    # Import libries
    from PIL import Image
    import numpy as np
    from threading import Thread

    def con(in_path, out_path):
        fail = False
        try:
            img = Image.open(in_path)
            img = img.resize((res,res),Image.ANTIALIAS)
            img = np.array(img)
            np.save(out_path, img)
            status = "Saved"
        except FileNotFoundError:
            fail = True
            status = "Failed. Image not found."
        log(f"converting '{in_path}' to '{out_path}'... {status}", a=0 if fail else 2)

    log(f"Converting images...", a=1)

    for i in range(sl, sh + 1):
        in_path = f"Images/{name} ({i}).jpg"
        out_path = f"{dest}/{name} ({i}).npy"
        t = Thread(target=con, args=(in_path, out_path))
        t.start()

    log(f"Saving images...")



else:
    from PIL import Image
    import numpy as np
    from threading import Thread
    import os

    def convert_single(img_path, dest_path=None, resolution=64, r=False):
        try:
            full_img = Image.open(img_path)
            compressed_img = full_img.resize((resolution, resolution), Image.ANTIALIAS)
            npy_img = np.array(compressed_img)
            if r:
                np.save(dest_path, npy_img)
            else:
                return npy_img
        except FileNotFoundError:
            print(f"File not found:")
            print(f"\t{img_path}")

    def convert_bunch(dest_folder, root_img_folder, img_class, range_generator, resolution, t=True):
        if t:
            threads = []
            for img_id in range_generator:
                th = Thread(target=convert_single, args=(os.path.join(root_img_folder, img_class, f"{img_class} ({img_id}).jpg"), os.path.join(dest_folder, f"{img_class} ({img_id}).jpg"), resolution))
                threads.append(th)
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        else:
            for img_id in range_generator:
                convert_single(os.path.join(root_img_folder, img_class, f"{img_class} ({img_id}).jpg"), os.path.join(dest_folder, f"{img_class} ({img_id}).jpg"), resolution)

    def convert_heap(dest_folder, root_img_folder, img_classes, range_generator, resolution, t=True, st=True):
        if t:
            threads = []
            for img_class in img_classes:
                th = Thread(target=convert_bunch, args=(dest_folder, root_img_folder, img_class, range_generator, resolution, st))
                threads.append(th)
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        else:
            for img_class in img_classes:
                convert_bunch(dest_folder, root_img_folder, img_class, range_generator, resolution, st)
