import os
import convert_jpg_to_npy as tonpy
import global_data as gd
from generate_input_data import generate_input_data
from generate_output_data import generate_output_data

import numpy as np

def create_dir_struct(root_path, dataset_name):
    os.makedirs(os.path.join(root_path, dataset_name))
    os.makedirs(os.path.join(root_path, dataset_name, "Models"))
    os.makedirs(os.path.join(root_path, dataset_name, "Training", "data"))
    os.makedirs(os.path.join(root_path, dataset_name, "Testing", "data"))
    return os.path.join(root_path, dataset_name)

if __name__ == '__main__':
    import os
    import sys
    import time
    from logger import Logger

    log = Logger(1)

    def usage():
        log("\nUsage:", a=1)
        log(f"python {os.path.basename(__file__)} dest root-img-path name id-range img-classes -h [-vv | -v | -q]", a=1)
        log(f"\tdataset-name          : Name of the dataset to be created.", a=1)
        log(f"\ttraining-img-id-range : Path to the root of the image folder.", a=1)
        log(f"\testing-img-id-range   : Dataset root folder name.", a=1)
        log(f"\timg-classes           : Comma seperated (no whitespace) image class names to be initialised within dataset.", a=1)
        log(f"\t                        Eg: 'pencil,sharpner,marker,bottle' NOT: 'pencil, sharpner, marker, bottle'", a=1)
        log(f"\t                        Specify 'null' to not innitialise images.", a=1)
        log(f"\t-h                    : Display help (this).", a=1)
        log(f"\t-v                    : Verbose output (log to console). Default", a=1)
        log(f"\t-vv                   : Extra verbose output (logs everything).", a=1)
        log(f"\t-q                    : Quiet output (dont log to console).", a=1)
        log(f"", a=1)
        log("Abort", a=0)

    try:
        dataset_name = sys.argv[1]
        img_range_training = [int(i) for i in sys.argv[2].split('-')][:2]; img_range_training[1] += 1
        img_range_testing = [int(i) for i in sys.argv[3].split('-')][:2]; img_range_testing[1] += 1
        img_classes = sys.argv[4].split(",")
        if os.path.exists(os.path.join(gd.RootPaths.Datasets, dataset_name)): raise FileExistsError

        for class_name in img_classes:
            if not os.path.exists(os.path.join(gd.RootPaths.Images, class_name)): raise FileNotFoundError

        # Parse flags / parameters
        args = sys.argv
        if '-h' in args:
            log.v = 1
            usage()
            exit()
        if '-vv' in args:
            log.v = 2
        elif '-v' in args:
            log.v = 1
        elif '-q' in args:
            log.v = 0

        log(f"Initialising in 5 seconds. Press ctrl + c to cancel.", a=0)
        time.sleep(5)

    except ValueError as e:
        log(f"Invalid value passed.", a=0)
        log("\t", e, a=0)
        usage()
        exit()
    except IndexError as e:
        log(f"Expected 4 valid arguments.", a=0)
        log("\t", e, a=0)
        usage()
        exit()
    except FileExistsError:
        log(f"Dataset with that name already exists.", a=0)
        usage()
        exit()
    except FileNotFoundError:
        log(f"{img_classes} contains an invalid class name. 1 or more file(s) not found.", a=1)
        exit()
    except KeyboardInterrupt:
        log(f"Abort", a=0)
        exit()

    log(f"Creating directories...", a=1)
    dataset_root_folder = create_dir_struct(gd.RootPaths.Datasets, dataset_name)

    if img_classes[0] != "null":
        log(f"Transfering data...", a=1)
        tonpy.convert_heap(os.path.join(dataset_root_folder, "Training", "data"), gd.RootPaths.Images, img_classes, range(*img_range_training), 64, t=True, st=True)
        tonpy.convert_heap(os.path.join(dataset_root_folder, "Testing", "data"), gd.RootPaths.Images, img_classes, range(*img_range_testing), 64, t=True, st=True)

        np.save(os.path.join(dataset_root_folder, "Training", "input"), generate_input_data(os.path.join(dataset_root_folder, "Training", "data")))
        output, key = generate_output_data(os.path.join(dataset_root_folder, "Training", "data"))
        np.save(os.path.join(dataset_root_folder, "Training", "output"), output)
        np.save(os.path.join(dataset_root_folder, "Training", "key"), key)

        np.save(os.path.join(dataset_root_folder, "Testing", "input"), generate_input_data(os.path.join(dataset_root_folder, "Testing", "data")))
        output, key = generate_output_data(os.path.join(dataset_root_folder, "Testing", "data"))
        np.save(os.path.join(dataset_root_folder, "Testing", "output"), output)
        np.save(os.path.join(dataset_root_folder, "Testing", "key"), key)
