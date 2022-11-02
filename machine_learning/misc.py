from pathlib import Path
from shutil import move, make_archive, rmtree
from re import match
from os import mkdir, getcwd, listdir
from os.path import exists


def zip_run_name_files(run_name, output_dir, zip_dir):

    # Make sure a si_no_outliers directory exist
    if not exists(output_dir):
        raise OSError(f"Directory {output_dir} does not exist")

    # The directory where files are now
    current_dir = Path(getcwd()).absolute()

    # The directory to put files into
    working_dir = output_dir / run_name
    mkdir(working_dir)

    # Move all files from current dir to working dir
    for f in listdir():
        if match(run_name, f):
            move(current_dir / f, working_dir / f)

    # Zip the new directory
    if zip_dir:
        make_archive(working_dir, 'zip', working_dir)
        rmtree(working_dir)
