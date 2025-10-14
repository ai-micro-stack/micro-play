import os
import fnmatch


def find_files_with_ext(root_dir, extensions, exclude_subdirs):
    found_files = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for dirname in list(dirs):
            for pattern in exclude_subdirs:
                if fnmatch.fnmatch(dirname, pattern):
                    dirs.remove(dirname)
                    break
        for file in files:
            if extensions == ".*" or file.endswith(extensions):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    return found_files
