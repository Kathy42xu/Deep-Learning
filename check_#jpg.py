import os

def count_jpg_files_in_subfolders(parent_folder_path):
    total_jpg_count = 0

    # Loop through all the subfolders in the parent folder
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)

        # Check if the path is a directory before counting
        if os.path.isdir(folder_path):
            jpg_count = len([name for name in os.listdir(folder_path) if name.endswith('.jpg')])
            total_jpg_count += jpg_count
    
    return total_jpg_count

# Example usage
parent_folder_path = '/home/20074688d/jtt-master copy/jtt/cub/data/waterbird_complete95_forest2water2'
print(count_jpg_files_in_subfolders(parent_folder_path))
