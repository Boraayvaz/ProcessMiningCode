import subprocess
import sys
import os
import shutil
import ctypes
from PIL import Image  # for converting logo.jpg to logo.ico

# List of required libraries
required_libraries = [
    'tkinter', 'pandas', 'os', 'numpy', 'matplotlib', 'networkx', 'collections',
    'datetime', 'reportlab', 'random', 'time', 'threading', 'openpyxl', 'PIL', 'pywin32'
]

def check_python_version():
    if sys.version_info.major == 3:
        print(f"Python 3 is installed: {sys.version}")
    else:
        print("Python 3 is not installed or not being used.")
        sys.exit("Please install Python 3 to continue.")

def install_library(library):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {library}: {e}")
        sys.exit(1)

def check_and_install_libraries():
    for library in required_libraries:
        try:
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            print(f"{library} is missing. Installing...")
            install_library(library)

def copy_files_to_documents(source_folder):
    # Get the user's Documents folder
    documents_folder = os.path.join(os.path.expanduser("~"), "Documents")
    destination_folder = os.path.join(documents_folder, "ProsB")

    # Check if "ProsB" folder exists, if not create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy all files from source_folder to destination_folder
    try:
        for file_name in os.listdir(source_folder):
            full_file_name = os.path.join(source_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, destination_folder)
        print(f"All files have been copied to {destination_folder}.")
        return destination_folder
    except Exception as e:
        print(f"An error occurred while copying files: {e}")
        return None

def create_bat_file(destination_folder, program_name="ProgramCode.py"):
    bat_file_path = os.path.join(destination_folder, "run_program.bat")
    
    # Content of the batch file
    bat_content = f"""@echo off
cd /d "{destination_folder}"
python {program_name}
pause
"""

    try:
        # Writing the batch file content
        with open(bat_file_path, "w") as bat_file:
            bat_file.write(bat_content)
        print(f".bat file created successfully at {bat_file_path}")
        return bat_file_path
    except Exception as e:
        print(f"An error occurred while creating .bat file: {e}")
        return None

def convert_image_to_icon(image_path, icon_path):
    """Converts a .jpg image to a .ico file."""
    try:
        img = Image.open(image_path)
        img.save(icon_path, format='ICO', sizes=[(64, 64)])
        print(f"Icon created: {icon_path}")
        return icon_path
    except Exception as e:
        print(f"Failed to convert {image_path} to .ico: {e}")
        return None

def create_shortcut_to_desktop(bat_file_path, icon_path):
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')  # Desktop path
    shortcut_path = os.path.join(desktop, "ProsB.lnk")
    
    # Create shortcut using win32com.client
    try:
        from win32com.client import Dispatch
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(shortcut_path)
        shortcut.TargetPath = bat_file_path  # Point to the .bat file
        shortcut.WorkingDirectory = os.path.dirname(bat_file_path)  # Set the working directory
        shortcut.IconLocation = icon_path  # Set the icon file
        shortcut.save()
        print(f"Shortcut created on desktop: {shortcut_path}")
    except Exception as e:
        print(f"Failed to create shortcut: {e}")

if __name__ == "__main__":
    # 1. Check if Python 3 is installed
    check_python_version()

    # 2. Check if required libraries are installed, and install missing ones
    check_and_install_libraries()

    # 3. Copy files to "Documents/ProsB"
    source_folder = os.path.dirname(os.path.abspath(__file__))  # assuming files are in the same directory as the script
    destination_folder = copy_files_to_documents(source_folder)

    if destination_folder:
        # 4. Create a .bat file in the "Documents/ProsB" folder
        bat_file_path = create_bat_file(destination_folder)

        if bat_file_path:
            # 5. Convert logo.jpg to logo.ico
            logo_jpg_path = os.path.join(destination_folder, "logo.jpg")
            logo_ico_path = os.path.join(destination_folder, "logo.ico")
            icon_path = convert_image_to_icon(logo_jpg_path, logo_ico_path)

            if icon_path:
                # 6. Create a desktop shortcut that points to the .bat file
                create_shortcut_to_desktop(bat_file_path, icon_path)

    print("Setup completed successfully.")
