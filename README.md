# Lab exercises for COMP0088 Introduction to Machine Learning

**WARNING: This repository is obsolete. Current COMP0088 students should use the lab exercises from the [colab](https://github.com/comp0088/colab) repo instead. These files are really only kept around for historical interest.**

-------

## TL;DR

* Install [Git](https://git-scm.com) (if you don't already have it)
* Install [Python 3](https://www.python.org/downloads/) (if you don't already have it - AVOID PYTHON 3.10) 
* Download [Visual Studio Code](https://code.visualstudio.com/), an easy-to-use editor
* Clone this repository to your local machine:
    ```sh
    git clone https://github.com/comp0088/labs.git
    ``` 
* Open the cloned folder in Visual Studio Code: File > Open Folder > select the cloned repository folder
* Open a new terminal in Visual Studio Code and make sure you are in the folder 'labs'
* Create and activate a virtual environment:
    ```
    cd labs
    python3 -m venv venvcomp0088
    # On Unix/MacOS:
    source venvcomp0088/bin/activate
    # On Windows:
    .\venvcomp0088\Scripts\activate
    # Upgrade pip:
    python -m pip install --upgrade pip
    ```
    If you get the error "python3 : The term 'python3' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the     name, or if a path was included, verify that the path is correct and try again", try using:
    ```
    python -m venv venvcomp0088
    ```
    If you are encountering issues, you can create a conda environment
    ```
    conda create -n venvcomp0088 python=3.9 -y
    conda activate venvcomp0088
    ```
* Install Python package requirements:
    ```sh
    pip install -r requirements.txt
    ```
    In case packages are missing, you can just run
    ```sh
    pip install _package_
    ```
* On the bottom left of you VS Code window you sould see a "Select Interpreter" button:
    * Click on the button
    * Select "Enter interpreter path..."
    * Select "Find..."
        * Go to your virtual environment folder in "labs" > Scripts > python.exe > Select Interpreter (Windows) 
        * Go to your virtual environment folder in "labs" > bin > python.exe (or just python if you don't see the .exe extension)> Select Interpreter (Unix/MacOS) 
    * Insted of "Select Interpreter" you should now see "3.x.x('venvcomp0088':venv)
   
* Read the week's lab exercises document `lab_N.pdf`
* Add your code to the script `week_N.py`
* Run the script from the command line to see your code in action:
    ```sh
    python week_N.py
    ```


## About

This repository contains lab exercises for the [COMP0088 Introduction to Machine Learning](https://moodle.ucl.ac.uk/course/view.php?id=1442) module for taught MSc students at UCL, delivered in Autumn 2021. Exercises are designed to be attempted in the on-campus lab sessions on Thursday mornings, though you are free to do additional work in your own time if you wish.

Lab attendance will be monitored, but the exercises are **not graded**. They are intended as learning experiences, to help you understand and apply different machine learning algorithms. You are welcome to discuss and help each other with these tasks and to ask for assistance and clarification from the TAs, but there is nothing to be gained by simply copying each others' work.

### Contents

Exercises for week *N* are specified in document `lab_N.pdf`. Skeleton code for the exercises is provided in the script `week_N.py`. You should add you solution code to this file. The script can be run at the command line like this:
```sh
python week_N.py
```
Each week's script will print some progress messages and generate an output file called `week_N.pdf` containing a number of plots. An example showing (approximately) what the finished output should look like is included in the exercises document.

In addition to the spec and script for each week, there are a few other files in the repo:

* `README.md`: this file.
* `plotting_examples.py`: a script containing a few examples of plotting with Matplotlib for those unfamiliar with this.
* `utils.py`: a library of utility functions that are used by the script code, or are available to be used by your own solution code.
* `requirements.txt`: a list of additional Python packages to install.
* `LICENSE`: text of the MIT License that applies to all code and documentation in this repository. (Summary: in the unlikely event that you have any reason to do so, you are free to reuse this material for any purpose you like.)


## Cloning & Updating

In order to copy the repo you need to have a working installation of the Git version control system. If you don't already have this, you can [download it here](https://git-scm.com).

Choose a convenient location for your working directory and download an initial copy of the repository with `git`:
```sh
git clone https://github.com/comp0088/labs.git
```
New lab exercises will be added to the repo each week. You can download updates with the following commands:
```sh
git fetch
git merge origin/main
```
Note however that `git` may report issues when you try to merge our upstream changes with your own if you have uncommitted changes in your directory, or if you have made changes that conflict with changes we have made in the main repo.

We will do our best to avoid making any changes that are likely to cause conflicts. You can generally assume that the `week_N.py` files will not be updated and can be freely edited and committed. (You should not assume this about `utils.py` or `requirements.txt` -- try to avoid editing these files if you can.) If conflicting changes do become necessary, for example to fix significant errors in one of the supplied scripts, we will announce it on Moodle and explain what you'll need to do about it.

We recommend using `git` to track your own changes as you work on the exercises. Commit your work at appropriate intervals and only `fetch/merge` new changes when your own changes are up to date.

However, if you have made changes that you don't want to commit for some reason, you can also temporarily get them out of the way using [`git stash`](https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning):
```sh
git stash push
git fetch
git merge origin/main
git stash apply
```

## Python Setup

The exercises require a local installation of Python 3, along with a number of additional packages for numerical programming, plotting and machine learning. We suggest using the latest stable release of Python 3.9 (currently 3.9.7) from [python.org](https://www.python.org/downloads/). (Python 3.10 has recently been released but many students have had issues with PyPI dependencies not yet being up to date, so install this version with caution.) If you know and prefer [Anaconda](https://www.anaconda.com/products/individual-d)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html) you are welcome to use that instead -- the included `comp88.yml` will install most dependencies, but you will also need to install the appropriate version of PyTorch for your system. Use [this configuration selector](https://pytorch.org/get-started/locally/) to determine the command to run.

(Although we recommend a more recent Python, the code has also been tested on Python 3.6.8, which is the version currently installed on some of the CS lab machines. It is possible, albeit suboptimal, to set up and run the exercises on one of those machines via SSH.)

### Virtual Environments

The package requirements for the lab exercises are pretty vanilla, but we strongly recommend working in a dedicated [virtual environment](https://docs.python.org/3/tutorial/venv.html) in order to avoid any conflicts or compatibility issues with any other Python work you may be doing.

There are several options for how and where to set up such a virtual environment. If you already have experience doing so then feel free to use any configuration you are comfortable with. If you haven't done this before and/or would rather not think about it, follow the default setup instructions below.

#### Default virtual environment setup

A straightforward way to configure your virtual environment is to store it in a hidden subdirectory of your working directory (ie, the directory containing this repository). Once you have cloned the repo as described above, change into the directory:
```sh
cd labs
```
Initialise a new virtual environment:
```sh
python3 -m venv .venv
```
(The name `.venv` is a reasonably common convention that should be recognised by Python-aware editors such as VSCode and PyCharm, but you can use a different and more informative name if you wish. In that case, also replace `.venv` with your chosen name in the commands below.)

**Make the virtual environment active**. This mean it will be used for any python commands or scripts you execute in the current shell. Activation occurs only for the specific terminal window you do the activation in, and it ends when you close the window (or issue the comment `deactivate`). So you'll need to do this every time you open a new terminal that you want to run lab scripts from.

The command you use to activate the environment varies depending on your operating system and shell.

On Unix-esque systems (Linux and MacOS) running a Bourne shell variant such as `bash` or `zsh` (this is usually the default), use:
```sh
source .venv/bin/activate
```
(If you are instead running a variant of the C-shell, `csh` or `tcsh`, you should instead `source .venv/bin/activate.csh`.)

On Windows using the standard `CMD.EXE` terminal, use:
```sh
.venv\Scripts\activate.bat
```
(If you are running PowerShell, you should instead run `.venv\Scripts\Activate.ps1`. However, note that the default Windows configuration for PowerShell [blocks script execution](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1) as a security feature. You'll need to [bypass this](https://superuser.com/questions/106360/how-to-enable-execution-of-powershell-scripts) before being able to activate the virtual environment in PowerShell. It is possible that you may be using PowerShell without knowing it -- the Windows versions of both VSCode and PyCharm can use PowerShell as their terminal.)

As a special case, if you are using the MINGW64 `bash` terminal that is installed by Git (among others) on Windows, then it is masquerading as a Unix environment but your virtual environment will have been set up with Windows naming conventions, so you need to use the following hybrid command:
```sh
source .venv/Scripts/activate
```
When the virtual environment is active, your commmand prompt will be modified with the prefix `(.venv)`.

### Installing Required Packages

With your virtual environment active, you should be able to install all required packages using `pip`, like this:
```sh
pip install -r requirements.txt 
```


## Working Environment

If you are an experienced Python coder with a preferred development environment, you should use that. Note, however, that the exercises are not really designed to be run within Jupyter Notebook. You will likely be able to if you really insist, but in general there will be no benefit to doing so and it probably won't be very convenient.

If you don't have any existing preference, we recommend using a Python-aware editor such as [VS Code](https://code.visualstudio.com) (be sure to install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)) or the [PyCharm IDE](https://www.jetbrains.com/pycharm/) (the Pro version is free to students), together with an [IPython](https://ipython.readthedocs.io/) interactive session running in a terminal window for testing and debugging.


## Feedback

Please post questions, comments, issues or bug reports to the [COMP0088 Moodle forum](https://moodle.ucl.ac.uk/mod/hsuforum/view.php?id=3184621) or raise them with the TAs during your lab sessions.

