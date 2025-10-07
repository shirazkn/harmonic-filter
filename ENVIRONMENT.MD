Some handy tips for installing and managing Python environments.

## On MacOS~
- Install Homebrew
- Install Direnv using `brew install direnv`. You also need to add something like `eval "$(direnv hook bash)"` to `~/.bashrc` (if using a Bash shell). Check online.
- `brew install python3.13`
- Go to the project directory, e.g., "`cd ~/harmonic-filter`"
- `echo "layout python python3.13" >> .envrc`
- `direnv allow`

Run `which python` and `which pip` to ensure that both point to the Python environment installed within the `harmonic-filter` directory. Run the `python` command inside and outside the directory to double-check.


## On Ubuntu~
- `sudo apt-get update`
- `sudo apt-get install direnv`, then add `eval "$(direnv hook bash)"` to `~/.bashrc`. Check online.
- `sudo add-apt-repository ppa:deadsnakes/ppa`
- `sudo apt update`
- `sudo apt install python3.13-venv` (without the `-venv`, it doesn't install pip)
- Same as MacOS.

## Installing Requirements
- `pip install -r requirements.txt`
