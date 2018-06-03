SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH="${PYTHONPATH}:${SHELL_FOLDER}/python"
python3 tests/mnist_dlsys.py -l -m mlp
