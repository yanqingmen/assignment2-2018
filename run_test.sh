SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH="${PYTHONPATH}:${SHELL_FOLDER}/python"
nosetests -v tests/test_tvm_op.py
