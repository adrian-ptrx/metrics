# cleanup
rm -rf dist build *.egg*

#create whl
python setup.py sdist bdist_wheel

