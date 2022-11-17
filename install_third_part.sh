cd /home/ma-user/modelarts/user-job-dir/WS3/utils/nearest_neighbors || exit
which python
python setup.py develop
cp /home/ma-user/modelarts/user-job-dir/WS3/utils/nearest_neighbors/nearest_neighbors.cpython-37m-aarch64-linux-gnu.so \
  /home/ma-user/modelarts/user-job-dir/WS3/utils/nearest_neighbors/lib/python/
ls /home/ma-user/modelarts/user-job-dir/WS3/utils/nearest_neighbors/lib/python/
echo "installation finish"
pip3 list
