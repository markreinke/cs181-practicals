yum update -y
yum update -y gcc
yum update -y gcc-c++
yum update -y gcc-gfortran
yum groupinstall "Development Tools"
yum update -y python-devel libpng-devel freetype-devel
yum update -y libxml12 libxml12-devel libxslt libxslt-devel
yum update -y python-setuptools
pip install --upgrade --no-cache-dir wheel
pip install --upgrade --no-cache-dir cython
pip install --upgrade --no-cache-dir pycparser
pip install --upgrade --no-cache-dir toolz
pip install --upgrade --no-cache-dir pytz

/bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=3072
/sbin/mkswap /var/swap.1
/sbin/swapon /var/swap.1

pip install --upgrade --no-cache-dir pandas


yum install -y numpy-f2py
yum install -y atlas atlas-devel
yum install -y blas blas-devel
yum install -y lapack lapack-devel
pip install --upgrade --no-cache-dir scipy

pip install --upgrade --no-cache-dir scikit-learn
pip install --upgrade --no-cache-dir pyprind