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

### Various Installation
yum install -y numpy-f2py
yum install -y atlas atlas-devel
yum install -y blas blas-devel
yum install -y lapack lapack-devel

### Install Python3.5 
yum install -y openssl-devel
wget https://www.python.org/ftp/python/3.5.0/Python-3.5.0.tgz
tar xf Python-3.5.0.tgz
cd Python-3.5.0
./configure
make
make install


/usr/local/bin/pip3.5 install pandas
/usr/local/bin/pip3.5 install scipy
/usr/local/bin/pip3.5 install numpy
/usr/local/bin/pip3.5 install boto
/usr/local/bin/pip3.5 install pyprind

swapoff /var/swap.1
rm /var/swap.1

