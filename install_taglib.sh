wget https://taglib.org/releases/taglib-2.1.1.tar.gz
tar -xvzf ./taglib-2.1.1
cd taglib-2.1.1
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON ..
make
sudo make install