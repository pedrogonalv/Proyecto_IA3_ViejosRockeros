SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

apt update
apt upgrade
apt install nano cmake libcurl4-openssl-dev
cd ${SCRIPTPATH}
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
cd ..
pip install -r requirements.txt



