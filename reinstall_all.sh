echo -e "\e\033[0;33m[REINSTALL]\e[0m Uninstalling MXNet" && \
echo " " && \
pip uninstall mxnet && \
echo " " && \
echo -e "\e\033[0;33m[REINSTALL]\e[0m Uninstalling bytescheduler" && \
echo " " && \
pip uninstall bytescheduler && \

echo " " && \
echo -e "\e\033[0;33m[REINSTALL]\e[0m Building MXNet" && \
echo " " && \
cd /research/d2/jc1901/proposed-scheduler/incubator-mxnet-1.5.x/build && \
ninja && \

echo " " && \
echo -e "\e\033[0;33m[REINSTALL]\e[0m Installing MXNet" && \
echo " " && \
cd ../python && \
pip install -e . && \

echo " " && \
echo -e "\e\033[0;33m[REINSTALL]\e[0m Installing Bytescheduler" && \
echo " " && \
cd /research/d2/jc1901/proposed-scheduler/bytescheduler && \
./install_bytescheduler.sh && \

echo -e "\e\033[0;33m[REINSTALL]\e[0m Please cd /research/d2/jc1901/proposed-scheduler/bytescheduler/examples/mxnet-image-classification"
