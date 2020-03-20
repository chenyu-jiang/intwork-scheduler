export USE_BYTESCHEDULER=1
export BYTESCHEDULER_WITH_MXNET=1
export BYTESCHEDULER_WITHOUT_PYTORCH=1
export MXNET_ROOT=/research/d2/jc1901/proposed-scheduler/incubator-mxnet-1.5.x

cd /research/d2/jc1901/proposed-scheduler/bytescheduler && \
echo " " && \
echo -e "\e\033[0;33m[REINSTALL BTS]\e[0m Uninstalling ByteScheduler" && \
echo " " && \
pip uninstall bytescheduler && \

echo " " && \
echo -e "\e\033[0;33m[REINSTALL BTS]\e[0m Installing ByteScheduler" && \
echo " " && \
python setup.py install && \

echo " " && \
echo -e "\e\033[0;33m[REINSTALL BTS]\e[0m Please cd /research/d2/jc1901/proposed-scheduler/bytescheduler/examples/mxnet-image-classification "
