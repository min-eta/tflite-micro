make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
cp gen/cortex_m_generic_cortex-m33_release/lib/libtensorflow-microlite.a libtflm_r.a
