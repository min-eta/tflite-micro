make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=debug TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn microlite
cp gen/cortex_m_generic_cortex-m33_debug/lib/libtensorflow-microlite.a libtflm_d.a
