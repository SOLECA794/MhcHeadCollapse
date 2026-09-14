#!/bin/bash
# env-runner: atomic build+install+test on remote 910. usage: bash verify.sh <ver>
VER=$1
rm -rf /tmp/$VER /tmp/${VER}_inst
mkdir -p /tmp/$VER/tests
cd /tmp && tar xzf $VER.tgz -C /tmp/$VER
cp ~/mhc_tests/* /tmp/$VER/tests/ 2>/dev/null
cd /tmp/$VER
sed -i "s/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/" code/CMakeLists.txt
sed -i "s/.AddConfig(\"ascend910b\")/.AddConfig(\"ascend910_93\")/" code/op_host/mhc_head_collapse.cpp
source /home/developer/Ascend/cann/set_env.sh
mkdir -p build && cd build
cmake ../code > /dev/null 2>&1 && make -j8 > /dev/null 2>&1 && make binary -j8 > /tmp/${VER}_bin.log 2>&1 && make package > /dev/null 2>&1 && chmod +x custom_opp_ubuntu_aarch64.run && ./custom_opp_ubuntu_aarch64.run --install-path=/tmp/${VER}_inst > /dev/null 2>&1
echo BUILD=$?
CANN=/home/developer/Ascend/cann
g++ -std=c++17 -O2 -o /tmp/mhc_test_$VER /tmp/$VER/tests/mhc_correctness.cpp -I$CANN/include -I/tmp/${VER}_inst/vendors/custom/op_api/include -L$CANN/lib64 -L/tmp/${VER}_inst/vendors/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi -lpthread 2>/dev/null
g++ -std=c++17 -O2 -o /tmp/mhc_bench_$VER /tmp/$VER/tests/mhc_bench.cpp -I$CANN/include -I/tmp/${VER}_inst/vendors/custom/op_api/include -L$CANN/lib64 -L/tmp/${VER}_inst/vendors/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi -lpthread 2>/dev/null
source /tmp/${VER}_inst/vendors/custom/bin/set_env.bash
for a in "4 4 1 fp16" "4 4 2 fp16" "4 4 4 fp16" "8 64 1 fp16" "8 64 2 fp16" "8 128 8 fp16" "8 512 1 fp16" "4 4 2 fp32"; do /tmp/mhc_test_$VER $a; done
echo ===BENCH
for a in "4 4 2 fp16" "8 64 1 fp16" "8 128 8 fp16"; do /tmp/mhc_bench_$VER $a; echo; done
echo ===DONE
