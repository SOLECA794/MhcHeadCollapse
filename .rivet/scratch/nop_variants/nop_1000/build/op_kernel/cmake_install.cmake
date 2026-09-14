# Install script for directory: /workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/op_kernel

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom/op_impl/ai_core/tbe/custom_impl/dynamic/" TYPE DIRECTORY FILES "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/ascendc_kernels/binary/dynamic/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom/op_impl/ai_core/tbe//kernel/ascend910b/" TYPE DIRECTORY FILES "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/ascendc_kernels/binary/ascend910b/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom/op_impl/ai_core/tbe//kernel/config/" TYPE DIRECTORY FILES "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/ascendc_kernels/binary/config/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom/op_impl/ai_core/tbe/config/ascend910b" TYPE FILE FILES "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom/framework/plugin" TYPE FILE FILES "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/npu_supported_ops.json")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants/nop_1000/build/op_kernel/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
