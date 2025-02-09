# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This script will download MKLML

message(STATUS "Downloading MKLML...")

set(MKLDNN_RELEASE v0.19)
set(MKLML_RELEASE_FILE_SUFFIX 2019.0.5.20190502)

set(MKLML_LNX_MD5 dfcea335652dbf3518e1d02cab2cea97)
set(MKLML_WIN_MD5 ff8c5237570f03eea37377ccfc95a08a)
set(MKLML_MAC_MD5 0a3d83ec1fed9ea318e8573bb5e14c24)

if(MSVC)
  set(MKL_NAME "mklml_win_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.zip"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip"
       EXPECTED_MD5 "${MKLML_WIN_MD5}" SHOW_PROGRESS)
  file(DOWNLOAD "https://github.com/apache/incubator-mxnet/releases/download/utils/7z.exe"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe"
       EXPECTED_MD5 "E1CF766CF358F368EC97662D06EA5A4C" SHOW_PROGRESS)

  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z2.exe" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")
  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mklml/7z.exe"
                  "x" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.zip" "-o${CMAKE_CURRENT_BINARY_DIR}/mklml/" "-y")

  set(MKL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKL_ROOT path to ${MKL_ROOT}")

  include_directories(${MKL_ROOT}/include)

elseif(APPLE)
  set(MKL_NAME "mklml_mac_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "${MKLML_MAC_MD5}" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")

  message(STATUS "Setting MKL_ROOT path to ${MKL_ROOT}")
  include_directories(${MKL_ROOT}/include)

elseif(UNIX)
  set(MKL_NAME "mklml_lnx_${MKLML_RELEASE_FILE_SUFFIX}")

  file(DOWNLOAD "https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_RELEASE}/${MKL_NAME}.tgz"
       "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
       EXPECTED_MD5 "${MKLML_LNX_MD5}" SHOW_PROGRESS)
  execute_process(COMMAND "tar" "-xzf" "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}.tgz"
                  "-C" "${CMAKE_CURRENT_BINARY_DIR}/mklml/")

  set(MKL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/mklml/${MKL_NAME}")
  message(STATUS "Setting MKL_ROOT path to ${MKL_ROOT}")
  include_directories(${MKL_ROOT}/include)

else()
  message(FATAL_ERROR "Wrong platform")
endif()
