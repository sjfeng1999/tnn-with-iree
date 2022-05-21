include(CMakeParseArguments)

set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(_TBLGEN_INCLUDE_DIRS ${IREE_SOURCE_DIR})

list(APPEND _TBLGEN_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${IREE_SOURCE_DIR}/compiler/src
  ${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include
)
list(TRANSFORM _TBLGEN_INCLUDE_DIRS PREPEND "-I")


function(mlir_tblgen_library)
  cmake_parse_arguments(
    _INS
    "SHARED"
    "TARGET;TBLGEN"
    "TD_FILES;OUTS"
    ${ARGN}
  )
  if(${_INS_TBLGEN} MATCHES "IREE")
    set(_TBLGEN "IREE")
  else()
    set(_TBLGEN "MLIR")
  endif()

  set(LLVM_TARGET_DEFINITIONS ${_INS_TD_FILES})

  set(_OUTPUTS)
  while(_INS_OUTS)
    list(GET _INS_OUTS 0 _COMMAND)
    list(REMOVE_AT _INS_OUTS 0)
    list(GET _INS_OUTS 0 _FILE)
    list(REMOVE_AT _INS_OUTS 0)
    tablegen(${_TBLGEN} ${_FILE} ${_COMMAND} ${_TBLGEN_INCLUDE_DIRS})
    list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
  endwhile()

  add_custom_target(${_INS_TARGET}_target DEPENDS ${_OUTPUTS})
  set_target_properties(${_INS_TARGET}_target  PROPERTIES FOLDER "Tablegenning")

  add_library(${_INS_TARGET} INTERFACE)
  add_dependencies(${_INS_TARGET} ${_INS_TARGET}_target)
endfunction(mlir_tblgen_library)

