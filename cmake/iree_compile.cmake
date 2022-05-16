
include(CMakeParseArguments)

function(iree_compile_vmfb)
  cmake_parse_arguments(
    _INS
    ""
    "TARGET;TRANSLATE_TOOL"
    "FLAGS;DEPS"
    ${ARGN}
  )

  add_custom_target(
    ${_INS_TARGET}
    COMMAND ${_INS_TRANSLATE_TOOL} ${_INS_FLAGS}
    BYPRODUCTS ${_INS_PRODUCT}
    DEPENDS ${_INS_DEPS}
  )
endfunction(iree_compile_vmfb)

