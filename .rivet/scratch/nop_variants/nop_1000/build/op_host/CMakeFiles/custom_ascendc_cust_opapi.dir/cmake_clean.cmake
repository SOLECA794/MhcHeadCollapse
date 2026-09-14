file(REMOVE_RECURSE
  "libcust_opapi.pdb"
  "libcust_opapi.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/custom_ascendc_cust_opapi.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
