function(operator_unittest arg)
    add_executable(test_operator_${arg}
        test_operator_${arg}.cpp
    )
    add_dependencies(test_operator_${arg}
        Ext_SkyPat
    )
    target_link_libraries(test_operator_${arg}
        onnc-runtime
        skypat
    )
add_test(Operator_${arg} test_operator_${arg})
endfunction(operator_unittest)


operator_unittest("and")
operator_unittest("add")
operator_unittest("acos")
operator_unittest("asin")
operator_unittest("abs")
operator_unittest("ceil")
operator_unittest("log")
operator_unittest("equal")
