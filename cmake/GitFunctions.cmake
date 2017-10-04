if(__git_functions)
    return()
endif()
set(__git_functions YES)

function( get_branch_from_git )
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE   git_result
        OUTPUT_VARIABLE   git_branch
        ERROR_VARIABLE    git_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if( NOT git_result EQUAL 0 )
        message( FATAL_ERROR "Failed to execute Git: ${git_error}" )
    endif()
    
    set( GIT_BRANCH ${git_branch} PARENT_SCOPE )
endfunction( get_branch_from_git )

