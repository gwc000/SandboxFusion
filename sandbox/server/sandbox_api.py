# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tabnanny import check
import traceback
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
import base64
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sandbox.configs.run_config import RunConfig
from sandbox.runners import (
    CODE_RUNNERS,
    CellRunResult,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
    RunJupyterRequest,
    run_jupyter,
)
from sandbox.utils.mounted_oj import (
    MountedOJCaseResult,
    VERDICT_AC,
    VERDICT_ERROR,
    judge_cases_from_disk,
    run_generator_from_paths,
    run_solution_cases_from_dir,
    resolve_data_root,
    resolve_generation_data_root,
    run_program_from_disk,
)
from sandbox.utils.execution import max_concurrency

sandbox_router = APIRouter()
logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()
DEFAULT_MOUNTED_OJ_MAX_CONCURRENCY = max(1, min(config.sandbox.max_concurrency, 8))
MOUNTED_OJ_MAX_CONCURRENCY = int(
    os.getenv('SANDBOX_MOUNTED_OJ_MAX_CONCURRENCY', DEFAULT_MOUNTED_OJ_MAX_CONCURRENCY))


class RunCodeRequest(BaseModel):
    compile_timeout: float = Field(10, description='compile timeout for compiled languages')
    run_timeout: float = Field(10, description='code run timeout')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    code: str = Field(..., examples=['print("hello")'], description='the code to run')
    stdin: Optional[str] = Field(None, examples=[''], description='optional string to pass into stdin')
    language: Language = Field(..., examples=['python'], description='the language or execution mode to run the code')
    files: Dict[str, Optional[str]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: List[str] = Field([], description='a list of file paths to fetch after code execution')
    ## 新增参数 用于校验
    argv: Optional[List[str]] = Field([], examples=['["1", "2", "3"]'], description='optional list of arguments to pass into the code')
    check_code: Optional[str] = Field(None, examples=['print("hello")'], description='the code to check')


class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class CheckCodeResult(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None

class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}
    check_result: Optional[CheckCodeResult] = None


class RunJupyterResponse(BaseModel):
    status: RunStatus
    message: str
    driver: Optional[CommandRunResult] = None
    cells: List[CellRunResult] = []
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


class RunMountedOJRequest(BaseModel):
    problem_id: str = Field(..., description='problem directory name under the mounted OJ data root')
    case_ids: List[str | int] | str | int = Field(
        ...,
        description='one or more case ids defined in problem.json, or "all" to run all cases',
    )
    code: str = Field(..., description='the code to judge')
    language: Literal['cpp', 'java', 'py3', 'python'] = Field(
        'cpp', description='supported mounted OJ languages: cpp, java, py3 (python alias accepted)'
    )
    data_dir: Optional[str] = Field(None, description='optional override for the mounted OJ data root')
    compile_timeout: float = Field(30, description='compile timeout in seconds')
    run_timeout: Optional[float] = Field(None, description='optional per-case run timeout in seconds')
    time_limit_multiplier: float = Field(
        1.0, description='used with problem.json time_limit_ms when run_timeout is not explicitly provided')
    memory_limit_MB: Optional[int] = Field(None, description='optional override for problem.json memory_limit_mb')
    enable_msvc_i64_compat: bool = Field(
        False,
        description='when true, rewrite legacy MSVC-style %I64* stdio format specifiers in submitted cpp code',
    )
    include_details: bool = Field(
        False,
        description='when true, include per-case run_result/check_result details in the response',
    )


class RunMountedOJResponse(BaseModel):
    status: RunStatus
    message: str
    problem_id: str
    data_dir: str
    compile_result: Optional[CommandRunResult] = None
    checker_compile_result: Optional[CommandRunResult] = None
    cases: List[MountedOJCaseResult] = []
    total_score: float = 0.0
    max_score: float = 0.0
    executor_pod_name: Optional[str] = None


class RunMountedProgramRequest(BaseModel):
    problem_id: str = Field(..., description='problem directory name under the mounted generation data root')
    language: Literal['cpp', 'java', 'py3', 'python'] = Field(
        'cpp', description='supported mounted execution languages: cpp, java, py3 (python alias accepted)'
    )
    code: Optional[str] = Field(None, description='inline code to compile and run')
    source_path: Optional[str] = Field(
        None,
        description='relative source path under the problem directory when code is not provided',
    )
    problem_files: List[str] = Field(
        default_factory=list,
        description='additional relative files under the problem directory to copy into the sandbox workdir',
    )
    data_dir: Optional[str] = Field(
        None,
        description='optional override for the mounted generation data root; defaults to the generation root',
    )
    compile_timeout: float = Field(30, description='compile timeout in seconds')
    run_timeout: float = Field(30, description='run timeout in seconds')
    memory_limit_MB: Optional[int] = Field(None, description='optional override for problem.json memory_limit_mb')
    stdin: Optional[str] = Field('', description='optional stdin string')
    argv: List[str] = Field(default_factory=list, description='optional argv list passed to the program')
    fetch_files: List[str] = Field(
        default_factory=list,
        description='relative files to fetch from the sandbox workdir after execution',
    )
    save_stdout_path: Optional[str] = Field(
        None,
        description='optional relative path under the problem directory where stdout should be persisted',
    )
    return_stdout: bool = Field(
        False,
        description='when false, suppress stdout in the response after optionally persisting it to disk',
    )
    enable_msvc_i64_compat: bool = Field(
        False,
        description='when true, rewrite legacy MSVC-style %I64* stdio format specifiers in cpp code',
    )


class RunMountedProgramResponse(RunCodeResponse):
    problem_id: str
    data_dir: str
    saved_stdout_path: Optional[str] = None


class RunGeneratorRequest(BaseModel):
    generator_path: str = Field(..., description='absolute path to generator.cpp under the mounted generation root')
    testlib_path: str = Field(..., description='absolute path to testlib.h')
    argv: List[str] = Field(default_factory=list, description='argv passed to the generator binary')
    output_path: str = Field(..., description='absolute path where generator stdout should be persisted')
    compile_timeout: float = Field(30, description='compile timeout in seconds')
    run_timeout: float = Field(30, description='run timeout in seconds')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    data_dir: Optional[str] = Field(None, description='optional override for the mounted generation data root')


class RunGeneratorExecInfo(BaseModel):
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None


class RunGeneratorResponse(BaseModel):
    success: bool
    output: str = ''
    error: str = ''
    exec_info: RunGeneratorExecInfo = Field(default_factory=RunGeneratorExecInfo)
    executor_pod_name: Optional[str] = None


class RunSolutionCaseResponse(BaseModel):
    case_id: str
    success: bool
    output: str = ''
    error: str = ''
    run_result: Optional[CommandRunResult] = None


class RunSolutionRequest(BaseModel):
    code: str = Field(..., description='the submission source code')
    language: Literal['cpp', 'java', 'py3', 'python'] = Field(
        'cpp', description='supported languages: cpp, java, py3 (python alias accepted)'
    )
    input_path: str = Field(..., description='absolute path to the directory containing *.in files')
    compile_timeout: float = Field(30, description='compile timeout in seconds')
    run_timeout: float = Field(30, description='run timeout in seconds')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    data_dir: Optional[str] = Field(None, description='optional override for the mounted generation data root')
    enable_msvc_i64_compat: bool = Field(
        False,
        description='when true, rewrite legacy MSVC-style %I64* stdio format specifiers in cpp code',
    )


class RunSolutionResponse(BaseModel):
    success: bool
    compile_result: Optional[CommandRunResult] = None
    cases: List[RunSolutionCaseResponse] = Field(default_factory=list)
    error: str = ''
    executor_pod_name: Optional[str] = None


def _strip_case_details(case_results: List[MountedOJCaseResult]) -> List[MountedOJCaseResult]:
    return [
        case.model_copy(update={'run_result': None, 'check_result': None})
        for case in case_results
    ]


def parse_run_status(result: CodeRunResult) -> Tuple[RunStatus, str]:
    outcomes = []
    retcodes = []
    err_msgs = []
    if result.compile_result is not None:
        outcomes.append(result.compile_result.status)
        err_msgs.append(result.compile_result.stderr or '')
        if result.compile_result.return_code is not None:
            retcodes.append(result.compile_result.return_code)
    if result.run_result is not None:
        outcomes.append(result.run_result.status)
        err_msgs.append(result.run_result.stderr or '')
        if result.run_result.return_code is not None:
            retcodes.append(result.run_result.return_code)

    for o, m in zip(outcomes, err_msgs):
        if o == CommandRunStatus.Error:
            return RunStatus.SandboxError, m
    if any([o == CommandRunStatus.TimeLimitExceeded for o in outcomes]):
        return RunStatus.Failed, ''
    if any([r != 0 for r in retcodes]):
        return RunStatus.Failed, ''
    # no error, no tle and no non-zero return codes -> success
    return RunStatus.Success, ''


@sandbox_router.post("/run_code", response_model=RunCodeResponse, tags=['sandbox'])
async def run_code(request: RunCodeRequest):
    resp = RunCodeResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    try:
        logger.debug(
            f'start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)'
        )
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f'exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


@sandbox_router.post("/run_check_code", response_model=RunCodeResponse, tags=['sandbox'])
async def run_check_code(request: RunCodeRequest):
    '''
    实现基于testlib的代码执行（支持多语言）+ 基于checker(cpp)的校验。

    checker(https://codeforces.com/testlib)
    '''
    resp = RunCodeResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    try:
        logger.debug(f'start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)')
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f'exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError
        return resp

    # 编译失败, 直接返回
    if resp.compile_result is not None and (
        resp.compile_result.status != CommandRunStatus.Finished 
        or resp.compile_result.return_code != 0):
        return resp

    # 执行失败, 直接返回
    if resp.run_result.status != CommandRunStatus.Finished:
        return resp 

    # 执行成功，对stdout进行校验
    try:
        check_args = request.model_dump()
        check_args['code'] = request.check_code
        check_args['files']['input.txt']  = base64.b64encode(request.stdin.encode()).decode()
        check_args['files']['output.txt'] = base64.b64encode(resp.run_result.stdout.encode()).decode()
        logger.debug(f'start check with code ```\n{check_args["code"][:100]}\n``` and files {list(check_args["files"].keys())}...(memory_limit: {request.memory_limit_MB}MB)')
        check_exec_result = await CODE_RUNNERS['cpp_check'](CodeRunArgs(**check_args))
    
        check_status, check_message = parse_run_status(check_exec_result)

        check_result = CheckCodeResult(
            status=check_status,
            message='' if check_status != RunStatus.SandboxError else check_message,
            compile_result=check_exec_result.compile_result,
            run_result=check_exec_result.run_result,
        )

        resp.check_result = check_result

    except Exception as e:
        message = f'exception on checking code {request.check_code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


@sandbox_router.post("/run_oj_cases", response_model=RunMountedOJResponse, tags=['sandbox'])
@max_concurrency(MOUNTED_OJ_MAX_CONCURRENCY)
async def run_oj_cases(request: RunMountedOJRequest):
    if request.case_ids in (None, [], ''):
        raise HTTPException(status_code=400, detail='case_ids must not be empty')
    if request.time_limit_multiplier <= 0:
        raise HTTPException(status_code=400, detail='time_limit_multiplier must be positive')
    if isinstance(request.case_ids, (str, int)):
        request_case_ids = [str(request.case_ids)]
    else:
        request_case_ids = [str(case_id) for case_id in request.case_ids]

    resp = RunMountedOJResponse(
        status=RunStatus.Success,
        message='',
        problem_id=request.problem_id,
        data_dir='',
        executor_pod_name=os.environ.get('MY_POD_NAME'),
    )
    try:
        data_root = resolve_data_root(request.data_dir)
        resp.data_dir = str(data_root)
        logger.debug(
            'start processing mounted OJ request',
            problem_id=request.problem_id,
            case_ids=request_case_ids,
            data_dir=resp.data_dir,
            language=request.language,
            enable_msvc_i64_compat=request.enable_msvc_i64_compat,
        )
        _, compile_result, checker_compile_result, case_results = await judge_cases_from_disk(
            data_root=data_root,
            problem_id=request.problem_id,
            case_ids=request.case_ids,
            code=request.code,
            compile_timeout=request.compile_timeout,
            run_timeout=request.run_timeout,
            time_limit_multiplier=request.time_limit_multiplier,
            memory_limit_mb=request.memory_limit_MB,
            enable_msvc_i64_compat=request.enable_msvc_i64_compat,
            language=request.language,
        )
        resp.compile_result = compile_result
        resp.checker_compile_result = checker_compile_result
        resp.cases = case_results if request.include_details else _strip_case_details(case_results)
        resp.total_score = float(sum(case.score for case in case_results))
        resp.max_score = float(sum(case.max_score for case in case_results))

        if any(
            result is not None and result.status == CommandRunStatus.Error
            for result in (compile_result, checker_compile_result)
        ) or any(case.verdict == VERDICT_ERROR for case in case_results):
            resp.status = RunStatus.SandboxError
            resp.message = 'sandbox error while compiling or running mounted OJ cases'
        elif all(case.verdict == VERDICT_AC for case in case_results):
            resp.status = RunStatus.Success
        else:
            resp.status = RunStatus.Failed
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        message = f'exception on mounted OJ request {request.problem_id}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


@sandbox_router.post("/run_generator", response_model=RunGeneratorResponse, tags=['sandbox'])
@max_concurrency(MOUNTED_OJ_MAX_CONCURRENCY)
async def run_generator(request: RunGeneratorRequest):
    resp = RunGeneratorResponse(
        success=False,
        output='',
        error='',
        executor_pod_name=os.environ.get('MY_POD_NAME'),
    )
    try:
        data_root = resolve_generation_data_root(request.data_dir)
        compile_result, run_result, output = await run_generator_from_paths(
            data_root=data_root,
            generator_path=request.generator_path,
            testlib_path=request.testlib_path,
            argv=request.argv,
            output_path=request.output_path,
            compile_timeout=request.compile_timeout,
            run_timeout=request.run_timeout,
            memory_limit_mb=request.memory_limit_MB,
        )
        resp.exec_info = RunGeneratorExecInfo(
            compile_result=compile_result,
            run_result=run_result,
        )
        compile_ok = (
            compile_result is not None and
            compile_result.status == CommandRunStatus.Finished and
            compile_result.return_code == 0
        )
        run_ok = (
            run_result is not None and
            run_result.status == CommandRunStatus.Finished and
            run_result.return_code == 0
        )
        resp.success = bool(compile_ok and run_ok)
        resp.output = output
        if not resp.success:
            compile_err = compile_result.stderr if compile_result is not None else ''
            run_err = run_result.stderr if run_result is not None else ''
            parts = [part for part in [compile_err, run_err] if part]
            resp.error = '; '.join(parts) if parts else 'generator execution failed'
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        message = f'exception on run_generator request {request.generator_path}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.error = message

    return resp


@sandbox_router.post("/run_solution", response_model=RunSolutionResponse, tags=['sandbox'])
@max_concurrency(MOUNTED_OJ_MAX_CONCURRENCY)
async def run_solution(request: RunSolutionRequest):
    resp = RunSolutionResponse(
        success=False,
        error='',
        executor_pod_name=os.environ.get('MY_POD_NAME'),
    )
    try:
        data_root = resolve_generation_data_root(request.data_dir)
        compile_result, case_results = await run_solution_cases_from_dir(
            data_root=data_root,
            code=request.code,
            language=request.language,
            input_path=request.input_path,
            compile_timeout=request.compile_timeout,
            run_timeout=request.run_timeout,
            memory_limit_mb=request.memory_limit_MB,
            enable_msvc_i64_compat=request.enable_msvc_i64_compat,
        )
        resp.compile_result = compile_result
        resp.cases = [RunSolutionCaseResponse(**case) for case in case_results]
        compile_ok = (
            compile_result is not None and
            compile_result.status == CommandRunStatus.Finished and
            compile_result.return_code == 0
        )
        resp.success = bool(compile_ok)
        if not compile_ok:
            resp.error = compile_result.stderr if compile_result is not None else 'solution compilation failed'
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        message = f'exception on run_solution request {request.input_path}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.error = message

    return resp


@sandbox_router.post("/run_mounted_program", response_model=RunMountedProgramResponse, tags=['sandbox'])
@max_concurrency(MOUNTED_OJ_MAX_CONCURRENCY)
async def run_mounted_program(request: RunMountedProgramRequest):
    resp = RunMountedProgramResponse(
        status=RunStatus.Success,
        message='',
        problem_id=request.problem_id,
        data_dir='',
        executor_pod_name=os.environ.get('MY_POD_NAME'),
    )
    try:
        data_root = resolve_generation_data_root(request.data_dir)
        resp.data_dir = str(data_root)
        logger.debug(
            'start processing mounted program request',
            problem_id=request.problem_id,
            data_dir=resp.data_dir,
            language=request.language,
            source_path=request.source_path,
            argv=request.argv,
            fetch_files=request.fetch_files,
        )
        _, compile_result, run_result, files = await run_program_from_disk(
            data_root=data_root,
            problem_id=request.problem_id,
            language=request.language,
            compile_timeout=request.compile_timeout,
            run_timeout=request.run_timeout,
            memory_limit_mb=request.memory_limit_MB,
            stdin=request.stdin,
            argv=request.argv,
            code=request.code,
            source_path=request.source_path,
            problem_files=request.problem_files,
            fetch_files=request.fetch_files,
            save_stdout_path=request.save_stdout_path,
            return_stdout=request.return_stdout,
            enable_msvc_i64_compat=request.enable_msvc_i64_compat,
        )
        resp.compile_result = compile_result
        resp.run_result = run_result
        resp.files = files
        resp.saved_stdout_path = request.save_stdout_path
        resp.status, message = parse_run_status(CodeRunResult(
            compile_result=compile_result,
            run_result=run_result,
            files=files,
        ))
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        message = f'exception on mounted program request {request.problem_id}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


@sandbox_router.post("/run_jupyter", name='Run Code in Jupyter', response_model=RunJupyterResponse, tags=['sandbox'])
async def run_jupyter_handler(request: RunJupyterRequest):
    resp = RunJupyterResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    code_repr = "\n".join(request.cells)[:100]
    try:
        logger.debug(
            f'start processing jupyter request with code ```\n{code_repr}\n``` and files {list(request.files.keys())}...'
        )
        result = await run_jupyter(request)
        resp.driver = result.driver
        if result.status != CommandRunStatus.Finished:
            resp.status = RunStatus.Failed
        else:
            resp.status = RunStatus.Success
            resp.cells = result.cells
            resp.files = result.files
    except Exception as e:
        message = f'exception on running jupyter {code_repr}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp
