import json
import os
import shlex
import shutil
import sys
import tempfile
import math
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from sandbox.configs.run_config import RunConfig
from sandbox.runners.base import build_preexec_fn, run_command_bare
from sandbox.runners.major import get_cpp_rt_flags, get_python_rt_env
from sandbox.runners.types import CommandRunResult, CommandRunStatus
from sandbox.utils.execution import get_tmp_dir
from sandbox.utils.extraction import find_java_public_class_name

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()
CPP_STD = 'c++17'
MSVC_I64_COMPAT_REPLACEMENTS = (
    ('%I64d', '%lld'),
    ('%I64i', '%lli'),
    ('%I64u', '%llu'),
    ('%I64x', '%llx'),
    ('%I64X', '%llX'),
    ('%I64o', '%llo'),
)

VERDICT_AC = 'AC'
VERDICT_WA = 'WA'
VERDICT_TLE = 'TLE'
VERDICT_RE = 'RE'
VERDICT_CE = 'CE'
VERDICT_CHECKER_CE = 'CHECKER_CE'
VERDICT_ERROR = 'ERROR'
CPU_LIMIT_SIGNAL_RETURN_CODES = {-signal.SIGXCPU}
SUPPORTED_MOUNTED_OJ_LANGUAGES = ('cpp', 'java', 'py3', 'python')
DEFAULT_GENERATION_DATA_ROOT = Path(
    os.getenv('OJ_GENERATION_DATA_ROOT', '/volume/lzchai/mounted_oj_cco_test_generation')
).expanduser().resolve()


class MountedOJCheckerSpec(BaseModel):
    source: str = 'checker.cpp'
    files: List[str] = Field(default_factory=list)
    argv: List[str] = Field(default_factory=lambda: ['input.txt', 'output.txt', 'answer.txt'])


class MountedOJCaseSpec(BaseModel):
    id: str | int
    input: str
    answer: str
    score: float = 1.0


class MountedOJProblemSpec(BaseModel):
    problem_id: Optional[str] = None
    time_limit_ms: int = 1000
    memory_limit_mb: int = -1
    shared_files: List[str] = Field(default_factory=list)
    checker: Optional[MountedOJCheckerSpec] = None
    test_cases: List[MountedOJCaseSpec]


class MountedOJCaseResult(BaseModel):
    case_id: str
    passed: bool
    verdict: str
    score: float = 0.0
    max_score: float = 1.0
    input_path: Optional[str] = None
    answer_path: Optional[str] = None
    run_result: Optional[CommandRunResult] = None
    check_result: Optional[CommandRunResult] = None


def _validate_identifier(raw_value: str | int, field_name: str) -> str:
    value = str(raw_value)
    if not value or value in {'.', '..'}:
        raise ValueError(f'invalid {field_name}: {raw_value!r}')
    if '/' in value or '\\' in value:
        raise ValueError(f'{field_name} must not contain path separators: {raw_value!r}')
    return value


def _resolve_under(base_dir: Path, relative_path: str) -> Path:
    candidate = (base_dir / relative_path).resolve()
    base_resolved = base_dir.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValueError(f'path escapes base directory: {relative_path!r}')
    return candidate


def _resolve_relative_subpath(base_dir: Path, relative_path: str, field_name: str) -> Path:
    raw_path = (relative_path or '').strip()
    if not raw_path:
        raise ValueError(f'{field_name} must not be empty')
    rel = Path(raw_path)
    if rel.is_absolute():
        raise ValueError(f'{field_name} must be relative: {relative_path!r}')
    if any(part in {'', '.', '..'} for part in rel.parts):
        raise ValueError(f'invalid {field_name}: {relative_path!r}')
    return _resolve_under(base_dir, raw_path)


def resolve_problem_dir(data_root: Path, problem_id: str) -> Path:
    return _resolve_relative_subpath(data_root, problem_id, 'problem_id')


def resolve_generation_file_path(data_root: Path, absolute_path: str, field_name: str) -> Path:
    raw_path = (absolute_path or '').strip()
    if not raw_path:
        raise ValueError(f'{field_name} must not be empty')
    candidate = Path(raw_path).expanduser().resolve()
    root_resolved = data_root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f'{field_name} must be under generation data root: {candidate}')
    return candidate


def resolve_data_root(data_dir: Optional[str]) -> Path:
    root = data_dir or os.getenv('OJ_DATA_ROOT') or config.dataset.oj_data_root
    if not root:
        raise ValueError('OJ data root is not configured. Set request.data_dir, OJ_DATA_ROOT, or config.dataset.oj_data_root.')
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f'OJ data root does not exist or is not a directory: {root_path}')
    return root_path


def resolve_generation_data_root(data_dir: Optional[str]) -> Path:
    root_path = Path(data_dir).expanduser().resolve() if data_dir else DEFAULT_GENERATION_DATA_ROOT
    if not root_path.is_dir():
        raise ValueError(f'generation data root does not exist or is not a directory: {root_path}')
    return root_path


def load_problem_spec(data_root: Path, problem_id: str) -> Tuple[Path, MountedOJProblemSpec, Dict[str, MountedOJCaseSpec]]:
    problem_dir = resolve_problem_dir(data_root, problem_id)
    manifest_path = problem_dir / 'problem.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'problem manifest not found: {manifest_path}')

    with open(manifest_path, 'r', encoding='utf-8') as f:
        problem = MountedOJProblemSpec(**json.load(f))

    if problem.problem_id is not None and str(problem.problem_id) != str(problem_id):
        raise ValueError(
            f'problem.json problem_id mismatch: expected {problem_id!r}, got {problem.problem_id!r}')

    case_map = {}
    for case in problem.test_cases:
        case_id = _validate_identifier(case.id, 'case_id')
        if case_id in case_map:
            raise ValueError(f'duplicate case id in manifest: {case_id!r}')
        case_map[case_id] = case

    return problem_dir, problem, case_map


def _copy_problem_files(problem_dir: Path, work_dir: Path, relative_paths: List[str]) -> None:
    for rel_path in relative_paths:
        src = _resolve_under(problem_dir, rel_path)
        if not src.is_file():
            raise FileNotFoundError(f'problem asset not found: {src}')
        dst = work_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def _prepare_case_dir(case_dir: Path, input_path: Path, answer_path: Path) -> Tuple[Path, Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    linked_input_path = case_dir / 'input.txt'
    linked_answer_path = case_dir / 'answer.txt'
    output_path = case_dir / 'output.txt'

    for path in (linked_input_path, linked_answer_path, output_path):
        if path.exists() or path.is_symlink():
            path.unlink()

    try:
        os.link(input_path, linked_input_path)
    except OSError:
        linked_input_path.symlink_to(input_path)

    try:
        os.link(answer_path, linked_answer_path)
    except OSError:
        linked_answer_path.symlink_to(answer_path)
    return linked_input_path, linked_answer_path, output_path


async def _compile_cpp(source_path: Path,
                       output_path: Path,
                       timeout: float,
                       cwd: Path,
                       memory_limit_mb: int = -1,
                       extra_flags: Optional[List[str]] = None) -> CommandRunResult:
    flags = await get_cpp_rt_flags()
    compile_flags = [f"-std={CPP_STD}"]
    if extra_flags:
        compile_flags.extend(extra_flags)
    compile_flags.extend(flags)
    command = (
        f'g++ {" ".join(compile_flags)} {shlex.quote(source_path.name)} -o {shlex.quote(output_path.name)}'
    ).strip()
    return await run_command_bare(
        command,
        timeout=timeout,
        cwd=str(cwd),
        preexec_fn=build_preexec_fn(memory_limit_mb),
    )


async def _run_binary(binary_path: Path,
                      cwd: Path,
                      stdin_path: Path,
                      output_path: Path,
                      timeout: float,
                      memory_limit_mb: int = -1,
                      cpu_limit_s: Optional[int] = None,
                      argv: Optional[List[str]] = None) -> CommandRunResult:
    command = shlex.quote(str(binary_path))
    if argv:
        command = f"{command} {' '.join(shlex.quote(arg) for arg in argv)}"
    return await run_command_bare(
        command,
        timeout=timeout,
        stdin_path=str(stdin_path),
        stdout_path=str(output_path),
        cwd=str(cwd),
        preexec_fn=build_preexec_fn(memory_limit_mb, cpu_limit_s=cpu_limit_s),
    )


async def _run_checker_binary(checker_path: Path, cwd: Path, argv: List[str], timeout: float,
                              memory_limit_mb: int = -1, cpu_limit_s: Optional[int] = None) -> CommandRunResult:
    escaped_argv = ' '.join(shlex.quote(arg) for arg in argv)
    command = shlex.quote(str(checker_path))
    if escaped_argv:
        command = f'{command} {escaped_argv}'
    return await run_command_bare(
        command,
        timeout=timeout,
        cwd=str(cwd),
        preexec_fn=build_preexec_fn(memory_limit_mb, cpu_limit_s=cpu_limit_s),
    )


def normalize_mounted_oj_language(language: str) -> str:
    normalized = (language or 'cpp').lower()
    if normalized == 'py3':
        return 'python'
    if normalized in {'cpp', 'java', 'python'}:
        return normalized
    raise ValueError(
        f'unsupported mounted OJ language: {language!r}. '
        f'Supported values are {SUPPORTED_MOUNTED_OJ_LANGUAGES!r}'
    )


async def _compile_java(source_path: Path,
                        timeout: float,
                        cwd: Path,
                        memory_limit_mb: int = -1,
                        classpath_entries: Optional[List[str]] = None) -> CommandRunResult:
    classpath_flag = ''
    if classpath_entries:
        classpath_flag = f" -cp {shlex.quote(':'.join(classpath_entries))}"
    command = f'javac{classpath_flag} {shlex.quote(source_path.name)}'.strip()
    return await run_command_bare(
        command,
        timeout=timeout,
        cwd=str(cwd),
        preexec_fn=build_preexec_fn(memory_limit_mb),
    )


def _build_java_runtime_command(classpath_entries: List[str], memory_limit_mb: int = -1) -> str:
    effective_memory_mb = memory_limit_mb if memory_limit_mb and memory_limit_mb > 0 else 256

    # Keep JVM startup conservative so Java can initialize reliably under RLIMIT_AS / RLIMIT_DATA.
    if effective_memory_mb <= 320:
        heap_mb = max(24, min(64, effective_memory_mb // 5))
        metaspace_mb = 24
        code_cache_mb = 8
        compressed_class_space_mb = 8
    elif effective_memory_mb <= 768:
        heap_mb = max(48, min(96, effective_memory_mb // 4))
        metaspace_mb = 32
        code_cache_mb = 12
        compressed_class_space_mb = 12
    else:
        heap_mb = max(64, min(160, effective_memory_mb // 3))
        metaspace_mb = 48
        code_cache_mb = 16
        compressed_class_space_mb = 16

    xms_mb = min(8, heap_mb)
    initial_code_cache_mb = min(8, code_cache_mb)

    jvm_flags = [
        f'-Xms{xms_mb}m',
        f'-Xmx{heap_mb}m',
        '-Xss256k',
        f'-XX:ReservedCodeCacheSize={code_cache_mb}m',
        f'-XX:InitialCodeCacheSize={initial_code_cache_mb}m',
        f'-XX:CompressedClassSpaceSize={compressed_class_space_mb}m',
        f'-XX:MaxMetaspaceSize={metaspace_mb}m',
        '-XX:+UseSerialGC',
        '-XX:TieredStopAtLevel=1',
        '-XX:-UsePerfData',
        '-Xshare:off',
    ]
    classpath = shlex.quote(':'.join(classpath_entries))
    return f"java {' '.join(jvm_flags)} -cp {classpath} -ea Main"


def _get_python_runtime_command() -> tuple[str, Dict[str, str]]:
    try:
        return 'python', get_python_rt_env('sandbox-runtime')
    except Exception as e:
        fallback_python = shutil.which('python3') or shutil.which('python') or sys.executable
        logger.warning(
            'failed to resolve sandbox-runtime python, falling back to host python',
            error=str(e),
            fallback_python=fallback_python,
        )
        return shlex.quote(fallback_python), {}


async def _compile_python(source_path: Path,
                          timeout: float,
                          cwd: Path,
                          memory_limit_mb: int = -1) -> CommandRunResult:
    python_command, extra_env = _get_python_runtime_command()
    return await run_command_bare(
        f'{python_command} -m py_compile {shlex.quote(source_path.name)}',
        timeout=timeout,
        cwd=str(cwd),
        extra_env=extra_env,
        preexec_fn=build_preexec_fn(memory_limit_mb),
    )


async def _run_command_with_files(command: str,
                                  cwd: Path,
                                  stdin_path: Path,
                                  output_path: Path,
                                  timeout: float,
                                  memory_limit_mb: int = -1,
                                  cpu_limit_s: Optional[int] = None,
                                  extra_env: Optional[Dict[str, str]] = None,
                                  argv: Optional[List[str]] = None) -> CommandRunResult:
    full_command = command
    if argv:
        full_command = f"{full_command} {' '.join(shlex.quote(arg) for arg in argv)}"
    return await run_command_bare(
        full_command,
        timeout=timeout,
        stdin_path=str(stdin_path),
        stdout_path=str(output_path),
        cwd=str(cwd),
        extra_env=extra_env,
        preexec_fn=build_preexec_fn(memory_limit_mb, cpu_limit_s=cpu_limit_s),
    )


async def _prepare_solution_runner(language: str,
                                   code: str,
                                   work_dir: Path,
                                   compile_timeout: float,
                                   enable_msvc_i64_compat: bool = False) -> tuple[Optional[CommandRunResult], Optional[callable]]:
    normalized_language = normalize_mounted_oj_language(language)

    if normalized_language == 'cpp':
        solution_src = work_dir / 'solution.cpp'
        solution_bin = work_dir / 'solution'
        solution_src.write_text(
            _rewrite_cpp_legacy_stdio_formats(code, enable_msvc_i64_compat),
            encoding='utf-8',
        )
        compile_result = await _compile_cpp(
            solution_src,
            solution_bin,
            compile_timeout,
            work_dir,
            extra_flags=['-O2', '-DONLINE_JUDGE'],
        )
        if compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0:
            return compile_result, None

        async def _runner(stdin_path: Path, output_path: Path, timeout: float,
                          memory_limit_mb: int = -1, cpu_limit_s: Optional[int] = None,
                          argv: Optional[List[str]] = None) -> CommandRunResult:
            return await _run_binary(
                solution_bin,
                work_dir,
                stdin_path=stdin_path,
                output_path=output_path,
                timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                cpu_limit_s=cpu_limit_s,
                argv=argv,
            )

        return compile_result, _runner

    if normalized_language == 'java':
        runtime_java_dir = Path(__file__).resolve().parents[2] / 'runtime' / 'java'
        classpath_entries = ['.']
        javatuples_jar = runtime_java_dir / 'javatuples-1.2.jar'
        if javatuples_jar.is_file():
            classpath_entries.append(str(javatuples_jar))

        class_name = find_java_public_class_name(code) or 'Main'
        solution_src = work_dir / f'{class_name}.java'
        solution_src.write_text(code, encoding='utf-8')
        compile_result = await _compile_java(
            solution_src,
            compile_timeout,
            work_dir,
            classpath_entries=classpath_entries,
        )
        if compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0:
            return compile_result, None

        async def _runner(stdin_path: Path, output_path: Path, timeout: float,
                          memory_limit_mb: int = -1, cpu_limit_s: Optional[int] = None,
                          argv: Optional[List[str]] = None) -> CommandRunResult:
            # For Java, rely on JVM flags rather than RLIMIT_AS / RLIMIT_DATA.
            # RLIMIT-based address-space caps are too strict for JVM startup on low-memory OJ problems.
            return await _run_command_with_files(
                _build_java_runtime_command(classpath_entries, memory_limit_mb=memory_limit_mb).replace(' -ea Main', f' -ea {class_name}'),
                work_dir,
                stdin_path=stdin_path,
                output_path=output_path,
                timeout=timeout,
                memory_limit_mb=-1,
                cpu_limit_s=cpu_limit_s,
                argv=argv,
            )

        return compile_result, _runner

    solution_src = work_dir / 'solution.py'
    solution_src.write_text(code, encoding='utf-8')
    compile_result = await _compile_python(solution_src, compile_timeout, work_dir)
    if compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0:
        return compile_result, None

    python_command, python_extra_env = _get_python_runtime_command()

    async def _runner(stdin_path: Path, output_path: Path, timeout: float,
                      memory_limit_mb: int = -1, cpu_limit_s: Optional[int] = None,
                      argv: Optional[List[str]] = None) -> CommandRunResult:
        return await _run_command_with_files(
            f'{python_command} {shlex.quote(solution_src.name)}',
            work_dir,
            stdin_path=stdin_path,
            output_path=output_path,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            cpu_limit_s=cpu_limit_s,
            extra_env=python_extra_env,
            argv=argv,
        )

    return compile_result, _runner


def _run_verdict(run_result: CommandRunResult) -> str:
    if run_result.status == CommandRunStatus.TimeLimitExceeded:
        return VERDICT_TLE
    if run_result.status == CommandRunStatus.Error:
        return VERDICT_ERROR
    if run_result.return_code in CPU_LIMIT_SIGNAL_RETURN_CODES:
        return VERDICT_TLE
    if run_result.return_code != 0:
        return VERDICT_RE
    return VERDICT_AC


def _checker_verdict(check_result: CommandRunResult) -> str:
    if check_result.status == CommandRunStatus.TimeLimitExceeded:
        return VERDICT_CHECKER_CE
    if check_result.status == CommandRunStatus.Error:
        return VERDICT_ERROR
    if check_result.return_code == 0:
        return VERDICT_AC
    if check_result.return_code in (1, 2, 7):
        return VERDICT_WA
    if check_result.return_code == 3:
        message = (check_result.stderr or '').lower()
        if 'better' in message or 'optimal' in message:
            return VERDICT_AC
        return VERDICT_CHECKER_CE
    return VERDICT_CHECKER_CE


def _plain_compare(actual: str, expected: str) -> bool:
    return actual.split() == expected.split()


def _rewrite_cpp_legacy_stdio_formats(code: str, enable_compat: bool) -> str:
    if not enable_compat:
        return code

    rewritten = code
    for old, new in MSVC_I64_COMPAT_REPLACEMENTS:
        rewritten = rewritten.replace(old, new)
    return rewritten


def _iter_tokens(path: Path, chunk_size: int = 64 * 1024):
    pending = ''
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = pending + chunk
            if data and not data[-1].isspace():
                split_at = len(data)
                while split_at > 0 and not data[split_at - 1].isspace():
                    split_at -= 1
                if split_at == 0:
                    pending = data
                    continue
                emit = data[:split_at]
                pending = data[split_at:]
            else:
                emit = data
                pending = ''
            for token in emit.split():
                yield token
    if pending:
        for token in pending.split():
            yield token


def _plain_compare_files(actual_path: Path, expected_path: Path) -> bool:
    actual_tokens = _iter_tokens(actual_path)
    expected_tokens = _iter_tokens(expected_path)
    while True:
        actual = next(actual_tokens, None)
        expected = next(expected_tokens, None)
        if actual != expected:
            return False
        if actual is None:
            return True


def _resolve_checker_argv(argv: List[str], input_path: Path, output_path: Path, answer_path: Path) -> List[str]:
    replacements = {
        'input.txt': str(input_path),
        'output.txt': str(output_path),
        'answer.txt': str(answer_path),
    }
    return [replacements.get(arg, arg) for arg in argv]


def _is_all_case_selector(raw_value: str | int) -> bool:
    return isinstance(raw_value, str) and raw_value.lower() == 'all'


def normalize_case_ids(case_ids: List[str | int] | str | int,
                       problem: MountedOJProblemSpec,
                       case_map: Dict[str, MountedOJCaseSpec]) -> List[str]:
    if isinstance(case_ids, (str, int)):
        raw_case_ids = [case_ids]
    else:
        raw_case_ids = list(case_ids)

    if not raw_case_ids:
        raise ValueError('case_ids must not be empty')

    if len(raw_case_ids) == 1 and _is_all_case_selector(raw_case_ids[0]):
        return [str(case.id) for case in problem.test_cases]

    if any(_is_all_case_selector(case_id) for case_id in raw_case_ids):
        raise ValueError('case_ids="all" must be used alone')

    normalized_case_ids = [_validate_identifier(case_id, 'case_id') for case_id in raw_case_ids]
    for case_id in normalized_case_ids:
        if case_id not in case_map:
            raise FileNotFoundError(f'case id {case_id!r} not found in problem')
    return normalized_case_ids


def _failed_case_results(case_ids: List[str], verdict: str,
                         case_map: Dict[str, MountedOJCaseSpec]) -> List[MountedOJCaseResult]:
    return [
        MountedOJCaseResult(
            case_id=case_id,
            passed=False,
            verdict=verdict,
            score=0.0,
            max_score=float(case_map[case_id].score),
        )
        for case_id in case_ids
    ]


async def judge_cases_from_disk(
    data_root: Path,
    problem_id: str,
    case_ids: List[str | int] | str | int,
    code: str,
    compile_timeout: float,
    run_timeout: Optional[float] = None,
    time_limit_multiplier: float = 1.0,
    memory_limit_mb: Optional[int] = None,
    enable_msvc_i64_compat: bool = False,
    language: str = 'cpp',
) -> Tuple[MountedOJProblemSpec, Optional[CommandRunResult], Optional[CommandRunResult], List[MountedOJCaseResult]]:
    problem_dir, problem, case_map = load_problem_spec(data_root, problem_id)
    normalized_case_ids = normalize_case_ids(case_ids, problem, case_map)

    effective_memory_limit = problem.memory_limit_mb if memory_limit_mb is None else memory_limit_mb
    tl_s = max(problem.time_limit_ms * time_limit_multiplier / 1000.0, 0.001)
    cpu_limit_s = max(1, math.ceil(tl_s))
    effective_run_timeout = run_timeout if run_timeout is not None else max(tl_s * 2 + 2.0, 10.0)
    checker_timeout = max(effective_run_timeout, 10.0)

    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir_name:
        work_dir = Path(tmp_dir_name)
        _copy_problem_files(problem_dir, work_dir, problem.shared_files)

        compile_result, solution_runner = await _prepare_solution_runner(
            language=language,
            code=code,
            work_dir=work_dir,
            compile_timeout=compile_timeout,
            enable_msvc_i64_compat=enable_msvc_i64_compat,
        )
        if compile_result is not None and (
            compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0
        ):
            return problem, compile_result, None, _failed_case_results(normalized_case_ids, VERDICT_CE, case_map)
        if solution_runner is None:
            return problem, compile_result, None, _failed_case_results(normalized_case_ids, VERDICT_ERROR, case_map)

        checker_compile_result = None
        checker_bin = None
        checker_argv = None
        if problem.checker is not None:
            _copy_problem_files(problem_dir, work_dir, [problem.checker.source] + problem.checker.files)
            checker_src = work_dir / problem.checker.source
            checker_bin = work_dir / 'checker'
            checker_argv = list(problem.checker.argv)
            checker_compile_result = await _compile_cpp(
                checker_src,
                checker_bin,
                compile_timeout,
                work_dir,
                extra_flags=['-O2', f'-I{work_dir}'],
            )
            if checker_compile_result.status != CommandRunStatus.Finished or checker_compile_result.return_code != 0:
                return (
                    problem,
                    compile_result,
                    checker_compile_result,
                    _failed_case_results(normalized_case_ids, VERDICT_CHECKER_CE, case_map),
                )

        case_results: List[MountedOJCaseResult] = []
        for case_id in normalized_case_ids:
            case = case_map[case_id]
            case_max_score = float(case.score)
            input_path = _resolve_under(problem_dir, case.input)
            answer_path = _resolve_under(problem_dir, case.answer)
            case_dir = work_dir / f'case_{case_id}'
            checker_input_path, checker_answer_path, output_path = _prepare_case_dir(case_dir, input_path, answer_path)

            run_result = await solution_runner(
                stdin_path=input_path,
                output_path=output_path,
                timeout=effective_run_timeout,
                memory_limit_mb=effective_memory_limit,
                cpu_limit_s=cpu_limit_s,
            )
            run_verdict = _run_verdict(run_result)
            if run_verdict != VERDICT_AC:
                case_results.append(
                    MountedOJCaseResult(
                        case_id=case_id,
                        passed=False,
                        verdict=run_verdict,
                        score=0.0,
                        max_score=case_max_score,
                        input_path=case.input,
                        answer_path=case.answer,
                        run_result=run_result,
                    ))
                continue

            check_result = None
            verdict = VERDICT_AC
            if checker_bin is not None and checker_argv is not None:
                resolved_checker_argv = _resolve_checker_argv(
                    checker_argv,
                    checker_input_path,
                    output_path,
                    checker_answer_path,
                )
                check_result = await _run_checker_binary(
                    checker_bin,
                    case_dir,
                    resolved_checker_argv,
                    timeout=checker_timeout,
                    memory_limit_mb=effective_memory_limit,
                    # Checkers are trusted problem assets; avoid killing them with contestant CPU limits.
                    cpu_limit_s=None,
                )
                verdict = _checker_verdict(check_result)
            elif not _plain_compare_files(output_path, answer_path):
                verdict = VERDICT_WA

            case_results.append(
                MountedOJCaseResult(
                    case_id=case_id,
                    passed=verdict == VERDICT_AC,
                    verdict=verdict,
                    score=case_max_score if verdict == VERDICT_AC else 0.0,
                    max_score=case_max_score,
                    input_path=case.input,
                    answer_path=case.answer,
                    run_result=run_result,
                    check_result=check_result,
                ))

        return problem, compile_result, checker_compile_result, case_results


async def run_program_from_disk(
    data_root: Path,
    problem_id: str,
    language: str,
    compile_timeout: float,
    run_timeout: float,
    memory_limit_mb: Optional[int] = None,
    stdin: Optional[str] = None,
    argv: Optional[List[str]] = None,
    code: Optional[str] = None,
    source_path: Optional[str] = None,
    problem_files: Optional[List[str]] = None,
    fetch_files: Optional[List[str]] = None,
    save_stdout_path: Optional[str] = None,
    return_stdout: bool = False,
    enable_msvc_i64_compat: bool = False,
) -> tuple[MountedOJProblemSpec, Optional[CommandRunResult], Optional[CommandRunResult], Dict[str, str]]:
    problem_dir, problem, _ = load_problem_spec(data_root, problem_id)
    effective_memory_limit = problem.memory_limit_mb if memory_limit_mb is None else memory_limit_mb

    source_code = code
    additional_files = list(problem_files or [])
    if source_code is None:
        if not source_path:
            raise ValueError('either code or source_path must be provided')
        resolved_source = _resolve_under(problem_dir, source_path)
        if not resolved_source.is_file():
            raise FileNotFoundError(f'source file not found: {resolved_source}')
        source_code = resolved_source.read_text(encoding='utf-8')
        additional_files.append(source_path)

    fetch_files = list(fetch_files or [])
    for rel_path in additional_files + fetch_files:
        _validate_identifier(rel_path.replace('/', '_').replace('\\', '_'), 'relative_path_guard')

    save_stdout_resolved: Optional[Path] = None
    if save_stdout_path:
        save_stdout_resolved = _resolve_under(problem_dir, save_stdout_path)
        save_stdout_resolved.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir_name:
        work_dir = Path(tmp_dir_name)
        _copy_problem_files(problem_dir, work_dir, problem.shared_files)
        if additional_files:
            _copy_problem_files(problem_dir, work_dir, additional_files)

        compile_result, program_runner = await _prepare_solution_runner(
            language=language,
            code=source_code,
            work_dir=work_dir,
            compile_timeout=compile_timeout,
            enable_msvc_i64_compat=enable_msvc_i64_compat,
        )
        if compile_result is not None and (
            compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0
        ):
            return problem, compile_result, None, {}
        if program_runner is None:
            return problem, compile_result, None, {}

        stdin_path = work_dir / 'stdin.txt'
        output_path = work_dir / 'stdout.txt'
        stdin_path.write_text(stdin or '', encoding='utf-8')
        if output_path.exists():
            output_path.unlink()

        run_result = await program_runner(
            stdin_path=stdin_path,
            output_path=output_path,
            timeout=run_timeout,
            memory_limit_mb=effective_memory_limit,
            cpu_limit_s=None,
            argv=argv or [],
        )

        if (
            save_stdout_resolved is not None and
            output_path.is_file()
        ):
            save_stdout_resolved.write_bytes(output_path.read_bytes())

        if run_result is not None and not return_stdout:
            run_result = run_result.model_copy(update={'stdout': None})

        encoded_files: Dict[str, str] = {}
        for rel_path in fetch_files:
            resolved = _resolve_under(work_dir, rel_path)
            if not resolved.is_file():
                continue
            encoded_files[rel_path] = base64.b64encode(resolved.read_bytes()).decode('utf-8')

        return problem, compile_result, run_result, encoded_files


async def run_generator_from_paths(
    data_root: Path,
    generator_path: str,
    testlib_path: str,
    argv: List[str],
    output_path: str,
    compile_timeout: float,
    run_timeout: float,
    memory_limit_mb: int = -1,
) -> tuple[Optional[CommandRunResult], Optional[CommandRunResult], str]:
    resolved_generator_path = resolve_generation_file_path(data_root, generator_path, 'generator_path')
    resolved_output_path = resolve_generation_file_path(data_root, output_path, 'output_path')
    resolved_testlib_path = Path(testlib_path).expanduser().resolve()
    if not resolved_generator_path.is_file():
        raise FileNotFoundError(f'generator source not found: {resolved_generator_path}')
    if not resolved_testlib_path.is_file():
        raise FileNotFoundError(f'testlib.h not found: {resolved_testlib_path}')

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir_name:
        work_dir = Path(tmp_dir_name)
        local_generator_path = work_dir / 'generator.cpp'
        local_generator_path.write_bytes(resolved_generator_path.read_bytes())
        (work_dir / 'testlib.h').write_bytes(resolved_testlib_path.read_bytes())

        compile_result = await _compile_cpp(
            local_generator_path,
            work_dir / 'gen',
            compile_timeout,
            work_dir,
            memory_limit_mb=memory_limit_mb,
            extra_flags=['-O2', '-DONLINE_JUDGE', f'-I{work_dir}'],
        )
        if compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0:
            return compile_result, None, ''

        stdin_path = work_dir / 'stdin.txt'
        stdin_path.write_text('', encoding='utf-8')
        stdout_path = work_dir / 'stdout.txt'
        if stdout_path.exists():
            stdout_path.unlink()

        run_result = await _run_binary(
            work_dir / 'gen',
            work_dir,
            stdin_path=stdin_path,
            output_path=stdout_path,
            timeout=run_timeout,
            memory_limit_mb=memory_limit_mb,
            cpu_limit_s=None,
            argv=argv or [],
        )

        output_text = ''
        if stdout_path.is_file():
            output_bytes = stdout_path.read_bytes()
            resolved_output_path.write_bytes(output_bytes)
            output_text = output_bytes.decode('utf-8', errors='replace')

        return compile_result, run_result, output_text


async def run_solution_cases_from_dir(
    data_root: Path,
    code: str,
    language: str,
    input_path: str,
    compile_timeout: float,
    run_timeout: float,
    memory_limit_mb: int = -1,
    enable_msvc_i64_compat: bool = False,
) -> tuple[Optional[CommandRunResult], List[Dict[str, object]]]:
    resolved_input_dir = resolve_generation_file_path(data_root, input_path, 'input_path')
    if not resolved_input_dir.is_dir():
        raise FileNotFoundError(f'input_path is not a directory: {resolved_input_dir}')

    case_files = sorted(
        [path for path in resolved_input_dir.iterdir() if path.is_file() and path.suffix == '.in'],
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem),
    )
    if not case_files:
        raise FileNotFoundError(f'no .in files found under input_path: {resolved_input_dir}')

    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir_name:
        work_dir = Path(tmp_dir_name)
        compile_result, solution_runner = await _prepare_solution_runner(
            language=language,
            code=code,
            work_dir=work_dir,
            compile_timeout=compile_timeout,
            enable_msvc_i64_compat=enable_msvc_i64_compat,
        )
        if compile_result is not None and (
            compile_result.status != CommandRunStatus.Finished or compile_result.return_code != 0
        ):
            return compile_result, []
        if solution_runner is None:
            return compile_result, []

        case_results: List[Dict[str, object]] = []
        for case_file in case_files:
            output_path = work_dir / f'{case_file.stem}.stdout'
            if output_path.exists():
                output_path.unlink()
            run_result = await solution_runner(
                stdin_path=case_file,
                output_path=output_path,
                timeout=run_timeout,
                memory_limit_mb=memory_limit_mb,
                cpu_limit_s=None,
            )
            output_text = ''
            if output_path.is_file():
                output_text = output_path.read_text(encoding='utf-8', errors='replace')
            case_results.append({
                'case_id': case_file.stem,
                'success': (
                    run_result.status == CommandRunStatus.Finished and
                    (run_result.return_code or 0) == 0
                ),
                'output': output_text,
                'error': '' if (
                    run_result.status == CommandRunStatus.Finished and
                    (run_result.return_code or 0) == 0
                ) else (run_result.stderr or ''),
                'run_result': run_result,
            })

        return compile_result, case_results
