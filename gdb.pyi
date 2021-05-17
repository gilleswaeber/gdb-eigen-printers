"""
gdb.pyi - gdb Python API typing file

Source: https://github.com/metal-ci/test @312dfbec (08.09.2020)
------------------------------------
Copyright 2020 metal.ci

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from typing import List, Tuple, Union, Optional, Iterator, Callable, Dict, overload

PYTHONDIR : str

def execute(command, from_tty=False, to_string=False) -> None: ...
def breakpoints() -> List[Breakpoint]: ...
def rbreak (regex, minsyms=False, throttle=None, symtabs:List[Symtab] = [])  -> List[Breakpoint]: ...
def parameter(parameter: str):...
def history(number: int) -> Value : ...
def convenience_variable(name: str) -> Union[None,Value]  : ...
def set_convenience_variable(name: str, value) -> None: ...
def parse_and_eval(expression: str) -> Value: ...
def find_pc_line(pc)-> Symtab_and_line : ...
def post_event(event: callable) -> None :...

class __stream: ...
STDOUT : __stream
STDERR : __stream
STDLOG : __stream

def write(string: str, stream:__stream = STDOUT): ...
def flush(stream:__stream = STDOUT): ...
def target_charset() -> str: ...
def target_wide_charset () -> str: ...
def solib_name(address: int) -> str: ...
def decode_line(expression: str = None) -> Tuple[str, Union[None,List[Symtab_and_line]]] : ...
def prompt_hook(current_prompt): ...


class error(RuntimeError):...
class MemoryError(error): ...
class GdbError(Exception): ...

class Value:
    address: Optional[Value]
    is_optimized_out: bool
    type: Type
    is_lazy: bool
    def __init__(self, value, type: Type = None): ...

    def cast(self, type: Type) -> Value:...
    def dereference(self) -> Value: ...
    def referenced_value (self) -> Value: ...
    def reference_value  (self) -> Value: ...
    def const_value  (self) -> Value: ...
    def dynamic_cast (self, type: Type) -> Value:...
    def reinterpret_cast (self, type: Type) -> Value:...
    def format_string (self, *args) -> str: ...
    def string(self, encoding:str=None, errors=None, length: int=None) -> str:...
    def lazy_string (self, encoding:str=None, length: int=None) -> str:...
    def fetch_lazy (self) -> None: ...

    def __getitem__(self, item) -> Value: ...
    def __add__(self, other) -> Value: ...
    def __radd__(self, other) -> Value: ...
    def __sub__(self, other) -> Value: ...
    def __rsub__(self, other) -> Value: ...
    def __int__(self) -> int: ...
    def __float__(self) -> int: ...

def lookup_type(name, block: Block = None): Type

class Type:
    alignof: int
    code: __type_code
    name: Optional[str]
    sizeof: int
    tag: Optional[str]
    objfile: Optional[Objfile]
    def fields(self) -> List[Field]: ...
    def array(self, n1: int, n2:int = None) -> Type:...
    def vector(self, n1: int, n2:int = None) -> Type:...
    def const(self) -> Type: ...
    def volatile(self) -> Type: ...
    def unqualified (self) -> Type: ...
    def range(self) -> Tuple[Type, Type]: ...
    def reference(self) -> Type: ...
    def pointer(self) -> Type: ...
    def strip_typedefs(self) -> Type: ...
    def target (self) -> Type: ...
    def template_argument (self, n: int, block:Block = None) -> Union[Value, Type]: ...
    def optimized_out (self) -> Value: ...

class Field:
    bitpos: int
    enumval: Optional[int]
    name: Optional[str]
    artificial: bool
    is_base_class: bool
    bitsize: int
    type: Optional[Type]
    parent_type: Optional[Type]

class __type_code:...

TYPE_CODE_PTR : __type_code
TYPE_CODE_ARRAY : __type_code
TYPE_CODE_STRUCT : __type_code
TYPE_CODE_UNION : __type_code
TYPE_CODE_ENUM : __type_code
TYPE_CODE_FLAGS : __type_code
TYPE_CODE_FUNC : __type_code
TYPE_CODE_INT : __type_code
TYPE_CODE_FLT : __type_code
TYPE_CODE_VOID : __type_code
TYPE_CODE_SET : __type_code
TYPE_CODE_RANGE : __type_code
TYPE_CODE_STRING : __type_code
TYPE_CODE_BITSTRING : __type_code
TYPE_CODE_ERROR : __type_code
TYPE_CODE_METHOD : __type_code
TYPE_CODE_METHODPTR : __type_code
TYPE_CODE_MEMBERPTR : __type_code
TYPE_CODE_REF : __type_code
TYPE_CODE_RVALUE_REF : __type_code
TYPE_CODE_CHAR : __type_code
TYPE_CODE_BOOL : __type_code
TYPE_CODE_COMPLEX : __type_code
TYPE_CODE_TYPEDEF : __type_code
TYPE_CODE_NAMESPACE : __type_code
TYPE_CODE_DECFLOAT : __type_code
TYPE_CODE_INTERNAL_FUNCTION : __type_code

def default_visualizer (value: Value): ...
pretty_printers: list

class FrameFilter:
    def filter(self, iterator: Iterator): ...
    name: str
    enabled: bool
    priority: int

def inferiors() -> List[Inferior]: ...
def selected_inferior() -> Inferior: ...

class Inferior:
    num: ...
    pid: int
    was_attached: bool
    progspace: Progspace
    def is_valid (self) -> bool: ...
    def threads(self) -> List[InferiorThread]: ...
    def architecture(self) -> Architecture: ...
    def read_memory  (self, address: int, length: int) -> memoryview: ...
    def write_memory (self, address: int, buffer: Union[str, bytes], length: int) -> None: ...
    def search_memory(self, address: int, length: int, pattern: Union[str, bytes]) -> int: ...
    def thread_from_handle(self, handle) -> InferiorThread: ...


class ThreadEvent: inferior_thread: Optional[InferiorThread]
class ContinueEvent(ThreadEvent): ...
class ExitedEvent: exit_code: Optional[int]; inferior: Inferior
class StopEvent(ThreadEvent): pass
class SignalEvent(StopEvent): stop_signal: str
class BreakpointEvent(SignalEvent): breakpoints: List[Breakpoint]; breakpoint: Breakpoint
class ClearObjFilesEvent: progspace: Progspace
class InferiorCallPreEvent:  tpid: int; address: int
class InferiorCallPostEvent: tpid: int; address: int
class MemoryChangedEvent: address: int; lenth: int
class RegisterChangedEvent: frame: Frame; regnum: int
class NewInferiorEvent:     inferior: Inferior
class InferiorDeletedEvent: inferior: Inferior
class NewThreadEvent(ThreadEvent): inferior_thread: InferiorThread
class NewObjFileEvent: new_objfile: Objfile

class EventRegistry:
    def connect(self, object) -> None: ...
    def disconnect(self, object) -> None: ...

class __events:
    class __cont(EventRegistry):
        def connect   (self, object: Callable[[ContinueEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[ContinueEvent], None]) -> None: ...
    cont: __cont

    class __exited(EventRegistry):
        def connect   (self, object: Callable[[ExitedEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[ExitedEvent], None]) -> None: ...
    exited: __exited

    class __stop(EventRegistry):
        def connect   (self, object: Callable[[Union[SignalEvent, BreakpointEvent]], None]) -> None: ...
        def disconnect(self, object: Callable[[Union[SignalEvent, BreakpointEvent]], None]) -> None: ...
    stop: __stop

    class __new_objfile(EventRegistry):
        def connect   (self, object: Callable[[NewObjFileEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[NewObjFileEvent], None]) -> None: ...
    new_objfile: __new_objfile

    class __clear_objfiles(EventRegistry):
        def connect   (self, object: Callable[[ClearObjFilesEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[ClearObjFilesEvent], None]) -> None: ...
    clear_objfiles: __clear_objfiles

    class __inferior_call(EventRegistry):
        def connect   (self, object: Callable[[Union[InferiorCallPreEvent, InferiorCallPostEvent]], None]) -> None: ...
        def disconnect(self, object: Callable[[Union[InferiorCallPreEvent, InferiorCallPostEvent]], None]) -> None: ...
    inferior_call: __inferior_call

    class __memory_changed(EventRegistry):
        def connect   (self, object: Callable[[MemoryChangedEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[MemoryChangedEvent], None]) -> None: ...
    memory_changed: __memory_changed

    class __breakpoint(EventRegistry):
        def connect   (self, object: Callable[[Breakpoint], None]) -> None: ...
        def disconnect(self, object: Callable[[Breakpoint], None]) -> None: ...
    breakpoint_created:__breakpoint
    breakpoint_modified:__breakpoint
    breakpoint_deleted:__breakpoint

    class __before_prompt(EventRegistry):
        def connect   (self, object: Callable[[], None]) -> None: ...
        def disconnect(self, object: Callable[[], None]) -> None: ...
    before_prompt: __before_prompt

    class __new_inferior(EventRegistry):
        def connect   (self, object: Callable[[NewInferiorEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[NewInferiorEvent], None]) -> None: ...
    new_inferior: __new_inferior

    class __inferior_deleted(EventRegistry):
        def connect   (self, object: Callable[[InferiorDeletedEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[InferiorDeletedEvent], None]) -> None: ...
    inferior_deleted: __inferior_deleted

    class __new_thread(EventRegistry):
        def connect   (self, object: Callable[[NewThreadEvent], None]) -> None: ...
        def disconnect(self, object: Callable[[NewThreadEvent], None]) -> None: ...
    new_thread: __new_thread

events: __events

def selected_thread() -> InferiorThread : ...

class InferiorThread:
    name: Optional[str]
    num: int
    global_num: int
    ptid: int
    inferior: Inferior
    def is_valid(self) -> bool : ...
    def switch(self) -> None: ...
    def is_stopped(self) -> bool : ...
    def is_running(self) -> bool : ...
    def is_exited(self) -> bool : ...
    def handle(self) -> bytes : ...

def start_recording(method: str=None, format: str=None) -> Record : ...
def current_recording() -> Optional[Record]: ...
def stop_recording() -> None: ...

class Record:
    method: str
    format: str
    begin: Instruction
    end: Instruction
    replay_position: Optional[Instruction]
    instruction_history: List[Instruction]
    function_call_history: List[RecordFunctionSegment]
    def goto(self, instruction: Instruction) -> None: ...

class Instruction:
    pc: int
    data: memoryview
    decoded: str
    size: int

class RecordInstruction(Instruction):
    number: int
    sal: Symtab_and_line
    is_speculative: bool

class RecordGap:
    number: int
    error_code: int
    error_string: str

class RecordFunctionSegment:
    number: int
    symbol: Symbol
    level: Optional[int]
    instructions: List[Union[RecordInstruction , RecordGap]]
    up: Optional[RecordFunctionSegment]
    prev: Optional[RecordFunctionSegment]
    next: Optional[RecordFunctionSegment]

class Command:
    def __init__(self, name:str, command_class:__command_class=None, completer_class: __complete_class=None, prefix:bool=None): ...
    def dont_repeat(self)->None:...
    def invoke(self, argument:str, from_tty:bool): ...
    def complete(self, text:str, word: str):...

class __command_class: ...

COMMAND_NONE : __command_class
COMMAND_RUNNING : __command_class
COMMAND_DATA : __command_class
COMMAND_STACK : __command_class
COMMAND_FILES : __command_class
COMMAND_SUPPORT : __command_class
COMMAND_STATUS : __command_class
COMMAND_BREAKPOINTS : __command_class
COMMAND_TRACEPOINTS : __command_class
COMMAND_USER : __command_class
COMMAND_OBSCURE : __command_class
COMMAND_MAINTENANCE : __command_class

class __complete_class: ...
COMPLETE_NONE : __complete_class
COMPLETE_FILENAME : __complete_class
COMPLETE_LOCATION : __complete_class
COMPLETE_COMMAND : __complete_class
COMPLETE_SYMBOL : __complete_class
COMPLETE_EXPRESSION : __complete_class


class Parameter:
    def __init__ (self, name: str, command_class: __command_class, parameter_class: __parameter_class, enum_sequence: List[str] = None): ...
    set_doc: str
    show_doc: str
    value:...

    def get_set_string(self) -> None:...
    def get_show_string(self, svalue: str) -> None: ...

class __parameter_class: pass

PARAM_BOOLEAN : __parameter_class
PARAM_AUTO_BOOLEAN : __parameter_class
PARAM_UINTEGER : __parameter_class
PARAM_INTEGER : __parameter_class
PARAM_STRING : __parameter_class
PARAM_STRING_NOESCAPE : __parameter_class
PARAM_OPTIONAL_FILENAME : __parameter_class
PARAM_FILENAME : __parameter_class
PARAM_ZINTEGER : __parameter_class
PARAM_ZUINTEGER : __parameter_class
PARAM_ZUINTEGER_UNLIMITED : __parameter_class
PARAM_ENUM : __parameter_class

class Function:
    def __init__(self, name: str): ...
    def invoke(self, *args): ...

def current_progspace () -> Progspace: ...
def progspaces() -> List[Progspace]: ...

class Progspace:
    filename: str
    pretty_printers: list
    type_printers: list
    frame_filters: Dict[str, FrameFilter]
    def block_for_pc(self, pc: int) -> Optional[Block]: ...
    def find_pc_line (self, pc: int) -> Optional[Symtab_and_line ]: ...
    def is_valid(self) -> bool: ...
    def objfiles(self) -> List[Objfile]: ...
    def solid_name(self, address: int) -> str:...

def current_objfile() -> Objfile: ...
def objfiles() -> List[Objfile]: ...
def lookup_objfile(name: Union[str, int], by_build_id: bool = None) -> Objfile: ...

class Objfile:
    filename: Optional[str]
    username: Optional[str]
    owner: Optional[Objfile]
    build_id: Optional[str]
    progspace: Progspace
    pretty_printers: list
    type_printers: list
    frame_filters: List[FrameFilter]
    def is_valid(self) -> bool: ...
    def add_separate_debug_file (self, file: str) -> None: ...
    def lookup_global_symbol(self, name: str, domain: str = None) -> Optional[Symbol]: ...
    def lookup_static_symbol(self, name: str, domain: str = None) -> Optional[Symbol]: ...

def selected_frame()-> Frame:...
def newest_frame ()-> Frame:...
def frame_stop_reason_string (reason: Union[int, '__frame_unwind_stop_reason']) -> str: ...
def invalidate_cached_frames() -> None: ...

class Frame:
    def is_valid(self) -> bool: ...
    def name(self) -> str: ...
    def architecture(self) -> Architecture: ...
    def type(self) -> __frame_type:...
    def unwind_stop_reason(self) -> __frame_unwind_stop_reason: ...
    def pc(self) -> int: ...
    def block(self) -> Block :...
    def function(self) -> Symbol: ...
    def older(self) -> Frame: ...
    def newer(self) -> Frame: ...
    def find_sal(self) -> Symtab_and_line: ...
    def read_register(self, register: str) -> Value:...
    def read_var(self, variable: Union[Symbol, str], block: Block = None) -> Symbol :...
    def select(self) -> None: ...

class __frame_type: ...
NORMAL_FRAME :__frame_type
DUMMY_FRAME :__frame_type
INLINE_FRAME :__frame_type
TAILCALL_FRAME :__frame_type
SIGTRAMP_FRAME :__frame_type
ARCH_FRAME :__frame_type
SENTINEL_FRAME :__frame_type

class __frame_unwind_stop_reason: ...
FRAME_UNWIND_NO_REASON: __frame_unwind_stop_reason
FRAME_UNWIND_NULL_ID: __frame_unwind_stop_reason
FRAME_UNWIND_OUTERMOST: __frame_unwind_stop_reason
FRAME_UNWIND_UNAVAILABLE: __frame_unwind_stop_reason
FRAME_UNWIND_INNER_ID: __frame_unwind_stop_reason
FRAME_UNWIND_SAME_ID: __frame_unwind_stop_reason
FRAME_UNWIND_NO_SAVED_PC: __frame_unwind_stop_reason
FRAME_UNWIND_MEMORY_ERROR: __frame_unwind_stop_reason
FRAME_UNWIND_FIRST_ERROR: __frame_unwind_stop_reason

def block_for_pc(pc: int) -> Block: ...

class Block:
    def is_valid(self) -> bool: ...
    def __getitem__(self, idx: int) -> Symbol:...
    start: int
    end: int
    function: Symbol
    superblock: Optional[Block]
    global_block: Block
    static_block: Block
    is_global: bool
    is_static: bool



def lookup_symbol(name: str, block: Block=None, domain: __domain_type = None)-> Symbol: ...
def lookup_global_symbol(name: str, domain: __domain_type = None)-> Symbol: ...
def lookup_static_symbol(name: str, domain: __domain_type = None)-> Symbol: ...

class Symbol:
    type: Optional[Type]
    symtab: Symtab
    line: int
    name: str
    linkage_name: str
    print_name: str
    addr_class: __symbol_address
    needs_frame: bool
    is_argument: bool
    is_constant: bool
    is_function: bool
    is_variable: bool

    def is_valid(self) -> bool: ...
    def value(self, frame:Frame= None) -> Value:...


class __domain_type:...
SYMBOL_UNDEF_DOMAIN: __domain_type
SYMBOL_VAR_DOMAIN: __domain_type
SYMBOL_STRUCT_DOMAIN: __domain_type
SYMBOL_LABEL_DOMAIN: __domain_type
SYMBOL_MODULE_DOMAIN: __domain_type
SYMBOL_COMMON_BLOCK_DOMAIN: __domain_type

class __symbol_address:...
SYMBOL_LOC_UNDEF : __symbol_address
SYMBOL_LOC_CONST : __symbol_address
SYMBOL_LOC_STATIC : __symbol_address
SYMBOL_LOC_REGISTER : __symbol_address
SYMBOL_LOC_ARG : __symbol_address
SYMBOL_LOC_REF_ARG : __symbol_address
SYMBOL_LOC_REGPARM_ADDR : __symbol_address
SYMBOL_LOC_LOCAL : __symbol_address
SYMBOL_LOC_TYPEDEF : __symbol_address
SYMBOL_LOC_BLOCK : __symbol_address
SYMBOL_LOC_CONST_BYTES : __symbol_address
SYMBOL_LOC_UNRESOLVED : __symbol_address
SYMBOL_LOC_OPTIMIZED_OUT : __symbol_address
SYMBOL_LOC_COMPUTED : __symbol_address
SYMBOL_LOC_COMPUTED : __symbol_address

class Symtab_and_line:
    symtab: Symtab
    pc: int
    last: int
    line: int
    def is_valid(self) -> bool:...

class Symtab:
    filename: str
    objfile: Objfile
    producer: str
    def is_valid(self) -> bool:...
    def fullname(self) -> str: ...

    def global_block(self) -> Block: ...
    def static_block(self) -> Block: ...
    def linetable(self) -> LineTableEntry:...

class LineTableEntry:
    line: int
    pc: int

class LineTable:
    def line(self, line: int) -> List[LineTableEntry]:...
    def has_line(self, line: int) -> bool:...
    def source_lines(self) -> List[int]: ...

class __breakpoint_type:...
BP_BREAKPOINT = __breakpoint_type
BP_WATCHPOINT = __breakpoint_type
BP_HARDWARE_WATCHPOINT = __breakpoint_type
BP_READ_WATCHPOINT = __breakpoint_type
BP_ACCESS_WATCHPOINT = __breakpoint_type

class __watchpoint_type:...
WP_READ = __watchpoint_type
WP_WRITE = __watchpoint_type
WP_ACCESS = __watchpoint_type

class Breakpoint:
    @overload
    def __init__(self, spec:str, type: __breakpoint_type =None, wp_class:__watchpoint_type=WP_WRITE,
                 internal: bool = False, temporary:bool = False, qualified: bool=False):...

    @overload
    def __init__(self, source:str=None, function:str=None, label:str=None, line:int=None,
                 internal: bool = False, temporary:bool = False, qualified: bool=False):...

    def stop(self) -> None:...

    def is_valid(self) -> bool:...
    def delete(self) -> None: ...
    enabled: bool
    silent: bool
    pending: bool

    thread: Optional[int]
    task: Optional[int]
    ignore_count: int
    number: int
    type: __breakpoint_type
    visible: bool
    temporary: bool
    location: Optional[str]
    expression: Optional[str]
    condition: Optional[str]
    commands: Optional[str]

class FinishBreakpoint:
    def __init__(self, frame: Frame=None, internal:bool=False):...
    def out_of_scope(self):...
    return_value: Optional[Value]

class LazyString:
    def value(self)-> Value:...
    address: int
    length: int
    encoding: str
    type: Type

class Architecture:
    def name(self) -> str: ...
    def disassemble(self, start_pc: int, end_pc: int = None, count: int = None) -> dict:...
