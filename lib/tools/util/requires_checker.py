import functools
import subprocess
from importlib import import_module


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
        'found, please install them first.'):
    """A decorator factory to check if prerequisites are satisfied.

    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.

    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call('which {}'.format(cmd), shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """A decorator to check if some python packages are installed.

    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """A decorator to check if some executable files are installed.

    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)