import os.path
import os
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError
from numpy.distutils.misc_util import appendpath
from numpy.distutils import log

def have_cython():
  try:
    import Cython
  except ImportError:
    return False
  return True

def generate_a_cython_source(self, base, ext_name, source, extension):
        if self.inplace or not have_cython():
            target_dir = os.path.dirname(base)
        else:
            target_dir = appendpath(self.build_src, os.path.dirname(base))
        target_file = os.path.join(target_dir, ext_name + '.c')
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target_file, 'newer'):
            if have_cython():
                import Cython.Compiler.Main
                log.info("cythonc:> %s: %s " % (target_dir, target_file))
                log.info("cwd %s " % (os.getcwd()))
                self.mkpath(target_dir)
                options = Cython.Compiler.Main.CompilationOptions(
                    defaults=Cython.Compiler.Main.default_options,
                    include_path=extension.include_dirs,
                    output_file=target_file)
                #log.info('\n'.join([s + ' ' + str(getattr(options, s)) for s in dir(options)]))
                # avoid calling compile_single, because it will give wrong module names.
                cython_result = Cython.Compiler.Main.compile([source],
                                                           options=options)
                if cython_result.num_errors != 0:
                    raise DistutilsError("%d errors while compiling %r with Cython" \
                          % (cython_result.num_errors, source))
            elif os.path.isfile(target_file):
                log.warn("Cython required for compiling %r but not available,"\
                         " using old target %r"\
                         % (source, target_file))
            else:
                raise DistutilsError("Cython required for compiling %r"\
                                     " but notavailable" % (source,))
        return target_file

from numpy.distutils.command import build_src
build_src.build_src.generate_a_pyrex_source = generate_a_cython_source
