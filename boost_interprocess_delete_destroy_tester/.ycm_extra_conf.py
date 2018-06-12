def FlagsForFile( filename, **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-std=c++14', '-I/usr/local/boost-1.60.0/include', '-L/usr/local/boost-1.60.0/lib', '-pthread' ],
}
